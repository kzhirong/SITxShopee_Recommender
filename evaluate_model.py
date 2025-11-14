#!/usr/bin/env python3
"""
Evaluate LLM-CTR Model on Test Set

This script loads a trained LLM-CTR checkpoint and evaluates it on the test set.
Calculates both AUC (Area Under ROC Curve) and Accuracy metrics.

Usage:
    # Evaluate Phase 2 model on x4 test set
    python evaluate_model.py --checkpoint checkpoints/llm_ctr_phase2/best_model.pt --dataset x4 --split test

    # Evaluate Phase 1 model on x1_sample20 test set
    python evaluate_model.py --checkpoint checkpoints/llm_ctr_phase1/best_model.pt --dataset x1_sample20 --split test
"""

import sys
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import argparse
import importlib
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss

# FuxiCTR imports
from fuxictr.features import FeatureMap
from fuxictr.utils import load_config, load_dataset_config
from fuxictr.pytorch.dataloaders import RankDataLoader

# Add DeepFM source to path
sys.path.append('model_zoo/DeepFM/src')
from projector import FeatureProjector


class LLM_CTR_Model(nn.Module):
    """LLM-enhanced CTR prediction model."""

    def __init__(self, embedding_layer, encoder, projector, llm, tokenizer, freeze_encoder=True):
        super().__init__()
        self.embedding_layer = embedding_layer
        self.encoder = encoder
        self.projector = projector
        self.llm = llm
        self.tokenizer = tokenizer

        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Always freeze embeddings and LLM
        for param in self.embedding_layer.parameters():
            param.requires_grad = False
        for param in self.llm.parameters():
            param.requires_grad = False

        # Get token IDs for "0" and "1"
        self.token_0 = tokenizer.encode("0", add_special_tokens=False)[0]
        self.token_1 = tokenizer.encode("1", add_special_tokens=False)[0]

    def forward(self, batch_dict, prompt_embeds):
        """Forward pass through the complete pipeline."""
        batch_size = prompt_embeds.shape[0]

        # 1. Feature IDs → Embeddings
        embedded = self.embedding_layer(batch_dict)  # [batch_size, num_fields, emb_dim]

        # 2. Embeddings → Encoder
        encoder_output = self.encoder(embedded)
        if isinstance(encoder_output, tuple):
            encoded = encoder_output[0]
        else:
            encoded = encoder_output

        # Convert to bfloat16 to match LLM dtype
        encoded = encoded.to(torch.bfloat16)

        # 3. Encoded → Projector
        projected = self.projector(encoded)  # [batch_size, num_fields, llm_dim]

        # 4. Concatenate [text + projected features]
        inputs_embeds = torch.cat([prompt_embeds, projected], dim=1)

        # 5. Feed to LLM
        llm_output = self.llm(inputs_embeds=inputs_embeds, return_dict=True)

        # 6. Get logits for next token prediction at the last position
        lm_logits = llm_output.logits  # [batch_size, seq_len, vocab_size]
        last_token_logits = lm_logits[:, -1, :]  # [batch_size, vocab_size]

        # Extract logits for tokens "0" and "1"
        logit_0 = last_token_logits[:, self.token_0]  # [batch_size]
        logit_1 = last_token_logits[:, self.token_1]  # [batch_size]

        # Stack into [batch_size, 2] for cross-entropy
        logits = torch.stack([logit_0, logit_1], dim=1)

        return logits


def prepare_prompt_embeddings(tokenizer, llm, prompt_template, batch_size, device):
    """Pre-compute and cache prompt embeddings for efficiency."""
    tokens = tokenizer(prompt_template, return_tensors="pt").to(device)

    with torch.no_grad():
        # Get embeddings for the prompt
        prompt_embeds = llm.get_input_embeddings()(tokens.input_ids)  # [1, seq_len, llm_dim]
        # Replicate for batch
        prompt_embeds = prompt_embeds.repeat(batch_size, 1, 1)  # [batch_size, seq_len, llm_dim]

    return prompt_embeds


def evaluate_testset(model, dataloader, device, feature_names, prompt_embeds_cache, dataset_name, return_raw=False):
    """Evaluate on test set and calculate AUC + Accuracy.

    Args:
        return_raw: If True, return raw predictions instead of computing metrics
    """
    model.eval()

    all_labels = []
    all_probs = []  # For AUC
    all_preds = []  # For Accuracy
    total_loss = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Evaluating {dataset_name} testset")

        for batch in pbar:
            # Move batch to device
            batch_dict = {feat: batch[feat].to(device) for feat in feature_names}
            labels = batch['label'].long().squeeze().to(device)

            # Get cached prompt embeddings
            batch_size = labels.shape[0]
            prompt_embeds = prompt_embeds_cache[:batch_size]

            # Forward pass
            logits = model(batch_dict, prompt_embeds)  # [batch_size, 2]
            loss = F.cross_entropy(logits, labels)

            # Get probabilities and predictions
            probs = F.softmax(logits, dim=1)[:, 1]  # Probability of class 1 (click)
            preds = logits.argmax(dim=1)

            # Collect results (convert bfloat16 to float32 first for NumPy compatibility)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.float().cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            total_loss += loss.item()

            # Update progress
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)

    # If return_raw, just return the raw data for combining later
    if return_raw:
        return {
            'labels': all_labels,
            'probs': all_probs,
            'preds': all_preds,
            'total_loss': total_loss,
            'num_batches': len(dataloader)
        }

    # Calculate metrics
    auc = roc_auc_score(all_labels, all_probs)
    accuracy = (all_preds == all_labels).mean() * 100
    logloss = log_loss(all_labels, all_probs)
    avg_loss = total_loss / len(dataloader)

    return {
        'auc': auc,
        'accuracy': accuracy,
        'logloss': logloss,
        'loss': avg_loss,
        'num_samples': len(all_labels)
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate LLM-CTR model on test set')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (e.g., checkpoints/llm_ctr_phase2/best_model.pt)')
    parser.add_argument('--dataset', type=str, default='x4',
                       help='Dataset to evaluate on. Use "all" for x1+x2+x4 combined, or specify: x1, x2, x4, x1_sample20, x2_sample20')
    parser.add_argument('--split', type=str, default='test', choices=['test', 'valid'],
                       help='Which split to evaluate on')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for evaluation')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID (-1 for CPU)')
    parser.add_argument('--clear-checkpoints', action='store_true',
                       help='Clear previous evaluation checkpoints and start fresh')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                       help='Directory to save evaluation checkpoints (e.g., /content/drive/MyDrive/eval_checkpoints). If not specified, saves next to model checkpoint.')

    args = parser.parse_args()

    # Handle "all" dataset option - evaluate on x1, x2, x4 combined
    if args.dataset == 'all':
        datasets_to_eval = ['x1', 'x2', 'x4']
        eval_combined = True
    else:
        datasets_to_eval = [args.dataset]
        eval_combined = False

    # Setup device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        # Enable optimizations
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print("=" * 80)
    print("LLM-CTR MODEL EVALUATION")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Dataset: {args.dataset}{' (x1 + x2 + x4 combined)' if eval_combined else ''}")
    print(f"  Split: {args.split}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Device: {device}")

    # Load checkpoint
    print(f"\n  Loading checkpoint...")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    print(f"  ✓ Checkpoint loaded")
    print(f"    Trained on dataset: {checkpoint.get('dataset', 'unknown')}")
    print(f"    Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"    Val Acc: {checkpoint.get('val_acc', 0):.2f}%")
    print(f"    Val Loss: {checkpoint.get('val_loss', 0):.4f}")

    # Get configuration from checkpoint
    prompt_template = checkpoint.get('prompt_template',
        "Based on the user's browsing behavior and ad interaction features, predict if they will click on this advertisement. Answer with 1 for click or 0 for no click:")
    feature_names = checkpoint.get('feature_names',
        [f'feat_{i}' for i in range(1, 22)] + ['hour', 'weekday', 'weekend'])

    # Load baseline model configuration
    print("\n" + "-" * 80)
    print("Loading model architecture")
    print("-" * 80)

    config_path = 'model_zoo/DeepFM/config'
    experiment_id = 'DeepFM_avazu_normalized'
    params = load_config(config_path, experiment_id)

    # Fix paths
    for key in ['data_root', 'test_data', 'train_data', 'valid_data']:
        if key in params and params[key]:
            params[key] = params[key].replace('../../', '')

    # Load unified feature map
    unified_data_dir = os.path.join(params['data_root'], 'avazu_unified')
    unified_feature_map_json = os.path.join(unified_data_dir, "feature_map.json")

    if not os.path.exists(unified_feature_map_json):
        raise FileNotFoundError(f"Unified feature map not found: {unified_feature_map_json}\n"
                              f"Please run training first to create it.")

    feature_map = FeatureMap('avazu_unified', unified_data_dir)
    feature_map.load(unified_feature_map_json, params)

    # Initialize DeepFM model (for embedding layer and encoder architecture)
    model_src_path = "model_zoo.DeepFM.src"
    src = importlib.import_module(model_src_path)
    model_class = getattr(src, params['model'])

    # Use x4_normalized for analyzer initialization
    model_params = params.copy()
    x4_data_dir = os.path.join(params['data_root'], 'avazu_x4_normalized')
    model_params['train_data'] = os.path.join(x4_data_dir, 'train.parquet')
    model_params['valid_data'] = os.path.join(x4_data_dir, 'valid.parquet')
    model_params['test_data'] = os.path.join(x4_data_dir, 'test.parquet')

    deepfm_model = model_class(feature_map, **model_params)
    embedding_layer = deepfm_model.embedding_layer
    encoder = deepfm_model.gen

    # Load LLM
    print("\n  Loading LLM (Qwen3-0.6B)...")

    # Try to use FlashAttention 2 if available and supported by GPU
    # Otherwise fall back to default attention
    llm_kwargs = {"torch_dtype": torch.bfloat16}
    use_flash_attn = False

    if device.type == 'cuda':
        try:
            import flash_attn  # noqa: F401 - Only checking if available, not using directly
            llm_kwargs["attn_implementation"] = "flash_attention_2"
            use_flash_attn = True
            print("  Attempting to use FlashAttention 2...")
        except ImportError:
            print("  FlashAttention 2 not installed, using default attention")
    else:
        print(f"  Running on {device}, using default attention")

    # Try loading with FlashAttention first, fallback if GPU doesn't support it
    try:
        llm = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-0.6B",
            **llm_kwargs
        ).to(device)
        if use_flash_attn:
            print("  ✓ Using FlashAttention 2 (GPU optimization enabled)")
    except (RuntimeError, ValueError) as e:
        if use_flash_attn and "flash" in str(e).lower():
            print(f"  ⚠ FlashAttention 2 not supported on this GPU: {e}")
            print("  Falling back to default attention...")
            # Remove FlashAttention and retry
            llm_kwargs.pop("attn_implementation", None)
            llm = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen3-0.6B",
                **llm_kwargs
            ).to(device)
            print("  ✓ Using default attention")
        else:
            raise  # Re-raise if it's a different error

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    print(f"  ✓ LLM loaded")

    # Create projector
    projector = FeatureProjector(
        feature_dim=params['embedding_dim'],
        llm_dim=llm.config.hidden_size,
        hidden_dim=512
    ).to(device).to(torch.bfloat16)

    # Create model
    model = LLM_CTR_Model(
        embedding_layer=embedding_layer,
        encoder=encoder,
        projector=projector,
        llm=llm,
        tokenizer=tokenizer,
        freeze_encoder=True  # Always frozen during evaluation
    ).to(device)

    # Load weights from checkpoint
    print(f"\n  Loading model weights from checkpoint...")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"  ✓ Model weights loaded")

    # Map dataset names to dataset_ids
    dataset_configs = {
        'x1': 'avazu_x1_normalized',
        'x2': 'avazu_x2_normalized',
        'x4': 'avazu_x4_normalized',
        'x1_sample20': 'avazu_x1_sample20',
        'x2_sample20': 'avazu_x2_sample20'
    }

    # Collect results from all datasets
    all_labels_combined = []
    all_probs_combined = []
    all_preds_combined = []
    total_loss_combined = 0
    total_batches = 0
    per_dataset_results = {}

    # Setup checkpoint directory for intermediate results
    if args.checkpoint_dir:
        # Use custom directory (e.g., Google Drive)
        eval_checkpoint_dir = Path(args.checkpoint_dir) / f'eval_checkpoints_{args.dataset}_{args.split}'
        print(f"\n  Using checkpoint directory: {eval_checkpoint_dir}")
    else:
        # Default: save next to model checkpoint
        output_dir = Path(args.checkpoint).parent
        eval_checkpoint_dir = output_dir / f'eval_checkpoints_{args.dataset}_{args.split}'

    # Clear checkpoints if requested
    if args.clear_checkpoints and eval_checkpoint_dir.exists():
        import shutil
        shutil.rmtree(eval_checkpoint_dir)
        print(f"\n  ✓ Cleared previous evaluation checkpoints")

    eval_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Checkpoint directory: {eval_checkpoint_dir}")

    # Evaluate each dataset
    for dataset_name in datasets_to_eval:
        # Check if this dataset was already evaluated (checkpoint exists)
        dataset_checkpoint_file = eval_checkpoint_dir / f'{dataset_name}_results.json'

        if dataset_checkpoint_file.exists() and eval_combined:
            # Try to load and validate checkpoint
            try:
                with open(dataset_checkpoint_file, 'r') as f:
                    cached_results = json.load(f)

                current_checkpoint_time = str(Path(args.checkpoint).stat().st_mtime)
                cached_checkpoint_time = cached_results.get('timestamp', '')

                if current_checkpoint_time != cached_checkpoint_time:
                    print(f"\n  ⚠ Cached results for {dataset_name} are from a different checkpoint")
                    print(f"    Cached: {cached_checkpoint_time}")
                    print(f"    Current: {current_checkpoint_time}")
                    print(f"    Re-evaluating {dataset_name}...")
                    # Don't use cache, continue to evaluation below
                else:
                    print("\n" + "-" * 80)
                    print(f"Loading cached results for {dataset_name} (already evaluated)")
                    print("-" * 80)

                    # Restore raw results from checkpoint
                    all_labels_combined.extend(cached_results['labels'])
                    all_probs_combined.extend(cached_results['probs'])
                    all_preds_combined.extend(cached_results['preds'])
                    total_loss_combined += cached_results['total_loss']
                    total_batches += cached_results['num_batches']
                    per_dataset_results[dataset_name] = {
                        'num_samples': cached_results['num_samples']
                    }
                    print(f"  ✓ Restored {dataset_name}: {cached_results['num_samples']:,} samples")
                    continue  # Skip to next dataset

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"\n  ⚠ Cached results for {dataset_name} are corrupted: {e}")
                print(f"    Deleting corrupt checkpoint and re-evaluating...")
                dataset_checkpoint_file.unlink()
                # Continue to evaluation below

        print("\n" + "-" * 80)
        print(f"Loading {dataset_name} {args.split} dataset")
        print("-" * 80)

        if dataset_name not in dataset_configs:
            raise ValueError(f"Unknown dataset: {dataset_name}. Valid: {list(dataset_configs.keys())}")

        dataset_id = dataset_configs[dataset_name]
        dataset_params = params.copy()
        dataset_params['dataset_id'] = dataset_id

        # Load dataset-specific config
        dataset_specific = load_dataset_config(config_path, dataset_id)
        dataset_params.update(dataset_specific)

        # Fix paths
        for key in ['data_root', 'test_data', 'train_data', 'valid_data']:
            if key in dataset_params and dataset_params[key]:
                dataset_params[key] = dataset_params[key].replace('../../', '')

        # Load feature map
        data_dir = os.path.join(dataset_params['data_root'], dataset_id)
        feature_map_json = os.path.join(data_dir, "feature_map.json")

        if not os.path.exists(feature_map_json):
            raise FileNotFoundError(f"Feature map not found: {feature_map_json}\n"
                                  f"Please preprocess {dataset_name} first.")

        dataset_feature_map = FeatureMap(dataset_id, data_dir)
        dataset_feature_map.load(feature_map_json, dataset_params)

        # Update paths to use parquet files instead of CSV
        dataset_params['train_data'] = os.path.join(data_dir, 'train.parquet')
        dataset_params['valid_data'] = os.path.join(data_dir, 'valid.parquet')
        dataset_params['test_data'] = os.path.join(data_dir, 'test.parquet')
        dataset_params['data_format'] = 'parquet'  # Important: tell it to use parquet

        # Create dataloader
        dataset_params['num_workers'] = 0
        dataset_params['batch_size'] = args.batch_size

        # Load the requested split
        if args.split == 'test':
            dataloader = RankDataLoader(dataset_feature_map, stage='test', **dataset_params).make_iterator()
        else:  # valid
            _, dataloader = RankDataLoader(dataset_feature_map, stage='train', **dataset_params).make_iterator()

        print(f"  ✓ Loaded {args.split} dataloader: ~{len(dataloader)} batches")

        # Pre-compute prompt embeddings
        print("\n  Pre-computing prompt embeddings...")
        prompt_embeds_cache = prepare_prompt_embeddings(
            tokenizer, llm, prompt_template, args.batch_size, device
        )
        print(f"  ✓ Prompt embeddings cached: {prompt_embeds_cache.shape}")

        # Evaluate
        print("\n" + "-" * 80)
        print(f"Evaluating on {dataset_name} {args.split} set")
        print("-" * 80)

        # Get raw results if combining, otherwise compute metrics directly
        raw_results = evaluate_testset(
            model, dataloader, device, feature_names,
            prompt_embeds_cache, dataset_name, return_raw=eval_combined
        )

        if eval_combined:
            # Accumulate raw results for combined metric calculation
            all_labels_combined.extend(raw_results['labels'])
            all_probs_combined.extend(raw_results['probs'])
            all_preds_combined.extend(raw_results['preds'])
            total_loss_combined += raw_results['total_loss']
            total_batches += raw_results['num_batches']
            per_dataset_results[dataset_name] = {
                'num_samples': len(raw_results['labels'])
            }
            print(f"  ✓ Completed {dataset_name}: {len(raw_results['labels']):,} samples")

            # Save checkpoint for this dataset (in case of crash)
            checkpoint_data = {
                'labels': raw_results['labels'].tolist(),
                'probs': raw_results['probs'].tolist(),
                'preds': raw_results['preds'].tolist(),
                'total_loss': raw_results['total_loss'],
                'num_batches': raw_results['num_batches'],
                'num_samples': len(raw_results['labels']),
                'dataset': dataset_name,
                'split': args.split,
                'timestamp': str(Path(args.checkpoint).stat().st_mtime)
            }
            with open(dataset_checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f)
            print(f"  ✓ Checkpoint saved: {dataset_checkpoint_file.name}")
        else:
            # Single dataset evaluation - use results directly
            results = raw_results

    # Calculate combined metrics if evaluating multiple datasets
    if eval_combined:
        print("\n" + "-" * 80)
        print("Computing combined metrics across all datasets")
        print("-" * 80)

        all_labels_combined = np.array(all_labels_combined)
        all_probs_combined = np.array(all_probs_combined)
        all_preds_combined = np.array(all_preds_combined)

        # Calculate combined metrics
        auc = roc_auc_score(all_labels_combined, all_probs_combined)
        accuracy = (all_preds_combined == all_labels_combined).mean() * 100
        logloss = log_loss(all_labels_combined, all_probs_combined)
        avg_loss = total_loss_combined / total_batches

        results = {
            'auc': auc,
            'accuracy': accuracy,
            'logloss': logloss,
            'loss': avg_loss,
            'num_samples': len(all_labels_combined),
            'per_dataset': per_dataset_results
        }

    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    if eval_combined:
        print(f"\nCombined Results (x1 + x2 + x4 {args.split} sets)")
        print(f"  Total Samples: {results['num_samples']:,}")
        for ds_name, ds_info in results['per_dataset'].items():
            print(f"    - {ds_name}: {ds_info['num_samples']:,} samples")
    else:
        print(f"\nDataset: {args.dataset} ({args.split} set)")
        print(f"  Samples: {results['num_samples']:,}")

    print(f"\nMetrics:")
    print(f"  AUC:      {results['auc']:.4f}")
    print(f"  Accuracy: {results['accuracy']:.2f}%")
    print(f"  LogLoss:  {results['logloss']:.4f}")
    print(f"  Loss:     {results['loss']:.4f}")

    # Save results
    output_dir = Path(args.checkpoint).parent
    results_file = output_dir / f'evaluation_{args.dataset}_{args.split}.json'

    eval_results = {
        'checkpoint': args.checkpoint,
        'dataset': args.dataset,
        'split': args.split,
        'metrics': results,
        'config': {
            'batch_size': args.batch_size,
            'device': str(device),
            'prompt_template': prompt_template
        }
    }

    with open(results_file, 'w') as f:
        json.dump(eval_results, f, indent=2)

    print(f"\n  ✓ Results saved to: {results_file}")

    # Clean up evaluation checkpoints after successful completion
    if eval_combined and eval_checkpoint_dir.exists():
        import shutil
        shutil.rmtree(eval_checkpoint_dir)
        print(f"  ✓ Cleaned up evaluation checkpoints")
        print(f"\nNote: If evaluation was interrupted, re-run the same command")
        print(f"      and it will resume from the last completed dataset.")

    print("=" * 80)


if __name__ == "__main__":
    main()
