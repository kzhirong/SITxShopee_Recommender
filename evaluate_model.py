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


def evaluate_testset(model, dataloader, device, feature_names, prompt_embeds_cache, dataset_name):
    """Evaluate on test set and calculate AUC + Accuracy."""
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

            # Collect results
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            total_loss += loss.item()

            # Update progress
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)

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
                       help='Dataset to evaluate on (e.g., x4, x1_sample20, x2_sample20)')
    parser.add_argument('--split', type=str, default='test', choices=['test', 'valid'],
                       help='Which split to evaluate on')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for evaluation')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID (-1 for CPU)')

    args = parser.parse_args()

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
    print(f"  Dataset: {args.dataset}")
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

    # Try to use FlashAttention 2 if available (A100+ only)
    # Otherwise fall back to default attention
    llm_kwargs = {"torch_dtype": torch.bfloat16}

    if device.type == 'cuda':
        try:
            import flash_attn
            llm_kwargs["attn_implementation"] = "flash_attention_2"
            print("  Using FlashAttention 2 (A100 optimization)")
        except ImportError:
            print("  FlashAttention 2 not available, using default attention")
    else:
        print(f"  Running on {device}, using default attention")

    llm = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B",
        **llm_kwargs
    ).to(device)
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

    # Load test dataset
    print("\n" + "-" * 80)
    print(f"Loading {args.dataset} {args.split} dataset")
    print("-" * 80)

    # Map dataset names to dataset_ids
    dataset_configs = {
        'x1': 'avazu_x1_normalized',
        'x2': 'avazu_x2_normalized',
        'x4': 'avazu_x4_normalized',
        'x1_sample20': 'avazu_x1_sample20',
        'x2_sample20': 'avazu_x2_sample20'
    }

    if args.dataset not in dataset_configs:
        raise ValueError(f"Unknown dataset: {args.dataset}. Valid: {list(dataset_configs.keys())}")

    dataset_id = dataset_configs[args.dataset]
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
                              f"Please preprocess {args.dataset} first.")

    dataset_feature_map = FeatureMap(dataset_id, data_dir)
    dataset_feature_map.load(feature_map_json, dataset_params)

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
    print(f"Evaluating on {args.dataset} {args.split} set")
    print("-" * 80)

    results = evaluate_testset(
        model, dataloader, device, feature_names,
        prompt_embeds_cache, args.dataset
    )

    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
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
    print("=" * 80)


if __name__ == "__main__":
    main()
