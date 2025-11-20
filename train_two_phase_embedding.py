#!/usr/bin/env python3
"""
Unified Two-Phase LLM-CTR Training Pipeline for Avazu Dataset

This script implements automatic two-phase training with fixed prompt embedding handling:
- Phase 1: Train projector only on x1 + x2 (encoder frozen, LLM frozen)
- Phase 2: Train projector + LLM on x4 (encoder frozen)
- Evaluation: Automatic evaluation on x4 test set

Key Features:
- Single script execution (no manual checkpoint passing)
- Token ID caching (NO stale embeddings bug!)
- Baseline checkpoint for encoder initialization
- Integrated evaluation

Usage:
    python train_two_phase_embedding.py \
        --baseline_checkpoint path/to/baseline.model \
        --phase1_epochs 1 \
        --phase2_epochs 1 \
        --gpu 0
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
from datetime import datetime
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss

# FuxiCTR imports
from fuxictr.features import FeatureMap
from fuxictr.utils import load_config
from fuxictr.pytorch.dataloaders import RankDataLoader

# Add DeepFM source to path
sys.path.append('model_zoo/DeepFM/src')
from projector import FeatureProjector


class LLM_CTR_Model(nn.Module):
    """
    LLM-enhanced CTR prediction model (Two-Phase variant).

    Architecture:
        Feature IDs → Embeddings → Encoder → Projector → [Text + Features] → LLM → Token Logits

    Training:
        Phase 1: Frozen encoder + LLM, train projector only
        Phase 2: Frozen encoder, train projector + LLM
    """

    def __init__(self, embedding_layer, encoder, projector, llm, tokenizer, prompt_template, freeze_llm=True):
        super().__init__()
        self.embedding_layer = embedding_layer
        self.encoder = encoder
        self.projector = projector
        self.llm = llm
        self.tokenizer = tokenizer

        # Always freeze encoder and embeddings (use baseline pre-trained weights)
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.embedding_layer.parameters():
            param.requires_grad = False

        # LLM: frozen in Phase 1, trainable in Phase 2
        for param in self.llm.parameters():
            param.requires_grad = not freeze_llm

        # Get token IDs for "0" and "1"
        self.token_0 = tokenizer.encode("0", add_special_tokens=False)[0]
        self.token_1 = tokenizer.encode("1", add_special_tokens=False)[0]

        print(f"Token ID for '0': {self.token_0}")
        print(f"Token ID for '1': {self.token_1}")

        # Cache tokenized prompt (token IDs only - NOT embeddings!)
        # Embeddings will be computed fresh each forward pass using current LLM weights
        print(f"\n  Caching tokenized prompt (token IDs only)...")
        print(f"  Prompt: '{prompt_template}'")
        self.prompt_token_ids = tokenizer(prompt_template, return_tensors="pt").input_ids
        print(f"  ✓ Cached {self.prompt_token_ids.shape[1]} prompt tokens")
        print(f"  Note: Embeddings computed fresh each batch (no stale embeddings!)")

    def forward(self, batch_dict):
        """Forward pass through the complete pipeline."""
        # 1. Feature IDs → Embeddings
        embedded = self.embedding_layer(batch_dict)
        batch_size = embedded.shape[0]

        # 2. Embeddings → Encoder
        encoder_output = self.encoder(embedded)
        if isinstance(encoder_output, tuple):
            encoded = encoder_output[0]
        else:
            encoded = encoder_output

        # Convert to bfloat16 to match LLM dtype
        encoded = encoded.to(torch.bfloat16)

        # 3. Encoded → Projector
        projected = self.projector(encoded)

        # 4. Compute prompt embeddings from cached token IDs
        # Token IDs are cached, but embeddings computed fresh using current LLM weights
        prompt_token_ids = self.prompt_token_ids.to(projected.device)
        prompt_embeds = self.llm.get_input_embeddings()(prompt_token_ids)
        prompt_embeds = prompt_embeds.repeat(batch_size, 1, 1)

        # 5. Concatenate [text + projected features]
        inputs_embeds = torch.cat([prompt_embeds, projected], dim=1)

        # 6. Feed to LLM
        llm_output = self.llm(inputs_embeds=inputs_embeds, return_dict=True)

        # 7. Get logits for next token prediction
        lm_logits = llm_output.logits
        last_token_logits = lm_logits[:, -1, :]

        # Extract logits for tokens "0" and "1"
        logit_0 = last_token_logits[:, self.token_0]
        logit_1 = last_token_logits[:, self.token_1]

        # Stack into [batch_size, 2] for cross-entropy
        logits = torch.stack([logit_0, logit_1], dim=1)

        return logits


def train_epoch(model, dataloader, optimizer, device, feature_names, epoch, phase, dataset_name, test_mode=False):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"Phase {phase} [{dataset_name}] - Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        # Test mode: only process 3 batches
        if test_mode and batch_idx >= 3:
            print(f"  [TEST MODE] Stopping after {batch_idx} batches")
            break

        optimizer.zero_grad(set_to_none=True)

        # Move batch to device
        batch_dict = {feat: batch[feat].to(device) for feat in feature_names}
        labels = batch['label'].long().squeeze().to(device)

        # Forward pass
        logits = model(batch_dict)

        # Cross-entropy loss
        loss = F.cross_entropy(logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.0 * correct / total:.2f}%'
        })

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    accuracy = 100.0 * correct / total if total > 0 else 0

    return avg_loss, accuracy


def evaluate(model, dataloader, device, feature_names, phase, dataset_name, split_name="Validation", test_mode=False):
    """Evaluate on validation/test set with AUC calculation."""
    model.eval()
    total_loss = 0
    all_labels = []
    all_probs = []
    all_preds = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Phase {phase} [{dataset_name}] - {split_name}")

        for batch_idx, batch in enumerate(pbar):
            # Test mode: only process 3 batches
            if test_mode and batch_idx >= 3:
                print(f"  [TEST MODE] Stopping after {batch_idx} batches")
                break

            # Move batch to device
            batch_dict = {feat: batch[feat].to(device) for feat in feature_names}
            labels = batch['label'].long().squeeze().to(device)

            # Forward pass
            logits = model(batch_dict)
            loss = F.cross_entropy(logits, labels)

            # Get probabilities for AUC
            probs = F.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)

            # Track metrics
            total_loss += loss.item()
            all_labels.extend(labels.float().cpu().numpy())
            all_probs.extend(probs.float().cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    # Calculate metrics
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    accuracy = (all_preds == all_labels).mean() * 100 if len(all_labels) > 0 else 0

    # Handle edge case where only one class is present (can happen in test mode)
    unique_labels = len(set(all_labels))
    if unique_labels < 2:
        print(f"\n  ⚠ Warning: Only {unique_labels} unique label(s) in evaluation data")
        print(f"  This is normal in test mode with only 3 batches")
        auc = 0.5  # Random baseline
        logloss = 0.693  # -log(0.5) for binary classification
    else:
        auc = roc_auc_score(all_labels, all_probs)
        logloss = log_loss(all_labels, all_probs)

    return avg_loss, accuracy, auc, logloss


def main():
    parser = argparse.ArgumentParser(description='Unified Two-Phase LLM-CTR Training')
    parser.add_argument('--phase1_epochs', type=int, default=1,
                       help='Epochs per dataset in Phase 1 (default: 1)')
    parser.add_argument('--phase2_epochs', type=int, default=1,
                       help='Epochs for Phase 2 (default: 1)')
    parser.add_argument('--phase1_batch_size', type=int, default=256,
                       help='Batch size for Phase 1 (default: 256)')
    parser.add_argument('--phase2_batch_size', type=int, default=32,
                       help='Batch size for Phase 2 (default: 32)')
    parser.add_argument('--phase1_lr', type=float, default=1e-3,
                       help='Learning rate for Phase 1 projector (default: 1e-3)')
    parser.add_argument('--phase2_lr', type=float, default=1e-4,
                       help='Learning rate for Phase 2 projector+LLM (default: 1e-4)')
    parser.add_argument('--baseline_checkpoint', type=str, required=True,
                       help='Path to baseline DeepFM checkpoint')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID (-1 for CPU)')
    parser.add_argument('--test_mode', action='store_true',
                       help='Run in test mode (only 3 batches per epoch for quick validation)')

    args = parser.parse_args()

    # Setup device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print("=" * 80)
    print("UNIFIED TWO-PHASE LLM-CTR TRAINING")
    print("=" * 80)

    if args.test_mode:
        print("\n⚠ ⚠ ⚠  TEST MODE ENABLED  ⚠ ⚠ ⚠")
        print("  Only 3 batches per epoch will be processed for quick validation")
        print("  This is NOT for actual training - only for testing the pipeline!")
        print("")

    print(f"\nConfiguration:")
    print(f"  Phase 1: Train projector on x1+x2 ({args.phase1_epochs} epoch each)")
    print(f"  Phase 2: Train projector+LLM on x4 ({args.phase2_epochs} epoch)")
    print(f"  Phase 1 batch size: {args.phase1_batch_size}")
    print(f"  Phase 2 batch size: {args.phase2_batch_size}")
    print(f"  Phase 1 LR: {args.phase1_lr}")
    print(f"  Phase 2 LR: {args.phase2_lr}")
    print(f"  Device: {device}")
    if args.test_mode:
        print(f"  Test mode: Only 3 batches per epoch")

    # Load baseline model configuration
    print("\n" + "-" * 80)
    print("STEP 1: Loading baseline encoder from checkpoint")
    print("-" * 80)

    config_path = 'model_zoo/DeepFM/config'
    experiment_id = 'DeepFM_avazu_normalized'
    params = load_config(config_path, experiment_id)

    # Fix paths
    for key in ['data_root', 'test_data', 'train_data', 'valid_data']:
        if key in params and params[key]:
            params[key] = params[key].replace('../../', '')

    # Load unified feature map (for x1, x2, x4 compatibility)
    unified_data_dir = os.path.join(params['data_root'], 'avazu_unified')
    unified_feature_map_json = os.path.join(unified_data_dir, "feature_map.json")

    if not os.path.exists(unified_feature_map_json):
        raise FileNotFoundError(
            f"Unified feature map not found: {unified_feature_map_json}\n"
            f"Please run create_unified_feature_map.py first."
        )

    feature_map = FeatureMap('avazu_unified', unified_data_dir)
    feature_map.load(unified_feature_map_json, params)

    print(f"  Features: {len(feature_map.features)}")
    print(f"  Embedding dim: {params['embedding_dim']}")

    # Load baseline DeepFM model to extract encoder
    model_src_path = "model_zoo.DeepFM.src"
    src = importlib.import_module(model_src_path)
    model_class = getattr(src, params['model'])

    # Use x4_normalized parquet for analyzer
    x4_data_dir = os.path.join(params['data_root'], 'avazu_x4_normalized')
    model_params = params.copy()
    model_params['train_data'] = os.path.join(x4_data_dir, 'train.parquet')
    model_params['valid_data'] = os.path.join(x4_data_dir, 'valid.parquet')
    model_params['test_data'] = os.path.join(x4_data_dir, 'test.parquet')

    deepfm_model = model_class(feature_map, **model_params)

    # Load baseline checkpoint
    import glob
    checkpoint_files = glob.glob(args.baseline_checkpoint)
    if not checkpoint_files:
        raise FileNotFoundError(f"Baseline checkpoint not found: {args.baseline_checkpoint}")
    checkpoint_path = checkpoint_files[0]

    print(f"  Loading baseline checkpoint: {checkpoint_path}")
    deepfm_model.load_weights(checkpoint_path)
    deepfm_model.eval()

    # Extract components
    embedding_layer = deepfm_model.embedding_layer
    encoder = deepfm_model.gen

    print(f"  ✓ Feature Encoder extracted (will be frozen)")
    print(f"  ✓ GEN Encoder extracted (will be frozen)")

    # Load LLM and tokenizer
    print("\n" + "-" * 80)
    print("STEP 2: Loading LLM (Qwen3-0.6B)")
    print("-" * 80)
    print("  ⚠ REQUIREMENT: FlashAttention 2 and Ampere+ GPU")

    # Check CUDA
    if device.type != 'cuda':
        raise RuntimeError(
            f"FlashAttention 2 requires CUDA GPU, but running on {device}."
        )

    # Check FlashAttention
    try:
        import flash_attn  # noqa: F401
        print("  ✓ FlashAttention 2 package found")
    except ImportError:
        raise ImportError(
            "FlashAttention 2 is not installed!\n"
            "Install with: pip install flash-attn --no-build-isolation"
        )

    # Load LLM
    llm = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    ).to(device)

    # Test FlashAttention
    print("  Testing FlashAttention 2 compatibility...")
    with torch.no_grad():
        test_input = torch.randn(1, 10, llm.config.hidden_size,
                                dtype=torch.bfloat16, device=device)
        try:
            _ = llm(inputs_embeds=test_input)
            print("  ✓ FlashAttention 2 working correctly")
        except RuntimeError as e:
            if "ampere" in str(e).lower() or "flash" in str(e).lower():
                raise RuntimeError(f"GPU does not support FlashAttention 2!\nError: {e}")
            else:
                raise

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    print(f"  ✓ LLM loaded (bfloat16)")
    print(f"  ✓ Tokenizer loaded")
    print(f"  LLM hidden size: {llm.config.hidden_size}")

    # Create projector
    print("\n" + "-" * 80)
    print("STEP 3: Creating projector")
    print("-" * 80)

    projector = FeatureProjector(
        feature_dim=params['embedding_dim'],
        llm_dim=llm.config.hidden_size,
        hidden_dim=512
    ).to(device).to(torch.bfloat16)

    print(f"  Projector: {params['embedding_dim']}D → {llm.config.hidden_size}D (bfloat16)")

    # Define prompt template
    prompt_template = "Based on the user's browsing behavior and ad interaction features, predict if they will click on this advertisement. Answer with 1 for click or 0 for no click:"

    # Feature names
    feature_names = [f'feat_{i}' for i in range(1, 22)] + ['hour', 'weekday', 'weekend']

    # Output directory
    save_dir = Path('checkpoints/llm_two_phase_embedding')
    save_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # PHASE 1: Train Projector (x1 + x2)
    # ========================================================================
    print("\n" + "=" * 80)
    print("PHASE 1: PROJECTOR WARM-UP (x1 + x2)")
    print("=" * 80)
    print(f"  Frozen: Feature Encoder + GEN + LLM")
    print(f"  Trainable: Projector only")
    print(f"  Learning rate: {args.phase1_lr}")

    # Create Phase 1 model
    model_phase1 = LLM_CTR_Model(
        embedding_layer=embedding_layer,
        encoder=encoder,
        projector=projector,
        llm=llm,
        tokenizer=tokenizer,
        prompt_template=prompt_template,
        freeze_llm=True  # Freeze LLM in Phase 1
    ).to(device)

    # Phase 1 optimizer (only projector trainable)
    optimizer_phase1 = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model_phase1.parameters()),
        lr=args.phase1_lr
    )

    # Count trainable parameters
    total_params = sum(p.numel() for p in model_phase1.parameters())
    trainable_params = sum(p.numel() for p in model_phase1.parameters() if p.requires_grad)
    print(f"\n  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen: {total_params - trainable_params:,}")

    # Train on x1 and x2 sequentially
    phase1_results = []
    phase1_datasets = ['x1', 'x2']

    for dataset_name in phase1_datasets:
        print(f"\n{'=' * 80}")
        print(f"Training on {dataset_name.upper()}")
        print(f"{'=' * 80}")

        # Load dataset
        dataset_id = f'avazu_{dataset_name}_normalized'
        from fuxictr.utils import load_dataset_config

        dataset_params = params.copy()
        dataset_params['dataset_id'] = dataset_id
        dataset_specific = load_dataset_config(config_path, dataset_id)
        dataset_params.update(dataset_specific)

        # Fix paths
        for key in ['data_root', 'test_data', 'train_data', 'valid_data']:
            if key in dataset_params and dataset_params[key]:
                dataset_params[key] = dataset_params[key].replace('../../', '')

        data_dir = os.path.join(dataset_params['data_root'], dataset_id)
        feature_map_json = os.path.join(data_dir, "feature_map.json")

        dataset_feature_map = FeatureMap(dataset_id, data_dir)
        dataset_feature_map.load(feature_map_json, dataset_params)

        dataset_params['num_workers'] = 0
        dataset_params['batch_size'] = args.phase1_batch_size
        dataset_params['train_data'] = os.path.join(data_dir, 'train.parquet')
        dataset_params['valid_data'] = os.path.join(data_dir, 'valid.parquet')

        train_gen, valid_gen = RankDataLoader(dataset_feature_map, stage='train', **dataset_params).make_iterator()

        print(f"  Train batches: {len(train_gen)}")
        print(f"  Valid batches: {len(valid_gen)}")

        # Train
        for epoch in range(1, args.phase1_epochs + 1):
            train_loss, train_acc = train_epoch(
                model_phase1, train_gen, optimizer_phase1, device, feature_names,
                epoch, 1, dataset_name, test_mode=args.test_mode
            )

            val_loss, val_acc, val_auc, val_logloss = evaluate(
                model_phase1, valid_gen, device, feature_names, 1, dataset_name, "Validation",
                test_mode=args.test_mode
            )

            print(f"\n{dataset_name.upper()} - Epoch {epoch} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            print(f"  Val AUC:    {val_auc:.4f} | Val LogLoss: {val_logloss:.4f}")

            phase1_results.append({
                'phase': 1,
                'dataset': dataset_name,
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_auc': val_auc,
                'val_logloss': val_logloss
            })

    # Save Phase 1 checkpoint
    phase1_checkpoint = {
        'model_state_dict': model_phase1.state_dict(),
        'optimizer_state_dict': optimizer_phase1.state_dict(),
        'prompt_template': prompt_template,
        'feature_names': feature_names,
        'results': phase1_results
    }
    phase1_path = save_dir / 'phase1_checkpoint.pt'
    torch.save(phase1_checkpoint, phase1_path)
    print(f"\n✓ Phase 1 checkpoint saved: {phase1_path}")

    # ========================================================================
    # PHASE 2: Train Projector + LLM (x4)
    # ========================================================================
    print("\n" + "=" * 80)
    print("PHASE 2: FINE-TUNE PROJECTOR + LLM (x4)")
    print("=" * 80)
    print(f"  Frozen: Feature Encoder + GEN")
    print(f"  Trainable: Projector + LLM")
    print(f"  Learning rate: {args.phase2_lr}")

    # Create Phase 2 model (LLM now trainable)
    model_phase2 = LLM_CTR_Model(
        embedding_layer=embedding_layer,
        encoder=encoder,
        projector=projector,
        llm=llm,
        tokenizer=tokenizer,
        prompt_template=prompt_template,
        freeze_llm=False  # Unfreeze LLM in Phase 2
    ).to(device)

    # Load Phase 1 checkpoint (projector weights)
    model_phase2.load_state_dict(phase1_checkpoint['model_state_dict'], strict=False)
    print(f"  ✓ Loaded Phase 1 checkpoint")

    # Phase 2 optimizer (projector + LLM trainable)
    optimizer_phase2 = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model_phase2.parameters()),
        lr=args.phase2_lr
    )

    # Count trainable parameters
    total_params = sum(p.numel() for p in model_phase2.parameters())
    trainable_params = sum(p.numel() for p in model_phase2.parameters() if p.requires_grad)
    print(f"\n  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen: {total_params - trainable_params:,}")

    # Load x4 dataset
    dataset_id = 'avazu_x4_normalized'
    dataset_params = params.copy()
    dataset_params['dataset_id'] = dataset_id

    data_dir = os.path.join(dataset_params['data_root'], dataset_id)
    feature_map_json = os.path.join(data_dir, "feature_map.json")

    dataset_feature_map = FeatureMap(dataset_id, data_dir)
    dataset_feature_map.load(feature_map_json, dataset_params)

    dataset_params['num_workers'] = 0
    dataset_params['batch_size'] = args.phase2_batch_size
    dataset_params['train_data'] = os.path.join(data_dir, 'train.parquet')
    dataset_params['valid_data'] = os.path.join(data_dir, 'valid.parquet')
    dataset_params['test_data'] = os.path.join(data_dir, 'test.parquet')

    train_gen, valid_gen = RankDataLoader(dataset_feature_map, stage='train', **dataset_params).make_iterator()
    test_gen = RankDataLoader(dataset_feature_map, stage='test', **dataset_params).make_iterator()

    print(f"\n  Train batches: {len(train_gen)}")
    print(f"  Valid batches: {len(valid_gen)}")
    print(f"  Test batches: {len(test_gen)}")

    # Train Phase 2
    phase2_results = []
    for epoch in range(1, args.phase2_epochs + 1):
        train_loss, train_acc = train_epoch(
            model_phase2, train_gen, optimizer_phase2, device, feature_names,
            epoch, 2, 'x4', test_mode=args.test_mode
        )

        val_loss, val_acc, val_auc, val_logloss = evaluate(
            model_phase2, valid_gen, device, feature_names, 2, 'x4', "Validation",
            test_mode=args.test_mode
        )

        print(f"\nPhase 2 - Epoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"  Val AUC:    {val_auc:.4f} | Val LogLoss: {val_logloss:.4f}")

        phase2_results.append({
            'phase': 2,
            'dataset': 'x4',
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_auc': val_auc,
            'val_logloss': val_logloss
        })

    # Save Phase 2 checkpoint
    phase2_checkpoint = {
        'model_state_dict': model_phase2.state_dict(),
        'optimizer_state_dict': optimizer_phase2.state_dict(),
        'prompt_template': prompt_template,
        'feature_names': feature_names,
        'results': phase2_results
    }
    phase2_path = save_dir / 'phase2_checkpoint.pt'
    torch.save(phase2_checkpoint, phase2_path)
    print(f"\n✓ Phase 2 checkpoint saved: {phase2_path}")

    # Save combined training results
    all_results = phase1_results + phase2_results
    results_file = save_dir / 'training_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"✓ Training results saved: {results_file}")

    # ========================================================================
    # EVALUATION: Test on x4
    # ========================================================================
    print("\n" + "=" * 80)
    print("EVALUATION: Testing on x4")
    print("=" * 80)

    test_loss, test_acc, test_auc, test_logloss = evaluate(
        model_phase2, test_gen, device, feature_names, 2, 'x4', "Test",
        test_mode=args.test_mode
    )

    print(f"\nTest Set Results (x4):")
    print(f"  Test Loss:    {test_loss:.4f}")
    print(f"  Test Acc:     {test_acc:.2f}%")
    print(f"  Test AUC:     {test_auc:.4f}")
    print(f"  Test LogLoss: {test_logloss:.4f}")

    # Save test results
    test_results = {
        'dataset': 'x4',
        'split': 'test',
        'loss': test_loss,
        'accuracy': test_acc,
        'auc': test_auc,
        'logloss': test_logloss,
        'timestamp': datetime.now().isoformat()
    }

    test_results_file = save_dir / 'evaluation_x4_test.json'
    with open(test_results_file, 'w') as f:
        json.dump(test_results, f, indent=2)

    print("\n" + "=" * 80)
    print("✅ UNIFIED TWO-PHASE TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {save_dir}")
    print(f"  Phase 1 checkpoint: {phase1_path}")
    print(f"  Phase 2 checkpoint: {phase2_path}")
    print(f"  Training results: {results_file}")
    print(f"  Test evaluation: {test_results_file}")
    print(f"\nFinal Test AUC: {test_auc:.4f}")


if __name__ == "__main__":
    main()
