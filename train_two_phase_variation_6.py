#!/usr/bin/env python3
"""
Two-Phase LLM-CTR Training (Variation 6) - Fine-tune Baseline Encoder

This variant combines aspects of single-phase and two-phase training:
- Architecture: Uses pre-trained baseline encoder (fine-tuned in Phase 1)
- Training: Two phases (Phase 1: encoder+projector, Phase 2: projector+LLM)

Phase 1: Fine-tune encoder + train projector on x1 + x2 (LLM frozen)
    - Feature Encoder: TRAINABLE (from baseline x4)
    - GEN Encoder: TRAINABLE (from baseline x4)
    - Projector: TRAINABLE (from scratch)
    - LLM: FROZEN
    - Data: x1 (1 epoch) → x2 (1 epoch)

Phase 2: Train projector + LLM on x4 (encoder frozen)
    - Feature Encoder: FROZEN (from Phase 1)
    - GEN Encoder: FROZEN (from Phase 1)
    - Projector: TRAINABLE (from Phase 1)
    - LLM: TRAINABLE
    - Data: x4 (1 epoch)

Key Features:
- Single script execution (Phase 1 → Phase 2 → Evaluation)
- Token ID caching (NO stale embeddings!)
- Encoder initialized from baseline (trained on x4)
- Encoder fine-tuned in Phase 1 on x1+x2
- Integrated evaluation on x4

Usage:
    python train_two_phase_variation_6.py \
        --baseline_checkpoint "model_zoo/DeepFM/Avazu/DeepFM_avazu_normalized/avazu_x4_normalized/DeepFM_avazu_normalized.model" \
        --phase1_batch_size 256 \
        --phase2_batch_size 32 \
        --phase1_lr 1e-3 \
        --phase2_lr 1e-4 \
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
    LLM-enhanced CTR prediction model (Variation 6).

    Architecture:
        Feature IDs → Embeddings → Encoder → Projector → [Text + Features] → LLM → Token Logits

    Training:
        Phase 1: Train encoder + projector, freeze LLM
        Phase 2: Freeze encoder, train projector + LLM
    """

    def __init__(self, embedding_layer, encoder, projector, llm, tokenizer, prompt_template, freeze_encoder=False, freeze_llm=True):
        super().__init__()
        self.embedding_layer = embedding_layer
        self.encoder = encoder
        self.projector = projector
        self.llm = llm
        self.tokenizer = tokenizer

        # Encoder: trainable in Phase 1, frozen in Phase 2
        for param in self.encoder.parameters():
            param.requires_grad = not freeze_encoder
        for param in self.embedding_layer.parameters():
            param.requires_grad = not freeze_encoder

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

        # Prepare batch
        batch_dict = {}
        for i, name in enumerate(feature_names):
            if isinstance(batch[name], torch.Tensor):
                batch_dict[name] = batch[name].to(device)
            else:
                batch_dict[name] = torch.tensor(batch[name]).to(device)

        labels = batch['label'].to(device)

        # Forward pass
        logits = model(batch_dict)

        # Compute loss
        loss = F.cross_entropy(logits, labels.long())

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        # Update progress bar
        avg_loss = total_loss / (batch_idx + 1)
        accuracy = 100.0 * correct / total
        pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'acc': f'{accuracy:.2f}%'})

    final_loss = total_loss / len(dataloader)
    final_acc = 100.0 * correct / total

    return final_loss, final_acc


def evaluate(model, dataloader, device, feature_names, phase="Evaluation", test_mode=False):
    """Evaluate the model."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"{phase}")
        for batch_idx, batch in enumerate(pbar):
            # Test mode: only process 3 batches
            if test_mode and batch_idx >= 3:
                print(f"  [TEST MODE] Stopping evaluation after {batch_idx} batches")
                break
            # Prepare batch
            batch_dict = {}
            for i, name in enumerate(feature_names):
                if isinstance(batch[name], torch.Tensor):
                    batch_dict[name] = batch[name].to(device)
                else:
                    batch_dict[name] = torch.tensor(batch[name]).to(device)

            labels = batch['label'].cpu().numpy()

            # Forward pass
            logits = model(batch_dict)
            # Convert to float32 before numpy (bfloat16 not supported by numpy)
            probs = F.softmax(logits, dim=1)[:, 1].float().cpu().numpy()

            all_preds.extend(probs)
            all_labels.extend(labels)

    # Calculate metrics
    # Handle edge case where only one class is present (can happen in test mode with few batches)
    unique_labels = len(set(all_labels))
    if unique_labels < 2:
        print(f"\n  ⚠ Warning: Only {unique_labels} unique label(s) in evaluation data")
        print(f"  This is normal in test mode with only 3 batches")
        auc = 0.5  # Random baseline
        logloss = 0.693  # -log(0.5) for binary
    else:
        auc = roc_auc_score(all_labels, all_preds)
        logloss = log_loss(all_labels, all_preds)

    return auc, logloss, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser(description='Two-Phase LLM-CTR Training (Variation 6)')

    # Training arguments
    parser.add_argument('--phase1_batch_size', type=int, default=256,
                       help='Batch size for Phase 1 (default: 256)')
    parser.add_argument('--phase2_batch_size', type=int, default=32,
                       help='Batch size for Phase 2 (default: 32)')
    parser.add_argument('--phase1_lr', type=float, default=1e-3,
                       help='Learning rate for Phase 1 (default: 1e-3)')
    parser.add_argument('--phase2_lr', type=float, default=1e-4,
                       help='Learning rate for Phase 2 (default: 1e-4)')

    # Model architecture
    parser.add_argument('--embedding_dim', type=int, default=16,
                       help='Embedding dimension (default: 16)')
    parser.add_argument('--encoder_hidden_units', type=int, nargs='+', default=[2000, 2000, 2000, 2000],
                       help='Hidden units for GEN encoder (default: [2000, 2000, 2000, 2000])')
    parser.add_argument('--projector_hidden_dim', type=int, default=2048,
                       help='Projector hidden dimension (default: 2048)')
    parser.add_argument('--llm_dim', type=int, default=896,
                       help='LLM embedding dimension for Qwen3-0.6B (default: 896)')

    # Paths
    parser.add_argument('--baseline_checkpoint', type=str, required=True,
                       help='Path to baseline DeepFM checkpoint')
    parser.add_argument('--output_dir', type=str, default='checkpoints/llm_two_phase_variation_6',
                       help='Directory to save checkpoints')

    # Device
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID (default: 0)')

    # Testing
    parser.add_argument('--test_mode', action='store_true',
                       help='Run in test mode (only 3 batches per phase for quick validation)')

    args = parser.parse_args()

    # Setup device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("TWO-PHASE LLM-CTR TRAINING (VARIATION 6)")
    if args.test_mode:
        print("⚠️  TEST MODE ENABLED - Only 3 batches per phase! ⚠️")
    print("=" * 80)
    print("\nTraining Configuration:")
    print(f"  Phase 1 (Encoder + Projector):")
    print(f"    - Datasets: x1 (1 epoch) → x2 (1 epoch)")
    print(f"    - Batch size: {args.phase1_batch_size}")
    print(f"    - Learning rate: {args.phase1_lr}")
    print(f"    - Trainable: Feature Encoder + GEN + Projector")
    print(f"    - Frozen: LLM")
    print(f"\n  Phase 2 (Projector + LLM):")
    print(f"    - Dataset: x4 (1 epoch)")
    print(f"    - Batch size: {args.phase2_batch_size}")
    print(f"    - Learning rate: {args.phase2_lr}")
    print(f"    - Trainable: Projector + LLM")
    print(f"    - Frozen: Feature Encoder + GEN")
    print(f"\n  Output: {args.output_dir}")
    print("=" * 80)

    # =========================================================================
    # STEP 1: Load Feature Map and Create Encoder from Scratch
    # =========================================================================
    print("\n[STEP 1] Creating encoder from scratch...")

    # Load config to get params
    config_path = 'model_zoo/DeepFM/config'
    experiment_id = 'DeepFM_avazu_normalized'
    params = load_config(config_path, experiment_id)

    # Fix paths
    for key in ['data_root', 'test_data', 'train_data', 'valid_data']:
        if key in params and params[key]:
            params[key] = params[key].replace('../../', '')

    # Override with command-line args
    params['embedding_dim'] = args.embedding_dim

    # Load feature map
    x4_data_dir = os.path.join(params['data_root'], 'avazu_x4_normalized')
    feature_map_path = os.path.join(x4_data_dir, 'feature_map.json')

    feature_map = FeatureMap('avazu_x4_normalized', x4_data_dir)
    feature_map.load(feature_map_path, params)
    print(f"  ✓ Loaded feature map: {feature_map.num_fields} fields, {len(feature_map.features)} features")

    # Load baseline DeepFM model to extract encoder
    print(f"\n  Loading baseline encoder from checkpoint...")
    model_src_path = "model_zoo.DeepFM.src"
    src = importlib.import_module(model_src_path)
    model_class = getattr(src, params['model'])

    # Create DeepFM model structure
    model_params = params.copy()
    model_params['train_data'] = os.path.join(x4_data_dir, 'train.parquet')
    model_params['valid_data'] = os.path.join(x4_data_dir, 'valid.parquet')
    model_params['test_data'] = os.path.join(x4_data_dir, 'test.parquet')

    deepfm_model = model_class(feature_map, **model_params)

    # Load baseline checkpoint weights
    import glob
    checkpoint_files = glob.glob(args.baseline_checkpoint)
    if not checkpoint_files:
        raise FileNotFoundError(f"Baseline checkpoint not found: {args.baseline_checkpoint}")
    checkpoint_path = checkpoint_files[0]

    print(f"  Loading weights from: {checkpoint_path}")
    deepfm_model.load_weights(checkpoint_path)
    deepfm_model.eval()

    # Extract encoder components (will be fine-tuned in Phase 1)
    embedding_layer = deepfm_model.embedding_layer
    encoder = deepfm_model.gen

    print(f"  ✓ Feature Encoder loaded from baseline (will be trainable in Phase 1)")
    print(f"  ✓ GEN Encoder loaded from baseline (will be trainable in Phase 1)")

    # =========================================================================
    # STEP 2: Load LLM with FlashAttention 2
    # =========================================================================
    print("\n[STEP 2] Loading LLM (Qwen3-0.6B) with FlashAttention 2...")

    llm = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=device
    )

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    # Get actual LLM hidden size (don't use command-line arg, use actual config)
    llm_hidden_size = llm.config.hidden_size
    print(f"  ✓ Loaded Qwen3-0.6B (actual hidden size: {llm_hidden_size})")

    # =========================================================================
    # STEP 3: Create Projector
    # =========================================================================
    print("\n[STEP 3] Creating projector...")

    # Projector takes each field's embedding and projects to LLM dimension
    # Input shape: [batch_size, num_fields, embedding_dim]
    # Output shape: [batch_size, num_fields, llm_dim]
    projector = FeatureProjector(
        feature_dim=params['embedding_dim'],
        hidden_dim=args.projector_hidden_dim,
        llm_dim=llm_hidden_size  # Use actual LLM hidden size, not args
    ).to(torch.bfloat16)

    print(f"  ✓ Created projector: {params['embedding_dim']} → {args.projector_hidden_dim} → {llm_hidden_size}")

    # =========================================================================
    # PHASE 1: Train Encoder + Projector on x1 + x2
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 1: Train Encoder + Projector (LLM frozen)")
    print("=" * 80)

    # Create Phase 1 model (encoder trainable, LLM frozen)
    prompt_template = "Predict the click-through rate (0 or 1):"
    model_phase1 = LLM_CTR_Model(
        embedding_layer,
        encoder,
        projector,
        llm,
        tokenizer,
        prompt_template,
        freeze_encoder=False,  # Encoder trainable
        freeze_llm=True        # LLM frozen
    ).to(device)

    # Phase 1 optimizer (encoder + projector)
    optimizer_phase1 = torch.optim.Adam([
        {'params': model_phase1.embedding_layer.parameters(), 'lr': args.phase1_lr},
        {'params': model_phase1.encoder.parameters(), 'lr': args.phase1_lr},
        {'params': model_phase1.projector.parameters(), 'lr': args.phase1_lr}
    ])

    # Train on x1 and x2
    for dataset_name in ['x1', 'x2']:
        print(f"\n--- Training on avazu_{dataset_name}_normalized ---")

        # Load dataset configuration
        data_dir = f'data/Avazu/avazu_{dataset_name}_normalized'

        # Setup dataset params
        dataset_params = params.copy()
        dataset_params['batch_size'] = args.phase1_batch_size
        dataset_params['train_data'] = os.path.join(data_dir, 'train.parquet')
        dataset_params['valid_data'] = os.path.join(data_dir, 'valid.parquet')

        # Create dataloaders
        train_gen, valid_gen = RankDataLoader(feature_map, stage='train', **dataset_params).make_iterator()

        # Train for 1 epoch
        train_loss, train_acc = train_epoch(
            model_phase1, train_gen, optimizer_phase1, device,
            feature_map.features.keys(), epoch=1, phase=1, dataset_name=dataset_name,
            test_mode=args.test_mode
        )

        # Validate
        val_auc, val_logloss, _, _ = evaluate(
            model_phase1, valid_gen, device, feature_map.features.keys(),
            phase=f"Phase 1 [{dataset_name}] Validation",
            test_mode=args.test_mode
        )

        print(f"\n  Phase 1 [{dataset_name}] Results:")
        print(f"    Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"    Val AUC: {val_auc:.4f}, Val LogLoss: {val_logloss:.4f}")

    # Save Phase 1 checkpoint
    phase1_checkpoint_path = os.path.join(args.output_dir, 'phase1_checkpoint.pt')
    torch.save({
        'embedding_layer': model_phase1.embedding_layer.state_dict(),
        'encoder': model_phase1.encoder.state_dict(),
        'projector': model_phase1.projector.state_dict(),
        'phase': 1,
        'config': vars(args)
    }, phase1_checkpoint_path)
    print(f"\n  ✓ Saved Phase 1 checkpoint: {phase1_checkpoint_path}")

    # =========================================================================
    # PHASE 2: Train Projector + LLM on x4
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 2: Train Projector + LLM (Encoder frozen)")
    print("=" * 80)

    # Create Phase 2 model (encoder frozen, LLM trainable)
    model_phase2 = LLM_CTR_Model(
        model_phase1.embedding_layer,  # Reuse from Phase 1
        model_phase1.encoder,          # Reuse from Phase 1
        model_phase1.projector,        # Reuse from Phase 1
        llm,
        tokenizer,
        prompt_template,
        freeze_encoder=True,   # Encoder frozen
        freeze_llm=False       # LLM trainable
    ).to(device)

    # Phase 2 optimizer (projector + LLM)
    optimizer_phase2 = torch.optim.Adam([
        {'params': model_phase2.projector.parameters(), 'lr': args.phase2_lr},
        {'params': model_phase2.llm.parameters(), 'lr': args.phase2_lr}
    ])

    # Load x4 dataset
    print("\n--- Training on avazu_x4_normalized ---")
    x4_data_dir = 'data/Avazu/avazu_x4_normalized'

    # Setup dataset params for x4
    x4_params = params.copy()
    x4_params['batch_size'] = args.phase2_batch_size
    x4_params['train_data'] = os.path.join(x4_data_dir, 'train.parquet')
    x4_params['valid_data'] = os.path.join(x4_data_dir, 'valid.parquet')
    x4_params['test_data'] = os.path.join(x4_data_dir, 'test.parquet')

    # Create dataloaders
    train_gen, valid_gen = RankDataLoader(feature_map, stage='train', **x4_params).make_iterator()

    # Train for 1 epoch
    train_loss, train_acc = train_epoch(
        model_phase2, train_gen, optimizer_phase2, device,
        feature_map.features.keys(), epoch=1, phase=2, dataset_name='x4',
        test_mode=args.test_mode
    )

    # Validate
    val_auc, val_logloss, _, _ = evaluate(
        model_phase2, valid_gen, device, feature_map.features.keys(),
        phase="Phase 2 [x4] Validation",
        test_mode=args.test_mode
    )

    print(f"\n  Phase 2 [x4] Results:")
    print(f"    Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"    Val AUC: {val_auc:.4f}, Val LogLoss: {val_logloss:.4f}")

    # Save Phase 2 checkpoint
    phase2_checkpoint_path = os.path.join(args.output_dir, 'phase2_checkpoint.pt')
    torch.save({
        'model': model_phase2.state_dict(),
        'phase': 2,
        'config': vars(args)
    }, phase2_checkpoint_path)
    print(f"\n  ✓ Saved Phase 2 checkpoint: {phase2_checkpoint_path}")

    # =========================================================================
    # EVALUATION: Test on x4
    # =========================================================================
    print("\n" + "=" * 80)
    print("EVALUATION: Test on x4")
    print("=" * 80)

    # Create test dataloader
    test_gen = RankDataLoader(feature_map, stage='test', **x4_params).make_iterator()

    test_auc, test_logloss, _, _ = evaluate(
        model_phase2, test_gen, device, feature_map.features.keys(),
        phase="Final Test [x4]",
        test_mode=args.test_mode
    )

    print(f"\n  Final Test Results:")
    print(f"    Test AUC: {test_auc:.4f}")
    print(f"    Test LogLoss: {test_logloss:.4f}")

    # Save evaluation results
    eval_results = {
        'test_auc': float(test_auc),
        'test_logloss': float(test_logloss),
        'timestamp': datetime.now().isoformat(),
        'config': vars(args)
    }

    eval_path = os.path.join(args.output_dir, 'evaluation_x4_test.json')
    with open(eval_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    print(f"\n  ✓ Saved evaluation results: {eval_path}")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nCheckpoints saved to: {args.output_dir}")
    print(f"  - phase1_checkpoint.pt (encoder + projector)")
    print(f"  - phase2_checkpoint.pt (full model)")
    print(f"  - evaluation_x4_test.json (test results)")


if __name__ == '__main__':
    main()
