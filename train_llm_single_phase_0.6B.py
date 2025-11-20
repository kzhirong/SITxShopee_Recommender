#!/usr/bin/env python3
"""
LLM-CTR Single-Phase Training Pipeline for Avazu Dataset

This script implements single-phase training on x4 only (simplified alternative to two-phase approach):
- Train Feature Encoder + GEN + Projector + LLM together on x4
- Integrated evaluation on x4 test set after training

Architecture:
    Feature IDs → Embeddings → Encoder (GEN) → Projector → [Text + Features] → LLM → Token Logits

Training:
    - Trained: Feature Encoder + Encoder (GEN) + Projector + LLM (all trained together from scratch)

Usage:
    python train_llm_single_phase_0.6B.py --epochs 1 --batch_size 256 --gpu 0
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

# Import FuxiCTR components for encoder
from fuxictr.pytorch.layers import FeatureEmbedding
from fuxictr.pytorch import GEN


class LLM_CTR_Model(nn.Module):
    """
    LLM-enhanced CTR prediction model (Single-Phase variant).

    Architecture:
        Feature IDs → Embeddings → Encoder → Projector → [Text + Features] → LLM → Token Logits

    Training:
        - Trained: Embeddings + Encoder + Projector + LLM (all trained together)
    """

    def __init__(self, embedding_layer, encoder, projector, llm, tokenizer, prompt_template):
        super().__init__()
        self.embedding_layer = embedding_layer
        self.encoder = encoder
        self.projector = projector
        self.llm = llm
        self.tokenizer = tokenizer

        # ALL components are trainable (nothing frozen)
        for param in self.encoder.parameters():
            param.requires_grad = True
        for param in self.embedding_layer.parameters():
            param.requires_grad = True
        for param in self.llm.parameters():
            param.requires_grad = True

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
        """
        Forward pass through the complete pipeline.

        Args:
            batch_dict: Dictionary of feature tensors

        Returns:
            logits: Binary classification logits [batch_size, 2] for [class_0, class_1]
        """
        # 1. Feature IDs → Embeddings
        embedded = self.embedding_layer(batch_dict)  # [batch_size, num_fields, emb_dim]
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
        projected = self.projector(encoded)  # [batch_size, num_fields, llm_dim]

        # 4. Compute prompt embeddings from cached token IDs
        # Token IDs are cached (never change), but embeddings computed fresh using current LLM weights
        # This ensures embeddings update as LLM trains (no stale embeddings!)
        prompt_token_ids = self.prompt_token_ids.to(projected.device)  # Move to correct device
        prompt_embeds = self.llm.get_input_embeddings()(prompt_token_ids)  # [1, seq_len, llm_dim] - uses CURRENT weights!
        prompt_embeds = prompt_embeds.repeat(batch_size, 1, 1)  # [batch_size, seq_len, llm_dim]

        # 5. Concatenate [text + projected features]
        inputs_embeds = torch.cat([prompt_embeds, projected], dim=1)

        # 6. Feed to LLM
        llm_output = self.llm(inputs_embeds=inputs_embeds, return_dict=True)

        # 7. Get logits for next token prediction at the last position
        lm_logits = llm_output.logits  # [batch_size, seq_len, vocab_size]
        last_token_logits = lm_logits[:, -1, :]  # [batch_size, vocab_size]

        # Extract logits for tokens "0" and "1"
        logit_0 = last_token_logits[:, self.token_0]  # [batch_size]
        logit_1 = last_token_logits[:, self.token_1]  # [batch_size]

        # Stack into [batch_size, 2] for cross-entropy
        logits = torch.stack([logit_0, logit_1], dim=1)

        return logits


def train_epoch(model, dataloader, optimizer, device, feature_names, epoch, test_mode=False):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"Training - Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        # Test mode: only process 3 batches
        if test_mode and batch_idx >= 3:
            print(f"  [TEST MODE] Stopping after {batch_idx} batches")
            break
        optimizer.zero_grad(set_to_none=True)

        # Move batch to device
        batch_dict = {feat: batch[feat].to(device) for feat in feature_names}
        labels = batch['label'].long().squeeze().to(device)  # [batch_size]

        # Forward pass (prompt token IDs cached, embeddings computed fresh!)
        logits = model(batch_dict)  # [batch_size, 2]

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

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def evaluate(model, dataloader, device, feature_names, split_name="Validation", test_mode=False):
    """Evaluate on validation/test set with AUC calculation."""
    model.eval()
    total_loss = 0
    all_labels = []
    all_probs = []
    all_preds = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Evaluating {split_name}")

        for batch_idx, batch in enumerate(pbar):
            # Test mode: only process 3 batches
            if test_mode and batch_idx >= 3:
                print(f"  [TEST MODE] Stopping evaluation after {batch_idx} batches")
                break
            # Move batch to device
            batch_dict = {feat: batch[feat].to(device) for feat in feature_names}
            labels = batch['label'].long().squeeze().to(device)

            # Forward pass (prompt token IDs cached, embeddings computed fresh!)
            logits = model(batch_dict)
            loss = F.cross_entropy(logits, labels)

            # Get probabilities for AUC
            probs = F.softmax(logits, dim=1)[:, 1]  # P(class=1)
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

    avg_loss = total_loss / len(dataloader)
    accuracy = (all_preds == all_labels).mean() * 100

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
    parser = argparse.ArgumentParser(description='Train LLM-CTR model (Single-Phase)')
    parser.add_argument('--epochs', type=int, default=100,
                    help='Number of training epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=256,
                    help='Batch size (default: 256)')
    parser.add_argument('--lr', type=float, default=1e-4,
                    help='Learning rate for LLM/projector (default: 1e-4)')
    parser.add_argument('--encoder_lr', type=float, default=1e-3,
                    help='Learning rate for encoder/embeddings (default: 1e-3)')
    parser.add_argument('--embedding_dim', type=int, default=16,
                    help='Embedding dimension (default: 16)')
    parser.add_argument('--patience', type=int, default=2,
                    help='Early stopping patience (default: 2)')
    parser.add_argument('--gpu', type=int, default=0,
                    help='GPU device ID (-1 for CPU)')

    # Testing
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
    print("LLM-CTR SINGLE-PHASE TRAINING (0.6B)")
    print("=" * 80)

    if args.test_mode:
        print("\n⚠ ⚠ ⚠  TEST MODE ENABLED  ⚠ ⚠ ⚠")
        print("  Only 3 batches per epoch will be processed for quick validation")
        print("  This is NOT for actual training - only for testing the pipeline!")
        print("")

    print(f"\nConfiguration:")
    print(f"  Model: Qwen3-0.6B")
    print(f"  Dataset: x4 unified (train + eval)")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate (LLM/Projector): {args.lr}")
    print(f"  Learning rate (Encoder/Embeddings): {args.encoder_lr}")
    print(f"  Embedding dimension: {args.embedding_dim}")
    print(f"  Device: {device}")
    print(f"  Training: Feature Encoder + GEN + Projector + LLM (all trained together)")
    if args.test_mode:
        print(f"  Test mode: Only 3 batches per epoch")

    # Load configuration for x4 dataset
    print("\n" + "-" * 80)
    print("STEP 1: Loading x4 dataset configuration")
    print("-" * 80)

    config_path = 'model_zoo/DeepFM/config'
    experiment_id = 'DeepFM_avazu_normalized'
    params = load_config(config_path, experiment_id)

    # Fix paths
    for key in ['data_root', 'test_data', 'train_data', 'valid_data']:
        if key in params and params[key]:
            params[key] = params[key].replace('../../', '')

    # Override embedding_dim if specified
    params['embedding_dim'] = args.embedding_dim

    # Load x4_normalized data
    x4_data_dir = os.path.join(params['data_root'], 'avazu_x4_normalized')
    x4_feature_map_json = os.path.join(x4_data_dir, "feature_map.json")

    if not os.path.exists(x4_feature_map_json):
        raise FileNotFoundError(
            f"x4_normalized feature map not found: {x4_feature_map_json}\n"
            f"Please ensure x4 dataset has been preprocessed."
        )

    # Load feature map
    feature_map = FeatureMap('avazu_x4_normalized', x4_data_dir)
    feature_map.load(x4_feature_map_json, params)

    print(f"  Features: {len(feature_map.features)}")
    print(f"  Embedding dim: {params['embedding_dim']}")

    # Initialize Feature Encoder and GEN from scratch
    print(f"\n  Initializing Feature Encoder and GEN from scratch...")

    embedding_layer = FeatureEmbedding(feature_map, params['embedding_dim'])

    # Remove embedding_dim from params dict to avoid duplicate argument error
    gen_params = {k: v for k, v in params.items() if k != 'embedding_dim'}
    encoder = GEN(feature_map, params['embedding_dim'], **gen_params)

    print(f"  ✓ Feature Encoder initialized (trainable)")
    print(f"  ✓ GEN Encoder initialized (trainable)")

    # Load LLM and tokenizer
    print("\n" + "-" * 80)
    print("STEP 2: Loading LLM (Qwen3-0.6B)")
    print("-" * 80)
    print("  ⚠ REQUIREMENT: FlashAttention 2 and Ampere+ GPU (A100, L4, H100, etc.)")

    # Check CUDA
    if device.type != 'cuda':
        raise RuntimeError(
            f"FlashAttention 2 requires CUDA GPU, but running on {device}.\n"
            f"This script requires an Ampere or newer GPU (A100, L4, H100, etc.)."
        )

    # Check FlashAttention installed
    try:
        import flash_attn  # noqa: F401
        print("  ✓ FlashAttention 2 package found")
    except ImportError:
        raise ImportError(
            "FlashAttention 2 is not installed!\n"
            "Install with: pip install flash-attn --no-build-isolation\n"
            "This script requires FlashAttention 2 to run."
        )

    # Load LLM with FlashAttention 2
    llm = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    ).to(device)

    # Test FlashAttention compatibility
    print("  Testing FlashAttention 2 compatibility...")
    with torch.no_grad():
        test_input = torch.randn(1, 10, llm.config.hidden_size,
                                dtype=torch.bfloat16, device=device)
        try:
            _ = llm(inputs_embeds=test_input)
            print("  ✓ FlashAttention 2 working correctly")
        except RuntimeError as e:
            if "ampere" in str(e).lower() or "flash" in str(e).lower():
                raise RuntimeError(
                    f"GPU does not support FlashAttention 2!\n"
                    f"Error: {e}\n\n"
                    f"FlashAttention 2 requires Ampere or newer GPU architecture:\n"
                    f"  ✓ Supported: A100, L4, H100, RTX 3090, RTX 4090, etc.\n"
                    f"  ✗ Not supported: V100, T4, K80, P100, etc.\n\n"
                    f"Your GPU does not meet the requirements."
                )
            else:
                raise

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    print(f"  ✓ LLM loaded with FlashAttention 2 (bfloat16)")
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

    # Create LLM-CTR model (prompt_template cached during initialization)
    model = LLM_CTR_Model(
        embedding_layer=embedding_layer,
        encoder=encoder,
        projector=projector,
        llm=llm,
        tokenizer=tokenizer,
        prompt_template=prompt_template
    ).to(device)

    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,} (100% - all components trainable)")

    # Setup optimizer with different learning rates for different components
    # Higher LR for encoder/embeddings (1e-3), lower for LLM/projector (1e-4)
    optimizer = torch.optim.Adam([
        {'params': model.embedding_layer.parameters(), 'lr': args.encoder_lr},
        {'params': model.encoder.parameters(), 'lr': args.encoder_lr},
        {'params': model.projector.parameters(), 'lr': args.lr},
        {'params': model.llm.parameters(), 'lr': args.lr}
    ])

    print(f"\n  Optimizer configured:")
    print(f"    Embedding layer: lr={args.encoder_lr}")
    print(f"    GEN encoder: lr={args.encoder_lr}")
    print(f"    Projector: lr={args.lr}")
    print(f"    LLM: lr={args.lr}")

    # Load training data
    print("\n" + "-" * 80)
    print("STEP 4: Loading x4 dataset")
    print("-" * 80)

    # Create dataloaders
    params['num_workers'] = 0  # Colab compatibility
    params['batch_size'] = args.batch_size

    # Set data paths for x4
    params['train_data'] = os.path.join(x4_data_dir, 'train.parquet')
    params['valid_data'] = os.path.join(x4_data_dir, 'valid.parquet')
    params['test_data'] = os.path.join(x4_data_dir, 'test.parquet')

    train_gen, valid_gen = RankDataLoader(feature_map, stage='train', **params).make_iterator()
    test_gen = RankDataLoader(feature_map, stage='test', **params).make_iterator()

    print(f"  ✓ Train batches: {len(train_gen)}")
    print(f"  ✓ Valid batches: {len(valid_gen)}")
    print(f"  ✓ Test batches: {len(test_gen)}")

    # Feature names
    feature_names = [f'feat_{i}' for i in range(1, 22)] + ['hour', 'weekday', 'weekend']
    print(f"  Features: feat_1 to feat_21 + hour, weekday, weekend (24 features)")

    # Training loop with early stopping
    print("\n" + "-" * 80)
    print("STEP 5: Training")
    print("-" * 80)
    print(f"\n  Training for up to {args.epochs} epochs with early stopping (patience={args.patience})")
    print(f"  NOTE: Prompt token IDs cached (fast), embeddings computed fresh each batch")
    print(f"        This ensures embeddings update as LLM trains (no stale embeddings!)")

    results = []
    best_val_auc = 0.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'=' * 80}")
        print(f"EPOCH {epoch}/{args.epochs}")
        print(f"{'=' * 80}")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_gen, optimizer, device, feature_names, epoch, test_mode=args.test_mode
        )

        # Validate
        val_loss, val_acc, val_auc, val_logloss = evaluate(
            model, valid_gen, device, feature_names, "Validation", test_mode=args.test_mode
        )

        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"  Val AUC:    {val_auc:.4f}")
        print(f"  Val LogLoss: {val_logloss:.4f}")

        # Save results
        results.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_auc': val_auc,
            'val_logloss': val_logloss
        })

        # Save best model and early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0  # Reset patience counter

            save_dir = Path('checkpoints/llm_single_phase_0.6B')
            save_dir.mkdir(parents=True, exist_ok=True)

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_auc,
                'val_loss': val_loss,
                'args': vars(args),
                'prompt_template': prompt_template,
                'feature_names': feature_names
            }

            save_path = save_dir / 'best_model.pt'
            torch.save(checkpoint, save_path)
            print(f"  ✓ Saved best model to {save_path} (AUC improved: {val_auc:.4f})")
        else:
            patience_counter += 1
            print(f"  ⚠ No improvement (patience: {patience_counter}/{args.patience})")

            # Early stopping
            if patience_counter >= args.patience:
                print(f"\n{'=' * 80}")
                print(f"Early stopping triggered! No improvement for {args.patience} epochs.")
                print(f"Best validation AUC: {best_val_auc:.4f}")
                print(f"{'=' * 80}")
                break

    # Save training results
    save_dir = Path('checkpoints/llm_single_phase_0.6B')
    save_dir.mkdir(parents=True, exist_ok=True)
    results_file = save_dir / 'training_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("STEP 6: Evaluating on Test Set")
    print("=" * 80)

    # Evaluate on test set
    test_loss, test_acc, test_auc, test_logloss = evaluate(
        model, test_gen, device, feature_names, "Test", test_mode=args.test_mode
    )

    print(f"\nTest Set Results:")
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
    print("✅ TRAINING AND EVALUATION COMPLETE")
    print("=" * 80)
    print(f"\nBest validation AUC: {best_val_auc:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"\nResults saved to:")
    print(f"  Training: {results_file}")
    print(f"  Test evaluation: {test_results_file}")
    print(f"  Model checkpoint: {save_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
