#!/usr/bin/env python3
"""
LLM-CTR Training Pipeline for Avazu Dataset

This script implements two-phase training:
- Phase 1: Train projector only (frozen encoder)
- Phase 2: Fine-tune encoder + projector

Uses LLM token prediction for "0" and "1" tokens instead of a separate prediction head.

Usage:
    # Phase 1: Train projector only
    python train_llm_ctr.py --phase 1 --datasets x1 x2 --epochs 5 --batch_size 32

    # Phase 2: Fine-tune encoder + projector
    python train_llm_ctr.py --phase 2 --datasets x1 x2 --epochs 3 --batch_size 32
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
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import json
from datetime import datetime

# FuxiCTR imports
from fuxictr.features import FeatureMap
from fuxictr.utils import load_config
from fuxictr.pytorch.dataloaders import RankDataLoader

# Add DeepFM source to path
sys.path.append('model_zoo/DeepFM/src')
from projector import FeatureProjector


class LLM_CTR_Model(nn.Module):
    """
    LLM-enhanced CTR prediction model.

    Architecture:
        Feature IDs → Embeddings → Encoder → Projector → [Text + Features] → LLM → Token Logits
    """

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

        print(f"Token ID for '0': {self.token_0}")
        print(f"Token ID for '1': {self.token_1}")

    def forward(self, batch_dict, prompt_embeds):
        """
        Forward pass through the complete pipeline.

        Args:
            batch_dict: Dictionary of feature tensors
            prompt_embeds: Pre-computed text prompt embeddings [batch_size, seq_len, llm_dim]

        Returns:
            logits: Binary classification logits [batch_size, 2] for [class_0, class_1]
        """
        batch_size = prompt_embeds.shape[0]

        # 1. Feature IDs → Embeddings
        embedded = self.embedding_layer(batch_dict)  # [batch_size, num_fields, emb_dim]

        # 2. Embeddings → Encoder
        encoder_output = self.encoder(embedded)
        if isinstance(encoder_output, tuple):
            encoded = encoder_output[0]
        else:
            encoded = encoder_output

        # 3. Encoded → Projector
        projected = self.projector(encoded)  # [batch_size, num_fields, llm_dim]

        # 4. Concatenate [text + projected features]
        inputs_embeds = torch.cat([prompt_embeds, projected], dim=1)

        # 5. Feed to LLM
        llm_output = self.llm(inputs_embeds=inputs_embeds, return_dict=True)

        # 6. Get logits for next token prediction at the last position
        # LLM models typically have a language modeling head
        # We'll use the last hidden state and project to vocabulary
        lm_logits = self.llm.lm_head(llm_output.last_hidden_state)  # [batch_size, seq_len, vocab_size]

        # Get logits at the last position (after all inputs)
        last_token_logits = lm_logits[:, -1, :]  # [batch_size, vocab_size]

        # Extract logits for tokens "0" and "1"
        logit_0 = last_token_logits[:, self.token_0]  # [batch_size]
        logit_1 = last_token_logits[:, self.token_1]  # [batch_size]

        # Stack into [batch_size, 2] for cross-entropy
        logits = torch.stack([logit_0, logit_1], dim=1)

        return logits


def prepare_prompt_embeddings(tokenizer, llm, prompt_template, batch_size, device):
    """
    Pre-compute and cache prompt embeddings for efficiency.

    Args:
        tokenizer: LLM tokenizer
        llm: LLM model
        prompt_template: Text prompt string
        batch_size: Batch size to replicate for
        device: Device to place embeddings on

    Returns:
        prompt_embeds: [batch_size, seq_len, llm_dim]
    """
    tokens = tokenizer(prompt_template, return_tensors="pt").to(device)

    with torch.no_grad():
        # Get embeddings for the prompt
        prompt_embeds = llm.get_input_embeddings()(tokens.input_ids)  # [1, seq_len, llm_dim]

        # Replicate for batch
        prompt_embeds = prompt_embeds.repeat(batch_size, 1, 1)  # [batch_size, seq_len, llm_dim]

    return prompt_embeds


def train_epoch(model, dataloader, optimizer, device, feature_names, prompt_template, epoch, phase):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"Phase {phase} - Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        optimizer.zero_grad()

        # Move batch to device
        batch_dict = {feat: batch[feat].to(device) for feat in feature_names}
        labels = batch['click'].long().squeeze().to(device)  # [batch_size]

        # Prepare prompt embeddings for this batch
        batch_size = labels.shape[0]
        prompt_embeds = prepare_prompt_embeddings(
            model.tokenizer, model.llm, prompt_template, batch_size, device
        )

        # Forward pass
        logits = model(batch_dict, prompt_embeds)  # [batch_size, 2]

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


def evaluate(model, dataloader, device, feature_names, prompt_template, phase):
    """Evaluate on validation/test set."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Phase {phase} - Evaluating")

        for batch in pbar:
            # Move batch to device
            batch_dict = {feat: batch[feat].to(device) for feat in feature_names}
            labels = batch['click'].long().squeeze().to(device)

            # Prepare prompt embeddings
            batch_size = labels.shape[0]
            prompt_embeds = prepare_prompt_embeddings(
                model.tokenizer, model.llm, prompt_template, batch_size, device
            )

            # Forward pass
            logits = model(batch_dict, prompt_embeds)
            loss = F.cross_entropy(logits, labels)

            # Track metrics
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * correct / total:.2f}%'
            })

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='Train LLM-CTR model')
    parser.add_argument('--phase', type=int, required=True, choices=[1, 2],
                       help='Training phase: 1 (projector only) or 2 (encoder + projector)')
    parser.add_argument('--datasets', nargs='+', default=['x1', 'x2'],
                       help='Dataset versions to use (e.g., x1 x2)')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--checkpoint', type=str,
                       default='model_zoo/DeepFM/Avazu/DeepFM_avazu_normalized/avazu_x4_normalized_*/DeepFM_avazu_normalized.model',
                       help='Path pattern for baseline checkpoint')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID (-1 for CPU)')

    args = parser.parse_args()

    # Setup device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print("=" * 80)
    print(f"LLM-CTR TRAINING - PHASE {args.phase}")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Phase: {args.phase} ({'Projector only' if args.phase == 1 else 'Encoder + Projector'})")
    print(f"  Datasets: {', '.join(args.datasets)}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Device: {device}")

    # Load baseline model configuration
    print("\n" + "-" * 80)
    print("STEP 1: Loading baseline DeepFM model")
    print("-" * 80)

    config_path = 'model_zoo/DeepFM/config'
    experiment_id = 'DeepFM_avazu_normalized'
    params = load_config(config_path, experiment_id)

    # Fix paths
    for key in ['data_root', 'test_data', 'train_data', 'valid_data']:
        if key in params and params[key]:
            params[key] = params[key].replace('../../', '')

    # Load feature map
    data_dir = os.path.join(params['data_root'], params['dataset_id'])
    feature_map_json = os.path.join(data_dir, "feature_map.json")
    feature_map = FeatureMap(params['dataset_id'], data_dir)
    feature_map.load(feature_map_json, params)

    print(f"  Features: {len(feature_map.features)}")
    print(f"  Embedding dim: {params['embedding_dim']}")

    # Load full DeepFM model
    model_src_path = "model_zoo.DeepFM.src"
    src = importlib.import_module(model_src_path)
    model_class = getattr(src, params['model'])
    deepfm_model = model_class(feature_map, **params)

    # Find and load checkpoint
    import glob
    checkpoint_files = glob.glob(args.checkpoint)
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint found matching: {args.checkpoint}")
    checkpoint_path = checkpoint_files[0]

    print(f"  Loading checkpoint: {checkpoint_path}")
    deepfm_model.load_weights(checkpoint_path)
    deepfm_model.eval()

    # Extract components
    embedding_layer = deepfm_model.embedding_layer
    encoder = deepfm_model.gen

    print(f"  ✓ Embedding layer extracted")
    print(f"  ✓ Encoder (GEN) extracted")

    # Load LLM and tokenizer
    print("\n" + "-" * 80)
    print("STEP 2: Loading LLM (Qwen3-0.6B)")
    print("-" * 80)

    llm = AutoModel.from_pretrained("Qwen/Qwen3-0.6B").to(device)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    # Ensure LLM has lm_head
    if not hasattr(llm, 'lm_head'):
        # Load the full language model instead
        from transformers import AutoModelForCausalLM
        llm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B").to(device)

    print(f"  ✓ LLM loaded")
    print(f"  ✓ Tokenizer loaded")

    # Create projector
    print("\n" + "-" * 80)
    print("STEP 3: Creating projector")
    print("-" * 80)

    projector = FeatureProjector(
        feature_dim=params['embedding_dim'],
        llm_dim=llm.config.hidden_size,
        hidden_dim=512
    ).to(device)

    print(f"  Projector: {params['embedding_dim']}D → {llm.config.hidden_size}D")

    # Create LLM-CTR model
    freeze_encoder = (args.phase == 1)
    model = LLM_CTR_Model(
        embedding_layer=embedding_layer,
        encoder=encoder,
        projector=projector,
        llm=llm,
        tokenizer=tokenizer,
        freeze_encoder=freeze_encoder
    ).to(device)

    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen: {total_params - trainable_params:,}")

    # Setup optimizer
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )

    # Define prompt template
    prompt_template = "Based on the following user features, predict if the user will click on this ad. Answer with 1 for click or 0 for no click:"

    print(f"\n  Prompt: '{prompt_template}'")

    # Load training data
    print("\n" + "-" * 80)
    print("STEP 4: Loading training data")
    print("-" * 80)

    # TODO: For now, use x4 as placeholder. Need to implement multi-dataset loading
    # In the actual implementation, we'd load x1 and x2 and combine them

    params['num_workers'] = 0
    params['batch_size'] = args.batch_size

    train_gen, valid_gen = RankDataLoader(feature_map, stage='train', **params).make_iterator()

    # Get feature names (use normalized names)
    feature_names = [f'feat_{i}' for i in range(1, 22)]  # 21 features

    print(f"  Training batches: ~{len(train_gen)}")
    print(f"  Validation batches: ~{len(valid_gen)}")

    # Training loop
    print("\n" + "-" * 80)
    print(f"STEP 5: Training Phase {args.phase}")
    print("-" * 80)

    best_val_acc = 0.0
    results = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_gen, optimizer, device, feature_names, prompt_template, epoch, args.phase
        )

        # Validate
        val_loss, val_acc = evaluate(
            model, valid_gen, device, feature_names, prompt_template, args.phase
        )

        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

        # Save results
        results.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_dir = Path(f'checkpoints/llm_ctr_phase{args.phase}')
            save_dir.mkdir(parents=True, exist_ok=True)

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'args': vars(args),
                'prompt_template': prompt_template
            }

            save_path = save_dir / 'best_model.pt'
            torch.save(checkpoint, save_path)
            print(f"  ✓ Saved best model to {save_path}")

    # Save training results
    results_file = Path(f'checkpoints/llm_ctr_phase{args.phase}/training_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"✅ PHASE {args.phase} TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nBest validation accuracy: {best_val_acc:.2f}%")
    print(f"Results saved to: {results_file}")

    if args.phase == 1:
        print("\nNext step: Run Phase 2 to fine-tune encoder + projector")
        print(f"  python train_llm_ctr.py --phase 2 --datasets x1 x2 --epochs 3")


if __name__ == "__main__":
    main()
