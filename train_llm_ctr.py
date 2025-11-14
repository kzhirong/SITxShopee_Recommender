#!/usr/bin/env python3
"""
LLM-CTR Training Pipeline for Avazu Dataset

This script implements two-phase training with sequential multi-dataset support:
- Phase 1: Train projector only (frozen encoder) on x1 → x2
- Phase 2: Fine-tune encoder + projector on x1 → x2
- Evaluation: Test on x4

Uses LLM token prediction for "0" and "1" tokens instead of a separate prediction head.

Usage:
    # Phase 1: Train projector only on x1_sample20 and x2_sample20 sequentially
    python train_llm_ctr.py --phase 1 --datasets x1_sample20 x2_sample20 --epochs 5 --batch_size 128

    # Phase 2: Fine-tune encoder + projector on x1_sample20 and x2_sample20 sequentially
    python train_llm_ctr.py --phase 2 --datasets x1_sample20 x2_sample20 --epochs 3 --batch_size 128 --checkpoint checkpoints/llm_ctr_phase1/best_model.pt
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

    Training:
        - Phase 1: Freeze encoder, train projector only
        - Phase 2: Unfreeze encoder, train encoder + projector
        - Embeddings and LLM are always frozen
    """

    def __init__(self, embedding_layer, encoder, projector, llm, tokenizer, freeze_encoder=True):
        super().__init__()
        self.embedding_layer = embedding_layer
        self.encoder = encoder
        self.projector = projector
        self.llm = llm
        self.tokenizer = tokenizer

        # Freeze encoder if specified (Phase 1)
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

        # Convert to bfloat16 to match LLM dtype (encoder outputs float32)
        encoded = encoded.to(torch.bfloat16)

        # 3. Encoded → Projector
        projected = self.projector(encoded)  # [batch_size, num_fields, llm_dim]

        # 4. Concatenate [text + projected features]
        inputs_embeds = torch.cat([prompt_embeds, projected], dim=1)

        # 5. Feed to LLM
        llm_output = self.llm(inputs_embeds=inputs_embeds, return_dict=True)

        # 6. Get logits for next token prediction at the last position
        # AutoModelForCausalLM returns logits directly
        lm_logits = llm_output.logits  # [batch_size, seq_len, vocab_size]

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


def train_epoch(model, dataloader, optimizer, device, feature_names, prompt_embeds_cache, epoch, phase, dataset_name):
    """Train for one epoch on a single dataset."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"Phase {phase} [{dataset_name}] - Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        optimizer.zero_grad(set_to_none=True)

        # Move batch to device
        batch_dict = {feat: batch[feat].to(device) for feat in feature_names}
        labels = batch['label'].long().squeeze().to(device)  # [batch_size]

        # Get cached prompt embeddings (or slice if batch size differs)
        batch_size = labels.shape[0]
        prompt_embeds = prompt_embeds_cache[:batch_size]

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


def evaluate(model, dataloader, device, feature_names, prompt_embeds_cache, phase, dataset_name):
    """Evaluate on validation/test set."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Phase {phase} [{dataset_name}] - Evaluating")

        for batch in pbar:
            # Move batch to device
            batch_dict = {feat: batch[feat].to(device) for feat in feature_names}
            labels = batch['label'].long().squeeze().to(device)

            # Get cached prompt embeddings
            batch_size = labels.shape[0]
            prompt_embeds = prompt_embeds_cache[:batch_size]

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
                       help='Dataset versions to use (e.g., x1 x2). Sequential training.')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs PER DATASET')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--baseline_checkpoint', type=str,
                       default='model_zoo/DeepFM/Avazu/DeepFM_avazu_normalized/DeepFM_avazu_normalized.model',
                       help='Path pattern for baseline DeepFM checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to LLM-CTR checkpoint to resume from (for Phase 2)')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID (-1 for CPU)')

    args = parser.parse_args()

    # Setup device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        # Enable TF32 for faster matmul on Ampere GPUs (A100, A6000, etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print("=" * 80)
    print(f"LLM-CTR TRAINING - PHASE {args.phase}")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Phase: {args.phase} ({'Projector only' if args.phase == 1 else 'Encoder + Projector'})")
    print(f"  Datasets: {' → '.join(args.datasets)} (sequential training)")
    print(f"  Epochs per dataset: {args.epochs}")
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

    # IMPORTANT: First ensure x4_normalized parquet exists (needed for unified feature map)
    print(f"  Checking for x4_normalized parquet files...")
    x4_data_dir = os.path.join(params['data_root'], 'avazu_x4_normalized')
    x4_feature_map_json = os.path.join(x4_data_dir, "feature_map.json")

    # Create x4_normalized parquet files if they don't exist
    if not os.path.exists(x4_feature_map_json):
        print(f"  x4_normalized parquet files not found. Creating them...")
        from fuxictr.utils import load_dataset_config
        from fuxictr.preprocess import FeatureProcessor, build_dataset

        x4_params = params.copy()
        x4_params['dataset_id'] = 'avazu_x4_normalized'
        x4_specific = load_dataset_config(config_path, 'avazu_x4_normalized')
        x4_params.update(x4_specific)

        # Fix paths
        for key in ['data_root', 'test_data', 'train_data', 'valid_data']:
            if key in x4_params and x4_params[key]:
                x4_params[key] = x4_params[key].replace('../../', '')

        print(f"    Processing x4_normalized CSVs...")
        feature_encoder = FeatureProcessor(**x4_params)
        x4_params["train_data"], x4_params["valid_data"], x4_params["test_data"] = \
            build_dataset(feature_encoder, **x4_params)

        print(f"    ✓ x4_normalized parquet files created")
    else:
        print(f"    ✓ x4_normalized parquet files already exist")

    # Now create unified feature map (requires x4_normalized to exist)
    # This feature map has maximum vocab sizes across x1, x2, x4
    unified_data_dir = os.path.join(params['data_root'], 'avazu_unified')
    unified_feature_map_json = os.path.join(unified_data_dir, "feature_map.json")

    # Create unified feature map if it doesn't exist
    if not os.path.exists(unified_feature_map_json):
        print(f"  Creating unified feature map...")
        import subprocess
        result = subprocess.run(['python', 'create_unified_feature_map.py'],
                              capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr)
            raise RuntimeError("Failed to create unified feature map")

    # Load unified feature map
    feature_map = FeatureMap('avazu_unified', unified_data_dir)
    feature_map.load(unified_feature_map_json, params)

    print(f"  Features: {len(feature_map.features)}")
    print(f"  Embedding dim: {params['embedding_dim']}")

    # Initialize DeepFM model to extract embedding layer and encoder
    # The model's init_analyzer requires real data paths, so we use x4_normalized parquet files
    # (The analyzer only needs to read some data; which dataset doesn't matter)
    model_src_path = "model_zoo.DeepFM.src"
    src = importlib.import_module(model_src_path)
    model_class = getattr(src, params['model'])

    # Use x4_normalized parquet files for analyzer initialization
    model_params = params.copy()
    model_params['train_data'] = os.path.join(x4_data_dir, 'train.parquet')
    model_params['valid_data'] = os.path.join(x4_data_dir, 'valid.parquet')
    model_params['test_data'] = os.path.join(x4_data_dir, 'test.parquet')

    print(f"  Using parquet files for analyzer:")
    print(f"    Train: {model_params['train_data']}")
    print(f"    Valid: {model_params['valid_data']}")

    deepfm_model = model_class(feature_map, **model_params)

    # Find and load baseline checkpoint
    import glob
    checkpoint_files = glob.glob(args.baseline_checkpoint)
    if not checkpoint_files:
        raise FileNotFoundError(f"No baseline checkpoint found matching: {args.baseline_checkpoint}")
    checkpoint_path = checkpoint_files[0]

    print(f"  Loading baseline checkpoint: {checkpoint_path}")
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

    from transformers import AutoModelForCausalLM
    llm = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B",
        torch_dtype=torch.bfloat16,  # Use bfloat16 for A100 (2x memory reduction)
        attn_implementation="flash_attention_2"  # Use FlashAttention 2 for faster attention
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    print(f"  ✓ LLM loaded (bfloat16 + FlashAttention2)")
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
    ).to(device).to(torch.bfloat16)  # Match LLM dtype

    print(f"  Projector: {params['embedding_dim']}D → {llm.config.hidden_size}D (bfloat16)")

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

    # Load checkpoint if resuming (Phase 2)
    if args.checkpoint:
        print(f"\n  Loading LLM-CTR checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  ✓ Loaded checkpoint from epoch {checkpoint['epoch']}")

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

    # Define prompt template (Option A - Detailed)
    prompt_template = "Based on the user's browsing behavior and ad interaction features, predict if they will click on this advertisement. Answer with 1 for click or 0 for no click:"

    print(f"\n  Prompt: '{prompt_template}'")

    # Load training data
    print("\n" + "-" * 80)
    print("STEP 4: Loading training data (Sequential Multi-Dataset)")
    print("-" * 80)

    # Sequential multi-dataset training
    # We'll train on x1, then x2, sequentially
    dataset_configs = {
        'x1_sample20': 'avazu_x1_sample20',  # 20% sample for faster development
        'x2_sample20': 'avazu_x2_sample20'   # 20% sample for faster development
    }

    # PHASE 1: Preprocess all datasets first (create parquet files)
    print("\n  Phase 1: Preprocessing datasets (creating parquet files if needed)...")
    dataset_params_dict = {}  # Store params for each dataset

    for dataset_name in args.datasets:
        if dataset_name not in dataset_configs:
            raise ValueError(f"Unknown dataset: {dataset_name}. Valid: {list(dataset_configs.keys())}")

        dataset_id = dataset_configs[dataset_name]
        print(f"\n  Loading dataset: {dataset_name} ({dataset_id})")

        # Load base config from experiment_id, but override with dataset-specific params
        from fuxictr.utils import load_dataset_config
        dataset_params = params.copy()  # Start with base params

        # CRITICAL: Override dataset_id FIRST before loading dataset config
        # The base params has dataset_id='avazu_x4_normalized' which causes wrong data to be loaded!
        dataset_params['dataset_id'] = dataset_id

        # Load dataset-specific config (CSV paths, etc.)
        dataset_specific = load_dataset_config(config_path, dataset_id)
        dataset_params.update(dataset_specific)  # Override with dataset-specific paths

        # Fix paths
        for key in ['data_root', 'test_data', 'train_data', 'valid_data']:
            if key in dataset_params and dataset_params[key]:
                dataset_params[key] = dataset_params[key].replace('../../', '')

        # Load feature map for this dataset
        data_dir = os.path.join(dataset_params['data_root'], dataset_id)
        feature_map_json = os.path.join(data_dir, "feature_map.json")

        # Build dataset if feature_map doesn't exist (like run_expid.py)
        if not os.path.exists(feature_map_json) and dataset_params.get("data_format") == "csv":
            print(f"    Preprocessing {dataset_name} (building feature_map and H5 files)...")
            from fuxictr.preprocess import FeatureProcessor, build_dataset

            # Get original CSV row counts for validation
            import pandas as pd
            csv_train_path = dataset_params['train_data']
            print(f"    Counting rows in source CSV: {csv_train_path}")
            csv_row_count = sum(1 for _ in open(csv_train_path)) - 1  # -1 for header
            print(f"    Source CSV rows: {csv_row_count:,}")

            feature_encoder = FeatureProcessor(**dataset_params)
            dataset_params["train_data"], dataset_params["valid_data"], dataset_params["test_data"] = \
                build_dataset(feature_encoder, **dataset_params)
            print(f"    ✓ Dataset preprocessed")

            # CRITICAL: Validate parquet file has correct number of rows
            print(f"    Validating preprocessed parquet file...")
            parquet_train_path = dataset_params["train_data"]
            if os.path.exists(parquet_train_path):
                df_validate = pd.read_parquet(parquet_train_path)
                parquet_row_count = len(df_validate)
                print(f"    Parquet file rows: {parquet_row_count:,}")

                # Check if row counts match (allow small difference for header/format)
                row_diff = abs(parquet_row_count - csv_row_count)
                if row_diff > 100:  # Allow up to 100 rows difference for edge cases
                    print(f"\n{'=' * 80}")
                    print(f"❌ VALIDATION FAILED - WRONG DATA PREPROCESSED!")
                    print(f"{'=' * 80}")
                    print(f"Dataset: {dataset_name}")
                    print(f"Expected rows (from CSV): {csv_row_count:,}")
                    print(f"Actual rows (in parquet): {parquet_row_count:,}")
                    print(f"Difference: {row_diff:,} rows")
                    print(f"\nThis means FuxiCTR processed the WRONG source data!")
                    print(f"Check dataset_params['dataset_id'] = {dataset_params.get('dataset_id')}")
                    print(f"\nDeleting incorrect parquet files...")

                    # Clean up incorrect files
                    for split in ['train', 'valid', 'test']:
                        bad_file = os.path.join(data_dir, f'{split}.parquet')
                        if os.path.exists(bad_file):
                            os.remove(bad_file)
                            print(f"  Deleted: {bad_file}")
                    if os.path.exists(feature_map_json):
                        os.remove(feature_map_json)
                        print(f"  Deleted: {feature_map_json}")

                    raise RuntimeError(f"Preprocessing validation failed for {dataset_name}. "
                                     f"Parquet has {parquet_row_count:,} rows but CSV has {csv_row_count:,} rows. "
                                     f"Incorrect files have been deleted. Please fix the dataset_id override bug.")
                else:
                    print(f"    ✅ Validation passed! Row counts match (diff: {row_diff})")
            else:
                raise FileNotFoundError(f"Parquet file not created: {parquet_train_path}")
        elif os.path.exists(feature_map_json):
            # If feature_map exists, parquet files should exist too
            # Update data paths to point to parquet files
            dataset_params['train_data'] = os.path.join(data_dir, 'train.parquet')
            dataset_params['valid_data'] = os.path.join(data_dir, 'valid.parquet')
            dataset_params['test_data'] = os.path.join(data_dir, 'test.parquet')
        else:
            # Feature map doesn't exist and either not CSV format or preprocessing was skipped
            raise FileNotFoundError(
                f"Feature map not found for {dataset_name}: {feature_map_json}\n"
                f"Please ensure the dataset has been preprocessed. Check:\n"
                f"  1. CSV files exist at: {dataset_params.get('train_data')}\n"
                f"  2. data_format is 'csv': {dataset_params.get('data_format')}\n"
                f"  3. Run preprocessing manually if needed"
            )

        # Store params for Phase 2 (dataloader creation)
        dataset_params_dict[dataset_name] = {
            'dataset_id': dataset_id,
            'dataset_params': dataset_params,
            'data_dir': data_dir,
            'feature_map_json': feature_map_json
        }

        print(f"    ✓ {dataset_name} ready for loading")

    # PHASE 2: Load dataloaders for all preprocessed datasets
    print("\n  Phase 2: Creating dataloaders for all datasets...")
    dataloaders = {}

    for dataset_name in args.datasets:
        stored = dataset_params_dict[dataset_name]
        dataset_id = stored['dataset_id']
        dataset_params = stored['dataset_params']
        data_dir = stored['data_dir']
        feature_map_json = stored['feature_map_json']

        print(f"\n  Loading dataloader: {dataset_name} ({dataset_id})")

        # Load feature map
        dataset_feature_map = FeatureMap(dataset_id, data_dir)
        dataset_feature_map.load(feature_map_json, dataset_params)

        # Create dataloader
        dataset_params['num_workers'] = 0  # Use main process for Colab compatibility
        dataset_params['batch_size'] = args.batch_size

        train_gen, valid_gen = RankDataLoader(dataset_feature_map, stage='train', **dataset_params).make_iterator()

        dataloaders[dataset_name] = {
            'train': train_gen,
            'valid': valid_gen,
            'feature_map': dataset_feature_map
        }

        print(f"    ✓ Train batches: ~{len(train_gen)}")
        print(f"    ✓ Valid batches: ~{len(valid_gen)}")

    # Get feature names (use normalized names - all datasets have same schema)
    # Include all features: feat_1 to feat_21, hour, weekday, weekend
    feature_names = [f'feat_{i}' for i in range(1, 22)] + ['hour', 'weekday', 'weekend']  # 24 features

    print(f"\n  Total datasets for training: {len(args.datasets)}")
    print(f"  Feature names: feat_1 to feat_21 + hour, weekday, weekend (24 features)")

    # Training loop - SEQUENTIAL MULTI-DATASET
    print("\n" + "-" * 80)
    print(f"STEP 5: Training Phase {args.phase} (Sequential Multi-Dataset)")
    print("-" * 80)

    # Pre-compute prompt embeddings ONCE (huge performance boost!)
    print("\n  Pre-computing prompt embeddings...")
    prompt_embeds_cache = prepare_prompt_embeddings(
        tokenizer, llm, prompt_template, args.batch_size, device
    )
    print(f"  ✓ Prompt embeddings cached: {prompt_embeds_cache.shape}")

    best_val_acc = 0.0
    results = []

    # Train on each dataset sequentially
    for dataset_name in args.datasets:
        print(f"\n{'=' * 80}")
        print(f"TRAINING ON DATASET: {dataset_name.upper()}")
        print(f"{'=' * 80}")

        train_loader = dataloaders[dataset_name]['train']
        valid_loader = dataloaders[dataset_name]['valid']

        for epoch in range(1, args.epochs + 1):
            print(f"\nDataset: {dataset_name} | Epoch {epoch}/{args.epochs}")

            # Train
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, device, feature_names,
                prompt_embeds_cache, epoch, args.phase, dataset_name
            )

            # Validate
            val_loss, val_acc = evaluate(
                model, valid_loader, device, feature_names,
                prompt_embeds_cache, args.phase, dataset_name
            )

            print(f"\nDataset: {dataset_name} | Epoch {epoch} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

            # Save results
            results.append({
                'dataset': dataset_name,
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
                    'dataset': dataset_name,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'args': vars(args),
                    'prompt_template': prompt_template,
                    'feature_names': feature_names,
                    'unified_feature_map_path': 'model_zoo/DeepFM/Avazu/unified_feature_map.json'
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
        datasets_str = ' '.join(args.datasets)
        print(f"  python train_llm_ctr.py --phase 2 --datasets {datasets_str} --epochs 3 --checkpoint checkpoints/llm_ctr_phase1/best_model.pt")


if __name__ == "__main__":
    main()
