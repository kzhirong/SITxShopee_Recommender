#!/usr/bin/env python3
"""
Evaluation script to compare baseline DeepFM vs LLM-CTR models on Avazu x4 test set.

This script:
1. Loads the baseline DeepFM model
2. Loads the trained LLM-CTR model (Phase 1 or Phase 2)
3. Evaluates both on the x4 test set
4. Compares performance metrics (AUC, accuracy, log loss)
5. Generates comparison report

Usage:
    # Evaluate Phase 1 model
    python evaluate_comparison.py --phase 1

    # Evaluate Phase 2 model
    python evaluate_comparison.py --phase 2
"""

import sys
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import argparse
import importlib
from pathlib import Path
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
import numpy as np
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

# Import LLM-CTR model
from train_llm_ctr import LLM_CTR_Model, prepare_prompt_embeddings

from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM


def evaluate_baseline_deepfm(model, dataloader, device, feature_names):
    """Evaluate baseline DeepFM model."""
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Baseline DeepFM"):
            # Move batch to device
            batch_dict = {feat: batch[feat].to(device) for feat in feature_names}
            labels = batch['click'].squeeze().cpu().numpy()

            # Forward pass through DeepFM
            logits = model.forward(batch_dict)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_preds.extend(probs.flatten().tolist())
            all_labels.extend(labels.flatten().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate metrics
    auc = roc_auc_score(all_labels, all_preds)
    logloss = log_loss(all_labels, all_preds)
    binary_preds = (all_preds >= 0.5).astype(int)
    accuracy = accuracy_score(all_labels, binary_preds)

    return {
        'auc': auc,
        'logloss': logloss,
        'accuracy': accuracy,
        'predictions': all_preds,
        'labels': all_labels
    }


def evaluate_llm_ctr(model, dataloader, device, feature_names, prompt_template):
    """Evaluate LLM-CTR model."""
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating LLM-CTR"):
            # Move batch to device
            batch_dict = {feat: batch[feat].to(device) for feat in feature_names}
            labels = batch['click'].long().squeeze()

            # Prepare prompt embeddings
            batch_size = labels.shape[0]
            prompt_embeds = prepare_prompt_embeddings(
                model.tokenizer, model.llm, prompt_template, batch_size, device
            )

            # Forward pass
            logits = model(batch_dict, prompt_embeds)  # [batch_size, 2]

            # Convert logits to probabilities
            probs = F.softmax(logits, dim=1)[:, 1]  # Probability of class 1 (click)

            all_preds.extend(probs.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate metrics
    auc = roc_auc_score(all_labels, all_preds)
    logloss = log_loss(all_labels, all_preds)
    binary_preds = (all_preds >= 0.5).astype(int)
    accuracy = accuracy_score(all_labels, binary_preds)

    return {
        'auc': auc,
        'logloss': logloss,
        'accuracy': accuracy,
        'predictions': all_preds,
        'labels': all_labels
    }


def print_comparison(baseline_results, llm_ctr_results, phase):
    """Print comparison table."""
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS COMPARISON")
    print("=" * 80)

    print(f"\n{'Metric':<20} {'Baseline DeepFM':<20} {'LLM-CTR (Phase {phase})':<20} {'Improvement':<15}")
    print("-" * 80)

    # AUC
    auc_diff = llm_ctr_results['auc'] - baseline_results['auc']
    auc_pct = (auc_diff / baseline_results['auc']) * 100
    print(f"{'AUC':<20} {baseline_results['auc']:<20.6f} {llm_ctr_results['auc']:<20.6f} {auc_pct:>+.2f}%")

    # Log Loss (lower is better)
    ll_diff = baseline_results['logloss'] - llm_ctr_results['logloss']  # Reversed
    ll_pct = (ll_diff / baseline_results['logloss']) * 100
    print(f"{'Log Loss':<20} {baseline_results['logloss']:<20.6f} {llm_ctr_results['logloss']:<20.6f} {ll_pct:>+.2f}%")

    # Accuracy
    acc_diff = llm_ctr_results['accuracy'] - baseline_results['accuracy']
    acc_pct = (acc_diff / baseline_results['accuracy']) * 100
    print(f"{'Accuracy':<20} {baseline_results['accuracy']:<20.6f} {llm_ctr_results['accuracy']:<20.6f} {acc_pct:>+.2f}%")

    print("\n" + "=" * 80)

    # Determine winner
    if llm_ctr_results['auc'] > baseline_results['auc']:
        print("✅ LLM-CTR outperforms baseline DeepFM on AUC")
    else:
        print("⚠️  Baseline DeepFM outperforms LLM-CTR on AUC")


def main():
    parser = argparse.ArgumentParser(description='Compare baseline DeepFM vs LLM-CTR')
    parser.add_argument('--phase', type=int, required=True, choices=[1, 2],
                       help='LLM-CTR phase to evaluate: 1 or 2')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size for evaluation')
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
    print("MODEL COMPARISON EVALUATION")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  LLM-CTR Phase: {args.phase}")
    print(f"  Test dataset: Avazu x4")
    print(f"  Batch size: {args.batch_size}")
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

    # Load baseline DeepFM model
    model_src_path = "model_zoo.DeepFM.src"
    src = importlib.import_module(model_src_path)
    model_class = getattr(src, params['model'])
    baseline_model = model_class(feature_map, **params)

    # Find and load baseline checkpoint
    import glob
    checkpoint_pattern = 'model_zoo/DeepFM/Avazu/DeepFM_avazu_normalized/avazu_x4_normalized_*/DeepFM_avazu_normalized.model'
    checkpoint_files = glob.glob(checkpoint_pattern)
    if not checkpoint_files:
        raise FileNotFoundError(f"No baseline checkpoint found matching: {checkpoint_pattern}")
    baseline_checkpoint_path = checkpoint_files[0]

    print(f"  Loading checkpoint: {baseline_checkpoint_path}")
    baseline_model.load_weights(baseline_checkpoint_path)
    baseline_model.to(device)
    baseline_model.eval()

    print(f"  ✓ Baseline DeepFM loaded")

    # Load LLM-CTR model
    print("\n" + "-" * 80)
    print(f"STEP 2: Loading LLM-CTR Phase {args.phase} model")
    print("-" * 80)

    # Extract components from baseline
    embedding_layer = baseline_model.embedding_layer
    encoder = baseline_model.gen

    # Load LLM and tokenizer
    llm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B").to(device)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    # Create projector
    projector = FeatureProjector(
        feature_dim=params['embedding_dim'],
        llm_dim=llm.config.hidden_size,
        hidden_dim=512
    ).to(device)

    # Create LLM-CTR model
    freeze_encoder = (args.phase == 1)
    llm_ctr_model = LLM_CTR_Model(
        embedding_layer=embedding_layer,
        encoder=encoder,
        projector=projector,
        llm=llm,
        tokenizer=tokenizer,
        freeze_encoder=freeze_encoder
    ).to(device)

    # Load trained checkpoint
    checkpoint_path = Path(f'checkpoints/llm_ctr_phase{args.phase}/best_model.pt')
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"LLM-CTR checkpoint not found: {checkpoint_path}")

    print(f"  Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    llm_ctr_model.load_state_dict(checkpoint['model_state_dict'])
    llm_ctr_model.eval()

    prompt_template = checkpoint['prompt_template']

    print(f"  ✓ LLM-CTR Phase {args.phase} loaded")
    print(f"  ✓ Validation accuracy during training: {checkpoint['val_acc']:.2f}%")

    # Load test data
    print("\n" + "-" * 80)
    print("STEP 3: Loading test data (Avazu x4)")
    print("-" * 80)

    params['num_workers'] = 0
    params['batch_size'] = args.batch_size

    test_gen = RankDataLoader(feature_map, stage='test', **params).make_iterator()

    # Get feature names (normalized)
    feature_names = [f'feat_{i}' for i in range(1, 22)]  # 21 features

    print(f"  Test batches: ~{len(test_gen)}")

    # Evaluate baseline DeepFM
    print("\n" + "-" * 80)
    print("STEP 4: Evaluating baseline DeepFM")
    print("-" * 80)

    baseline_results = evaluate_baseline_deepfm(baseline_model, test_gen, device, feature_names)

    print(f"\nBaseline DeepFM Results:")
    print(f"  AUC: {baseline_results['auc']:.6f}")
    print(f"  Log Loss: {baseline_results['logloss']:.6f}")
    print(f"  Accuracy: {baseline_results['accuracy']:.6f}")

    # Evaluate LLM-CTR
    print("\n" + "-" * 80)
    print(f"STEP 5: Evaluating LLM-CTR Phase {args.phase}")
    print("-" * 80)

    # Reload test data (iterator was consumed)
    test_gen = RankDataLoader(feature_map, stage='test', **params).make_iterator()

    llm_ctr_results = evaluate_llm_ctr(llm_ctr_model, test_gen, device, feature_names, prompt_template)

    print(f"\nLLM-CTR Phase {args.phase} Results:")
    print(f"  AUC: {llm_ctr_results['auc']:.6f}")
    print(f"  Log Loss: {llm_ctr_results['logloss']:.6f}")
    print(f"  Accuracy: {llm_ctr_results['accuracy']:.6f}")

    # Print comparison
    print_comparison(baseline_results, llm_ctr_results, args.phase)

    # Save results to file
    print("\n" + "-" * 80)
    print("STEP 6: Saving results")
    print("-" * 80)

    results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'phase': args.phase,
        'baseline_deepfm': {
            'auc': float(baseline_results['auc']),
            'logloss': float(baseline_results['logloss']),
            'accuracy': float(baseline_results['accuracy'])
        },
        'llm_ctr': {
            'auc': float(llm_ctr_results['auc']),
            'logloss': float(llm_ctr_results['logloss']),
            'accuracy': float(llm_ctr_results['accuracy'])
        },
        'improvements': {
            'auc_diff': float(llm_ctr_results['auc'] - baseline_results['auc']),
            'auc_pct': float((llm_ctr_results['auc'] - baseline_results['auc']) / baseline_results['auc'] * 100),
            'logloss_diff': float(baseline_results['logloss'] - llm_ctr_results['logloss']),
            'logloss_pct': float((baseline_results['logloss'] - llm_ctr_results['logloss']) / baseline_results['logloss'] * 100),
            'accuracy_diff': float(llm_ctr_results['accuracy'] - baseline_results['accuracy']),
            'accuracy_pct': float((llm_ctr_results['accuracy'] - baseline_results['accuracy']) / baseline_results['accuracy'] * 100)
        }
    }

    results_file = Path(f'checkpoints/llm_ctr_phase{args.phase}/evaluation_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"  ✓ Results saved to {results_file}")

    print("\n" + "=" * 80)
    print("✅ EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
