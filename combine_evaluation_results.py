#!/usr/bin/env python3
"""
Combine Individual Evaluation Results

This script combines evaluation results from x1, x2, and x4 test sets
and calculates the final combined AUC and LogLoss metrics.

Usage:
    python combine_evaluation_results.py [--results-dir DIR]
"""

import json
import argparse
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss


def load_individual_results(results_dir, datasets=['x1', 'x2', 'x4']):
    """Load individual evaluation results from JSON files."""
    all_results = {}

    for dataset in datasets:
        result_file = results_dir / f'evaluation_{dataset}_test.json'

        if not result_file.exists():
            print(f"⚠ Warning: {result_file} not found, skipping {dataset}")
            continue

        with open(result_file, 'r') as f:
            data = json.load(f)
            all_results[dataset] = data
            print(f"✓ Loaded {dataset}: {data['metrics']['num_samples']:,} samples")

    return all_results


def extract_metrics(all_results):
    """Extract individual metrics from each dataset."""
    print("\n" + "=" * 80)
    print("INDIVIDUAL DATASET RESULTS")
    print("=" * 80)

    for dataset, data in all_results.items():
        metrics = data['metrics']
        print(f"\n{dataset.upper()} Results:")
        print(f"  Samples:  {metrics['num_samples']:,}")
        print(f"  AUC:      {metrics['auc']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.2f}%")
        print(f"  LogLoss:  {metrics['logloss']:.4f}")
        print(f"  Loss:     {metrics['loss']:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Combine individual evaluation results')
    parser.add_argument('--results-dir', type=str,
                       default='checkpoints/llm_ctr_phase2',
                       help='Directory containing evaluation result files')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    print("=" * 80)
    print("COMBINING EVALUATION RESULTS")
    print("=" * 80)
    print(f"\nSearching for results in: {results_dir}")
    print()

    # Load individual results
    all_results = load_individual_results(results_dir)

    if len(all_results) == 0:
        print("\n❌ Error: No evaluation result files found!")
        print(f"\nExpected files in {results_dir}:")
        print("  - evaluation_x1_test.json")
        print("  - evaluation_x2_test.json")
        print("  - evaluation_x4_test.json")
        return

    # Show individual results
    extract_metrics(all_results)

    # Calculate simple weighted average (NOT combined AUC)
    print("\n" + "=" * 80)
    print("WEIGHTED AVERAGE METRICS (across datasets)")
    print("=" * 80)

    total_samples = sum(data['metrics']['num_samples'] for data in all_results.values())

    weighted_auc = sum(
        data['metrics']['auc'] * data['metrics']['num_samples']
        for data in all_results.values()
    ) / total_samples

    weighted_accuracy = sum(
        data['metrics']['accuracy'] * data['metrics']['num_samples']
        for data in all_results.values()
    ) / total_samples

    weighted_logloss = sum(
        data['metrics']['logloss'] * data['metrics']['num_samples']
        for data in all_results.values()
    ) / total_samples

    print(f"\nTotal Samples: {total_samples:,}")
    for dataset, data in all_results.items():
        print(f"  - {dataset}: {data['metrics']['num_samples']:,} samples")

    print(f"\nWeighted Average Metrics:")
    print(f"  AUC:      {weighted_auc:.4f}")
    print(f"  Accuracy: {weighted_accuracy:.2f}%")
    print(f"  LogLoss:  {weighted_logloss:.4f}")

    print("\n" + "=" * 80)
    print("⚠ IMPORTANT NOTE:")
    print("=" * 80)
    print("""
The metrics above are WEIGHTED AVERAGES of individual dataset metrics.

For the TRUE COMBINED AUC and LogLoss, you need to combine the raw
predictions from all datasets and recalculate metrics on the combined data.

This requires the individual prediction files (not just the summary metrics).

To get TRUE combined metrics, use:
    python evaluate_model.py --dataset all --split test
""")

    # Save summary
    output_file = results_dir / 'evaluation_summary.json'
    summary = {
        'individual_results': {
            dataset: data['metrics']
            for dataset, data in all_results.items()
        },
        'weighted_averages': {
            'total_samples': total_samples,
            'auc': weighted_auc,
            'accuracy': weighted_accuracy,
            'logloss': weighted_logloss
        },
        'note': 'These are weighted averages. For true combined AUC/LogLoss, use --dataset all'
    }

    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Summary saved to: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
