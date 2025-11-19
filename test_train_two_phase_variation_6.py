#!/usr/bin/env python3
"""
Quick sanity test for train_two_phase_variation_6.py
Tests all phases with minimal data to catch bugs before full training.

Usage:
    python test_train_two_phase_variation_6.py --gpu 0
"""

import sys
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description='Test two-phase variation 6 pipeline')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    args = parser.parse_args()

    print("="*80)
    print("SANITY TEST: Two-Phase Variation 6")
    print("Testing with minimal data (2 batches per phase)")
    print("="*80)

    # Modify train_two_phase_variation_6.py to add early exit
    test_code = """
import sys
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# Monkey-patch tqdm to limit iterations
original_tqdm = __import__('tqdm').tqdm
class LimitedTqdm:
    def __init__(self, iterable, *args, **kwargs):
        self.iterable = iterable
        self.iter_obj = iter(iterable)
        self.counter = 0
        self.max_iters = 2  # Only 2 batches for testing
        self.pbar = original_tqdm(range(self.max_iters), *args, **kwargs)

    def __iter__(self):
        return self

    def __next__(self):
        if self.counter >= self.max_iters:
            raise StopIteration
        self.counter += 1
        self.pbar.update(1)
        return next(self.iter_obj)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.pbar.close()

# Replace tqdm
import tqdm
tqdm.tqdm = LimitedTqdm

# Now run the actual training script
exec(open('train_two_phase_variation_6.py').read())
"""

    # Save test script
    with open('/tmp/test_variation_6_quick.py', 'w') as f:
        f.write(test_code)

    # Run with small batch size
    cmd = [
        sys.executable, '/tmp/test_variation_6_quick.py',
        '--phase1_batch_size', '4',
        '--phase2_batch_size', '4',
        '--phase1_lr', '1e-3',
        '--phase2_lr', '1e-4',
        '--embedding_dim', '16',
        '--encoder_hidden_units', '100', '100',  # Smaller network
        '--projector_hidden_dim', '128',
        '--output_dir', '/tmp/test_variation_6_output',
        '--gpu', str(args.gpu)
    ]

    try:
        subprocess.run(cmd, check=True)
        print("\n" + "="*80)
        print("✓ SANITY TEST PASSED!")
        print("All phases completed without errors")
        print("="*80)
        return 0
    except subprocess.CalledProcessError as e:
        print("\n" + "="*80)
        print("✗ SANITY TEST FAILED!")
        print(f"Error in pipeline: {e}")
        print("="*80)
        return 1

if __name__ == '__main__':
    sys.exit(main())
