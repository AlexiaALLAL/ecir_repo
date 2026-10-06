"""
Script to analyze NaN batches saved during training.

Usage:
    python analyze_nan_batches.py --batch_path /path/to/nan_batch_*.pt
"""

import torch
import argparse
import os
from pathlib import Path


def analyze_nan_batch(batch_path):
    """Analyze a saved NaN batch and print diagnostic information."""
    print(f"\n{'='*80}")
    print(f"Analyzing: {batch_path}")
    print(f"{'='*80}\n")
    
    # Load the batch
    data = torch.load(batch_path, map_location='cpu')
    
    # Print basic info
    print(f"Epoch: {data.get('epoch', 'N/A')}")
    print(f"Step: {data.get('step', 'N/A')}")
    print(f"Reason: {data.get('reason', 'N/A')}")
    
    if 'loss_value' in data:
        print(f"Loss value: {data['loss_value']}")
    
    if 'grad_norm' in data:
        print(f"Gradient norm: {data['grad_norm']}")
    
    # Analyze batch data
    print(f"\n--- Batch Contents ---")
    batch = data.get('batch', {})
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"\n{key}:")
            print(f"  Shape: {value.shape}")
            print(f"  Dtype: {value.dtype}")
            print(f"  Device: {value.device}")
            print(f"  Min: {value.min().item():.6f}")
            print(f"  Max: {value.max().item():.6f}")
            print(f"  Mean: {value.float().mean().item():.6f}")
            print(f"  Has NaN: {torch.isnan(value).any().item()}")
            print(f"  Has Inf: {torch.isinf(value).any().item()}")
            
            # Count NaN/Inf
            num_nan = torch.isnan(value).sum().item()
            num_inf = torch.isinf(value).sum().item()
            if num_nan > 0:
                print(f"  NaN count: {num_nan} ({100 * num_nan / value.numel():.2f}%)")
            if num_inf > 0:
                print(f"  Inf count: {num_inf} ({100 * num_inf / value.numel():.2f}%)")
    
    # Analyze outputs if available
    if 'outputs' in data and data['outputs'] is not None:
        print(f"\n--- Model Outputs ---")
        outputs = data['outputs']
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"\n{key}:")
                print(f"  Shape: {value.shape}")
                print(f"  Has NaN: {torch.isnan(value).any().item()}")
                print(f"  Has Inf: {torch.isinf(value).any().item()}")
                if value.numel() > 0:
                    print(f"  Min: {value.min().item():.6f}")
                    print(f"  Max: {value.max().item():.6f}")
    
    # Analyze gradient statistics if available
    if 'grad_stats' in data:
        print(f"\n--- Gradient Statistics ---")
        grad_stats = data['grad_stats']
        
        # Find layers with NaN or Inf gradients
        problematic_layers = []
        for name, stats in grad_stats.items():
            if stats['has_nan'] or stats['has_inf']:
                problematic_layers.append((name, stats))
        
        if problematic_layers:
            print(f"\nProblematic layers ({len(problematic_layers)}):")
            for name, stats in problematic_layers:
                print(f"\n  {name}:")
                print(f"    Mean: {stats['mean']:.6f}")
                print(f"    Max: {stats['max']:.6f}")
                print(f"    Min: {stats['min']:.6f}")
                print(f"    Has NaN: {stats['has_nan']}")
                print(f"    Has Inf: {stats['has_inf']}")
        else:
            print("\nNo layers with NaN/Inf gradients found in stats")
            print(f"Total layers tracked: {len(grad_stats)}")
    
    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Analyze NaN batches from training')
    parser.add_argument('--batch_path', type=str, help='Path to a specific NaN batch file')
    parser.add_argument('--batch_dir', type=str, help='Directory containing NaN batch files')
    args = parser.parse_args()
    
    batch_files = []
    
    if args.batch_path:
        batch_files.append(args.batch_path)
    elif args.batch_dir:
        # Find all NaN batch files in directory
        batch_dir = Path(args.batch_dir)
        batch_files = list(batch_dir.glob('nan_batch_*.pt'))
    else:
        print("Please provide either --batch_path or --batch_dir")
        return
    
    if not batch_files:
        print(f"No NaN batch files found")
        return
    
    print(f"Found {len(batch_files)} NaN batch file(s)")
    
    for batch_file in sorted(batch_files):
        analyze_nan_batch(batch_file)


if __name__ == '__main__':
    main()
