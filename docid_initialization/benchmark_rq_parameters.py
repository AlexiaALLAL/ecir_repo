# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "datasets",
#     "pandas",
#     "sentence-transformers",
#     "scikit-learn",
#     "scipy",
#     "matplotlib",
# ]
# ///

"""
Benchmark different RQ parameters by training codebooks, generating docids, 
and evaluating correlation metrics.
"""

import os
import json
import subprocess
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from itertools import product
from tqdm import tqdm

def run_command(command: str, description: str):
    """Run a shell command and print status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print(f"{'='*60}")
    result = subprocess.run(command, shell=True, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"❌ Error running: {description}")
        raise RuntimeError(f"Command failed: {command}")
    print(f"✅ Completed: {description}")
    return result

def train_codebook(dataset_name: str, subset: str, split: str, model_name: str, num_codebooks: int, 
                   codebook_size: int, max_samples: int) -> str:
    """Train a codebook with specified parameters and return the codebook path."""
    # Generate expected codebook path
    dataset_slug = dataset_name.replace('/', '_')
    model_slug = model_name.replace('/', '_').replace('-', '_')
    samples_str = f"{max_samples//1000}k" if max_samples else "all"
    codebook_path = f"codebooks/rq_{dataset_slug}_{model_slug}_nc{num_codebooks}_cs{codebook_size}_n{samples_str}.pkl"
    
    # Check if codebook already exists
    if os.path.exists(codebook_path):
        print(f"\n✅ Codebook already exists: {codebook_path}")
        print(f"   Skipping training for nc={num_codebooks}, cs={codebook_size}")
        return codebook_path
    
    command = (
        f"uv run docid_initialization/residual_quantizer.py "
        f"--dataset_name {dataset_name} "
        f"--subset {subset} "
        f"--split {split} "
        f"--embedding_model {model_name} "
        f"--num_codebooks {num_codebooks} "
        f"--codebook_size {codebook_size} "
        f"--max_samples {max_samples}"
    )
    
    run_command(command, f"Training codebook (nc={num_codebooks}, cs={codebook_size})")
    
    return codebook_path


def generate_docids(dataset_name: str, subset: str, split: str, codebook_path: str,
                    model_name: str, max_samples: int) -> str:
    """Generate docids using the specified codebook and return the dataset path."""
    # Expected dataset path (based on codebook name)
    codebook_name = os.path.splitext(os.path.basename(codebook_path))[0]
    docid_path = f"docids/{codebook_name}_dataset"
    
    # Check if docid dataset already exists
    if os.path.exists(docid_path):
        print(f"\n✅ DocID dataset already exists: {docid_path}")
        print(f"   Skipping docid generation")
        return docid_path
    
    command = (
        f"uv run docid_initialization/generate_docids.py "
        f"--dataset_name {dataset_name} "
        f"--subset {subset} "
        f"--split {split} "
        f"--max_samples {max_samples} "
        f"--docid_method residual_quantization "
        f"--generator_params codebook_path={codebook_path},embedding_model={model_name} "
        f"--batch_size 1024 "
        f"--no-ensure_unique"
    )
    
    run_command(command, f"Generating docids from {codebook_path}")
    
    return docid_path


def evaluate_docids(dataset_name: str, subset: str, split: str, docid_path: str, distance_metric: str, 
                    codebook_size: int, sample_size: int, n_pairs: int, no_analyze_prefix: bool,
                    test_embedding_model: str) -> dict:
    """Evaluate docids and return metrics."""

    # Expected metrics path
    docid_filename = docid_path.strip('/').split('/')[-1].replace('_dataset', '')
    test_embedding_model_sanitized = test_embedding_model.replace('/', '_').replace('-', '_')
    metrics_path = f"docid_results/{docid_filename}/metrics_{distance_metric}_{test_embedding_model_sanitized}.json"

    # Check if metrics already exist
    if os.path.exists(metrics_path):
        print(f"\n✅ Metrics already exist: {metrics_path}")
        print(f"   Skipping evaluation for {distance_metric}")

    else:
        command = (
            f"uv run docid_initialization/test_generated_docids.py "
            f"--dataset_name {dataset_name} "
            f"--subset {subset} "
            f"--split {split} "
            f"--docid_path {docid_path} "
            f"--sample_size {sample_size} "
            f"--n_pairs {n_pairs} "
            f"--distance_metric {distance_metric} "
            f"--codebook_size {codebook_size} "
            f"{'--no_analyze_prefix' if no_analyze_prefix else ''} "
            f" --test_embedding_model {test_embedding_model}"
        )
        run_command(command, f"Evaluating docids with {distance_metric}")    
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    return metrics

def plot_correlation_heatmaps(results_df: pd.DataFrame, output_dir: str):
    """
    Create heatmaps showing correlation vs. num_codebooks and codebook_size.
    One subplot for each distance metric.
    """
    distance_metrics = results_df['distance_metric'].unique()
    
    # Plot Spearman correlation
    fig, axes = plt.subplots(1, len(distance_metrics), figsize=(7*len(distance_metrics), 6))
    if len(distance_metrics) == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, distance_metrics):
        # Filter data for this metric
        metric_df = results_df[results_df['distance_metric'] == metric]
        
        # Create pivot table for heatmap
        pivot = metric_df.pivot_table(
            values='spearman_correlation',
            index='num_codebooks',
            columns='codebook_size',
            aggfunc='mean'
        )
        
        # Plot heatmap
        im = ax.imshow(pivot.values, cmap='viridis', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        
        ax.set_xlabel('Codebook Size', fontsize=12)
        ax.set_ylabel('Number of Codebooks', fontsize=12)
        ax.set_title(f'Spearman Correlation\n({metric})', fontsize=14)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation', fontsize=11)
        
        # Add text annotations
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                text = ax.text(j, i, f'{pivot.values[i, j]:.3f}',
                             ha="center", va="center", color="white", fontsize=10)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'spearman_heatmaps.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Spearman heatmap saved to: {plot_path}")
    plt.close()
    
    # Plot Pearson correlation
    fig, axes = plt.subplots(1, len(distance_metrics), figsize=(7*len(distance_metrics), 6))
    if len(distance_metrics) == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, distance_metrics):
        # Filter data for this metric
        metric_df = results_df[results_df['distance_metric'] == metric]
        
        # Create pivot table for heatmap
        pivot = metric_df.pivot_table(
            values='pearson_correlation',
            index='num_codebooks',
            columns='codebook_size',
            aggfunc='mean'
        )
        
        # Plot heatmap
        im = ax.imshow(pivot.values, cmap='viridis', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        
        ax.set_xlabel('Codebook Size', fontsize=12)
        ax.set_ylabel('Number of Codebooks', fontsize=12)
        ax.set_title(f'Pearson Correlation\n({metric})', fontsize=14)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation', fontsize=11)
        
        # Add text annotations
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                text = ax.text(j, i, f'{pivot.values[i, j]:.3f}',
                             ha="center", va="center", color="white", fontsize=10)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'pearson_heatmaps.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Pearson heatmap saved to: {plot_path}")
    plt.close()

def plot_correlation_lines(results_df: pd.DataFrame, output_dir: str):
    """
    Create line plots showing correlation trends.
    """
    # Spearman correlation trends
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Correlation vs. num_codebooks (grouped by codebook_size and metric)
    for (cs, metric), group in results_df.groupby(['codebook_size', 'distance_metric']):
        group_sorted = group.sort_values('num_codebooks')
        ax1.plot(group_sorted['num_codebooks'], 
                group_sorted['spearman_correlation'],
                marker='o', label=f'{metric}, cs={cs}')
    
    ax1.set_xlabel('Number of Codebooks', fontsize=12)
    ax1.set_ylabel('Spearman Correlation', fontsize=12)
    ax1.set_title('Spearman Correlation vs. Number of Codebooks', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Correlation vs. codebook_size (grouped by num_codebooks and metric)
    for (nc, metric), group in results_df.groupby(['num_codebooks', 'distance_metric']):
        group_sorted = group.sort_values('codebook_size')
        ax2.plot(group_sorted['codebook_size'], 
                group_sorted['spearman_correlation'],
                marker='o', label=f'{metric}, nc={nc}')
    
    ax2.set_xlabel('Codebook Size', fontsize=12)
    ax2.set_ylabel('Spearman Correlation', fontsize=12)
    ax2.set_title('Spearman Correlation vs. Codebook Size', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'spearman_trends.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Spearman line plots saved to: {plot_path}")
    plt.close()
    
    # Pearson correlation trends
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Correlation vs. num_codebooks (grouped by codebook_size and metric)
    for (cs, metric), group in results_df.groupby(['codebook_size', 'distance_metric']):
        group_sorted = group.sort_values('num_codebooks')
        ax1.plot(group_sorted['num_codebooks'], 
                group_sorted['pearson_correlation'],
                marker='o', label=f'{metric}, cs={cs}')
    
    ax1.set_xlabel('Number of Codebooks', fontsize=12)
    ax1.set_ylabel('Pearson Correlation', fontsize=12)
    ax1.set_title('Pearson Correlation vs. Number of Codebooks', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Correlation vs. codebook_size (grouped by num_codebooks and metric)
    for (nc, metric), group in results_df.groupby(['num_codebooks', 'distance_metric']):
        group_sorted = group.sort_values('codebook_size')
        ax2.plot(group_sorted['codebook_size'], 
                group_sorted['pearson_correlation'],
                marker='o', label=f'{metric}, nc={nc}')
    
    ax2.set_xlabel('Codebook Size', fontsize=12)
    ax2.set_ylabel('Pearson Correlation', fontsize=12)
    ax2.set_title('Pearson Correlation vs. Codebook Size', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'pearson_trends.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Pearson line plots saved to: {plot_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark RQ parameters: train codebooks, generate docids, evaluate metrics"
    )
    parser.add_argument("--dataset_name", type=str, 
                       default="mteb/msmarco",
                       help="Dataset name")
    parser.add_argument("--subset", type=str, default="corpus",
                       help="Subset of the dataset, if applicable")
    parser.add_argument("--split", type=str, default="corpus",
                       help="Dataset split to use")
    parser.add_argument("--model_name", type=str,
                       default="all-MiniLM-L6-v2",
                       help="Embedding model name")
    parser.add_argument("--test_embedding_model", type=str,
                       default="all-MiniLM-L6-v2",
                       help="Embedding model name for testing/evaluation")
    parser.add_argument("--num_codebooks_list", type=str,
                       default="3,5,7,10",
                       help="Comma-separated list of num_codebooks to test")
    parser.add_argument("--codebook_size_list", type=str,
                       default="64,128,256",
                       help="Comma-separated list of codebook_size to test")
    parser.add_argument("--distance_metrics", type=str,
                       default="rq_simple,rq_weighted",
                       help="Comma-separated list of distance metrics")
    parser.add_argument("--training_samples", type=int, default=100000,
                       help="Number of samples for training codebooks")
    parser.add_argument("--eval_samples", type=int, default=100000,
                       help="Number of samples for evaluation")
    parser.add_argument("--n_pairs", type=int, default=5000,
                       help="Number of pairs for correlation computation")
    parser.add_argument("--output_dir", type=str, default="rq_benchmark_results",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Parse parameter lists
    num_codebooks_list = [int(x) for x in args.num_codebooks_list.split(',')]
    codebook_size_list = [int(x) for x in args.codebook_size_list.split(',')]
    distance_metrics = args.distance_metrics.split(',')
    
    print("\n" + "="*60)
    print("RQ PARAMETER BENCHMARK")
    print("="*60)
    print(f"Dataset: {args.dataset_name}")
    print(f"Model: {args.model_name}")
    print(f"Num codebooks: {num_codebooks_list}")
    print(f"Codebook sizes: {codebook_size_list}")
    print(f"Distance metrics: {distance_metrics}")
    print(f"Training samples: {args.training_samples}")
    print(f"Eval samples: {args.eval_samples}")
    print("="*60)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Store all results
    results = []
    
    # Iterate over all parameter combinations
    total_configs = len(num_codebooks_list) * len(codebook_size_list)
    config_idx = 0
    
    for num_codebooks, codebook_size in tqdm(product(num_codebooks_list, codebook_size_list), total=total_configs):
        config_idx += 1
        print(f"\n{'#'*60}")
        print(f"CONFIG {config_idx}/{total_configs}: nc={num_codebooks}, cs={codebook_size}")
        print(f"{'#'*60}")
        
        # Step 1: Train codebook
        codebook_path = train_codebook(
            dataset_name=args.dataset_name,
            subset=args.subset,
            split=args.split,
            model_name=args.model_name,
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
            max_samples=args.training_samples
        )
        
        # Step 2: Generate docids
        docid_path = generate_docids(
            dataset_name=args.dataset_name,
            subset=args.subset,
            split=args.split,
            codebook_path=codebook_path,
            model_name = args.model_name,
            max_samples=args.eval_samples
        )
        
        # Step 3: Evaluate with each distance metric
        for distance_metric in distance_metrics:
            print(f"\n--- Evaluating with {distance_metric} ---")
            # only analyze prefix if the is the largest nb_codebooks            
            metrics = evaluate_docids(
                dataset_name=args.dataset_name,
                subset=args.subset,
                split=args.split,
                docid_path=docid_path,
                distance_metric=distance_metric,
                codebook_size=codebook_size,
                sample_size=args.eval_samples,
                n_pairs=args.n_pairs,
                no_analyze_prefix=(num_codebooks != max(num_codebooks_list)),
                test_embedding_model=args.test_embedding_model
            )
            
            # Extract key metrics
            result = {
                'num_codebooks': num_codebooks,
                'codebook_size': codebook_size,
                'distance_metric': distance_metric,
                'spearman_correlation': metrics['correlation']['spearman_correlation'],
                'spearman_pvalue': metrics['correlation']['spearman_pvalue'],
                'pearson_correlation': metrics['correlation']['pearson_correlation'],
                'pearson_pvalue': metrics['correlation']['pearson_pvalue'],
                'uniqueness_ratio': metrics['uniqueness']['uniqueness_ratio'],
                'mean_docid_length': metrics['docid_length_stats']['mean_length']
            }
            results.append(result)
            
            print(f"  Spearman: {result['spearman_correlation']:.4f}")
            print(f"  Pearson: {result['pearson_correlation']:.4f}")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results to CSV
    results_csv = os.path.join(args.output_dir, 'benchmark_results.csv')
    results_df.to_csv(results_csv, index=False)
    print(f"\n✅ Results saved to: {results_csv}")
    
    # Create visualizations
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    plot_correlation_heatmaps(results_df, args.output_dir)
    plot_correlation_lines(results_df, args.output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    print("\nBest configurations by Spearman correlation:")
    top_5_spearman = results_df.nlargest(5, 'spearman_correlation')
    for idx, row in top_5_spearman.iterrows():
        print(f"  nc={row['num_codebooks']}, cs={row['codebook_size']}, "
              f"metric={row['distance_metric']}: {row['spearman_correlation']:.4f}")
    
    print("\nBest configurations by Pearson correlation:")
    top_5_pearson = results_df.nlargest(5, 'pearson_correlation')
    for idx, row in top_5_pearson.iterrows():
        print(f"  nc={row['num_codebooks']}, cs={row['codebook_size']}, "
              f"metric={row['distance_metric']}: {row['pearson_correlation']:.4f}")
    
    print("\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
