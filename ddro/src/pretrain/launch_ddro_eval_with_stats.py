"""
Script to run DDRO evaluation multiple times and compute statistics.
Calculates mean and 95% confidence intervals for all metrics.
"""
import os
import json
import argparse
import subprocess
import re
import numpy as np
from scipy import stats
from typing import Dict, List
import pandas as pd
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--encoding", default="pq", type=str, help="docid method atomic/pq/url")
parser.add_argument("--scale", default="top_300k", type=str, help="scale: top_300k, rand_100k, etc.")
parser.add_argument("--dataset", default="msmarco", type=str, help="dataset: msmarco/nq")
parser.add_argument("--num_runs", default=5, type=int, help="number of evaluation runs")
parser.add_argument("--output_dir", default="eval_stats", type=str, help="directory to save statistics")
parser.add_argument("--model_checkpoint", required=True, type=str, help="model checkpoint to evaluate")
parser.add_argument("--num_beams", default=10, type=int, help="number of beams for evaluation (overrides default based on dataset/encoding)")

args = parser.parse_args()

# Load correct config file based on dataset
config_file_path = f"src/scripts/configs/config_{args.dataset}.json" if args.dataset == "nq" else "src/scripts/configs/config.json"
config_file = json.load(open(config_file_path, "r"))

# Get config for the specified encoding
config = config_file[args.encoding]
encoding, add_doc_num, max_docid_length, use_origin_head = (
    config["encoding"], 
    config["add_doc_num"], 
    config["max_docid_length"], 
    config["use_origin_head"]
)

code_dir = os.getcwd()

# Determine num_beams based on dataset and encoding
# num_beams = (
#     80  if (args.dataset == "msmarco" and args.encoding == "pq") else
#     100 if (args.dataset == "nq" and args.encoding == "pq") else
#     50  if (args.dataset == "nq" and args.encoding == "url") else
#     15
# )
num_beams = args.num_beams

model = "t5_128_1"
cur_data = "query_dev"
use_docid_rank = "True"
operation = "testing"
max_seq_length = 64
model_name = "DDRO"
top_or_rand, scale = args.scale.split("_")


def parse_metrics_from_log(log_content: str) -> Dict[str, float]:
    """
    Parse all metrics from the evaluation log output.
    Parses individual metrics from the FINAL RESULTS section.
    """
    metrics = {}
    
    # Parse each metric individually - more robust than complex multi-line regex
    # Format in log: "MRR@10: 0.4345 | MRR: 0.4355"
    metric_patterns = {
        'MRR@10': r'MRR@10:\s*([\d.]+)',
        'MRR': r'MRR:\s*([\d.]+)',
        'P@1': r'P@1:\s*([\d.]+)',
        'P@10': r'P@10:\s*([\d.]+)',
        'P@20': r'P@20:\s*([\d.]+)',
        'P@100': r'P@100:\s*([\d.]+)',
        'R@1': r'R@1:\s*([\d.]+)',
        'R@10': r'R@10:\s*([\d.]+)',
        'R@100': r'R@100:\s*([\d.]+)',
        'R@1000': r'R@1000:\s*([\d.]+)',
        'Hit@1': r'Hit@1:\s*([\d.]+)',
        'Hit@5': r'Hit@5:\s*([\d.]+)',
        'Hit@10': r'Hit@10:\s*([\d.]+)',
        'Hit@100': r'Hit@100:\s*([\d.]+)'
    }
    
    for metric_name, pattern in metric_patterns.items():
        match = re.search(pattern, log_content)
        if match:
            metrics[metric_name] = float(match.group(1))
    
    return metrics


def calculate_confidence_interval(values: List[float], confidence: float = 0.95) -> tuple:
    """
    Calculate mean and confidence interval for a list of values.
    
    Args:
        values: List of metric values
        confidence: Confidence level (default 0.95 for 95%)
    
    Returns:
        tuple: (mean, ci_lower, ci_upper, std_dev)
    """
    n = len(values)
    mean = np.mean(values)
    std_dev = np.std(values, ddof=1)  # Sample standard deviation
    
    # Use t-distribution for small sample sizes
    t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
    margin_of_error = t_value * (std_dev / np.sqrt(n))
    
    ci_lower = mean - margin_of_error
    ci_upper = mean + margin_of_error
    
    return mean, ci_lower, ci_upper, std_dev


def run_single_evaluation(run_id: int) -> Dict[str, float]:
    """
    Run a single evaluation and return the metrics.
    """
    print(f"\n{'='*80}")
    print(f"Starting evaluation run {run_id + 1}/{args.num_runs}")
    print(f"{'='*80}\n")
    
    # Create log directory if it doesn't exist
    log_dir = f"{code_dir}/logs/{args.dataset}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Dynamically construct file paths based on dataset and encoding
    base_path = f"{code_dir}/resources"
    
    if args.dataset == "nq":
        test_file = f"{base_path}/datasets/processed/nq-data/test_data/{cur_data}.{model}.{encoding}_nq.json"
        docid_filename = f"t5_512_{encoding}_docids.txt"
        docid_file = f"{base_path}/datasets/processed/nq-data/encoded_docid/{docid_filename}"
    else:  # msmarco
        encoding_filename = "url" if encoding == "url_title" else encoding
        test_file = f"{base_path}/datasets/processed/msmarco-data/eval_data/{cur_data}.{encoding_filename}.jsonl"
        encoding_for_docid = "url" if encoding == "url_title" else args.encoding
        docid_file = f"{base_path}/datasets/processed/msmarco-data/encoded_docid/{encoding_for_docid}_docid.txt"
    
    model_path = args.model_checkpoint
    log_path = f"logs/{args.dataset}/dpo_{model_name}_{encoding}_run{run_id + 1}.log"
    
    # Verify files exist (only on first run)
    if run_id == 0:
        files_to_check = {
            "Test file": test_file,
            "Docid file": docid_file,
            "Checkpoint": model_path
        }
        
        print("File validation:")
        all_files_exist = True
        for file_desc, file_path in files_to_check.items():
            if os.path.exists(file_path):
                print(f"[FOUND] {file_desc}: {file_path}")
            else:
                print(f"[NOT FOUND] {file_desc}: {file_path}")
                all_files_exist = False
        
        if not all_files_exist:
            print("\nERROR: Some required files are missing. Please check the paths above.")
            return {}
        print()
    
    # Run evaluation
    cmd = f"""python src/pretrain/eval_ddro_docid_ranking.py \
        --per_gpu_batch_size 4 \
        --save_path {model_path} \
        --log_path {log_path} \
        --pretrain_model_path t5-base \
        --test_file_path {test_file} \
        --docid_path {docid_file} \
        --dataset_script_dir src/data/data_scripts \
        --dataset_cache_dir {code_dir}/negs_tutorial_cache \
        --num_beams {num_beams} \
        --add_doc_num {add_doc_num} \
        --max_seq_length {max_seq_length} \
        --max_docid_length {max_docid_length} \
        --operation {operation} \
        --use_docid_rank {use_docid_rank}"""
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.stdout:
        print(f"\n--- STDOUT from run {run_id + 1} ---")
        print(result.stdout[-1000:])  # Last 1000 chars
    
    
    # Read the CSV file to extract metrics (much more reliable than parsing text!)
    csv_path = log_path.replace(".log", ".csv")
    
    if os.path.exists(csv_path):
        try:
            # Read the CSV file created by the evaluation script
            df = pd.read_csv(csv_path)
            
            # Calculate mean for each metric column
            metrics = {
                'MRR@10': df['MRR@10'].mean() if 'MRR@10' in df.columns else None,
                'MRR': df['MRR'].mean() if 'MRR' in df.columns else None,
                'P@1': df['P@1'].mean() if 'P@1' in df.columns else None,
                'P@10': df['P@10'].mean() if 'P@10' in df.columns else None,
                'P@20': df['P@20'].mean() if 'P@20' in df.columns else None,
                'P@100': df['P@100'].mean() if 'P@100' in df.columns else None,
                'R@1': df['R@1'].mean() if 'R@1' in df.columns else None,
                'R@10': df['R@10'].mean() if 'R@10' in df.columns else None,
                'R@100': df['R@100'].mean() if 'R@100' in df.columns else None,
                'R@1000': df['R@1000'].mean() if 'R@1000' in df.columns else None,
                'Hit@1': df['Hit@1'].mean() if 'Hit@1' in df.columns else None,
                'Hit@5': df['Hit@5'].mean() if 'Hit@5' in df.columns else None,
                'Hit@10': df['Hit@10'].mean() if 'Hit@10' in df.columns else None,
                'Hit@100': df['Hit@100'].mean() if 'Hit@100' in df.columns else None
            }
            
            # Remove None values
            metrics = {k: v for k, v in metrics.items() if v is not None}
            
            if metrics:
                print(f"✓ Run {run_id + 1} completed. MRR@10: {metrics.get('MRR@10', 0):.4f}")
                return metrics
            else:
                print(f"✗ Warning: No metrics found in CSV for run {run_id + 1}")
                return {}
                
        except Exception as e:
            print(f"✗ Error reading CSV file for run {run_id + 1}: {e}")
            return {}
    else:
        print(f"✗ Error: CSV file not found for run {run_id + 1}: {csv_path}")
        return {}


def compute_statistics(all_metrics: List[Dict[str, float]]) -> pd.DataFrame:
    """
    Compute mean, confidence intervals, and standard deviations for all metrics.
    """
    if not all_metrics:
        return pd.DataFrame()
    
    # Get all metric names
    metric_names = list(all_metrics[0].keys())
    
    stats_data = []
    for metric_name in metric_names:
        values = [run_metrics[metric_name] for run_metrics in all_metrics if metric_name in run_metrics]
        
        if len(values) > 0:
            mean, ci_lower, ci_upper, std_dev = calculate_confidence_interval(values)
            
            stats_data.append({
                'Metric': metric_name,
                'Mean': mean,
                'Std Dev': std_dev,
                'CI 95% Lower': ci_lower,
                'CI 95% Upper': ci_upper,
                'Min': min(values),
                'Max': max(values),
                'Num Runs': len(values)
            })
    
    return pd.DataFrame(stats_data)


def main():
    print(f"\n{'='*80}")
    print(f"DDRO Evaluation with Statistical Analysis")
    print(f"{'='*80}")
    print(f"Dataset: {args.dataset}")
    print(f"Encoding: {args.encoding}")
    print(f"Scale: {args.scale}")
    print(f"Number of runs: {args.num_runs}")
    print(f"Num beams: {num_beams}")
    print(f"{'='*80}\n")
    
    # Run evaluations
    all_metrics = []
    for i in tqdm(range(args.num_runs), desc="Running evaluations", unit="run"):
        metrics = run_single_evaluation(i)
        if metrics:
            all_metrics.append(metrics)
    
    if not all_metrics:
        print("\nERROR: No successful evaluation runs. Cannot compute statistics.")
        return
    
    print(f"\n{'='*80}")
    print(f"Computing statistics from {len(all_metrics)} successful runs...")
    print(f"All metrics from runs:")
    for i, run_metrics in enumerate(all_metrics):
        print(f"Run {i + 1}: " + ", ".join(f"{k}: {v:.4f}" for k, v in run_metrics.items()))
    print(f"{'='*80}\n")
    
    # Compute statistics
    stats_df = compute_statistics(all_metrics)
    
    # Create output directory
    output_dir = f"{code_dir}/{args.output_dir}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{output_dir}/eval_stats_{args.dataset}_{args.encoding}_{timestamp}.csv"
    stats_df.to_csv(output_file, index=False)
    
    # Also save raw data
    raw_file = f"{output_dir}/eval_raw_{args.dataset}_{args.encoding}_{timestamp}.csv"
    raw_df = pd.DataFrame(all_metrics)
    raw_df.to_csv(raw_file, index=False)
    
    # Print results
    print("\n" + "="*80)
    print("STATISTICAL RESULTS")
    print("="*80)
    print(stats_df.to_string(index=False))
    print("="*80)
    
    print(f"\nResults saved to:")
    print(f"  Statistics: {output_file}")
    print(f"  Raw data: {raw_file}")
    
    # Print formatted summary for key metrics
    print("\n" + "="*80)
    print("KEY METRICS SUMMARY (Mean ± 95% CI)")
    print("="*80)
    
    key_metrics = ['R@1', 'R@10', 'MRR@10']
    for metric in key_metrics:
        row = stats_df[stats_df['Metric'] == metric]
        if not row.empty:
            mean = row['Mean'].values[0]
            ci_lower = row['CI 95% Lower'].values[0]
            ci_upper = row['CI 95% Upper'].values[0]
            print(f"{metric:12s}: {mean*100:.2f} [{ci_lower*100:.2f}, {ci_upper*100:.2f}]")
    
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
