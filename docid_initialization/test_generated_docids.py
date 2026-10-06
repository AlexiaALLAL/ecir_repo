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

"""This takes a generated docid file and measure performances of the different docid generation methods."""

from datasets import load_dataset
import uuid
from typing import Callable, Dict, Any
import os
import argparse
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
from collections import Counter
import json
from datasets import load_from_disk
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots

from utils import download_corpus


def retrieve_generated_docids(dataset_name, subset, split, docid_path: str, max_samples: int=10000) -> pd.DataFrame: # TODO: use something else than pd.df
    """Load the generated DocIDs from a CSV file, and concat them with corpus."""
    docids = load_from_disk(docid_path)
    docids = docids.select(range(max_samples)).to_pandas()
    print(f"Loaded {len(docids)} DocIDs from {docid_path}")
    print(f"Columns in loaded DocID file: {docids.columns.tolist()}, with types {docids.dtypes.to_dict()}")
    print(f"Sample of loaded DocID DataFrame:")
    print(docids.head())
    dataset = download_corpus(dataset_name, subset, split)
    dataset = dataset.select(range(len(docids)))  # Ensure same length

    dataset_df = pd.DataFrame(dataset)
    dataset_df['passage_id'] = dataset_df['passage_id'].astype(int)
    docids['passage_id'] = docids['passage_id'].astype(int)

    print(f"Columns in loaded corpus DataFrame: {dataset_df.columns.tolist()}, with types {dataset_df.dtypes.to_dict()}")
    print(f"Sample of loaded corpus DataFrame:")
    print(dataset_df.head())
    merged_df = dataset_df.merge(docids, on='passage_id', how='left')
    print(f"Merged DataFrame has {len(merged_df)} rows and columns: {merged_df.columns.tolist()}")
    return merged_df


def compute_docid_similarity_edit_distance(docid1: str, docid2: str) -> float:
    """
    Compute similarity between two DocIDs using edit distance (normalized).
    Returns a value between 0 (completely different) and 1 (identical).
    """
    from difflib import SequenceMatcher
    return SequenceMatcher(None, docid1, docid2).ratio()


def compute_docid_similarity_hamming(docid1: str, docid2: str) -> float:
    """
    Compute similarity using Hamming distance (normalized).
    Compares character by character. Returns 1 - (hamming_distance / max_length).
    """
    max_len = max(len(docid1), len(docid2))
    if max_len == 0:
        return 1.0
    
    # Pad shorter string
    docid1_padded = docid1.ljust(max_len)
    docid2_padded = docid2.ljust(max_len)
    
    hamming_dist = sum(c1 != c2 for c1, c2 in zip(docid1_padded, docid2_padded))
    return 1.0 - (hamming_dist / max_len)


def compute_docid_similarity_rq_weighted(docid1: str, docid2: str, codebook_size: int = 256) -> float:
    """
    Compute similarity for Residual Quantization DocIDs using the first
    different token in the code sequence.
    For example, if code1=(1,9,4,2) and code2=(1,9,5,3), the first differing code is at position 2 (0-based),
    so distance is codebook_size ** (nb_codebooks - i - 1), normalized over max distance.
    
    Args:
        docid1, docid2: DocID strings in format "42-17-231-99"
        codebook_size: Size of each codebook (default: 256)
    
    Returns:
        Similarity between 0 (very different) and 1 (identical)
    """
    codes1 = [int(c) for c in docid1.split(' - ')]
    codes2 = [int(c) for c in docid2.split(' - ')]
    
    nb_codebooks = len(codes1)
    max_distance = codebook_size ** nb_codebooks - 1
    
    weighted_distance = 0  # Default for identical docids
    for i, (c1, c2) in enumerate(zip(codes1, codes2)):
        if c1 != c2:
            weighted_distance = codebook_size ** (nb_codebooks - i - 1)
            break
    
    similarity = 1.0 - (weighted_distance / max_distance)
    return similarity

def compute_docid_similarity_rq_first_different_id(docid1: str, docid2: str) -> float:
    """
    Compute similarity for Residual Quantization DocIDs based on the position of the first differing code.
    For example, if code1=(1,9,4,2) and code2=(1,9,5,3), the first differing code is at position 2 (0-based),
    so similarity = 2 / nb_codebooks.
    """
    codes1 = [int(c) for c in docid1.split(' - ')]
    codes2 = [int(c) for c in docid2.split(' - ')]
    
    nb_codebooks = len(codes1)
    
    for i, (c1, c2) in enumerate(zip(codes1, codes2)):
        if c1 != c2:
            return i / nb_codebooks
    
    return 1.0  # Identical DocIDs


def compute_docid_similarity_embedding_cosine(docid1: str, docid2: str) -> float:
    """
    Compute cosine similarity for embedding-based DocIDs.
    The docids are quantized embeddings in format "127-131-89-..."
    
    Args:
        docid1, docid2: DocID strings representing quantized embeddings
    
    Returns:
        Cosine similarity between the two quantized embedding vectors
    """
    # Parse quantized values
    vec1 = np.array([int(c) for c in docid1.split(' - ')])
    vec2 = np.array([int(c) for c in docid2.split(' - ')])
    
    # Compute cosine similarity
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def get_docid_similarity_function(distance_metric: str, codebook_size: int = 256):
    """
    Return the appropriate DocID similarity function based on the metric name.
    
    Args:
        distance_metric: Name of the distance metric
        codebook_size: Size of codebook for RQ-weighted metric
    
    Returns:
        Function that computes similarity between two DocIDs
    """
    metrics = {
        'edit_distance': compute_docid_similarity_edit_distance,
        'hamming': compute_docid_similarity_hamming,
        'rq_weighted': lambda d1, d2: compute_docid_similarity_rq_weighted(d1, d2, codebook_size),
        'rq_simple': compute_docid_similarity_rq_first_different_id,
        'embedding_cosine': compute_docid_similarity_embedding_cosine,
    }
    
    if distance_metric not in metrics:
        raise ValueError(f"Unknown distance metric: {distance_metric}. Choose from: {list(metrics.keys())}")
    
    return metrics[distance_metric]


def compute_embedding_similarity(embeddings: np.ndarray, idx1: int, idx2: int) -> float:
    """Compute cosine similarity between two embeddings."""
    return cosine_similarity(embeddings[idx1].reshape(1, -1), embeddings[idx2].reshape(1, -1))[0][0]


def evaluate_docid_relevance(merged_df: pd.DataFrame, embeddings: np.ndarray, 
                            n_pairs: int = 5000, 
                            distance_metric: str = 'edit_distance', codebook_size: int = 256):
    """
    Evaluate DocID quality by comparing semantic similarity with DocID similarity.
    
    Args:
        merged_df: DataFrame with 'passage', 'docid' columns
        embeddings: Precomputed embeddings aligned with merged_df
        n_pairs: Number of document pairs to compare
        distance_metric: Distance metric to use ('edit_distance', 'hamming', 'rq_weighted', 'rq_simple')
        codebook_size: Size of codebook for RQ-weighted metric (default: 256)
        
    Returns:
        Dict with evaluation metrics
    """
    print("Evaluating DocID relevance")
    
    # Filter out duplicate docids to avoid artificially high similarity
    original_size = len(merged_df)
    sample_df = merged_df.drop_duplicates(subset='docid', keep='first').reset_index(drop=True)
    
    # Also filter embeddings to match
    unique_indices = merged_df.drop_duplicates(subset='docid', keep='first').index.tolist()
    filtered_embeddings = embeddings[unique_indices]
    
    n_duplicates = original_size - len(sample_df)
    if n_duplicates > 0:
        print(f"⚠️  Removed {n_duplicates} duplicate docids from analysis (kept first occurrence)")
        print(f"   Analyzing {len(sample_df)} unique docids out of {original_size} total documents")
    
    print(f"\nComputing similarity metrics for {n_pairs} random pairs...")
    
    # Generate random pairs
    np.random.seed(42)
    n_docs = len(sample_df)
    pairs_idx = np.random.randint(0, n_docs, size=(n_pairs, 2))
    
    embedding_similarities = []
    docid_similarities = []
    
    for idx1, idx2 in pairs_idx:
        if idx1 == idx2:
            continue
            
        # Compute embedding similarity
        emb_sim = compute_embedding_similarity(filtered_embeddings, idx1, idx2)
        embedding_similarities.append(emb_sim)
        
        # Compute DocID similarity using selected metric
        docid1 = str(sample_df.iloc[idx1]['docid'])
        docid2 = str(sample_df.iloc[idx2]['docid'])
        similarity_fn = get_docid_similarity_function(distance_metric, codebook_size)
        docid_sim = similarity_fn(docid1, docid2)
        docid_similarities.append(docid_sim)
    
    embedding_similarities = np.array(embedding_similarities)
    docid_similarities = np.array(docid_similarities)
    
    # Compute correlation metrics
    spearman_corr, spearman_pval = stats.spearmanr(embedding_similarities, docid_similarities)
    pearson_corr, pearson_pval = stats.pearsonr(embedding_similarities, docid_similarities)
    
    # Compute basic statistics
    metrics = {
        'distance_metric': distance_metric,
        'codebook_size': codebook_size if distance_metric == 'rq_weighted' else None,
        'n_duplicates_removed': n_duplicates,
        'n_unique_docids_analyzed': len(sample_df),
        'correlation': {
            'spearman_correlation': float(spearman_corr),
            'spearman_pvalue': float(spearman_pval),
            'pearson_correlation': float(pearson_corr),
            'pearson_pvalue': float(pearson_pval),
        },
        'embedding_similarity_stats': {
            'mean': float(np.mean(embedding_similarities)),
            'std': float(np.std(embedding_similarities)),
            'min': float(np.min(embedding_similarities)),
            'max': float(np.max(embedding_similarities)),
        },
        'docid_similarity_stats': {
            'mean': float(np.mean(docid_similarities)),
            'std': float(np.std(docid_similarities)),
            'min': float(np.min(docid_similarities)),
            'max': float(np.max(docid_similarities)),
        },
        'uniqueness': {
            'total_docids': original_size,
            'unique_docids': len(sample_df),
            'uniqueness_ratio': len(sample_df) / original_size,
        },
        'docid_length_stats': {
            'mean_length': float(sample_df['docid'].str.len().mean()),
            'std_length': float(sample_df['docid'].str.len().std()),
            'min_length': int(sample_df['docid'].str.len().min()),
            'max_length': int(sample_df['docid'].str.len().max()),
        }
    }
    
    return metrics


def analyze_prefix_similarity(merged_df: pd.DataFrame, embeddings: np.ndarray, 
                             pairs_per_prefix: int = 1000,
                             output_path: str = None):
    """
    Analyze how semantic similarity varies with shared DocID prefix length.
    
    For each prefix length (1 to max_prefix_length codes), sample pairs of documents
    that share the same prefix and compute the mean embedding similarity.
    
    Args:
        merged_df: DataFrame with 'docid' column and embeddings aligned by index
        embeddings: Array of embeddings aligned with merged_df
        pairs_per_prefix: Number of random pairs to sample per prefix length
        output_path: Path to save the plot (if None, returns data without saving)
    
    Returns:
        Dict with prefix lengths and corresponding mean similarities
    """
    print("\nAnalyzing prefix-based semantic similarity...")
    
    # Filter out duplicate docids
    original_size = len(merged_df)
    unique_indices = merged_df.drop_duplicates(subset='docid', keep='first').index.tolist()
    merged_df = merged_df.iloc[unique_indices].reset_index(drop=True)
    embeddings = embeddings[unique_indices]
    
    n_duplicates = original_size - len(merged_df)
    if n_duplicates > 0:
        print(f"⚠️  Removed {n_duplicates} duplicate docids from prefix analysis")
    
    # Parse all docids to get codes
    docids = merged_df['docid'].values
    all_codes = []
    for docid in docids:
        codes = docid.split(' - ')
        all_codes.append(codes)
    
    # Determine max prefix length
    max_prefix_length = max(len(codes) for codes in all_codes)
    print(f"Max prefix length: {max_prefix_length}")
    
    # Build prefix index: prefix -> list of document indices
    # We'll build this for each prefix length
    prefix_similarities = {}
    
    for prefix_len in range(0, max_prefix_length + 1):
        print(f"Processing prefix length {prefix_len}/{max_prefix_length}...")
        
        # Build prefix -> doc_indices mapping
        prefix_to_docs = {}
        for doc_idx, codes in enumerate(all_codes):
            prefix = ' - '.join(codes[:prefix_len])
            if prefix not in prefix_to_docs:
                prefix_to_docs[prefix] = []
            prefix_to_docs[prefix].append(doc_idx)
        
        # Sample pairs with same prefix and compute similarities
        similarities = []
        
        # Get prefixes with at least 2 documents
        valid_prefixes = [p for p, docs in prefix_to_docs.items() if len(docs) >= 2]
        
        if len(valid_prefixes) == 0:
            print(f"  Warning: No prefixes with 2+ documents at length {prefix_len}")
            prefix_similarities[prefix_len] = None
            break
        
        print(f"  Found {len(valid_prefixes)} prefixes with 2+ documents.")
        print(f"Each prefix has on average {np.mean([len(prefix_to_docs[p]) for p in valid_prefixes]):.2f} documents.")
        
        # Sample pairs
        np.random.seed(42 + prefix_len)  # Different seed per prefix length
        
        weights = np.array([len(prefix_to_docs[p]) for p in valid_prefixes])
        weights = weights / weights.sum()  # Normalize to probabilities
        
        # Limit number of pairs to avoid oversampling small prefix sets
        # For each prefix, we can sample at most C(n,2) = n*(n-1)/2 unique pairs
        max_possible_pairs = sum(len(docs) * (len(docs) - 1) // 2 for docs in prefix_to_docs.values() if len(docs) >= 2)
        n_pairs_to_sample = min(pairs_per_prefix, max_possible_pairs)
        
        if n_pairs_to_sample < pairs_per_prefix:
            print(f"  ⚠️  Limited to {n_pairs_to_sample} pairs (max possible unique pairs)")
        
        sampled_pairs = set()  # Track sampled pairs to avoid duplicates
        
        for _ in range(n_pairs_to_sample * 10):  # Allow some attempts for duplicate avoidance
            if len(sampled_pairs) >= n_pairs_to_sample:
                break
                
            # Randomly select a prefix with probability proportional to its size
            prefix = np.random.choice(valid_prefixes, p=weights)
            docs = prefix_to_docs[prefix]
            
            # Randomly select two different documents with this prefix
            idx1, idx2 = np.random.choice(docs, size=2, replace=False)
            
            # Store as sorted tuple to avoid (i,j) and (j,i) duplicates
            pair = tuple(sorted([idx1, idx2]))
            
            if pair not in sampled_pairs:
                sampled_pairs.add(pair)
                # Compute embedding similarity
                emb_sim = compute_embedding_similarity(embeddings, idx1, idx2)
                similarities.append(emb_sim)
        
        # Compute mean similarity and 95% confidence interval for this prefix length
        mean_sim = np.mean(similarities)
        std_sim = np.std(similarities)
        
        # 95% confidence interval using t-distribution
        n_samples = len(similarities)
        if n_samples > 1:
            # Degrees of freedom
            df = n_samples - 1
            # t-value for 95% confidence interval
            t_value = stats.t.ppf(0.975, df)  # 0.975 for two-tailed 95% CI
            # Standard error of the mean
            se_sim = std_sim / np.sqrt(n_samples)
            # Margin of error
            ci_95 = t_value * se_sim
        else:
            ci_95 = float('nan')  # Not enough samples for CI
        
        prefix_similarities[prefix_len] = {
            'mean': float(mean_sim),
            'std': float(std_sim),
            'ci_95': float(ci_95),
            'n_prefixes': len(valid_prefixes),
            'mean_docs_per_prefix': float(np.mean([len(prefix_to_docs[p]) for p in valid_prefixes])),
            'n_pairs': len(similarities)
        }
        print(f"  Prefix length {prefix_len}: mean={mean_sim:.4f}, 95% CI=±{ci_95:.4f}, n_pairs={len(similarities)}")
    
    # Create plot
    if output_path:
        plt.figure(figsize=(10, 6))
        
        # Extract data for plotting
        prefix_lengths = []
        mean_sims = []
        ci_95_sims = []
        
        for prefix_len in sorted(prefix_similarities.keys()):
            if prefix_similarities[prefix_len] is not None:
                prefix_lengths.append(prefix_len)
                mean_sims.append(prefix_similarities[prefix_len]['mean'])
                ci_95_sims.append(prefix_similarities[prefix_len]['ci_95'])  # Use 95% CI
        
        # Plot mean with error bars (95% confidence interval)
        plt.errorbar(prefix_lengths, mean_sims, yerr=ci_95_sims, 
                    marker='o', capsize=5, capthick=2, linewidth=2, markersize=8)
        
        plt.xlabel('Prefix Length (Number of Codes)', fontsize=12)
        plt.ylabel('Mean Embedding Similarity', fontsize=12)
        plt.title('Semantic Similarity vs. Shared DocID Prefix Length', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ Prefix similarity plot saved to: {output_path}")
        plt.close()
    
    return prefix_similarities


def write_metrics_to_file(metrics: Dict[str, Any], output_path: str):
    """Write evaluation metrics to a text file."""
    with open(output_path, 'w') as f:
        f.write(f"{'='*60}\n")
        f.write("DOCID EVALUATION METRICS\n")
        f.write(f"{'='*60}\n\n")
        
        f.write(f"DISTANCE METRIC: {metrics.get('distance_metric', 'unknown')}\n")
        if metrics.get('codebook_size'):
            f.write(f"CODEBOOK SIZE: {metrics['codebook_size']}\n")
        f.write("\n")
        
        f.write("CORRELATION METRICS (DocID similarity vs Semantic similarity)\n")
        f.write(f"  Spearman correlation: {metrics['correlation']['spearman_correlation']:.4f} (p={metrics['correlation']['spearman_pvalue']:.4e})\n")
        f.write(f"  Pearson correlation:  {metrics['correlation']['pearson_correlation']:.4f} (p={metrics['correlation']['pearson_pvalue']:.4e})\n")
        f.write(f"  → Higher correlation means DocIDs better preserve semantic similarity\n")
        
        f.write(f"\nEMBEDDING SIMILARITY STATISTICS\n")
        f.write(f"  Mean: {metrics['embedding_similarity_stats']['mean']:.4f}\n")
        f.write(f"  Std:  {metrics['embedding_similarity_stats']['std']:.4f}\n")
        f.write(f"  Range: [{metrics['embedding_similarity_stats']['min']:.4f}, {metrics['embedding_similarity_stats']['max']:.4f}]\n")
        
        f.write(f"\nDOCID SIMILARITY STATISTICS\n")
        f.write(f"  Mean: {metrics['docid_similarity_stats']['mean']:.4f}\n")
        f.write(f"  Std:  {metrics['docid_similarity_stats']['std']:.4f}\n")
        f.write(f"  Range: [{metrics['docid_similarity_stats']['min']:.4f}, {metrics['docid_similarity_stats']['max']:.4f}]\n")
        
        f.write(f"\nUNIQUENESS METRICS\n")
        f.write(f"  Total DocIDs: {metrics['uniqueness']['total_docids']}\n")
        f.write(f"  Unique DocIDs: {metrics['uniqueness']['unique_docids']}\n")
        f.write(f"  Uniqueness ratio: {metrics['uniqueness']['uniqueness_ratio']:.4f}\n")
        
        f.write(f"\nDOCID LENGTH STATISTICS\n")
        f.write(f"  Mean length: {metrics['docid_length_stats']['mean_length']:.2f}\n")
        f.write(f"  Std length:  {metrics['docid_length_stats']['std_length']:.2f}\n")
        f.write(f"  Range: [{metrics['docid_length_stats']['min_length']}, {metrics['docid_length_stats']['max_length']}]\n")
        
        f.write(f"\n{'='*60}\n")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Document ID generation methods.")
    parser.add_argument("--dataset_name", type=str, default="sentence-transformers/msmarco", help="Name of the dataset to download from HuggingFace (default: sentence-transformers/msmarco).")
    parser.add_argument("--subset", type=str, default=None, help="Subset of the dataset, if applicable.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use (default: train).")
    parser.add_argument("--docid_path", type=str, required=True, help="Path to the dataset folder containing generated DocIDs.")
    parser.add_argument("--sample_size", type=int, default=10000, help="Number of documents to sample for evaluation (default: 1000).")
    parser.add_argument("--n_pairs", type=int, default=5000, help="Number of document pairs to compare (default: 5000).")
    parser.add_argument("--distance_metric", type=str, default="edit_distance", 
                       choices=['edit_distance', 'hamming', 'rq_weighted', 'rq_simple', 'embedding_cosine'],
                       help="Distance metric for DocID similarity (default: edit_distance). Use 'rq_weighted' or 'rq_simple' for residual quantization DocIDs, 'embedding_cosine' for embedding-based DocIDs.")
    parser.add_argument("--codebook_size", type=int, default=256, 
                       help="Codebook size for rq_weighted distance metric (default: 256).")
    parser.add_argument("--no_analyze_prefix", action="store_true", default=False, 
                        help="Skip prefix-based similarity analysis.")
    parser.add_argument("--test_embedding_model", type=str, default="all-MiniLM-L6-v2", 
                       help="SentenceTransformer model to use for embeddings during testing (default: all-MiniLM-L6-v2).")
    args = parser.parse_args()

    print("Starting DocID testing script with arguments:", args)
    sample_df = retrieve_generated_docids(args.dataset_name, args.subset, args.split, args.docid_path, args.sample_size)
    print("Sample of merged DataFrame with generated DocIDs:")
    print(sample_df.head())
    
    # Create output directory
    docid_filename = args.docid_path.strip('/').split('/')[-1].replace('_dataset', '')
    output_dir = os.path.join('docid_results', docid_filename)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")
    
    # Compute embeddings once for both evaluations
    print(f"Loading embedding model...")
    model = SentenceTransformer(args.test_embedding_model) # can be BAAI/bge-small-en-v1.5 when using embedding-based docids
    
    print(f"Computing embeddings for {len(sample_df)} passages...")
    embeddings = model.encode(
        sample_df['passage'].tolist(), 
        batch_size=1024,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    
    # Evaluate DocID relevance (standard metrics)
    print("\n" + "="*60)
    print("EVALUATING DOCID CORRELATION METRICS")
    print("="*60)
    metrics = evaluate_docid_relevance(
        sample_df,
        embeddings,
        n_pairs=args.n_pairs,
        distance_metric=args.distance_metric,
        codebook_size=args.codebook_size
    )
    
    # Save standard metrics
    test_embedding_model_sanitized = args.test_embedding_model.replace('/', '_').replace('-', '_')
    txt_output_path = os.path.join(output_dir, 'metrics_' + args.distance_metric + '_' + test_embedding_model_sanitized + '.txt')
    write_metrics_to_file(metrics, txt_output_path)
    print(f"\n✅ Metrics saved to: {txt_output_path}")
    
    json_output_path = os.path.join(output_dir, 'metrics_' + args.distance_metric + '_' + test_embedding_model_sanitized + '.json')
    with open(json_output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ JSON metrics saved to: {json_output_path}")
    
    # Analyze prefix-based similarity (for RQ and embedding DocIDs with structured format)
    if not args.no_analyze_prefix and args.distance_metric in ['rq_weighted', 'rq_simple', 'embedding_cosine']:
        print("\n" + "="*60)
        print("ANALYZING PREFIX-BASED SEMANTIC SIMILARITY")
        print("="*60)
        
        plot_output_path = os.path.join(output_dir, f'prefix_similarity_plot_{test_embedding_model_sanitized}.png')
        prefix_results = analyze_prefix_similarity(
            sample_df,
            embeddings,
            pairs_per_prefix=1000,
            output_path=plot_output_path
        )
        
        # Save prefix analysis results to JSON
        prefix_json_path = os.path.join(output_dir, f'prefix_similarity_{test_embedding_model_sanitized}.json')
        with open(prefix_json_path, 'w') as f:
            json.dump(prefix_results, f, indent=2)
        print(f"✅ Prefix similarity data saved to: {prefix_json_path}")
    else:
        print("\nSkipping prefix-based similarity analysis (either disabled or not applicable for selected distance metric).")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)