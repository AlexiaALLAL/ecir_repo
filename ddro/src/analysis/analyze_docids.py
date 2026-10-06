import os
import json
import csv
import argparse
import collections
import numpy as np
import torch
from tqdm import tqdm
import gzip
import wandb
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import random
from sentence_transformers import SentenceTransformer
from array import array
import time


parser = argparse.ArgumentParser(description="Generate document IDs using encoding methods")
parser.add_argument("--encoding", default="unknown_encoding", type=str, help="docid method: atomic/pq/rq-kmeans/rq-module/rq-ir/url/summary")
parser.add_argument("--doc_embed_path", type=str, required=False, help="Path to input doc embedding file")
parser.add_argument("--docid_path", type=str, required=True, help="Path to input docid file")
parser.add_argument("--nb_pairs", default=10000, type=int, help="Number of pairs to use for correlation calculation")
parser.add_argument("--nb_triplets", default=10000, type=int, help="Number of triplets to use for ranking preservation calculation")
parser.add_argument("--nb_queries_ranking_preservation", default=-10000, type=int, help="Number of queries to use for query-document ranking preservation calculation")
parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility")
parser.add_argument("--corpus_path", type=str, required=False, help="Path to corpus file (JSONL or TSV format) to get document titles")
parser.add_argument("--num_samples_per_token", default=10, type=int, help="Number of random documents to sample for each first token")
parser.add_argument("--num_codebooks", type=int, default=None, help="L : Number of codebooks used (for RQ/PQ methods)")
parser.add_argument("--nb_subspaces", type=int, default=None, help="C : Number of subspaces used (for RQ/PQ methods)")
parser.add_argument("--codebook_size", type=int, default=None, help="V : Size of each codebook (for RQ/PQ methods)")
parser.add_argument("--model_path", type=str, required=False, help="Path to trained quantization model checkpoint (for model-based evaluation)")
parser.add_argument("--query_embed_path", type=str, required=False, help="Path to query embeddings file (for model-based IR evaluation)")
parser.add_argument("--qrel_path", type=str, required=False, help="Path to qrels file (for model-based IR evaluation)")
parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for encoding embeddings (reduce if memory issues occur)")
parser.add_argument("--results_file", type=str, default="src/analysis/all_analysis_results.csv", help="Path to cumulative results file")
parser.add_argument("--num_samples_for_robustness", type=int, default=100, help="Number of document pairs to sample for robustness evaluation")
parser.add_argument("--embedding_model", type=str, default="sentence-transformers/gtr-t5-base", help="SentenceTransformer model to use for embedding-based robustness evaluation")
parser.add_argument("--pretrain_model_path", type=str, default="t5-base", help="Path to T5 model for vocab_size (used for docid shifting)")
parser.add_argument("--dataset", type=str, default="msmarco", help="Dataset name (msmarco or nq) for organizing output files")
parser.add_argument("--metrics", type=str, nargs="+", default=["all"], 
                    help="List of metrics to compute. Options: all, uniqueness, entropy_gini, correlation, "
                         "ranking_preservation, shared_tokens, query_ranking, query_closeness, robustness, "
                         "robustness_ranking_preservation, sample_docs, plot_distributions. Default: all")


args = parser.parse_args()

# Available metrics
AVAILABLE_METRICS = {
    'uniqueness',           # Uniqueness ratio of docids
    'entropy_gini',         # Entropy and Gini index per token level
    'correlation',          # Correlation between embedding and docid similarity
    'shared_tokens',        # Embedding similarity for pairs with shared tokens, mean over k
    'shared_tokens_new',    # Embedding similarity for pairs with shared tokens, mean over (k, doc)
    'ranking_preservation', # Ranking preservation for random triplets
    'query_ranking',        # Query-doc ranking preservation (requires model)
    'query_closeness',      # Query-doc closeness (requires model)
    'robustness',           # Robustness to text reformulation (requires corpus and model)
    'robustness_ranking_preservation', # Ranking preservation for reformulated docs vs random docs
    'sample_docs',          # Save sample documents by first token
    'plot_distributions'    # Plot code-level distributions
}

# Parse metrics argument
if 'all' in args.metrics:
    METRICS_TO_COMPUTE = AVAILABLE_METRICS.copy()
else:
    METRICS_TO_COMPUTE = set(args.metrics)
    # Validate metrics
    invalid_metrics = METRICS_TO_COMPUTE - AVAILABLE_METRICS
    if invalid_metrics:
        raise ValueError(f"Invalid metrics: {invalid_metrics}. Available metrics: {AVAILABLE_METRICS}")

print(f"\nMetrics to compute: {sorted(METRICS_TO_COMPUTE)}")

def smart_open(path, mode="rt", encoding="utf-8"):
    if "b" in mode:
        return gzip.open(path, mode=mode) if str(path).endswith(".gz") else open(path, mode=mode)
    if mode == "r":
        mode = "rt"
    return gzip.open(path, mode=mode, encoding=encoding) if str(path).endswith(".gz") else open(path, mode=mode, encoding=encoding)

def normalize_docid(docid: str) -> str:
    """Normalize a docid to the bracketed lowercase form used in the docid file."""
    return "[{}]".format(str(docid).lower().strip('[').strip(']'))

def get_encoded_docid(docid_path, all_docid=None, token_to_id=None):
    encoded_docid = {}
    with smart_open(docid_path) as fr:
        for line in fr:
            docid, encode = line.strip().split("\t")
            docid = normalize_docid(docid)
            encoded_docid[docid] = encode
    return encoded_docid

def load_original_docids(docid_path, target_ids):
    """
    Load original docids for the target documents.
    
    Args:
        docid_path: Path to encoded docid file
        target_ids: Set of ids to load
        
    Returns:
        dict: {docid: encoded_docid_string}
    """
    encoded_docid = {}
    
    with smart_open(docid_path) as fr:
        for line in fr:
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue
            docid, encoded = parts
            docid = normalize_docid(docid)
            
            if docid in target_ids:
                encoded_docid[docid] = encoded
    
    print(f"Loaded {len(encoded_docid)} original docids")
    return encoded_docid

def get_uniqueness_ratio(encoded_docid):
    unique_docids = set(encoded_docid.values())
    uniqueness_ratio = len(unique_docids) / len(encoded_docid) if len(encoded_docid) > 0 else 0
    return uniqueness_ratio

def load_doc_embeddings(doc_embed_path):
    docid_2_idx, idx_2_docid = {}, {}
    doc_embeddings = []
    open_func = gzip.open if doc_embed_path.endswith('.gz') else open
    with open_func(doc_embed_path, 'rt', encoding='utf-8') as fr:
        for line in tqdm(fr, desc="Loading document vectors"):
            did, demb = line.strip().split('\t')
            d_embedding = [float(x) for x in demb.split(',')]
            docid_2_idx[did] = len(docid_2_idx)
            idx_2_docid[docid_2_idx[did]] = did
            doc_embeddings.append(d_embedding)
    return docid_2_idx, idx_2_docid, np.array(doc_embeddings, dtype=np.float32)

def load_query_embeddings(query_embed_path):
    """
    Load query embeddings from file.
    Returns:
        qid_2_idx: dict mapping query ID to index
        idx_2_qid: dict mapping index to query ID
        query_embeddings: np.array of query embeddings
    """
    qid_2_idx, idx_2_qid = {}, {}
    query_embeddings = []
    open_func = gzip.open if query_embed_path.endswith('.gz') else open
    with open_func(query_embed_path, 'rt', encoding='utf-8') as fr:
        for line in tqdm(fr, desc="Loading query vectors"):
            qid, qemb = line.strip().split('\t')
            q_embedding = [float(x) for x in qemb.split(',')]
            qid_2_idx[qid] = len(qid_2_idx)
            idx_2_qid[qid_2_idx[qid]] = qid
            query_embeddings.append(q_embedding)
    return qid_2_idx, idx_2_qid, np.array(query_embeddings, dtype=np.float32)

def load_corpus(corpus_path):
    """
    Load corpus from JSONL or TSV format file.
    Expected formats:
    - JSONL: {"docid": "...", "title": "...", "text": "..."}
    - TSV: docid\ttitle\ttext
    Returns dict mapping docid to document info.
    """
    corpus = {}
    with smart_open(corpus_path) as fr:
        for line in tqdm(fr, desc="Loading corpus"):
            line = line.strip()
            if not line:
                continue
            
            try:
                # Try JSON format first
                item = json.loads(line)
                
            except json.JSONDecodeError:
                print(f"Line is not valid JSON: {line[:100]}...")
                continue
            docid = item.get('docid') or item.get('id')
            title = item.get('title', '')
            if docid:
                docid = normalize_docid(str(docid))
                corpus[docid] = {'title': title}
    print(f"Loaded {len(corpus)} documents from corpus.")
    print(f"Sample document from corpus: {next(iter(corpus.items()))}")
    return corpus

def load_corpus_subset(corpus_path, num_samples=100, seed=42):
    """
    Load a random subset of documents from the corpus.
    
    Returns:
        dict: {docid: {"title": str, "text": str}}
    """
    random.seed(seed)
    np.random.seed(seed)
    
    print(f"Loading corpus from {corpus_path}...")
    all_docs = []
    
    with smart_open(corpus_path) as fr:
        for line in tqdm(fr, desc="Reading corpus"):
            line = line.strip()
            if not line:
                continue
            
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping non-JSON line: {line[:100]}...")
                continue
            
            docid = normalize_docid(item.get('docid') or item.get('id', ''))
            title = item.get('title', '')
            text = item.get("body", "") or item.get("doc_tac", "")
            text = text.replace("\n", " ").replace("\t", " ").strip()
            
            if docid and text:
                all_docs.append({
                    'docid': docid,
                    'title': title,
                    'text': text
                })
    
    print(f"Loaded {len(all_docs)} documents from corpus")
    
    # Sample subset
    if num_samples < len(all_docs):
        sampled_docs = random.sample(all_docs, num_samples)
    else:
        print(f"Warning: num_samples={num_samples} is greater than total documents={len(all_docs)}. Using all documents.")
        sampled_docs = all_docs
    
    # Convert to dict
    corpus_dict = {doc['docid']: {'title': doc['title'], 'text': doc['text']} 
                   for doc in sampled_docs}
    
    print(f"Selected {len(corpus_dict)} documents for robustness testing")
    print(f"Sample docid: {list(corpus_dict.keys())[0]}, title: {corpus_dict[list(corpus_dict.keys())[0]]['title']}, text: {corpus_dict[list(corpus_dict.keys())[0]]['text'][:100]}...")
    return corpus_dict

def load_qrels(qrel_path):
    """
    Load qrels (relevance judgments) from file.
    Format: qid 0 docid relevance
    Returns:
        qrels: dict[qid] -> list of relevant docids
    """
    qrels = collections.defaultdict(list)
    open_func = gzip.open if qrel_path.endswith('.gz') else open
    with open_func(qrel_path, 'rt', encoding='utf-8') as fr:
        for line in tqdm(fr, desc="Loading qrels"):
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            qid, _, docid, relevance = parts[0], parts[1], parts[2], parts[3]
            qid = f"[{qid.lower()}]"
            docid = f"[{docid.lower()}]"
            if int(relevance) > 0:
                qrels[qid].append(docid)
    print(f"Loaded {len(qrels)} queries with relevance judgments")
    return dict(qrels)

def get_docid_similarity(docid1, docid2, docid_distance="hierarchical", C = None, L = None, V = None):
    assert docid_distance in ["hierarchical", "full", "mix"], "Unsupported docid_distance. Choose from 'hierarchical', 'full', or 'mix'."
    
    if docid_distance == "full":
        sim = 0
        for a, b in zip(docid1.split(','), docid2.split(',')):
            if a == b:
                sim += 1
        return sim/len(docid1.split(','))
    
    elif docid_distance == "hierarchical":
        sim = 0
        for a, b in zip(docid1.split(','), docid2.split(',')):
            if a == b:
                sim += 1
            else:
                break  # stop at the first mismatch for hierarchical distance
        return sim/len(docid1.split(','))

    elif docid_distance == "mix":
        assert C is not None and L is not None, "C and L must be provided for mix distance"
        sim = 0
        tokens1 = docid1.split(',')
        tokens2 = docid2.split(',')
        for c in range(C):
            simc = 0
            # docids are k(l,c) ordered like k(1,1), k(1,2), ..., k(1,C), k(2,1), ..., k(L,C)
            docid_c_1 = tokens1[c::C]
            docid_c_2 = tokens2[c::C]
            for a, b in zip(docid_c_1, docid_c_2):
                if a == b:
                    simc += 1
                else:
                    break  # stop at the first mismatch for this subspace
            sim += simc
        return sim / (C * L)

    else:
        raise ValueError(f"Unsupported docid_distance: {docid_distance}")

def get_correlation(embeddings, docid_2_idx, encoded_docid, nb_pairs=10000, C=None, L=None):
    """
    Get correlation between embeddings similarity and docid similarity. 
    Docid similarity is defined as the number of matching docid components. 
    For example, sim ([0,1,2,3], [0,1,4,3]) = 2 since the first two components match.
    Samples randomly nb_pairs of document pairs to compute the correlation to save time.
    """
    docids = list(encoded_docid.keys())
    docid_sims_hierarchical, docid_sims_full, docid_sims_mix, embed_sims = [], [], [], []
    for _ in tqdm(range(nb_pairs), desc="Computing docid vs embedding similarity"): 
        i, j = np.random.choice(len(docids), 2, replace=False)
        id_i, id_j = docids[i], docids[j]
        docid_i, docid_j = encoded_docid[id_i], encoded_docid[id_j]
        embedding_i, embedding_j = embeddings[docid_2_idx[id_i]], embeddings[docid_2_idx[id_j]]
        sim_hierarchical = get_docid_similarity(docid_i, docid_j, docid_distance="hierarchical")
        sim_full = get_docid_similarity(docid_i, docid_j, docid_distance="full")
        sim_mix = get_docid_similarity(docid_i, docid_j, docid_distance="mix", C=C, L=L)
        docid_sims_hierarchical.append(sim_hierarchical)
        docid_sims_full.append(sim_full)
        docid_sims_mix.append(sim_mix)
        embed_sim = np.dot(embedding_i, embedding_j) / (np.linalg.norm(embedding_i) * np.linalg.norm(embedding_j))
        embed_sims.append(embed_sim)
    correlation_hierarchical = np.corrcoef(docid_sims_hierarchical, embed_sims)[0, 1] if len(docid_sims_hierarchical) > 1 else 0
    correlation_full = np.corrcoef(docid_sims_full, embed_sims)[0, 1] if len(docid_sims_full) > 1 else 0
    correlation_mix = np.corrcoef(docid_sims_mix, embed_sims)[0, 1] if len(docid_sims_mix) > 1 else 0
    return correlation_hierarchical, correlation_full, correlation_mix

def get_ranking_preservation(embeddings, docid_2_idx, encoded_docid, nb_triplets=10000, C=None, L=None):
    """
    Measure if docid codes preserve ranking.
    For each triplet (d1, d2, d3) where d1 is the anchor:
    - Find which of d2, d3 is closest to d1 using embedding similarity
    - Find which of d2, d3 is closest to d1 using docid similarity
    - Check if both measures agree
    Returns the average agreement across all triplets (excluding ties).
    """
    docids = list(encoded_docid.keys())
    agreements_full = []
    agreements_hierarchical = []
    agreements_mix = []
    
    skipped_embedding_ties = 0
    skipped_docid_ties_full = 0
    skipped_docid_ties_hierarchical = 0
    skipped_docid_ties_mix = 0
    
    # Continue sampling until we have exactly nb_triplets valid triplets for EACH distance
    pbar = tqdm(total=nb_triplets * 3, desc=f"Computing ranking preservation")
    
    while len(agreements_full) < nb_triplets or len(agreements_hierarchical) < nb_triplets or len(agreements_mix) < nb_triplets:
        # Sample 3 distinct documents
        i, j, k = np.random.choice(len(docids), 3, replace=False)
        id_anchor, id_d2, id_d3 = docids[i], docids[j], docids[k]
        
        # Get embeddings
        emb_anchor = embeddings[docid_2_idx[id_anchor]]
        emb_d2 = embeddings[docid_2_idx[id_d2]]
        emb_d3 = embeddings[docid_2_idx[id_d3]]
        
        # Calculate embedding similarities
        emb_sim_d2 = np.dot(emb_anchor, emb_d2) / (np.linalg.norm(emb_anchor) * np.linalg.norm(emb_d2))
        emb_sim_d3 = np.dot(emb_anchor, emb_d3) / (np.linalg.norm(emb_anchor) * np.linalg.norm(emb_d3))
        
        # Skip if embedding similarity is a tie
        if abs(emb_sim_d2 - emb_sim_d3) < 1e-9:
            skipped_embedding_ties += 1
            continue
        
        # Determine which is closer by embedding
        closest_by_embedding = id_d2 if emb_sim_d2 > emb_sim_d3 else id_d3
        
        # Get docids
        docid_anchor = encoded_docid[id_anchor]
        docid_d2 = encoded_docid[id_d2]
        docid_d3 = encoded_docid[id_d3]
        
        # Handle full distance - only add if we haven't reached nb_triplets yet
        if len(agreements_full) < nb_triplets:
            docid_sim_d2_full = get_docid_similarity(docid_anchor, docid_d2, docid_distance="full")
            docid_sim_d3_full = get_docid_similarity(docid_anchor, docid_d3, docid_distance="full")
            if docid_sim_d2_full != docid_sim_d3_full:
                closest_by_docid_full = id_d2 if docid_sim_d2_full > docid_sim_d3_full else id_d3
                agreement_full = 1 if closest_by_embedding == closest_by_docid_full else 0
                agreements_full.append(agreement_full)
            else:
                skipped_docid_ties_full += 1
        
        # Handle hierarchical distance - only add if we haven't reached nb_triplets yet
        if len(agreements_hierarchical) < nb_triplets:
            docid_sim_d2_hierarchical = get_docid_similarity(docid_anchor, docid_d2, docid_distance="hierarchical")
            docid_sim_d3_hierarchical = get_docid_similarity(docid_anchor, docid_d3, docid_distance="hierarchical")
            if docid_sim_d2_hierarchical != docid_sim_d3_hierarchical:
                closest_by_docid_hierarchical = id_d2 if docid_sim_d2_hierarchical > docid_sim_d3_hierarchical else id_d3
                agreement_hierarchical = 1 if closest_by_embedding == closest_by_docid_hierarchical else 0
                agreements_hierarchical.append(agreement_hierarchical)
            else:
                skipped_docid_ties_hierarchical += 1
        
        # Handle mix distance - only add if we haven't reached nb_triplets yet
        if len(agreements_mix) < nb_triplets:
            docid_sim_d2_mix = get_docid_similarity(docid_anchor, docid_d2, docid_distance="mix", C=C, L=L)
            docid_sim_d3_mix = get_docid_similarity(docid_anchor, docid_d3, docid_distance="mix", C=C, L=L)
            if docid_sim_d2_mix != docid_sim_d3_mix:
                closest_by_docid_mix = id_d2 if docid_sim_d2_mix > docid_sim_d3_mix else id_d3
                agreement_mix = 1 if closest_by_embedding == closest_by_docid_mix else 0
                agreements_mix.append(agreement_mix)
            else:
                skipped_docid_ties_mix += 1
        
        # Update progress bar
        current_progress = len(agreements_full) + len(agreements_hierarchical) + len(agreements_mix)
        pbar.n = current_progress
        pbar.refresh()
    
    pbar.close()
    
    avg_agreement_full = np.mean(agreements_full) if agreements_full else 0
    avg_agreement_hierarchical = np.mean(agreements_hierarchical) if agreements_hierarchical else 0
    avg_agreement_mix = np.mean(agreements_mix) if agreements_mix else 0
    
    print(f"  Skipped {skipped_embedding_ties} triplets due to embedding ties")
    print(f"  Skipped {skipped_docid_ties_full} triplets due to docid ties (full distance)")
    print(f"  Skipped {skipped_docid_ties_hierarchical} triplets due to docid ties (hierarchical distance)")
    print(f"  Skipped {skipped_docid_ties_mix} triplets due to docid ties (mix distance)")
    print(f"  Evaluated {len(agreements_full)} triplets for full distance")
    print(f"  Evaluated {len(agreements_hierarchical)} triplets for hierarchical distance")
    print(f"  Evaluated {len(agreements_mix)} triplets for mix distance")
    
    return avg_agreement_full, avg_agreement_hierarchical, avg_agreement_mix

def save_sample_docs_by_first_token(encoded_docid, corpus, encoding, num_samples=10):
    """
    For each unique first token in docids, save num_samples random documents to a JSON file.
    Groups documents by their first token (before the first comma in the encoded docid).
    """
    output_path = f"src/analysis/first_tokens/{encoding}/sample_docs_by_first_token.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Group docids by first token
    first_token_groups = collections.defaultdict(list) # {first_token: [docid1, docid2, ...], ...}
    
    for doc_id, encoded_id in encoded_docid.items():
        first_token = encoded_id.split(',')[0]
        first_token_groups[first_token].append(doc_id)
    
    # Build data structure
    results = {
        "total_first_tokens": len(first_token_groups),
        "samples_per_token": num_samples,
        "samples_by_token": {}
    }
    
    for first_token in sorted(first_token_groups.keys()):
        doc_ids = first_token_groups[first_token]
        num_to_sample = min(num_samples, len(doc_ids))
        sampled_ids = np.random.choice(doc_ids, size=num_to_sample, replace=False)
        
        token_data = {
            "total_documents": len(doc_ids),
            "samples": []
        }
        
        for doc_id in sampled_ids:
            if doc_id in corpus:
                title = corpus[doc_id].get('title', 'N/A')
            else:
                title = 'Title not found in corpus'
            
            token_data["samples"].append({
                "docid": doc_id,
                "title": title
            })
        
        results["samples_by_token"][first_token] = token_data
    
    # Write to JSON file
    with open(output_path, 'w', encoding='utf-8') as fw:
        json.dump(results, fw, indent=2, ensure_ascii=False)
    
    print(f"Saved sample documents to {output_path}")

def _sort_code_tokens(tokens):
    def sort_key(token):
        token = token.strip()
        if token.startswith("[") and token.endswith("]"):
            token = token[1:-1]
        try:
            return (0, int(token))
        except ValueError:
            return (1, token)

    return sorted(tokens, key=sort_key)

def _normalized_entropy_from_counts(counts):
    """
    Compute normalized entropy in [0, 1] from a list of counts.
    Normalization uses log(K), where K is the number of unique tokens.
    """
    if not counts:
        return 0.0

    values = np.array(counts, dtype=np.float64)
    total = values.sum()
    if total <= 0:
        return 0.0

    probs = values / total
    probs = probs[probs > 0]
    if probs.size <= 1:
        return 0.0

    entropy = -np.sum(probs * np.log(probs))
    max_entropy = np.log(probs.size)
    if max_entropy <= 0:
        return 0.0
    return float(entropy / max_entropy)

def compute_gini_index(counts):
    """
    Compute the Gini index from a list of counts.
    Gini index measures inequality in the distribution.
    Returns a value in [0, 1] where 0 = perfect equality, 1 = perfect inequality.
    """
    if not counts or len(counts) == 0:
        return 0.0
    
    counts = np.array(counts, dtype=np.float64)
    if counts.sum() == 0:
        return 0.0
    
    # Sort counts
    sorted_counts = np.sort(counts)
    n = len(sorted_counts)
    
    # Compute Gini index
    cumsum = np.cumsum(sorted_counts)
    gini = (2 * np.sum((np.arange(1, n + 1)) * sorted_counts) - (n + 1) * cumsum[-1]) / (n * cumsum[-1])
    
    return float(gini)

def compute_entropy_and_gini_per_level(encoded_docid):
    """
    Compute normalized entropy and Gini index for each code level.
    Returns:
        entropies: list of normalized entropy values per level
        gini_indices: list of Gini index values per level
    """
    level_counters = []
    for encoded in encoded_docid.values():
        tokens = [tok.strip() for tok in encoded.split(',')]
        for level_idx, token in enumerate(tokens):
            if level_idx >= len(level_counters):
                level_counters.append(collections.Counter())
            level_counters[level_idx][token] += 1
    
    entropies = []
    gini_indices = []
    
    for level_idx, counter in enumerate(level_counters):
        counts = list(counter.values())
        
        # Compute normalized entropy
        normalized_entropy = _normalized_entropy_from_counts(counts)
        entropies.append(normalized_entropy)
        
        # Compute Gini index
        gini = compute_gini_index(counts)
        gini_indices.append(gini)
    
    return entropies, gini_indices

def build_index(docs, V=256, M=24):
    """
    Build an inverted index for the given documents.

    Args:
        docs: List of tuples (doc_id, code) where code is a list of tokens.
        V: Number of unique tokens.
        M: Number of positions in the code.

    Returns:
        index: A 2D list where index[token][pos] contains a list of doc_ids, 
                indicating which documents have the given token at the given position.
    """
    index = [[array('I') for _ in range(M)] for _ in range(V)]

    for doc_id, code in docs:
        for pos, token in enumerate(code):
            index[token][pos].append(doc_id)

    return index

def search(index, query):
    postings = []

    for pos, token in enumerate(query):
        if token is not None:
            postings.append(index[token][pos])

    if not postings:
        return []

    # optimisation cruciale :
    # commencer par les listes les plus petites
    postings.sort(key=len)

    result = postings[0]

    for p in postings[1:]:
        result = intersect_sorted(result, p)

        if not result:
            break

    return result

def intersect_sorted(a, b):
    """
    Compute the intersection of two sorted arrays.
    """
    i = j = 0
    out = array('I')

    while i < len(a) and j < len(b):
        va = a[i]
        vb = b[j]

        if va == vb:
            out.append(va)
            i += 1
            j += 1
        elif va < vb:
            i += 1
        else:
            j += 1

    return out

def compute_embedding_similarity_by_shared_tokens_old(embeddings, docid_2_idx, encoded_docid, 
                                                   nb_pairs=1000, seed=42, C=None, L=None):
    """
    Compute mean embedding similarity for document pairs sharing k tokens in their docids.
    
    For hierarchical matching: pairs share the first k tokens (prefix matching)
    For full matching: pairs share k tokens at random positions (random mask matching)
    
    Args:
        embeddings: numpy array of document embeddings
        docid_2_idx: dict mapping docid to index in embeddings
        encoded_docid: dict mapping docid to encoded docid string
        nb_pairs: number of pairs to sample for each k
        seed: random seed for reproducibility
        
    Returns:
        hierarchical_similarities: dict {k: mean_similarity} for prefix matching
        full_similarities: dict {k: mean_similarity} for random mask matching
        mix_similarities: dict {k: mean_similarity} for mix matching
    """
    np.random.seed(seed)
    
    # Parse all docids into token lists
    docid_tokens = {}
    num_tokens = None
    for docid, encoded in encoded_docid.items():
        tokens = tuple(encoded.split(','))
        docid_tokens[docid] = tokens
        if num_tokens is None:
            num_tokens = len(tokens)
        elif len(tokens) != num_tokens:
            # Handle variable length docids - use the minimum
            num_tokens = min(num_tokens, len(tokens))
        
    hierarchical_similarities = {}
    full_similarities = {}
    mix_similarities = {}
    
    print("\nComputing embedding similarity by shared tokens...")
    
    # Hierarchical: prefix matching
    print("  Hierarchical (prefix) matching:")
    for k in range(1, num_tokens + 1):
        # Group documents by first k tokens
        prefix_groups = collections.defaultdict(list) # {(t1, t2, ..., tk): [docid1, docid2, ...], ...}
        for docid, tokens in docid_tokens.items():
            if len(tokens) >= k:
                prefix = tokens[:k]
                prefix_groups[prefix].append(docid)
        
        # Sample pairs from groups with at least 2 documents
        valid_groups = [group for group in prefix_groups.values() if len(group) >= 2]
        
        if not valid_groups:
            hierarchical_similarities[k] = 0.0
            continue
        
        similarities = []
        for _ in range(nb_pairs):
            # Pick a random group
            group = valid_groups[np.random.randint(len(valid_groups))]
            # Pick two different documents from the group
            doc1, doc2 = np.random.choice(group, size=2, replace=False)
            
            # Compute embedding similarity
            emb1 = embeddings[docid_2_idx[doc1]]
            emb2 = embeddings[docid_2_idx[doc2]]
            sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            similarities.append(sim)
        
        mean_sim = np.mean(similarities)
        hierarchical_similarities[k] = mean_sim
        print(f"    k={k}: {mean_sim:.4f} (avg similarity for {len(similarities)} pairs)")
    
    # Full: random mask matching
    print("  Full (random mask) matching:")
    for k in range(1, num_tokens + 1):
        similarities = []
        # Compute groupings once per mask, then sample multiple pairs per mask
        num_masks_to_try = max(100, nb_pairs // 10)  # Enough masks to get sufficient pairs
        pairs_per_mask = 10  # Sample multiple pairs per mask to amortize cost
        
        for _ in tqdm(range(num_masks_to_try), desc=f"    k={k}", leave=False):
            if len(similarities) >= nb_pairs:
                break
            
            # Pick k random positions
            mask_positions = tuple(sorted(np.random.choice(num_tokens, size=k, replace=False)))
            
            # Find documents matching at these positions (computed ONCE per mask)
            mask_groups = collections.defaultdict(list) # {(token_i, token_j, ...): [docid1, docid2, ...], ...}
            for docid, tokens in docid_tokens.items():
                if len(tokens) >= num_tokens:
                    masked_tokens = tuple(tokens[i] for i in mask_positions)
                    mask_groups[masked_tokens].append(docid)
            
            # Find groups with at least 2 documents
            valid_groups = [group for group in mask_groups.values() if len(group) >= 2]
            
            if not valid_groups:
                continue
            
            # Sample MULTIPLE pairs from this mask to amortize the grouping cost
            for _ in range(pairs_per_mask):
                if len(similarities) >= nb_pairs:
                    break
                if len(valid_groups) <= pairs_per_mask:
                    print(f"      Warning: Only {len(valid_groups)} valid groups for k={k} mask, may get the same pair multiple times.")
                
                # Pick a random group
                group = valid_groups[np.random.randint(len(valid_groups))]
                # Pick two different documents from the group
                doc1, doc2 = np.random.choice(group, size=2, replace=False)
                
                # Compute embedding similarity
                emb1 = embeddings[docid_2_idx[doc1]]
                emb2 = embeddings[docid_2_idx[doc2]]
                sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                similarities.append(sim)

        mean_sim = np.mean(similarities) if similarities else 0.0
        full_similarities[k] = mean_sim
        print(f"    k={k}: {mean_sim:.4f} (avg similarity for {len(similarities)} pairs)")
    
    # Mix: hierarchical + random mask matching
    print("  Mix (hierarchical + random mask) matching:")
    for k in range(1, num_tokens + 1):
        similarities = []
        num_masks_to_try = max(100, nb_pairs // 10)  # Enough masks to get sufficient pairs
        pairs_per_mask = 10  # Sample multiple pairs per mask to amortize cost

        for _ in tqdm(range(num_masks_to_try), desc=f"    k={k}", leave=False):
            if len(similarities) >= nb_pairs:
                break

            # Sample a list of ki random number of masked tokens per subspace such that sum(ki) = k
            ki_values = [0] * C
            while sum(ki_values) < k:
                idx = np.random.randint(C)
                if ki_values[idx] < L:  # Ensure we don't exceed the number of levels
                    ki_values[idx] += 1
            # Build mask positions based on ki_values
            mask_positions = []
            for c in range(C):
                positions_c = [i for i in range(c, num_tokens, C)][:ki_values[c]]
                mask_positions.extend(positions_c)
            mask_positions = tuple(sorted(mask_positions))

            # Find documents matching at these positions
            mask_groups = collections.defaultdict(list) # {(token_i, token_j, ...): [docid1, docid2, ...], ...}
            for docid, tokens in docid_tokens.items():
                if len(tokens) >= num_tokens:
                    masked_tokens = tuple(tokens[i] for i in mask_positions)
                    mask_groups[masked_tokens].append(docid)
            
            # Find groups with at least 2 documents
            valid_groups = [group for group in mask_groups.values() if len(group) >= 2]
            if not valid_groups:
                continue

            # Sample MULTIPLE pairs from this mask to amortize the grouping cost
            for _ in range(pairs_per_mask):
                if len(similarities) >= nb_pairs:
                    break
                if len(valid_groups) <= pairs_per_mask:
                    print(f"      Warning: Only {len(valid_groups)} valid groups for k={k} mask, may get the same pair multiple times.")
                

                # Pick a random group
                group = valid_groups[np.random.randint(len(valid_groups))]
                # Pick two different documents from the group
                doc1, doc2 = np.random.choice(group, size=2, replace=False)
                # Compute embedding similarity
                emb1 = embeddings[docid_2_idx[doc1]]
                emb2 = embeddings[docid_2_idx[doc2]]
                sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                similarities.append(sim)
        mean_sim = np.mean(similarities) if similarities else 0.0
        mix_similarities[k] = mean_sim
        print(f"    k={k}: {mean_sim:.4f} (avg similarity for {len(similarities)} pairs)")

    return hierarchical_similarities, full_similarities, mix_similarities

def compute_embedding_similarity_by_shared_tokens(embeddings, docid_2_idx, encoded_docid, 
                                                   nb_pairs=1000, seed=42, C=None, L=None, V=None):
    """
    Compute mean embedding similarity for document pairs sharing k tokens in their docids.
    Uses inverted index for efficient document lookup.
    
    For hierarchical matching: pairs share the first k tokens (prefix matching)
    For full matching: pairs share k tokens at random positions (random mask matching)
    
    Args:
        embeddings: numpy array of document embeddings
        docid_2_idx: dict mapping docid to index in embeddings
        encoded_docid: dict mapping docid to encoded docid string
        nb_pairs: number of pairs to sample for each k
        seed: random seed for reproducibility
        C: number of subspaces (for mix matching)
        L: number of codebooks (for mix matching)
        V: number of unique tokens (for building index)
        
    Returns:
        hierarchical_similarities: dict {k: mean_similarity} for prefix matching
        full_similarities: dict {k: mean_similarity} for random mask matching
        mix_similarities: dict {k: mean_similarity} for mix matching
    """
    np.random.seed(seed)
    
    # Parse all docids into token lists and build index
    docs = []
    docid_list = []  # Keep track of docid order
    all_token_values = set()
    
    for docid, encoded in encoded_docid.items():
        tokens = [int(t) for t in encoded.split(',')]
        docs.append((len(docid_list), tokens))  # Use index as doc_id for the index
        docid_list.append(docid)
        all_token_values.update(tokens)
    
    # Build inverted index with appropriate V (max token value) and M (num positions)
    num_tokens = L * C
    max_token = max(all_token_values)
    min_token = min(all_token_values)
    
    print(f"Token value range: [{min_token}, {max_token}], num_tokens: {num_tokens}")
    
    # Build index using actual token range (not the codebook_size parameter!)
    index = build_index(docs, V=max_token+1, M=num_tokens)
    print(f"Built inverted index with V={max_token+1}, M={num_tokens} for {len(docs)} documents")
        
    hierarchical_similarities = {}
    full_similarities = {}
    mix_similarities = {}
    
    print("\nComputing embedding similarity by shared tokens...")
    
    # Hierarchical: prefix matching
    print("  Hierarchical (prefix) matching:")
    time_start = time.time()
    pairs_per_anchor = 20  # Sample multiple pairs per anchor to reduce searches
    max_attempts = nb_pairs * 50  # Prevent infinite loop

    
    for k in range(1, num_tokens + 1):
        similarities = []
        attempts = 0
        
        while len(similarities) < nb_pairs and attempts < max_attempts:
            attempts += 1
            # Pick a random document
            anchor_idx = np.random.randint(len(docs))
            anchor_tokens = docs[anchor_idx][1]
            
            # Create query: first k tokens, rest are None (wildcards)
            query = anchor_tokens[:k] + [None] * (num_tokens - k)
            
            # Find all documents with this prefix
            matching_indices = search(index, query)
            
            # Need at least 2 documents (including anchor)
            if len(matching_indices) < 2:
                continue
            
            # Sample multiple pairs from this anchor to reduce number of searches
            num_pairs_to_sample = min(pairs_per_anchor, nb_pairs - len(similarities))
            for _ in range(num_pairs_to_sample):
                # Pick two different documents from the matches
                if len(matching_indices) < 2:
                    break
                idx1, idx2 = np.random.choice(matching_indices, size=2, replace=False)
                docid1, docid2 = docid_list[idx1], docid_list[idx2]
                
                # Compute embedding similarity
                emb1 = embeddings[docid_2_idx[docid1]]
                emb2 = embeddings[docid_2_idx[docid2]]
                sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                similarities.append(sim)
        
        mean_sim = np.mean(similarities)
        hierarchical_similarities[k] = mean_sim
        print(f"    k={k}: {mean_sim:.4f} (avg similarity for {len(similarities)} pairs, {attempts} attempts)")
    print(f"  Hierarchical matching took {(time.time() - time_start) / 60:.2f} minutes")

    # Full: random mask matching
    print("  Full (random mask) matching:")
    start_time = time.time()
    for k in range(1, num_tokens + 1):
        similarities = []
        attempts = 0
        
        while len(similarities) < nb_pairs and attempts < max_attempts:
            attempts += 1
            # Pick k random positions
            mask_positions = sorted(np.random.choice(num_tokens, size=k, replace=False))
            
            # Pick a random document
            anchor_idx = np.random.randint(len(docs))
            anchor_tokens = docs[anchor_idx][1]
            
            # Create query: None everywhere except at mask positions
            query = [None] * num_tokens
            for pos in mask_positions:
                query[pos] = anchor_tokens[pos]
            
            # Find all documents matching at these positions
            matching_indices = search(index, query)
            
            # Need at least 2 documents
            if len(matching_indices) < 2:
                continue
            
            # Sample multiple pairs from this anchor to reduce number of searches
            num_pairs_to_sample = min(pairs_per_anchor, nb_pairs - len(similarities))
            for _ in range(num_pairs_to_sample):
                # Pick two different documents from the matches
                if len(matching_indices) < 2:
                    break
                idx1, idx2 = np.random.choice(matching_indices, size=2, replace=False)
                docid1, docid2 = docid_list[idx1], docid_list[idx2]
                
                # Compute embedding similarity
                emb1 = embeddings[docid_2_idx[docid1]]
                emb2 = embeddings[docid_2_idx[docid2]]
                sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                similarities.append(sim)

        mean_sim = np.mean(similarities)
        full_similarities[k] = mean_sim
        print(f"    k={k}: {mean_sim:.4f} (avg similarity for {len(similarities)} pairs, {attempts} attempts)")
    print(f"  Full matching took {(time.time() - start_time) / 60:.2f} minutes")

    # Mix: hierarchical + random mask matching (subspace-aware)
    print("  Mix (hierarchical + random mask) matching:")
    time_start = time.time()
    if C is None or L is None:
        print("    Skipping mix matching (C or L not provided)")
        for k in range(1, num_tokens + 1):
            mix_similarities[k] = 0.0
    else:
        for k in range(1, num_tokens + 1):
            similarities = []
            attempts = 0

            while len(similarities) < nb_pairs and attempts < max_attempts:
                attempts += 1
                
                # Sample ki values: number of masked tokens per subspace such that sum(ki) = k
                ki_values = [0] * C
                for _ in range(k):
                    # Randomly pick a subspace that hasn't reached L tokens yet
                    valid_subspaces = [c for c in range(C) if ki_values[c] < L]
                    if not valid_subspaces:
                        break
                    c = np.random.choice(valid_subspaces)
                    ki_values[c] += 1
                
                # Build mask positions based on ki_values
                # Tokens are ordered: k(1,1), k(1,2), ..., k(1,C), k(2,1), ..., k(L,C)
                mask_positions = []
                for c in range(C):
                    # Positions for subspace c: c, c+C, c+2C, ...
                    positions_c = [c + l*C for l in range(L)][:ki_values[c]]
                    mask_positions.extend(positions_c)
                mask_positions = sorted(mask_positions)
                
                if not mask_positions:
                    continue
                
                # Pick a random document
                anchor_idx = np.random.randint(len(docs))
                anchor_tokens = docs[anchor_idx][1]
                
                # Create query: None everywhere except at mask positions
                query = [None] * num_tokens
                for pos in mask_positions:
                    if pos < len(anchor_tokens):
                        query[pos] = anchor_tokens[pos]
                
                # Find all documents matching at these positions
                matching_indices = search(index, query)
                
                # Need at least 2 documents
                if len(matching_indices) < 2:
                    continue
                
                # Sample multiple pairs from this anchor to reduce number of searches
                num_pairs_to_sample = min(pairs_per_anchor, nb_pairs - len(similarities))
                for _ in range(num_pairs_to_sample):
                    # Pick two different documents from the matches
                    if len(matching_indices) < 2:
                        break
                    idx1, idx2 = np.random.choice(matching_indices, size=2, replace=False)
                    docid1, docid2 = docid_list[idx1], docid_list[idx2]
                    
                    # Compute embedding similarity
                    emb1 = embeddings[docid_2_idx[docid1]]
                    emb2 = embeddings[docid_2_idx[docid2]]
                    sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                    similarities.append(sim)
            
            mean_sim = np.mean(similarities) if similarities else 0.0
            mix_similarities[k] = mean_sim
            print(f"    k={k}: {mean_sim:.4f} (avg similarity for {len(similarities)} pairs, {attempts} attempts)")
    print(f"  Mix matching took {(time.time() - time_start) / 60:.2f} minutes")

    return hierarchical_similarities, full_similarities, mix_similarities

def compute_embedding_similarity_by_shared_tokens_mean(embeddings, docid_2_idx, encoded_docid, 
                                                   nb_pairs=1000, seed=42, C=None, L=None, V=None):
    """
    Compute mean embedding similarity for document pairs sharing k tokens in their docids.
    Returns the mean similarity over nb_pairs of (k, doc) sampled randomly.
    Uses inverted index for efficient document lookup.
    
    For hierarchical matching: pairs share the first k tokens (prefix matching)
    For full matching: pairs share k tokens at random positions (random mask matching)
    For mix matching: pairs share k tokens sampled according to the mix strategy (subspace-aware random mask)
    
    Args:
        embeddings: numpy array of document embeddings
        docid_2_idx: dict mapping docid to index in embeddings
        encoded_docid: dict mapping docid to encoded docid string
        nb_pairs: number of pairs to sample for each k
        seed: random seed for reproducibility
        C: number of subspaces (for mix matching)
        L: number of codebooks (for mix matching)
        V: number of unique tokens (for building index)
        
    Returns:
        hierarchical_similarities: dict {k: mean_similarity} for prefix matching
        full_similarities: dict {k: mean_similarity} for random mask matching
        mix_similarities: dict {k: mean_similarity} for mix matching
    """
    np.random.seed(seed)
    
    # Parse all docids into token lists and build index
    docs = []
    docid_list = []  # Keep track of docid order
    all_token_values = set()
    
    for docid, encoded in encoded_docid.items():
        tokens = [int(t) for t in encoded.split(',')]
        docs.append((len(docid_list), tokens))  # Use index as doc_id for the index
        docid_list.append(docid)
        all_token_values.update(tokens)
    
    # Build inverted index with appropriate V (max token value) and M (num positions)
    num_tokens = L * C
    max_token = max(all_token_values)
    min_token = min(all_token_values)
    
    print(f"Token value range: [{min_token}, {max_token}], num_tokens: {num_tokens}")
    
    # Build index using actual token range (not the codebook_size parameter!)
    index = build_index(docs, V=max_token+1, M=num_tokens)
    print(f"Built inverted index with V={max_token+1}, M={num_tokens} for {len(docs)} documents")
        
    hierarchical_similarities = []
    full_similarities = []
    mix_similarities = []
    
    print("\nComputing embedding similarity by shared tokens...")
    
    time_start = time.time()
    pairs_per_anchor = 1  # Sample multiple pairs per anchor to reduce searches
    attempts = 0

    # HIERARCHICAL MATCHING
    while len(hierarchical_similarities) < nb_pairs:
        #pick a random k and document
        k = np.random.randint(1, num_tokens + 1)
        anchor_idx = np.random.randint(len(docs))
        anchor_tokens = docs[anchor_idx][1]

        # Pick k random positions
        mask_positions = sorted(np.random.choice(num_tokens, size=k, replace=False))
        
        
        query_hierarchical = anchor_tokens[:k] + [None] * (num_tokens - k)
        matching_indices = search(index, query_hierarchical)
        
        # Need at least 2 documents (including anchor)
        if len(matching_indices) < 2:
            continue
        
        # Sample multiple pairs from this anchor to reduce number of searches
        num_pairs_to_sample = min(pairs_per_anchor, nb_pairs - len(hierarchical_similarities))
        for _ in range(num_pairs_to_sample):
            # Pick two different documents from the matches
            if len(matching_indices) < 2:
                break
            idx1, idx2 = np.random.choice(matching_indices, size=2, replace=False)
            docid1, docid2 = docid_list[idx1], docid_list[idx2]
            
            # Compute embedding similarity
            emb1 = embeddings[docid_2_idx[docid1]]
            emb2 = embeddings[docid_2_idx[docid2]]
            sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            hierarchical_similarities.append(sim)
        
    # FULL MATCHING
    while len(full_similarities) < nb_pairs:
        attempts += 1
        #pick a random k and document
        k = np.random.randint(1, num_tokens + 1)
        anchor_idx = np.random.randint(len(docs))
        anchor_tokens = docs[anchor_idx][1]
        query_full = [None] * num_tokens
        for pos in mask_positions:
            query_full[pos] = anchor_tokens[pos]
        
        # Find all documents matching at these positions
        matching_indices = search(index, query_full)
        
        # Need at least 2 documents
        if len(matching_indices) < 2:
            continue
        
        # Sample multiple pairs from this anchor to reduce number of searches
        num_pairs_to_sample = min(pairs_per_anchor, nb_pairs - len(full_similarities))
        for _ in range(num_pairs_to_sample):
            # Pick two different documents from the matches
            if len(matching_indices) < 2:
                break
            idx1, idx2 = np.random.choice(matching_indices, size=2, replace=False)
            docid1, docid2 = docid_list[idx1], docid_list[idx2]
            
            # Compute embedding similarity
            emb1 = embeddings[docid_2_idx[docid1]]
            emb2 = embeddings[docid_2_idx[docid2]]
            sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            full_similarities.append(sim)
        
    # MIX MATCHING
    while len(mix_similarities) < nb_pairs:
        attempts += 1
        #pick a random k and document
        k = np.random.randint(1, num_tokens + 1)
        anchor_idx = np.random.randint(len(docs))
        anchor_tokens = docs[anchor_idx][1]
        ki_values = [0] * C
        for _ in range(k):
            # Randomly pick a subspace that hasn't reached L tokens yet
            valid_subspaces = [c for c in range(C) if ki_values[c] < L]
            if not valid_subspaces:
                break
            c = np.random.choice(valid_subspaces)
            ki_values[c] += 1
        
        # Build mask positions based on ki_values
        # Tokens are ordered: k(1,1), k(1,2), ..., k(1,C), k(2,1), ..., k(L,C)
        mask_positions = []
        for c in range(C):
            # Positions for subspace c: c, c+C, c+2C, ...
            positions_c = [c + l*C for l in range(L)][:ki_values[c]]
            mask_positions.extend(positions_c)
        mask_positions = sorted(mask_positions)
        
        if not mask_positions:
            continue
        
        # Pick a random document
        anchor_idx = np.random.randint(len(docs))
        anchor_tokens = docs[anchor_idx][1]
        
        # Create query: None everywhere except at mask positions
        query = [None] * num_tokens
        for pos in mask_positions:
            if pos < len(anchor_tokens):
                query[pos] = anchor_tokens[pos]
        
        # Find all documents matching at these positions
        matching_indices = search(index, query)
        
        # Need at least 2 documents
        if len(matching_indices) < 2:
            continue
        
        # Sample multiple pairs from this anchor to reduce number of searches
        num_pairs_to_sample = min(pairs_per_anchor, nb_pairs - len(mix_similarities))
        for _ in range(num_pairs_to_sample):
            # Pick two different documents from the matches
            if len(matching_indices) < 2:
                break
            idx1, idx2 = np.random.choice(matching_indices, size=2, replace=False)
            docid1, docid2 = docid_list[idx1], docid_list[idx2]
            
            # Compute embedding similarity
            emb1 = embeddings[docid_2_idx[docid1]]
            emb2 = embeddings[docid_2_idx[docid2]]
            sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            mix_similarities.append(sim)

    print(f"  Matching took {(time.time() - time_start) / 60:.2f} minutes")

    return hierarchical_similarities, full_similarities, mix_similarities

def plot_code_level_distributions(encoded_docid, encoding):
    """
    Plot and save the distribution of code tokens for each code level.
    One image is saved per level.
    """
    output_dir = f"src/analysis/distribution_plots/{encoding}"
    os.makedirs(output_dir, exist_ok=True)

    level_counters = []
    for encoded in encoded_docid.values():
        tokens = [tok.strip() for tok in encoded.split(',')]
        for level_idx, token in enumerate(tokens):
            if level_idx >= len(level_counters):
                level_counters.append(collections.Counter())
            level_counters[level_idx][token] += 1

    summary = {
        "num_documents": len(encoded_docid),
        "num_levels": len(level_counters),
        "levels": {}
    }
    level_indices = []
    level_entropies = []

    for level_idx, counter in enumerate(level_counters):
        ordered_tokens = _sort_code_tokens(counter.keys())
        counts = [counter[token] for token in ordered_tokens]
        normalized_entropy = _normalized_entropy_from_counts(counts)

        fig_width = max(10, min(30, len(ordered_tokens) * 0.35))
        plt.figure(figsize=(fig_width, 5))
        plt.bar(ordered_tokens, counts)
        plt.title(f"Distribution at Code Level {level_idx}")
        plt.xlabel("Code token")
        plt.ylabel("Number of documents")
        if len(ordered_tokens) > 20:
            plt.xticks(rotation=90)
        plt.tight_layout()

        plot_path = os.path.join(output_dir, f"code_level_{level_idx}_distribution.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()

        summary["levels"][str(level_idx)] = {
            "num_unique_tokens": len(ordered_tokens),
            "normalized_entropy": normalized_entropy,
            "plot_path": plot_path,
            "token_counts": {token: counter[token] for token in ordered_tokens}
        }
        level_indices.append(level_idx)
        level_entropies.append(normalized_entropy)

    # Plot normalized entropy per code level
    entropy_plot_path = os.path.join(output_dir, "normalized_entropy_per_level.png")
    plt.figure(figsize=(10, 5))
    plt.plot(level_indices, level_entropies, marker="o")
    plt.ylim(0.0, 1.0)
    plt.title("Normalized Entropy by Code Level")
    plt.xlabel("Code level")
    plt.ylabel("Normalized entropy")
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(entropy_plot_path, dpi=150)
    plt.close()

    summary["normalized_entropy_plot_path"] = entropy_plot_path

    summary_path = os.path.join(output_dir, "code_level_distribution_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fw:
        json.dump(summary, fw, indent=2, ensure_ascii=False)

    print(f"Saved code-level distribution plots to {output_dir}")
    print(f"Saved normalized entropy plot to {entropy_plot_path}")
    print(f"Saved distribution summary to {summary_path}")

def plot_entropy_and_gini(entropies, gini_indices, encoding, L, C, V, dataset="msmarco"):
    """
    Plot entropy and gini index per token position in a single figure with two subplots.
    
    Args:
        entropies: List of entropy values per token position
        gini_indices: List of gini index values per token position
        encoding: Encoding method name (e.g., 'pq_nc16_cs512')
        L: Number of codebooks
        C: Number of subspaces
        V: Codebook size
        dataset: Dataset name (msmarco or nq) for organizing output files
    """
    output_dir = f"src/analysis/entropy_plots/{dataset}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Token positions (M)
    token_positions = list(range(len(entropies)))
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot entropy
    ax1.plot(token_positions, entropies, marker='o', linewidth=2, markersize=6, color='#2E86AB')
    ax1.set_ylim(0.0, 1.0)
    ax1.set_xlabel('Token Position (M)', fontsize=12)
    ax1.set_ylabel('Normalized Entropy', fontsize=12)
    ax1.set_title('Normalized Entropy per Token Position', fontsize=13, fontweight='bold')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.4)
    
    # Plot gini
    ax2.plot(token_positions, gini_indices, marker='s', linewidth=2, markersize=6, color='#A23B72')
    ax2.set_ylim(0.0, 1.0)
    ax2.set_xlabel('Token Position (M)', fontsize=12)
    ax2.set_ylabel('Gini Index', fontsize=12)
    ax2.set_title('Gini Index per Token Position', fontsize=13, fontweight='bold')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.4)
    
    # Overall title
    fig.suptitle(f'Token Distribution Metrics [{dataset.upper()}] - {encoding} (L={L}, C={C}, V={V})', fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save with encoding, L, C, V in filename
    plot_filename = f"entropy_gini_{encoding}.png"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved entropy and gini plot to {plot_path}")

def plot_shared_similarities(hierarchical_sims, full_sims, mix_sims, encoding, L, C, V, dataset="msmarco"):
    """
    Plot shared similarities (hierarchical, full, mix) against number of shared tokens.
    
    Args:
        hierarchical_sims: Dict {k: mean_similarity} for hierarchical matching
        full_sims: Dict {k: mean_similarity} for full matching
        mix_sims: Dict {k: mean_similarity} for mix matching
        encoding: Encoding method name (e.g., 'pq_nc16_cs512')
        L: Number of codebooks
        C: Number of subspaces
        V: Codebook size
        dataset: Dataset name (msmarco or nq) for organizing output files
    """
    output_dir = f"src/analysis/shared_sim_plots/{dataset}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract token positions (k values) and similarities
    token_positions = sorted(hierarchical_sims.keys())
    hier_values = [hierarchical_sims[k] for k in token_positions]
    full_values = [full_sims[k] for k in token_positions]
    mix_values = [mix_sims[k] for k in token_positions]
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot all three similarities
    ax.plot(token_positions, hier_values, marker='o', linewidth=2, markersize=5, 
            color='#2E86AB', label='Hierarchical', alpha=0.8)
    ax.plot(token_positions, full_values, marker='s', linewidth=2, markersize=5, 
            color='#A23B72', label='Full', alpha=0.8)
    ax.plot(token_positions, mix_values, marker='^', linewidth=2, markersize=5, 
            color='#F18F01', label='Mix', alpha=0.8)
    
    ax.set_xlabel('Number of Shared Tokens (k)', fontsize=12)
    ax.set_ylabel('Mean Embedding Similarity', fontsize=12)
    ax.set_title(f'Embedding Similarity by Shared Tokens [{dataset.upper()}] - {encoding} (L={L}, C={C}, V={V})', 
                 fontsize=13, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=11, loc='best')
    
    plt.tight_layout()
    
    # Save with encoding, L, C, V in filename
    plot_filename = f"shared_sim_{encoding}.png"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved shared similarities plot to {plot_path}")

def save_per_token_metrics(entropies, gini_indices, hierarchical_sims, full_sims, mix_sims, 
                           encoding, L, C, V, dataset="msmarco"):
    """
    Save entropy, gini, and shared similarity metrics in a single CSV file.
    
    Args:
        entropies: List of entropy values per token position
        gini_indices: List of gini index values per token position
        hierarchical_sims: Dict {k: mean_similarity} for hierarchical matching
        full_sims: Dict {k: mean_similarity} for full matching
        mix_sims: Dict {k: mean_similarity} for mix matching
        encoding: Encoding method name
        L: Number of codebooks
        C: Number of subspaces
        V: Codebook size
        dataset: Dataset name (msmarco or nq) for organizing output files
    """
    output_dir = f"src/analysis/per_token_results/{dataset}"
    os.makedirs(output_dir, exist_ok=True)
    
    csv_filename = f"per_token_metrics_{encoding}_L{L}_C{C}_V{V}.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['token_position', 'normalized_entropy', 'gini_index', 
                        'hierarchical_similarity', 'full_similarity', 'mix_similarity'])
        
        # Entropy and gini are indexed from 0 to L*C-1
        # Shared similarities are indexed from 1 to L*C (number of shared tokens)
        # So we'll write rows for positions 0 to L*C-1
        # For position i, shared similarity corresponds to k=i+1
        for pos in range(len(entropies)):
            k = pos + 1  # Number of shared tokens
            hier_sim = hierarchical_sims.get(k, '')
            full_sim = full_sims.get(k, '')
            mix_sim = mix_sims.get(k, '')
            
            writer.writerow([pos, entropies[pos], gini_indices[pos], 
                           hier_sim, full_sim, mix_sim])
    
    print(f"Saved per-token metrics to {csv_path}")

def load_trained_model(model_path, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Load a trained quantization model from a checkpoint.
    Supports models saved from generate_encoded_docids.py:
    - PyTorch models (rq-module, rq-ir): .pth files
    - Pickle models (pq, rq-kmeans, pq-rq): .pkl files
    
    Returns:
        model: The loaded model (RQVAE, PQ, RQ-kmeans, or PQRQ)
        model_type: String indicating the model type
        config: Model configuration (dict or RQVAEConfig)
    """
    print(f"Loading model from {model_path}")
    
    # Add data_prep path to sys.path for importing residual_quantizer modules
    data_prep_path = str(Path(__file__).parent.parent / 'data' / 'data_prep')
    if data_prep_path not in sys.path:
        sys.path.insert(0, data_prep_path)
    
    # Check file extension to determine model type
    if model_path.endswith('.pth'):
        print("Detected .pth file. Assuming RQVAE model.")
        # PyTorch model (RQVAE)
        # Import here to avoid issues if module is not needed
        from residual_quantizer_module import RQVAE, RQVAEConfig
        
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Reconstruct config from checkpoint
        saved_config_dict = checkpoint["config"]
        config = RQVAEConfig(**saved_config_dict)
        
        # Create and load model
        model = RQVAE(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        
        print(f"Loaded RQVAE model. Config: L={config.codebook_quantization_levels}, C={config.codebook_size}")
        return model, 'rqvae', config
        
    elif model_path.endswith('.pkl'):
        # Pickle model (PQ, RQ-kmeans, or PQRQ)
        import pickle
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        if 'pq' in model_data:
            print("Detected PQ model in pickle file.")
            # Product Quantization model
            print(f"Loaded PQ model. M={model_data['pq'].M}, Ks={model_data['pq'].Ks}")
            return model_data['pq'], 'pq', model_data
        elif 'rq_steps' in model_data and 'nb_subspaces' in model_data:
            print("Detected PQRQ model in pickle file.")
            # PQRQ model - reload using the load method
            from residual_quantizer import PQRQ
            
            pqrq = PQRQ.load(model_path)
            print(f"Loaded PQRQ model. nb_subspaces={pqrq.nb_subspaces}, rq_steps={pqrq.rq_steps}, codebook_size={pqrq.codebook_size}")
            return pqrq, 'pq-rq', model_data
        elif 'codebooks' in model_data:
            print("Detected RQ-kmeans model in pickle file.")
            # RQ-kmeans model - reload using the load method
            from residual_quantizer import ResidualQuantizerKmeans
            
            rq = ResidualQuantizerKmeans.load(model_path)
            print(f"Loaded RQ-kmeans model. L={rq.nb_codebooks}, C={rq.codebook_size}")
            return rq, 'rq-kmeans', model_data
        else:
            raise ValueError(f"Unknown pickle model format in {model_path}")
    else:
        raise ValueError(f"Unsupported model file format: {model_path}. Expected .pth or .pkl")

def encode_to_docid_vectors(model, model_type, embeddings, batch_size=1024, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Encode embeddings to docid code vectors using the trained model.
    
    Args:
        model: Trained quantization model (RQVAE, PQ, RQ-kmeans, or PQRQ)
        model_type: Type of model ('rqvae', 'pq', 'rq-kmeans', or 'pq-rq')
        embeddings: np.array of embeddings to encode
        batch_size: Batch size for encoding (used for RQVAE)
        device: Device to use for encoding (used for RQVAE)
    
    Returns:
        docid_vectors: np.array of shape (n_embeddings, n_codebooks) containing integer codes
    """
    if model_type == 'rqvae':
        # RQVAE model - process in batches with PyTorch
        all_codes = []
        
        model.eval()
        with torch.no_grad():
            for i in tqdm(range(0, len(embeddings), batch_size), desc="Encoding to docid vectors"):
                batch_embeddings = embeddings[i:i + batch_size]
                embedding_tensor = torch.tensor(batch_embeddings, dtype=torch.float32).to(device)
                
                # Get codes from model
                reconstructed, indices, loss_dict = model(embedding_tensor)
                
                # Stack indices to get [batch_size, num_codebooks]
                batch_codes = torch.stack(indices, dim=-1).cpu().numpy()
                all_codes.append(batch_codes)
        
        return np.vstack(all_codes)
    
    elif model_type == 'rq-kmeans':
        # RQ-kmeans model - encode in batches to avoid memory issues
        print(f"Encoding with {model_type.upper()} model...")
        all_codes = []
        
        for i in tqdm(range(0, len(embeddings), batch_size), desc="Encoding to docid vectors"):
            batch_embeddings = embeddings[i:i + batch_size]
            batch_codes = model.encode(batch_embeddings)
            all_codes.append(batch_codes)
        
        return np.vstack(all_codes)
    
    elif model_type == 'pq-rq':
        # PQRQ model - encode in batches to avoid memory issues
        print(f"Encoding with {model_type.upper()} model...")
        all_codes = []
        
        for i in tqdm(range(0, len(embeddings), batch_size), desc="Encoding to docid vectors"):
            batch_embeddings = embeddings[i:i + batch_size]
            batch_codes = model.encode(batch_embeddings)
            all_codes.append(batch_codes)
        
        return np.vstack(all_codes)
    
    elif model_type == 'pq':
        # PQ model - encode everything at once
        print(f"Encoding with {model_type.upper()} model...")
        return model.encode(embeddings)
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def compute_doc_and_query_vectors_and_qrels(model_path, doc_embed_path, query_embed_path, qrel_path, batch_size=1024):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Batch size for encoding: {batch_size}")
    
    # Load model
    model, model_type, config = load_trained_model(model_path, device=device)
    
    # Load embeddings
    docid_2_idx, idx_2_docid, doc_embeddings = load_doc_embeddings(doc_embed_path)
    print(f"Loaded {len(doc_embeddings)} document embeddings.")
    print(f"Loaded {len(docid_2_idx)} docid to index mappings.")
    qid_2_idx, idx_2_qid, query_embeddings = load_query_embeddings(query_embed_path)
    print(f"Loaded {len(query_embeddings)} query embeddings.")
    print(f"Loaded {len(qid_2_idx)} qid to index mappings.")
    # Load qrels
    qrels = load_qrels(qrel_path)
    
    # Encode to docid vectors
    print("Encoding documents to docid vectors...")
    doc_vectors = encode_to_docid_vectors(model, model_type, doc_embeddings, batch_size=batch_size, device=device)
    
    print("Encoding queries to docid vectors...")
    query_vectors = encode_to_docid_vectors(model, model_type, query_embeddings, batch_size=batch_size, device=device)

    return docid_2_idx, idx_2_docid, qid_2_idx, idx_2_qid, qrels, doc_vectors, query_vectors

def get_ranking_preservation_query_pos(docid_2_idx, idx_2_qid, qrels, doc_vectors, query_vectors, num_query_sample=-1, C=None, L=None):
    """
    Like get_ranking_preservation, but for query-doc+, doc_random triplets instead of random docs triplets.
    """
    agreements_full = []
    agreements_hierarchical = []
    agreements_mix = []
    skipped = 0

    if num_query_sample > 0:
        print(f"Sampling {num_query_sample} out of {len(query_vectors)} queries for evaluation...")
        query_indices = random.sample(range(len(query_vectors)), min(num_query_sample, len(query_vectors)))
    else:
        print(f"Evaluating on all {len(query_vectors)} queries...")
        query_indices = range(len(query_vectors))

    for q_idx in tqdm(query_indices, desc="Evaluating ranking preservation for query-doc+ pairs"):
        qid = idx_2_qid[q_idx]
        
        if qid not in qrels or len(qrels[qid]) == 0:
            print(f"Query {qid} has no relevant documents in qrels. Skipping.")
            skipped += 1
            continue
        
        relevant_docids = set(qrels[qid])
        
        for docid_pos in relevant_docids:
            if docid_pos not in docid_2_idx:
                # this is normal, since we kept only documents with a relevant query in train...
                skipped += 1
                continue
                
            d_idx = docid_2_idx[docid_pos]

            # get a random negative document that is not relevant to the query
            while True:
                docid_neg = random.choice(list(docid_2_idx.keys()))
                if docid_neg not in relevant_docids:
                    break
            d_idx_neg = docid_2_idx[docid_neg]

            sim_pos_full = get_docid_similarity(
                docid1=','.join(map(str, query_vectors[q_idx])),
                docid2=','.join(map(str, doc_vectors[d_idx])),
                docid_distance="full"
            )
            sim_pos_hier = get_docid_similarity(
                docid1=','.join(map(str, query_vectors[q_idx])),
                docid2=','.join(map(str, doc_vectors[d_idx])),
                docid_distance="hierarchical"
            )
            sim_pos_mix = get_docid_similarity(
                docid1=','.join(map(str, query_vectors[q_idx])),
                docid2=','.join(map(str, doc_vectors[d_idx])),
                docid_distance="mix",
                C=C, L=L
            )
            sim_neg_full = get_docid_similarity(
                docid1=','.join(map(str, query_vectors[q_idx])),
                docid2=','.join(map(str, doc_vectors[d_idx_neg])),
                docid_distance="full"
            )
            sim_neg_hier = get_docid_similarity(
                docid1=','.join(map(str, query_vectors[q_idx])),
                docid2=','.join(map(str, doc_vectors[d_idx_neg])),
                docid_distance="hierarchical"
            )
            sim_neg_mix = get_docid_similarity(
                docid1=','.join(map(str, query_vectors[q_idx])),
                docid2=','.join(map(str, doc_vectors[d_idx_neg])),
                docid_distance="mix",
                C=C, L=L
            )
            
            if sim_pos_full > sim_neg_full:
                agreements_full.append(1)
            elif sim_pos_full == sim_neg_full:
                # ties are counted as half agreement
                agreements_full.append(0.5)
            else:
                agreements_full.append(0)
            
            if sim_pos_hier > sim_neg_hier:
                agreements_hierarchical.append(1)
            elif sim_pos_hier == sim_neg_hier:
                agreements_hierarchical.append(0.5)
            else:                
                agreements_hierarchical.append(0)

            if sim_pos_mix > sim_neg_mix:
                agreements_mix.append(1)
            elif sim_pos_mix == sim_neg_mix:
                agreements_mix.append(0.5)
            else:
                agreements_mix.append(0)

    avg_agreement_full = np.mean(agreements_full)
    avg_agreement_hierarchical = np.mean(agreements_hierarchical)
    avg_agreement_mix = np.mean(agreements_mix)

    print(f"Ranking preservation for query-doc+ pairs (full distance): {avg_agreement_full:.4f}")
    print(f"Ranking preservation for query-doc+ pairs (hierarchical distance): {avg_agreement_hierarchical:.4f}")
    print(f"Ranking preservation for query-doc+ pairs (mix distance): {avg_agreement_mix:.4f}")

    print(f"Number of ties in full distance: {len([x for x in agreements_full if x == 0.5])} out of {len(agreements_full)}")
    print(f"Number of ties in hierarchical distance: {len([x for x in agreements_hierarchical if x == 0.5])} out of {len(agreements_hierarchical)}")
    print(f"Number of ties in mix distance: {len([x for x in agreements_mix if x == 0.5])} out of {len(agreements_mix)}")
    agreements_full = [x for x in agreements_full if x != 0.5]
    agreements_hierarchical = [x for x in agreements_hierarchical if x != 0.5]
    agreements_mix = [x for x in agreements_mix if x != 0.5]
    avg_agreement_full_no_ties = np.mean(agreements_full)
    avg_agreement_hierarchical_no_ties = np.mean(agreements_hierarchical)
    avg_agreement_mix_no_ties = np.mean(agreements_mix)
    print(f"Ranking preservation for query-doc+ pairs (full distance, no ties): {avg_agreement_full_no_ties:.4f}")
    print(f"Ranking preservation for query-doc+ pairs (hierarchical distance, no ties): {avg_agreement_hierarchical_no_ties:.4f}")
    print(f"Ranking preservation for query-doc+ pairs (mix distance, no ties): {avg_agreement_mix_no_ties:.4f}")

    print(f"Number of skipped query-doc+ pairs: {skipped}")
    return avg_agreement_full, avg_agreement_hierarchical, avg_agreement_mix, avg_agreement_full_no_ties, avg_agreement_hierarchical_no_ties, avg_agreement_mix_no_ties, skipped

def evaluate_closeness_query_pos(docid_2_idx, idx_2_qid, qrels, doc_vectors, query_vectors, V=None, L=None, C=None):
    """
    Evaluate the distance between query docid and the positive doc docid.
    """    
    # compute distance between query docid and positive doc docid
    similarities_full = []
    similarities_hierarchical = []
    similarities_mix = []
    skipped=0
    for q_idx in tqdm(range(len(query_vectors)), desc="Evaluating query-docid closeness"):
        qid = idx_2_qid[q_idx]
        
        if qid not in qrels or len(qrels[qid]) == 0:
            print(f"Query {qid} has no relevant documents in qrels. Skipping.")
            skipped += 1
            continue
        
        relevant_docids = set(qrels[qid])
        
        for docid in relevant_docids:
            if docid not in docid_2_idx:
                # this is normal, since we kept only documents with a relevant query in train...
                skipped += 1
                continue
            d_idx = docid_2_idx[docid]
            sim_full = get_docid_similarity(
                docid1=','.join(map(str, query_vectors[q_idx])),
                docid2=','.join(map(str, doc_vectors[d_idx])),
                docid_distance="full"
            )
            sim_hier = get_docid_similarity(
                docid1=','.join(map(str, query_vectors[q_idx])),
                docid2=','.join(map(str, doc_vectors[d_idx])),
                docid_distance="hierarchical"
            )
            sim_mix = get_docid_similarity(
                docid1=','.join(map(str, query_vectors[q_idx])),
                docid2=','.join(map(str, doc_vectors[d_idx])),
                docid_distance="mix",
                C=C, L=L
            )
            similarities_full.append(sim_full)
            similarities_hierarchical.append(sim_hier)
            similarities_mix.append(sim_mix)
    avg_similarity_full = np.mean(similarities_full)
    avg_similarity_hierarchical = np.mean(similarities_hierarchical)
    avg_similarity_mix = np.mean(similarities_mix)

    print(f"Full sim(q, p+): {avg_similarity_full:.4f}")
    print(f"Hierarchical sim(q, p+): {avg_similarity_hierarchical:.4f}")
    print(f"Mix sim(q, p+): {avg_similarity_mix:.4f}")

    # random_sim_hierarchical = (1 - 1 / V ** L)/(V-1)
    # random_sim_full = 1 / V

    # print(f"Full sim(q, p+): {avg_similarity_full:.4f} vs random baseline: {random_sim_full:.4f}")
    # print(f"Hierarchical sim(q, p+): {avg_similarity_hierarchical:.4f} vs random baseline: {random_sim_hierarchical:.4f}")
    # sim_full_normalized = (avg_similarity_full - random_sim_full) / (1 - random_sim_full)
    # sim_hier_normalized = (avg_similarity_hierarchical - random_sim_hierarchical) / (1-random_sim_hierarchical)
    # print(f"Normalized full sim(q, p+): {sim_full_normalized:.4f}")
    # print(f"Normalized hierarchical sim(q, p+): {sim_hier_normalized:.4f}")
    print(f"Number of skipped queries/documents: {skipped}")
    # return avg_similarity_full, avg_similarity_hierarchical, random_sim_full, random_sim_hierarchical, sim_full_normalized, sim_hier_normalized
    return avg_similarity_full, avg_similarity_hierarchical, avg_similarity_mix, skipped

def reformulate_text_t5(text, model_name="Vamsi/T5_Paraphrase_Paws", max_length=256, 
                        num_return_sequences=1, top_k=120, top_p=0.95, 
                        return_all=False):
    """
    Reformulate text using T5 paraphrasing model from Hugging Face.
    
    Args:
        text: Original text
        model_name: Hugging Face model name for paraphrasing
        max_length: Maximum length of generated text (default: 256)
        num_return_sequences: Number of diverse paraphrases to generate (default: 1)
        top_k: Top-k filtering parameter (default: 120)
        top_p: Nucleus sampling parameter (default: 0.95)
        return_all: If True, return all paraphrases; if False, return the first one
        
    Returns:
        str or list: Reformulated text(s)
    """
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError:
        raise ImportError("Please install transformers: pip install transformers")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model and tokenizer (cached after first call)
    if not hasattr(reformulate_text_t5, 'model'):
        print(f"Loading paraphrasing model {model_name}...")
        reformulate_text_t5.tokenizer = AutoTokenizer.from_pretrained(model_name)
        reformulate_text_t5.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        reformulate_text_t5.model.eval()
    
    tokenizer = reformulate_text_t5.tokenizer
    model = reformulate_text_t5.model
    
    # Prepare input text (some models expect "paraphrase: " prefix)
    input_text = "paraphrase: " + text + " </s>"
    
    # Tokenize
    # encoding = tokenizer.encode_plus(
    encoding = tokenizer(
        input_text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    
    # Generate paraphrases using sampling (as recommended by model authors)
    # with torch.no_grad():
    #     outputs = model.generate(
    #         input_ids=input_ids,
    #         attention_mask=attention_mask,
    #         max_length=max_length,
    #         do_sample=True,           # Enable sampling (creates diversity)
    #         top_k=top_k,              # Top-k filtering
    #         top_p=top_p,              # Nucleus sampling
    #         early_stopping=True,
    #         num_return_sequences=num_return_sequences
    #     )
    
    with torch.no_grad():
        # with beam search (less diverse, more deterministic)
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_return_sequences,
            num_return_sequences=num_return_sequences,
            temperature=1.0,            # No temperature scaling for beam search
            early_stopping=True
        )
    
    # Decode all paraphrases
    paraphrases = []
    for output in outputs:
        line = tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        paraphrases.append(line)
    
    # Return first paraphrase by default (for robustness testing)
    # or all if requested
    if return_all:
        return paraphrases
    else:
        return paraphrases[0] if paraphrases else text

def reformulate_texts(corpus_dict):
    """
    Reformulate all texts in the corpus.
    
    Args:
        corpus_dict: Dict of {docid: {"title": str, "text": str}}
        method: Reformulation method ("t5-paraphrase", "backtranslation")
        **kwargs: Additional arguments for reformulation function
        
    Returns:
        dict: {docid: reformulated_text}
    """
    reformulated = {}
    print(f"Reformulating {len(corpus_dict)} documents using t5 paraphrase...")
    
    for docid, doc_info in tqdm(corpus_dict.items(), desc="Reformulating"):
        original_text = doc_info['text']
        reformulated_text = reformulate_text_t5(original_text)
        reformulated[docid] = reformulated_text
    
    print(f"Reformulation complete. Sample reformulated text:")
    for i in range(5):
        sample_docid = list(reformulated.keys())[i]
        print(f"DocID: {sample_docid}")
        print(f"Original: {corpus_dict[sample_docid]['text'][:200]}...")
        print(f"Reformulated: {reformulated[sample_docid][:200]}...")
        print("-"*40)
    
    return reformulated

def embed_texts(texts, docids, model_name='sentence-transformers/gtr-t5-base', batch_size=128):
    """
    Embed texts using sentence transformers.
    
    Args:
        texts: List of texts to embed
        docids: List of corresponding docids
        model_name: Name of the embedding model
        batch_size: Batch size for encoding
        
    Returns:
        dict: {docid: embedding_array}
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading embedding model {model_name} on {device}...")
    model = SentenceTransformer(model_name).to(device)
    
    print(f"Embedding {len(texts)} texts...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        device=device,
        convert_to_numpy=True
    )
    
    # Convert to dict
    embedding_dict = {docid: emb for docid, emb in zip(docids, embeddings)}

    print(f"Embedding complete. Sample embedding:")
    sample_docid = docids[0]
    print(f"DocID: {sample_docid}")
    print(f"Embedding vector (first 10 dimensions): {embedding_dict[sample_docid][:10]}...")
    
    return embedding_dict

def encode_embeddings_to_docids(embedding_dict, model_path, cluster_num, num_codebooks,
                                 batch_size=1024, pretrain_model_path="t5-base"):
    """
    Encode embeddings to docid strings using a trained model.
    
    Args:
        embedding_dict: Dict of {docid: embedding_array}
        model_path: Path to trained quantization model
        cluster_num: Size of each codebook (V)
        num_codebooks: Number of codebooks (L)
        batch_size: Batch size for encoding
        pretrain_model_path: Path to T5 model for vocab_size
        
    Returns:
        dict: {docid: encoded_docid_string}
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load T5 model to get vocab_size (for shifting)
    from transformers import T5ForConditionalGeneration
    t5_model = T5ForConditionalGeneration.from_pretrained(pretrain_model_path)
    vocab_size = t5_model.config.vocab_size
    print(f"T5 vocab_size: {vocab_size}")
    
    # Load quantization model
    model, model_type, config = load_trained_model(model_path, device=device)
    
    print(f"Using cluster_num={cluster_num}, num_codebooks={num_codebooks}")
    
    # Prepare embeddings in consistent order
    docids = list(embedding_dict.keys())
    embeddings = np.array([embedding_dict[docid] for docid in docids])
    
    # Encode to docid vectors
    docid_vectors = encode_to_docid_vectors(model, model_type, embeddings, batch_size=batch_size, device=device)

    # Convert to strings with proper shifting (same as generate_encoded_docids.py)
    encoded_docids = {}
    for docid, code_vector in zip(docids, docid_vectors):
        # Apply offset per codebook to avoid collisions
        new_doc_code = [int(x) + n_codebook * cluster_num for n_codebook, x in enumerate(code_vector)]
        # Shift by vocab_size to create new tokens
        code = ','.join(str(x + vocab_size) for x in new_doc_code)
        encoded_docids[docid] = code
    
    return encoded_docids

def compute_robustness_metrics(original_docids, reformulated_docids, V, L, C):
    """
    Compute metrics comparing original vs reformulated docids.
    
    Args:
        original_docids: Dict of {docid: original_encoded_docid}
        reformulated_docids: Dict of {docid: reformulated_encoded_docid}
        V: Size of each codebook
        L: Number of codebooks
        C: Number of subspaces
        
    Returns:
        dict: Metrics including mean similarities, exact match rate, etc.
    """
    hierarchical_sims = []
    full_sims = []
    mix_sims = []
    exact_matches = 0
    
    for docid in original_docids.keys():
        if docid not in reformulated_docids:
            print(f"Warning: DocID {docid} missing in reformulated docids. Skipping.")
            continue
        
        orig = original_docids[docid]
        reform = reformulated_docids[docid]
        
        # Compute similarities
        hier_sim = get_docid_similarity(orig, reform, docid_distance="hierarchical")
        full_sim = get_docid_similarity(orig, reform, docid_distance="full")
        mix_sim = get_docid_similarity(orig, reform, docid_distance="mix", C=C, L=L)
        
        hierarchical_sims.append(hier_sim)
        full_sims.append(full_sim)
        mix_sims.append(mix_sim)
        
        # Check exact match
        if orig == reform:
            exact_matches += 1
    
    num_docs = len(hierarchical_sims)

    print(f"Range of hierarchical similarities: [{min(hierarchical_sims):.4f}, {max(hierarchical_sims):.4f}]")
    print(f"Range of full similarities: [{min(full_sims):.4f}, {max(full_sims):.4f}]")
    print(f"Range of mix similarities: [{min(mix_sims):.4f}, {max(mix_sims):.4f}]")

    mean_full_sim = np.mean(full_sims) if full_sims else 0.0
    mean_hier_sim = np.mean(hierarchical_sims) if hierarchical_sims else 0.0
    mean_mix_sim = np.mean(mix_sims) if mix_sims else 0.0
    print(f"Mean hierarchical similarity: {mean_hier_sim:.4f}")
    print(f"Mean full similarity: {mean_full_sim:.4f}")
    print(f"Mean mix similarity: {mean_mix_sim:.4f}")
    
    # Compute random baselines
    # random_full = 1.0 / V
    # random_hierarchical = (1.0 - V**(-L)) / ((V - 1) * L)
    
    # Normalize similarities: (sim - random) / (1 - random)
    # This makes 0 = random performance, 1 = perfect match
    # normalized_hierarchical_sims = []
    # normalized_full_sims = []
    
    # for hier_sim, full_sim in zip(hierarchical_sims, full_sims):
    #     # Normalize hierarchical: convert to fraction first
    #     norm_hier = (hier_sim - random_hierarchical) / (1.0 - random_hierarchical) if random_hierarchical < 1.0 else 0.0
    #     normalized_hierarchical_sims.append(norm_hier)
        
    #     # Normalize full
    #     norm_full = (full_sim - random_full) / (1.0 - random_full) if random_full < 1.0 else 0.0
    #     normalized_full_sims.append(norm_full)
    
    # metrics = {
    #     'num_documents': int(num_docs),
    #     'exact_match_rate': float(exact_matches / num_docs if num_docs > 0 else 0),
        
    #     # Raw similarities
    #     'mean_hierarchical_similarity_raw': mean_hier_sim,
    #     'mean_full_similarity_raw': mean_full_sim,
    #     'mean_mix_similarity_raw': mean_mix_sim,
        
        # Normalized similarities (0 = random, 1 = perfect)
        # 'mean_hierarchical_similarity': float(np.mean(normalized_hierarchical_sims)) if normalized_hierarchical_sims else 0.0,
        # 'std_hierarchical_similarity': float(np.std(normalized_hierarchical_sims)) if normalized_hierarchical_sims else 0.0,
        # 'mean_full_similarity': float(np.mean(normalized_full_sims)) if normalized_full_sims else 0.0,
        # 'std_full_similarity': float(np.std(normalized_full_sims)) if normalized_full_sims else 0.0,
        # 'min_hierarchical_similarity': float(np.min(normalized_hierarchical_sims)) if normalized_hierarchical_sims else 0.0,
        # 'max_hierarchical_similarity': float(np.max(normalized_hierarchical_sims)) if normalized_hierarchical_sims else 0.0,
        # 'min_full_similarity': float(np.min(normalized_full_sims)) if normalized_full_sims else 0.0,
        # 'max_full_similarity': float(np.max(normalized_full_sims)) if normalized_full_sims else 0.0,
        
        # Random baselines for reference
        # 'random_hierarchical_baseline': float(random_hierarchical),
        # 'random_full_baseline': float(random_full),
    # }
    
    return mean_hier_sim, mean_full_sim, mean_mix_sim

def compute_robustness_ranking_preservation(original_docids, reformulated_docids, V, L, C):
    """
    Compute ranking preservation for robustness: probability that reformulated doc is closer
    to original than a random doc is.
    
    For each document:
    - Compute similarity between original and reformulated docid
    - Compute similarity between original and a random other docid
    - Check if reformulated is closer than random
    
    Args:
        original_docids: Dict of {docid: original_encoded_docid}
        reformulated_docids: Dict of {docid: reformulated_encoded_docid}
        V: Size of each codebook
        L: Number of codebooks
        C: Number of subspaces
        
    Returns:
        tuple: (hierarchical_preservation, full_preservation, mix_preservation)
    """
    agreements_hierarchical = []
    agreements_full = []
    agreements_mix = []
    
    docids_list = list(original_docids.keys())
    
    for i, docid in enumerate(docids_list):
        if docid not in reformulated_docids:
            continue
        
        orig = original_docids[docid]
        reform = reformulated_docids[docid]
        
        # Sample a random other document
        random_idx = np.random.randint(0, len(docids_list))
        while random_idx == i:  # Make sure it's different
            random_idx = np.random.randint(0, len(docids_list))
        random_docid = docids_list[random_idx]
        random_encoded = original_docids[random_docid]
        
        # Compute similarities
        sim_orig_reform_hier = get_docid_similarity(orig, reform, docid_distance="hierarchical")
        sim_orig_random_hier = get_docid_similarity(orig, random_encoded, docid_distance="hierarchical")
        
        sim_orig_reform_full = get_docid_similarity(orig, reform, docid_distance="full")
        sim_orig_random_full = get_docid_similarity(orig, random_encoded, docid_distance="full")
        
        sim_orig_reform_mix = get_docid_similarity(orig, reform, docid_distance="mix", C=C, L=L)
        sim_orig_random_mix = get_docid_similarity(orig, random_encoded, docid_distance="mix", C=C, L=L)
        
        # Check if reformulated is closer than random (skip ties)
        if sim_orig_reform_hier != sim_orig_random_hier:
            agreements_hierarchical.append(1 if sim_orig_reform_hier > sim_orig_random_hier else 0)
        
        if sim_orig_reform_full != sim_orig_random_full:
            agreements_full.append(1 if sim_orig_reform_full > sim_orig_random_full else 0)
        
        if sim_orig_reform_mix != sim_orig_random_mix:
            agreements_mix.append(1 if sim_orig_reform_mix > sim_orig_random_mix else 0)
    
    # Compute preservation rates
    preservation_hier = np.mean(agreements_hierarchical) if agreements_hierarchical else 0.0
    preservation_full = np.mean(agreements_full) if agreements_full else 0.0
    preservation_mix = np.mean(agreements_mix) if agreements_mix else 0.0
    
    print(f"\nRobustness Ranking Preservation Results:")
    print(f"  Evaluated {len(agreements_hierarchical)} triplets (hierarchical)")
    print(f"  Evaluated {len(agreements_full)} triplets (full)")
    print(f"  Evaluated {len(agreements_mix)} triplets (mix)")
    print(f"  Hierarchical preservation: {preservation_hier:.4f} ({preservation_hier*100:.2f}%)")
    print(f"  Full preservation: {preservation_full:.4f} ({preservation_full*100:.2f}%)")
    print(f"  Mix preservation: {preservation_mix:.4f} ({preservation_mix*100:.2f}%)")
    
    return preservation_hier, preservation_full, preservation_mix

def prepare_robustness_data(args):
    """
    Prepare data for robustness testing: load documents, reformulate them, and generate docids.
    This function is shared between robustness and robustness_ranking_preservation metrics.
    
    Returns:
        tuple: (original_docids, reformulated_docids) - both are dicts of {docid: encoded_docid}
    """
    print("\nPreparing robustness data (reformulating documents and generating docids)...")
    # Step 1: Load subset of documents
    corpus_dict = load_corpus_subset(args.corpus_path, num_samples=args.num_samples_for_robustness, seed=args.seed)
    target_docids = set(corpus_dict.keys())
    
    # Step 2: Load original docids
    docid_path = args.docid_path
    original_docids = load_original_docids(docid_path, target_docids)
    
    # Filter corpus to only documents with original docids
    corpus_dict = {docid: corpus_dict[docid] for docid in original_docids.keys()}
    print(f"Testing {len(corpus_dict)} documents with both text and original docids")
    
    # Step 3: Reformulate texts (with caching)
    # Create cache filename based on parameters
    cache_dir = "src/analysis/paraphrase_cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_filename = f"paraphrases_n{args.num_samples_for_robustness}_seed{args.seed}.json"
    cache_path = os.path.join(cache_dir, cache_filename)
    
    # Try to load from cache
    if os.path.exists(cache_path):
        print(f"Loading paraphrases from cache: {cache_path}")
        with open(cache_path, 'r', encoding='utf-8') as f:
            cached_data = json.load(f)
        
        # Filter to only the docids we need
        reformulated_texts = {docid: cached_data[docid] for docid in corpus_dict.keys() if docid in cached_data}
        
        # Check if we have all needed docids
        missing_docids = set(corpus_dict.keys()) - set(reformulated_texts.keys())
        if missing_docids:
            print(f"Warning: Cache missing {len(missing_docids)} docids. Regenerating those...")
            missing_corpus = {docid: corpus_dict[docid] for docid in missing_docids}
            new_reformulated = reformulate_texts(missing_corpus)
            reformulated_texts.update(new_reformulated)
            # Update cache
            cached_data.update(new_reformulated)
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cached_data, f, ensure_ascii=False, indent=2)
        else:
            print(f"Successfully loaded {len(reformulated_texts)} paraphrases from cache")
    else:
        print(f"No cache found. Generating paraphrases (this may take a while)...")
        reformulated_texts = reformulate_texts(corpus_dict)
        # Save to cache
        print(f"Saving paraphrases to cache: {cache_path}")
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(reformulated_texts, f, ensure_ascii=False, indent=2)
    
    # Step 4: Embed reformulated texts
    docids_list = list(reformulated_texts.keys())
    texts_list = [reformulated_texts[docid] for docid in docids_list]
    
    reformulated_embeddings = embed_texts(texts_list, docids_list, 
                                         model_name=args.embedding_model, 
                                         batch_size=args.batch_size)
    
    # Step 5: Encode to new docids
    reformulated_docids = encode_embeddings_to_docids(
        reformulated_embeddings, 
        args.model_path,
        cluster_num=args.codebook_size,
        num_codebooks=args.num_codebooks,
        batch_size=args.batch_size,
        pretrain_model_path=args.pretrain_model_path
    )
    
    print(f"Original docids sample:")
    for i, docid in enumerate(original_docids.keys()):
        if i >= 5:
            break
        print(f"DocID: {docid}, Original Encoded: {original_docids[docid]}, Reformulated Encoded: {reformulated_docids.get(docid, 'N/A')}")
    
    return original_docids, reformulated_docids

def robustness_test(args, original_docids=None, reformulated_docids=None):
    """
    Compute robustness similarity metrics.
    If original_docids and reformulated_docids are provided, use them.
    Otherwise, prepare the data first.
    """
    if original_docids is None or reformulated_docids is None:
        original_docids, reformulated_docids = prepare_robustness_data(args)
    
    # Compute metrics
    mean_hier_sim, mean_full_sim, mean_mix_sim = compute_robustness_metrics(
        original_docids, reformulated_docids, V=args.codebook_size, L=args.num_codebooks, C=args.nb_subspaces
    )
    
    return mean_hier_sim, mean_full_sim, mean_mix_sim

def print_robustness_metrics(metrics):
    """Print robustness metrics in a readable format."""
    print("\n" + "="*60)
    print("DOCID ROBUSTNESS TEST RESULTS")
    print("="*60)
    print(f"Configuration: V={metrics['cluster_num']}, L={metrics['num_codebooks']}")
    print(f"Number of documents tested: {metrics['num_documents']}")
    print(f"Exact match rate: {metrics['exact_match_rate']:.4f} ({metrics['exact_match_rate']*100:.2f}%)")
    
    print(f"\nRandom baselines:")
    print(f"  Hierarchical: {metrics['random_hierarchical_baseline']:.4f}")
    print(f"  Full: {metrics['random_full_baseline']:.4f}")
    
    print(f"\nNormalized similarities (0=random, 1=perfect):")
    print(f"  Hierarchical: {metrics['mean_hierarchical_similarity']:.4f} ± {metrics['std_hierarchical_similarity']:.4f}")
    print(f"    Range: [{metrics['min_hierarchical_similarity']:.4f}, {metrics['max_hierarchical_similarity']:.4f}]")
    print(f"  Full: {metrics['mean_full_similarity']:.4f} ± {metrics['std_full_similarity']:.4f}")
    print(f"    Range: [{metrics['min_full_similarity']:.4f}, {metrics['max_full_similarity']:.4f}]")
    
    print(f"\nRaw similarities (for reference):")
    print(f"  Hierarchical: {metrics['mean_hierarchical_similarity_raw']:.4f} ± {metrics['std_hierarchical_similarity_raw']:.4f}")
    print(f"  Full: {metrics['mean_full_similarity_raw']:.4f} ± {metrics['std_full_similarity_raw']:.4f}")
    print("="*60)

def save_analysis_results(results_file, **kwargs):
    """
    Save analysis results to a CSV file for comparison across different encodings.
    If a row with the same encoding configuration exists, update it with new metrics.
    Otherwise, append a new row.
    
    Matching criteria: encoding, nb_subspaces, num_codebooks, codebook_size, dataset
    
    Args:
        results_file: Path to the CSV file
        **kwargs: All metric values to save (e.g., encoding, num_codebooks, uniqueness_ratio, etc.)
    """
    import csv
    from datetime import datetime
    
    # Add timestamp
    result = {'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    # Format all kwargs: floats get .4f formatting, integers and strings stay as-is
    for key, value in kwargs.items():
        if isinstance(value, float):
            result[key] = f"{value:.4f}"
        else:
            result[key] = value
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    # Keys to match for row identification
    match_keys = ['encoding', 'nb_subspaces', 'num_codebooks', 'codebook_size', 'dataset']
    
    # Check if file exists
    file_exists = os.path.isfile(results_file)
    
    existing_rows = []
    existing_columns = []
    row_matched = False
    matched_idx = -1
    
    if file_exists:
        # Read all existing data
        with open(results_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            existing_columns = reader.fieldnames if reader.fieldnames else []
            existing_rows = list(reader)
        
        # Look for matching row
        for idx, row in enumerate(existing_rows):
            # Check if all match keys are equal
            match = all(
                row.get(key, '') == str(result.get(key, ''))
                for key in match_keys
            )
            if match:
                row_matched = True
                matched_idx = idx
                print(f"Found existing row matching {', '.join([f'{k}={result.get(k)}' for k in match_keys])}")
                print("Updating with new metrics...")
                break
    
    # Determine fieldnames
    if file_exists and existing_columns:
        fieldnames = existing_columns.copy()
        # Add any new columns from result
        for key in result.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    else:
        # New file: use all keys from result (timestamp first, then sorted)
        fieldnames = ['timestamp'] + sorted([k for k in result.keys() if k != 'timestamp'])
    
    # Update or append row
    if row_matched:
        # Update existing row with new values (keep old values for unspecified metrics)
        for key, value in result.items():
            existing_rows[matched_idx][key] = value
        # Add empty values for any new columns in old rows
        for row in existing_rows:
            for col in fieldnames:
                if col not in row:
                    row[col] = ''
    else:
        # Append new row
        new_row = {col: result.get(col, '') for col in fieldnames}
        existing_rows.append(new_row)
        print("Appending new row...")
    
    # Write all data back to file
    with open(results_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(existing_rows)
    
    if row_matched:
        print(f"Results updated in {results_file}")
    else:
        print(f"Results appended to {results_file}")


if __name__ == "__main__":
    # Set all random seeds for full reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        # Additional settings for reproducibility on CUDA
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Using random seed: {args.seed} (set for random, numpy, torch)")

    docid_path = args.docid_path
    encoded_docid = get_encoded_docid(docid_path)
    
    # Initialize results dictionary
    results = {
        'encoding': args.encoding,
        'nb_subspaces': args.nb_subspaces,
        'num_codebooks': args.num_codebooks,
        'codebook_size': args.codebook_size,
        'dataset': args.dataset,
        'nb_pairs': args.nb_pairs,
        'nb_triplets': args.nb_triplets,
        'seed': args.seed
    }
    
    # ==================== UNIQUENESS ====================
    if 'uniqueness' in METRICS_TO_COMPUTE:
        print("\n" + "="*60)
        print("COMPUTING UNIQUENESS RATIO")
        print("="*60)
        uniqueness_ratio = get_uniqueness_ratio(encoded_docid)
        print(f"Uniqueness ratio of generated docids: {uniqueness_ratio:.4f} ({len(set(encoded_docid.values()))} unique docids out of {len(encoded_docid)})")
        results['uniqueness_ratio'] = uniqueness_ratio
    
    # ==================== ENTROPY & GINI ====================
    if 'entropy_gini' in METRICS_TO_COMPUTE:
        print("\n" + "="*60)
        print("COMPUTING ENTROPY AND GINI INDEX")
        print("="*60)
        entropies, gini_indices = compute_entropy_and_gini_per_level(encoded_docid)
        mean_entropy = np.mean(entropies) if entropies else 0.0
        mean_gini = np.mean(gini_indices) if gini_indices else 0.0
        print(f"\nPer-level normalized entropy: {[f'{e:.4f}' for e in entropies]}")
        print(f"Mean normalized entropy across levels: {mean_entropy:.4f}")
        print(f"\nPer-level Gini index: {[f'{g:.4f}' for g in gini_indices]}")
        print(f"Mean Gini index across levels: {mean_gini:.4f}")
        plot_entropy_and_gini(entropies, gini_indices, args.encoding, L=args.num_codebooks, C=args.nb_subspaces, V=args.codebook_size, dataset=args.dataset)
        results['mean_entropy'] = mean_entropy
        results['mean_gini'] = mean_gini
    
    # Load embeddings if needed for any metric
    needs_embeddings = {'correlation', 'ranking_preservation', 'shared_tokens', 'shared_tokens_new'}
    if METRICS_TO_COMPUTE & needs_embeddings:
        if not args.doc_embed_path:
            print("\nWarning: doc_embed_path not provided. Skipping metrics that require embeddings.")
            METRICS_TO_COMPUTE -= needs_embeddings
        else:
            docid2idx, idx2docid, doc_embeddings = load_doc_embeddings(args.doc_embed_path)
    
    # ==================== CORRELATION ====================
    if 'correlation' in METRICS_TO_COMPUTE:
        print("\n" + "="*60)
        print("COMPUTING CORRELATION BETWEEN DOCID AND EMBEDDING SIMILARITY")
        print("="*60)
        correlation_hierarchical, correlation_full, correlation_mix = get_correlation(
            doc_embeddings, docid2idx, encoded_docid, nb_pairs=args.nb_pairs, 
            C=args.nb_subspaces, L=args.num_codebooks)
        print(f"Correlation between docid similarity and embedding similarity (hierarchical distance): {correlation_hierarchical:.4f}")
        print(f"Correlation between docid similarity and embedding similarity (full distance): {correlation_full:.4f}")
        print(f"Correlation between docid similarity and embedding similarity (mix distance): {correlation_mix:.4f}")
        results['correlation_hierarchical'] = correlation_hierarchical
        results['correlation_full'] = correlation_full
        results['correlation_mix'] = correlation_mix
    
    # ==================== SHARED TOKENS ====================
    if 'shared_tokens' in METRICS_TO_COMPUTE:
        print("\n" + "="*60)
        print("COMPUTING EMBEDDING SIMILARITY FOR PAIRS WITH SHARED TOKENS")
        print("="*60)
        hierarchical_shared_sims, full_shared_sims, mix_shared_sims = compute_embedding_similarity_by_shared_tokens(
            doc_embeddings, docid2idx, encoded_docid, nb_pairs=args.nb_triplets, seed=args.seed, 
            C=args.nb_subspaces, L=args.num_codebooks, V=args.codebook_size
        )
        mean_hierarchical_shared_sim = np.mean(list(hierarchical_shared_sims.values()))
        mean_full_shared_sim = np.mean(list(full_shared_sims.values()))
        mean_mix_shared_sim = np.mean(list(mix_shared_sims.values()))
        print(f"\nMean embedding similarity for pairs with shared tokens:")
        print(f"  Hierarchical (prefix): {mean_hierarchical_shared_sim:.4f}")
        print(f"  Full (random mask): {mean_full_shared_sim:.4f}")
        print(f"  Mix: {mean_mix_shared_sim:.4f}")
        results['mean_hierarchical_shared_sim'] = mean_hierarchical_shared_sim
        results['mean_full_shared_sim'] = mean_full_shared_sim
        results['mean_mix_shared_sim'] = mean_mix_shared_sim
    

    # ==================== SHARED TOKENS NEW ====================
    if 'shared_tokens_new' in METRICS_TO_COMPUTE:
        print("\n" + "="*60)
        print("COMPUTING EMBEDDING SIMILARITY FOR PAIRS WITH SHARED TOKENS (NEW : Mean over random (k, d) pairs)")
        print("="*60)
        hierarchical_shared_sims, full_shared_sims, mix_shared_sims = compute_embedding_similarity_by_shared_tokens_mean(
            doc_embeddings, docid2idx, encoded_docid, nb_pairs=args.nb_triplets, seed=args.seed, 
            C=args.nb_subspaces, L=args.num_codebooks, V=args.codebook_size
        )
        mean_hierarchical_shared_sim = np.mean(hierarchical_shared_sims)
        mean_full_shared_sim = np.mean(full_shared_sims)
        mean_mix_shared_sim = np.mean(mix_shared_sims)
        print(f"\nMean embedding similarity for pairs with shared tokens, mean over (k, doc) pairs:")
        print(f"  Hierarchical (prefix): {mean_hierarchical_shared_sim:.4f}")
        print(f"  Full (random mask): {mean_full_shared_sim:.4f}")
        print(f"  Mix: {mean_mix_shared_sim:.4f}")
        results['mean_hierarchical_shared_sim_new'] = mean_hierarchical_shared_sim
        results['mean_full_shared_sim_new'] = mean_full_shared_sim
        results['mean_mix_shared_sim_new'] = mean_mix_shared_sim
    
    # ==================== RANKING PRESERVATION ====================
    if 'ranking_preservation' in METRICS_TO_COMPUTE:
        print("\n" + "="*60)
        print("COMPUTING RANKING PRESERVATION")
        print("="*60)
        ranking_preservation_full, ranking_preservation_hierarchical, ranking_preservation_mix = get_ranking_preservation(
            doc_embeddings, docid2idx, encoded_docid, nb_triplets=args.nb_triplets, C=args.nb_subspaces, L=args.num_codebooks)
        print(f"Ranking preservation (hierarchical distance): {ranking_preservation_hierarchical:.4f}")
        print(f"Ranking preservation (full distance): {ranking_preservation_full:.4f}")
        print(f"Ranking preservation (mix distance): {ranking_preservation_mix:.4f}")
        results['ranking_preservation_hierarchical'] = ranking_preservation_hierarchical
        results['ranking_preservation_full'] = ranking_preservation_full
        results['ranking_preservation_mix'] = ranking_preservation_mix
    
    # Load model-based resources if needed
    needs_model = {'query_ranking', 'query_closeness', 'robustness'}
    if METRICS_TO_COMPUTE & needs_model:
        if not all([args.model_path, args.doc_embed_path, args.query_embed_path, args.qrel_path]):
            print("\nWarning: model_path, doc_embed_path, query_embed_path, or qrel_path not provided.")
            print("Skipping query-based metrics.")
            METRICS_TO_COMPUTE -= needs_model
        else:
            print("\n" + "="*60)
            print("LOADING MODEL FOR QUERY-BASED EVALUATION")
            print("="*60)
            docid_2_idx, idx_2_docid, qid_2_idx, idx_2_qid, qrels, doc_vectors, query_vectors = compute_doc_and_query_vectors_and_qrels(
                args.model_path, args.doc_embed_path, args.query_embed_path, args.qrel_path, batch_size=args.batch_size)
    
    # ==================== QUERY RANKING ====================
    if 'query_ranking' in METRICS_TO_COMPUTE:
        print("\n" + "="*60)
        print("COMPUTING QUERY-DOC RANKING PRESERVATION")
        print("="*60)
        (agreement_query_pos_full, agreement_query_pos_hierarchical, agreement_query_pos_mix, 
         agreement_query_pos_full_no_ties, agreement_query_pos_hierarchical_no_ties, agreement_query_pos_mix_no_ties, 
         skipped_query_doc_pairs) = get_ranking_preservation_query_pos(
            docid_2_idx, idx_2_qid, qrels, doc_vectors, query_vectors, 
            num_query_sample=args.nb_queries_ranking_preservation, C=args.nb_subspaces, L=args.num_codebooks
        )
        results['agreement_query_pos_hierarchical'] = agreement_query_pos_hierarchical
        results['agreement_query_pos_full'] = agreement_query_pos_full
        results['agreement_query_pos_mix'] = agreement_query_pos_mix
        results['agreement_query_pos_full_no_ties'] = agreement_query_pos_full_no_ties
        results['agreement_query_pos_hierarchical_no_ties'] = agreement_query_pos_hierarchical_no_ties
        results['agreement_query_pos_mix_no_ties'] = agreement_query_pos_mix_no_ties
        results['skipped_query_doc_pairs'] = skipped_query_doc_pairs
    
    # ==================== QUERY CLOSENESS ====================
    if 'query_closeness' in METRICS_TO_COMPUTE:
        print("\n" + "="*60)
        print("COMPUTING QUERY-DOC CLOSENESS")
        print("="*60)
        r_full, r_hier, r_mix, skipped = evaluate_closeness_query_pos(
            docid_2_idx, idx_2_qid, qrels, doc_vectors, query_vectors, 
            V=args.codebook_size, C=args.nb_subspaces, L=args.num_codebooks
        )
        results['r_hier'] = r_hier
        results['r_full'] = r_full
        results['r_mix'] = r_mix
    
    # ==================== ROBUSTNESS (SHARED DATA PREPARATION) ====================
    # Prepare robustness data once if either robustness metric is requested
    needs_robustness_data = {'robustness', 'robustness_ranking_preservation'}
    if METRICS_TO_COMPUTE & needs_robustness_data:
        if not args.corpus_path:
            print("\nWarning: corpus_path not provided. Skipping robustness metrics.")
            METRICS_TO_COMPUTE -= needs_robustness_data
        else:
            print("\n" + "="*60)
            print("PREPARING ROBUSTNESS DATA (REFORMULATING DOCUMENTS)")
            print("="*60)
            original_docids_robustness, reformulated_docids_robustness = prepare_robustness_data(args)
    
    # ==================== ROBUSTNESS SIMILARITY ====================
    if 'robustness' in METRICS_TO_COMPUTE:
        print("\n" + "="*60)
        print("COMPUTING ROBUSTNESS TO TEXT REFORMULATION (SIMILARITY)")
        print("="*60)
        mean_hier_sim_robustness, mean_full_sim_robustness, mean_mix_sim_robustness = robustness_test(
            args, original_docids_robustness, reformulated_docids_robustness
        )
        results['robustness_sim_hierarchical'] = mean_hier_sim_robustness
        results['robustness_sim_full'] = mean_full_sim_robustness
        results['robustness_sim_mix'] = mean_mix_sim_robustness
        results['nb_robustness_docs'] = args.num_samples_for_robustness
    
    # ==================== ROBUSTNESS RANKING PRESERVATION ====================
    if 'robustness_ranking_preservation' in METRICS_TO_COMPUTE:
        print("\n" + "="*60)
        print("COMPUTING ROBUSTNESS RANKING PRESERVATION")
        print("="*60)
        preservation_hier, preservation_full, preservation_mix = compute_robustness_ranking_preservation(
            original_docids_robustness, reformulated_docids_robustness, 
            V=args.codebook_size, L=args.num_codebooks, C=args.nb_subspaces
        )
        results['robustness_ranking_preservation_hierarchical'] = preservation_hier
        results['robustness_ranking_preservation_full'] = preservation_full
        results['robustness_ranking_preservation_mix'] = preservation_mix
    
    # ==================== SAMPLE DOCS ====================
    if 'sample_docs' in METRICS_TO_COMPUTE:
        if not args.corpus_path:
            print("\nWarning: corpus_path not provided. Skipping sample docs.")
        else:
            print("\n" + "="*60)
            print("SAVING SAMPLE DOCUMENTS BY FIRST TOKEN")
            print("="*60)
            corpus = load_corpus(args.corpus_path)
            save_sample_docs_by_first_token(encoded_docid, corpus, args.encoding, num_samples=args.num_samples_per_token)
    
    # ==================== PLOT DISTRIBUTIONS ====================
    if 'plot_distributions' in METRICS_TO_COMPUTE:
        print("\n" + "="*60)
        print("PLOTTING CODE-LEVEL DISTRIBUTIONS")
        print("="*60)
        plot_code_level_distributions(encoded_docid, args.encoding)
    
    # ==================== SAVE RESULTS ====================
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    save_analysis_results(results_file=args.results_file, **results)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print(f"Computed metrics: {sorted(METRICS_TO_COMPUTE)}")
    print(f"Results saved to: {args.results_file}")