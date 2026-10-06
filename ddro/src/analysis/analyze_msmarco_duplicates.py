import os
import json
import gzip
import argparse
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
from semhash import SemHash


parser = argparse.ArgumentParser(description="Analyze the duplicates in MSMARCO docids")
parser.add_argument('--input_path', type=str, required=True, help="Path to msmarco data.")
parser.add_argument("--doc_embed_path", type=str, required=False, help="Path to input doc embedding file")
parser.add_argument("--docid_path", default="output/encoded_docid.txt", type=str, help="Path with generated docids")
parser.add_argument('--dataset', choices=['msmarco', "msmarco-queries"], default='msmarco', required=False)
parser.add_argument('--semhash_threshold', type=float, default=0.9, required=False,
                    help="Semantic similarity threshold for semhash self-deduplication")
args = parser.parse_args()


def smart_open(path, mode='r', encoding=None):
    if path.endswith('.gz'):
        if 'b' not in mode:
            mode = mode.replace('r', 'rt').replace('w', 'wt')
            return gzip.open(path, mode, encoding=encoding or 'utf-8')
        return gzip.open(path, mode)
    return open(path, mode, encoding=encoding or 'utf-8')


def load_documents(input_path: str, is_nq: bool = False, is_query: bool = False):
    if is_nq:
        with smart_open(input_path) as f:
            df = pd.read_csv(f, sep='\t', header=None, names=[
                'query', 'id', 'long_answer', 'short_answer', 'title',
                'abstract', 'content', 'document_url', 'doc_tac', 'language'])
        texts = df['doc_tac'].fillna('').astype(str).tolist()
        ids = df['id'].astype(str).tolist()
        
    elif is_query:
        # input is a tsv files with qid and query_text columns
        texts, ids = [], []
        with smart_open(input_path) as f:
            for line in tqdm(f, desc="Loading queries"):
                parts = line.strip().split('\t')
                if len(parts) != 2:
                    print(f"WARNING: Skipping malformed line: {line.strip()}")
                    continue
                ids.append(parts[0].lower())
                texts.append(parts[1])
            
    else:
        texts, ids = [], []
        with smart_open(input_path) as f:
            for line in tqdm(f, desc="Loading documents"):
                item = json.loads(line.strip())
                ids.append(item['docid'].lower())
                texts.append(item['body'])
    return ids, texts


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



def get_encoded_docid(docid_path):
    encoded_docid = {}
    if docid_path is None:
        raise ValueError("docid_path cannot be None. Please provide a valid path to the encoded docid file.")
    else:
        with smart_open(docid_path) as fr:
            for line in tqdm(fr, desc="Loading encoded docids"):
                docid, encode = line.strip().split("\t")
                docid = "[{}]".format(docid.lower().strip('[').strip(']'))
                encoded_docid[docid] = encode
    return encoded_docid


def get_semhash_unique_texts(texts, threshold=0.9):
    """Return semantic unique text count using semhash; return None if unavailable."""
    semhash = SemHash.from_records(records=texts)
    result = semhash.self_deduplicate(threshold=threshold)

    return len(result.selected)

def analize_duplicates(doc_nums, texts, embeddings, docids, docid_2_idx, idx_2_docid):
    """
    Looks at the documents with same embeddings but different texts.
    Then looks at the documents with same docids but different texts and embeddings.
    Also checks if duplicates are all diferent or if it is one document that is duplicated multiple times.
    """
    # EMBEDDING DUPLICATES
    embedding_to_docs = {}
    embedding_to_docs_approx = {}
    for doc_num, text, embedding in tqdm(zip(doc_nums, texts, embeddings), total=len(doc_nums), desc="Analyzing duplicates"):
        embedding_tuple = tuple(embedding)
        embedding_tuple_approx = tuple(np.round(embedding, decimals=5))
        if embedding_tuple not in embedding_to_docs:
            embedding_to_docs[embedding_tuple] = []
        if embedding_tuple_approx not in embedding_to_docs_approx:
            embedding_to_docs_approx[embedding_tuple_approx] = []
        # check if the text is the same as any of the existing docs with the same embedding
        if all(existing_text != text for _, existing_text in embedding_to_docs[embedding_tuple]):
            embedding_to_docs[embedding_tuple].append((doc_num, text))
        if all(existing_text != text for _, existing_text in embedding_to_docs_approx[embedding_tuple_approx]):
            embedding_to_docs_approx[embedding_tuple_approx].append((doc_num, text))
    print(f"Number of unique embeddings: {len(embedding_to_docs)}")
    print(f"Number of unique embeddings (approx., 5 decimals): {len(embedding_to_docs_approx)}")
    
    
    # see how many docs have 2 duplicates, 3 duplicates, etc.
    duplicate_counts = {}
    for docs in embedding_to_docs.values():
        count = len(docs)
        if count > 1:
            if count not in duplicate_counts:
                duplicate_counts[count] = 0
            duplicate_counts[count] += 1

    duplicate_counts_approx = {}
    for docs in embedding_to_docs_approx.values():
        count = len(docs)
        if count > 1:
            if count not in duplicate_counts_approx:
                duplicate_counts_approx[count] = 0
            duplicate_counts_approx[count] += 1
    for count, num_docs in sorted(duplicate_counts.items()):
        print(f"{num_docs} embeddings have {count} duplicates.")
    for count, num_docs in sorted(duplicate_counts_approx.items()):
        print(f"[approx] {num_docs} embeddings have {count} duplicates.")

    # DOCID DUPLICATES
    docids_to_docs = {}
    for doc_num, docid in docids.items():
        if docid not in docids_to_docs:
            docids_to_docs[docid] = []
        # check if the embedding is the same as any of the existing docs with the same docid
        embedding = embeddings[docid_2_idx[doc_num]]
        text = texts[docid_2_idx[doc_num]]
        if all(not np.array_equal(existing_embedding, embedding) for _, existing_embedding, _ in docids_to_docs[docid]):
            docids_to_docs[docid].append((doc_num, embedding, text))
    print(f"Number of unique docids: {len(docids_to_docs)}")
    
    duplicate_counts = {}
    for docs in docids_to_docs.values():
        count = len(docs)
        if count > 1:
            if count not in duplicate_counts:
                duplicate_counts[count] = 0
            duplicate_counts[count] += 1
    for count, num_docs in sorted(duplicate_counts.items()):
        print(f"{num_docs} docids have {count} duplicates.")


def main():
    is_query = args.dataset == 'msmarco-queries'
    doc_nums, texts = load_documents(args.input_path, is_nq=False, is_query=is_query)
    docid_2_idx, idx_2_docid, embeddings = load_doc_embeddings(args.doc_embed_path)
    docids = get_encoded_docid(args.docid_path)

    total_docs = len(doc_nums)
    unique_docs = len(set(doc_nums))
    unique_texts = len(set(texts))
    unique_texts_semhash = get_semhash_unique_texts(texts, threshold=args.semhash_threshold)
    unique_embeddings = len({tuple(row) for row in embeddings})
    unique_embeddings_approx = len({tuple(np.round(row, decimals=5)) for row in embeddings})
    unique_docids = len(set(docids.values()))
    print(f"Total documents: {total_docs}, ratio: {unique_docs / total_docs:.4f}")
    print(f"Unique texts: {unique_texts}, ratio: {unique_texts / total_docs:.4f}")
    if unique_texts_semhash is not None:
        print(
            f"Unique texts (semhash, threshold={args.semhash_threshold}): "
            f"{unique_texts_semhash}, ratio: {unique_texts_semhash / total_docs:.4f}"
        )
    print(f"Unique embeddings (exact): {unique_embeddings}, ratio: {unique_embeddings / total_docs:.4f}")
    print(f"Unique embeddings (approx., 5 decimals): {unique_embeddings_approx}, ratio: {unique_embeddings_approx / total_docs:.4f}")
    print(f"Unique docids: {unique_docids}, ratio: {unique_docids / total_docs:.4f}")

    analize_duplicates(doc_nums, texts, embeddings, docids, docid_2_idx, idx_2_docid)

if __name__ == "__main__":
    main()
