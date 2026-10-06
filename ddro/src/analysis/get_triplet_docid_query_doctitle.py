"""
Generate a CSV with 3 columns:
  - encoded_docid  : the encoded docid of the positive document (from the docid file)
  - query_text     : the query string
  - doc_title      : the title of the positive document (from the corpus)

Required inputs:
  --docid_path   : .txt file mapping original doc numbers to encoded docids
                   Format per line: [docid]<TAB><encoded_docid_tokens>
  --qrels_path   : qrels file (space- or tab-delimited: qid _ docid rel)
  --queries_path : queries TSV file (qid<TAB>query_text)
  --corpus_path  : JSON Lines corpus file (each line: {"docid": ..., "title": ..., ...})
  --output_path  : output CSV file path
"""

import os
import csv
import gzip
import json
import argparse
from tqdm import tqdm


def smart_open(path, mode="rt", encoding="utf-8"):
    if path.endswith(".gz"):
        return gzip.open(path, mode=mode, encoding=encoding)
    return open(path, mode=mode, encoding=encoding)


def normalize_docid(docid: str) -> str:
    """Normalize a docid to the bracketed lowercase form used in the docid file."""
    return "[{}]".format(docid.strip("[]").lower())


def load_docid_mapping(docid_path: str) -> dict:
    """Load original doc number -> encoded docid mapping.

    File format (one entry per line):
        [docid]<TAB><encoded_docid>
    """
    mapping = {}
    i = 0
    with smart_open(docid_path) as f:
        for line in tqdm(f, desc="Loading docid mapping"):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            raw_docid, encoded = parts
            docid = normalize_docid(raw_docid)
            mapping[docid] = encoded
            if i < 10:
                print(f"Sample docid mapping: raw_docid={raw_docid}, normalized_docid={docid}, encoded={encoded}")
                i += 1
    return mapping


def load_qrels(qrels_path: str) -> list:
    """Load qrels as a list of (qid, docid) pairs for positive judgements.

    Supports both space-delimited (qid 0 docid 1) and tab-delimited formats.
    Only keeps entries with relevance >= 1.
    """
    pairs = []
    i = 0
    with smart_open(qrels_path) as f:
        for line in tqdm(f, desc="Loading qrels"):
            line = line.strip()
            if not line:
                continue
            # Try space split first (standard TREC format), fall back to tab
            parts = line.split()
            if len(parts) < 4:
                parts = line.split("\t")
            if len(parts) < 4:
                continue
            qid, _, docid, rel = parts[0], parts[1], parts[2], parts[3]
            try:
                if int(rel) >= 1:
                    pairs.append((qid, docid))
            except ValueError:
                continue
            if i < 10:
                print(f"Sample qrel: qid={qid}, docid={docid}, rel={rel}")
                i += 1
    return pairs


def load_queries(queries_path: str) -> dict:
    """Load query id -> query text mapping from a TSV file."""
    queries = {}
    i = 0
    with smart_open(queries_path) as f:
        for line in tqdm(f, desc="Loading queries"):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            qid, query_text = parts
            queries[qid] = query_text
            if i < 10:
                print(f"Sample query: qid={qid}, query_text={query_text}")
                i += 1
    return queries


def load_corpus(corpus_path: str) -> dict:
    """Load docid -> title mapping from a JSON Lines corpus file."""
    titles = {}
    texts = {}
    i = 0
    with smart_open(corpus_path) as f:
        for line in tqdm(f, desc="Loading corpus"):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line) # keys: 'docid', 'url', 'title', 'body', 'sents'
            except json.JSONDecodeError:
                continue
            raw_docid = item.get("docid") or item.get("id", "")
            title = item.get("title", "")
            text = item.get("body", "") or item.get("doc_tac", "")
            text = text.replace("\n", " ").replace("\t", " ").strip()
            if i < 10:
                print(f"Sample corpus item: docid={raw_docid}, text={text[:100]}...")
                i += 1
            if raw_docid:
                docid = normalize_docid(str(raw_docid))
                titles[docid] = title
                texts[docid] = text

    return titles, texts


def main():
    parser = argparse.ArgumentParser(
        description="Generate a CSV of (encoded_docid, query_text, doc_title) triplets."
    )
    parser.add_argument("--docid_path", required=True,
                        help="Path to the docid .txt file (doc_number -> encoded_docid).")
    parser.add_argument("--qrels_path", required=True,
                        help="Path to the qrels file.")
    parser.add_argument("--queries_path", required=True,
                        help="Path to the queries TSV file (qid<TAB>query_text).")
    parser.add_argument("--corpus_path", required=True,
                        help="Path to the corpus JSON Lines file.")
    parser.add_argument("--output_path", required=True,
                        help="Path to the output CSV file.")
    args = parser.parse_args()

    docid_mapping = load_docid_mapping(args.docid_path)
    qrels = load_qrels(args.qrels_path)
    queries = load_queries(args.queries_path)
    titles, texts = load_corpus(args.corpus_path)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)

    skipped = 0
    written = 0
    with open(args.output_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        # writer.writerow(["encoded_docid", "query_text", "doc_title"])
        writer.writerow(["id_doc", "doc_title", "doc_text", "id_query", "query_text", "docid"])

        for qid, raw_docid in tqdm(qrels, desc="Building CSV rows"):
            id_doc = normalize_docid(raw_docid)

            encoded_docid = docid_mapping.get(id_doc)
            query_text = queries.get(qid)
            doc_title = titles.get(id_doc)
            doc_text = texts.get(id_doc)

            if encoded_docid is None or query_text is None or doc_text is None:
                skipped += 1
                continue

            writer.writerow([id_doc, doc_title, doc_text, qid, query_text, encoded_docid])
            written += 1

    print(f"Done. Written: {written} rows, Skipped: {skipped} rows (missing data).")
    print(f"Output saved to: {args.output_path}")


if __name__ == "__main__":
    main()
