import os
import json
import argparse
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser(description="Merge MS MARCO training instances by stage.")
    parser.add_argument("--encoding", default="url_title", help="Document ID encoding method")
    parser.add_argument("--dataset", default="msmarco", help="Dataset name")
    parser.add_argument("--dedup", type=bool, default=False, help="Whether to use deduplicated data for NQ")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--cur_data", default="general_pretrain", choices=["general_pretrain", "search_pretrain", "finetune"], help="Training stage")
    parser.add_argument("--max_seq_length", type=int, default=128, help="Max sequence length")
    return parser.parse_args()


def main():
    args = parse_arguments()

    model = "t5"
    cur_data = args.cur_data
    encoding = args.encoding
    max_seq_length = args.max_seq_length
    dataset = args.dataset
    scale ="300k"
    cur_data = "query_dev"
    base_path = "resources/datasets/processed"

    if dataset == "nq" and args.dedup:
        output_dir = f"{base_path}/{dataset}-data-dedup/eval_data"
    else:
        output_dir = f"{base_path}/{dataset}-data/eval_data"

    os.makedirs(output_dir, exist_ok=True)

    if dataset == "msmarco":
        os.system(f" python src/data/data_prep/build_t5_data/generate_eval_instances.py \
            --max_seq_length {max_seq_length} \
            --pretrain_model_path {model}-base \
            --data_path resources/datasets/processed/{dataset}-docs-sents.top.300k.json \
            --docid_path resources/datasets/processed/{dataset}-data/encoded_docid/{encoding}_docid.txt \
            --query_path resources/datasets/raw/{dataset}-data/{dataset}-docdev-queries.tsv.gz \
            --qrels_path resources/datasets/raw/{dataset}-data/{dataset}-docdev-qrels.tsv.gz \
            --output_path {output_dir}/{cur_data}.{encoding}.jsonl \
            --current_data {cur_data} \
            --seed {args.seed}"
            )
    elif dataset == "nq":
        if args.dedup:
            os.system(f" python src/data/data_prep/build_t5_data/generate_eval_instances.py \
                --max_seq_length {max_seq_length} \
                --pretrain_model_path {model}-base \
                --data_path resources/datasets/processed/nq-msmarco-dedup/nq-merged-json/nq-docs-sents.json \
                --docid_path resources/datasets/processed/{dataset}-data/encoded_docid/{encoding}_docid.txt \
                --query_path resources/datasets/processed/nq-msmarco-dedup/nq_queries_dev.tsv.gz \
                --qrels_path resources/datasets/processed/nq-msmarco-dedup/nq_qrels_dev.tsv.gz \
                --output_path {output_dir}/{cur_data}.{encoding}.jsonl \
                --current_data {cur_data} \
                --seed {args.seed}"
                )
        else:
            os.system(f" python src/data/data_prep/build_t5_data/generate_eval_instances.py \
                --max_seq_length {max_seq_length} \
                --pretrain_model_path {model}-base \
                --data_path resources/datasets/processed/nq-msmarco/nq-merged-json/nq-docs-sents.json \
                --docid_path resources/datasets/processed/{dataset}-data/encoded_docid/{encoding}_docid.txt \
                --query_path resources/datasets/processed/nq-msmarco/nq_queries_dev.tsv.gz \
                --qrels_path resources/datasets/processed/nq-msmarco/nq_qrels_dev.tsv.gz \
                --output_path {output_dir}/{cur_data}.{encoding}.jsonl \
                --current_data {cur_data} \
                --seed {args.seed}"
                )
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
            

    print("write success")

if __name__ == '__main__':
    main()