import os
import json
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--encoding", default="pq", type=str, help="docid method atomic/pq/url, to get the config")
parser.add_argument("--encoding_exact_name", default="pq", type=str, help="encoding method to get the file paths")
parser.add_argument("--scale", default="top_300k", type=str, help="scale: top_300k, rand_100k, etc.")
parser.add_argument("--dataset", default="msmarco", type=str, help="dataset: msmarco/nq")
parser.add_argument("--num_beams", default=10, type=int, help="number of beams for evaluation (default: 10, can be increased for better performance at the cost of speed)")
parser.add_argument("--epoch", default=None, type=int, help="epoch to evaluate (default: None for latest checkpoint)")
parser.add_argument("--deduplicate", action="store_true", help="Whether to remove documents with duplicate titles between train and dev (default: false)")
parser.add_argument("--results_csv_file", default=None, type=str, help="CSV file to append results to (optional)")
parser.add_argument("--encoding_method", default=None, type=str, help="Encoding method for results CSV (pq, rq-kmeans, prq-module, etc.)")
parser.add_argument("--num_codebooks", default=None, type=int, help="Number of codebooks/subspaces for results CSV")
parser.add_argument("--nb_subspaces", default=None, type=int, help="Number of subspaces for results CSV (prq-module)")
parser.add_argument("--codebook_size", default=None, type=int, help="Codebook size for results CSV")

args = parser.parse_args()

# Load correct config file based on dataset
config_file_path = f"src/scripts/configs/config_{args.dataset}.json" if args.dataset == "nq" else "src/scripts/configs/config.json"
config_file = json.load(open(config_file_path, "r"))

# Get config for the specified encoding
config = config_file[args.encoding]
add_doc_num, max_docid_length, use_origin_head = (
    config["add_doc_num"], 
    config["max_docid_length"], 
    config["use_origin_head"]
)
encoding = args.encoding_exact_name

code_dir = os.getcwd()

# Determine num_beams based on dataset and encoding
# num_beams = (
#     80  if (args.dataset == "msmarco" and args.encoding == "pq") else
#     100 if (args.dataset == "nq" and args.encoding == "pq") else
#     50  if (args.dataset == "nq" and args.encoding == "url") else
#     10 # used to be 15 but default value is 10 for for training so keeping the same
#     # might need to optimize this hyperparameter
# )
num_beams = args.num_beams

## Test settings
print(f"=== Evaluation Configuration ===")
print(f"Dataset: {args.dataset}")
print(f"Encoding: {args.encoding}")
print(f"Config encoding: {encoding}")
print(f"Scale: {args.scale}")
print(f"Num beams: {num_beams}")
if args.epoch is not None:
    print(f"Epoch: {args.epoch}")
else:
    print(f"Epoch: Latest Checkpoint")
print(f"================================")

model = "t5_128_1"
cur_data = "query_dev"
use_docid_rank = "True"
operation = "testing"
max_seq_length = 64

model_name = "DDRO"
top_or_rand, scale = args.scale.split("_")

def main():
    # Create log directory if it doesn't exist
    log_dir = f"{code_dir}/logs/{args.dataset}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Dynamically construct file paths based on dataset and encoding
    base_path = f"{code_dir}/resources"
    
    if args.dataset == "nq":
        # NQ test file - uses encoding name from config (e.g., url_title or pq)
        encoding_filename = "url" if encoding == "url_title" else encoding
        if args.deduplicate:
            test_file = f"{base_path}/datasets/processed/nq-data-dedup/eval_data/{cur_data}.{encoding_filename}.jsonl"
        else:
            test_file = f"{base_path}/datasets/processed/nq-data/eval_data/{cur_data}.{encoding_filename}.jsonl"
        
        # NQ docid file - in encoded_docid subdirectory with _docids suffix
        encoding_for_docid = "url" if encoding == "url_title" else encoding
        docid_file = f"{base_path}/datasets/processed/nq-data/encoded_docid/{encoding_for_docid}_docid.txt"
        
    else:  # msmarco
        # MS MARCO test file - url_title -> url for filename
        encoding_filename = "url" if encoding == "url_title" else encoding
        test_file = f"{base_path}/datasets/processed/msmarco-data/eval_data/{cur_data}.{encoding_filename}.jsonl"
        
        # MS MARCO docid file
        encoding_for_docid = "url" if encoding == "url_title" else encoding
        docid_file = f"{base_path}/datasets/processed/msmarco-data/encoded_docid/{encoding_for_docid}_docid.txt"
    
    if args.epoch is not None:
        if args.dataset == "nq" and args.deduplicate:
            model_path = f"outputs/{args.dataset}-dedup/t5_128_1_top_300k_{encoding}_pretrain_search_finetune/model_epoch{args.epoch}.pt"
        else:
            model_path = f"outputs/{args.dataset}/t5_128_1_top_300k_{encoding}_pretrain_search_finetune/model_epoch{args.epoch}.pt"
    else:
        if args.dataset == "nq" and args.deduplicate:
            model_path = f"outputs/{args.dataset}-dedup/t5_128_1_top_300k_{encoding}_pretrain_search_finetune/model_final.pt"
        else:
            model_path = f"outputs/{args.dataset}/t5_128_1_top_300k_{encoding}_pretrain_search_finetune/model_final.pt"
    # model_path = "outputs/nq/t5_128_1_top_300k_pq_pretrain_search_finetune_/model_final.pt"
    # model_path = f"outputs-sft-msmarco/ddro/ddro_ckp_{args.encoding}_5epoch_lr1e-5_BETA_049/dpo_model_final.pkl"

    # Verify all files exist
    files_to_check = {
        "Test file": test_file,
        "Docid file": docid_file,
        "Checkpoint": model_path
    }
    
    print("\nFile validation:")
    all_files_exist = True
    for file_desc, file_path in files_to_check.items():
        if os.path.exists(file_path):
            print(f"[FOUND] {file_desc}: {file_path}")
        else:
            print(f"[NOT FOUND] {file_desc}: {file_path}")
            all_files_exist = False
    
    if not all_files_exist:
        print("\nERROR: Some required files are missing. Please check the paths above.")
        return
    
    print(f"\nStarting evaluation...\n")
    
    cmd = f"""python src/pretrain/eval_ddro_docid_ranking.py \
        --per_gpu_batch_size 4 \
        --save_path {model_path} \
        --log_path logs/{args.dataset}/dpo_{model_name}_{encoding}.log \
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
    
    if args.results_csv_file:
        cmd += f" --results_csv_file {args.results_csv_file}"
    if args.encoding_method:
        cmd += f" --encoding_method {args.encoding_method}"
    if args.num_codebooks is not None:
        cmd += f" --num_codebooks {args.num_codebooks}"
    if args.nb_subspaces is not None:
        cmd += f" --nb_subspaces {args.nb_subspaces}"
    if args.codebook_size is not None:
        cmd += f" --codebook_size {args.codebook_size}"
    
    os.system(cmd)
    print("\nEvaluation completed successfully")
    print("=== Evaluation Completed ===")


if __name__ == '__main__':
    main()