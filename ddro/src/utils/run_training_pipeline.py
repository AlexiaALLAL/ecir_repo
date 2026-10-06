import os
import json
import argparse
import subprocess


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoding", default="pq", type=str, help="DocID encoding method: pq or url (used to get config from config.json)")
    parser.add_argument("--encoding_name_exact", default="rq-module_ir_3e-3_infonce_L24_C256_etrue_w0.5_d32", type=str, help="Exact encoding name to use for training (should match the encoding used in the generated training data, and may include additional info like IR weight and loss type)")
    parser.add_argument("--dataset", default="msmarco", type=str, help="Dataset name (e.g., msmarco or nq)")
    parser.add_argument("--dedup", type=bool, default=False, help="Whether to use deduplicated data (only applicable for nq)")
    parser.add_argument("--scale", default="top_300k", type=str, help="Dataset scale: top_300k or rand_300k")
    parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume from (e.g., outputs/model/model_epoch8.pt)")
    parser.add_argument("--resume_wandb_run_id", type=str, default=None, help="WandB run ID to resume")
    parser.add_argument("--resume_epoch", type=int, default=0, help="Epoch number to resume from (training will continue from epoch+1)")
    parser.add_argument("--resume_stage", type=str, default="pretrain", choices=["pretrain", "search_pretrain", "finetune"], help="Which stage to resume (pretrain, search_pretrain, or finetune)")
    parser.add_argument("--only_current_stage", action="store_true", help="Run only the current stage (specified by --resume_stage), don't continue to next stages")
    parser.add_argument("--train_subset_size", type=int, default=None, help="Number of training samples to use (for faster iteration). If None, use entire dataset.")
    return parser.parse_args()


def load_config(encoding: str, scale: str, dataset: str):
    if dataset == "msmarco":
        config_file_path = "src/scripts/configs/config.json"
    elif dataset == "nq":
        config_file_path = "src/scripts/configs/config_nq.json"
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    with open(config_file_path, "r") as file:
        config_data = json.load(file)

    config = config_data[encoding]
    return config, config["add_doc_num"]


def run_stage(stage_name: str, model: str, load_model: str, all_data: str, cur_data: str,
              stage: str, load_ckpt: str, operation: str, max_seq_length: int, epoch: int,
              encoding: str, dataset:str, add_doc_num: int, max_docid_length: int, use_origin_head: str,
              save_every_n_epoch: int, top_or_rand: str, scale: str, resume_checkpoint: str = None, 
              resume_wandb_id: str = None, resume_epoch: int = 0, train_subset_size: int = None,
              seed: int = 42, dedup: bool = False
        ):

    code_dir = "src"
    log_dir = f"logs/{stage}"
    os.makedirs(log_dir, exist_ok=True)
    str_subset = f"_{train_subset_size}" if train_subset_size is not None else ''
    if dataset == "nq" and dedup:
        save_dir = f"outputs/{dataset}-dedup/{model}_{top_or_rand}_{scale}_{encoding}_{all_data}{str_subset}"
        train_path = f"resources/datasets/processed/{dataset}-data-dedup/train_data_{top_or_rand}_{scale}/{cur_data}.{model}.{encoding}.json"
        test_path = f"resources/datasets/processed/{dataset}-data-dedup/test_data_{top_or_rand}_{scale}/"
        eval_path = f"resources/datasets/processed/{dataset}-data-dedup/eval_data/query_dev.{encoding}.jsonl"
    else:
        save_dir = f"outputs/{dataset}/{model}_{top_or_rand}_{scale}_{encoding}_{all_data}{str_subset}"
        train_path = f"resources/datasets/processed/{dataset}-data/train_data_{top_or_rand}_{scale}/{cur_data}.{model}.{encoding}.json"
        test_path = f"resources/datasets/processed/{dataset}-data/test_data_{top_or_rand}_{scale}/"
        eval_path = f"resources/datasets/processed/{dataset}-data/eval_data/query_dev.{encoding}.jsonl"
    
    if dataset == "msmarco":
        doc_file_path = f"resources/datasets/processed/msmarco-data/msmarco-docs-sents.{top_or_rand}.{scale}.json"
    else:
        if dataset == "nq" and dedup:
            doc_file_path = f"resources/datasets/processed/nq-msmarco-dedup/nq-merged-json/nq-docs-sents.json"
        else:
            doc_file_path = f"resources/datasets/processed/nq-data/nq-docs-sents.json"
    docid_path = f"resources/datasets/processed/{dataset}-data/encoded_docid/{encoding}_docid.txt"

    if dataset == "nq" and dedup:
        model_ckpt_path = (
            f"outputs/{dataset}-dedup/{load_model}_{top_or_rand}_{scale}_{encoding}_pretrain{'_search' if stage == 'finetune' else ''}"
            f"{str_subset}/model_final.pt"
        )
    else:
        model_ckpt_path = (
            f"outputs/{dataset}/{load_model}_{top_or_rand}_{scale}_{encoding}_pretrain{'_search' if stage == 'finetune' else ''}"
            f"{str_subset}/model_final.pt"
        )
    print(f"Model checkpoint path: {model_ckpt_path}")


    if stage == "pretrain":
        stage_nb = "1"
    elif stage == "search_pretrain":
        stage_nb = "2"
    elif stage == "finetune":
        stage_nb = "3"
    wandb_run_name = f"sft-stage{stage_nb}_{encoding}_{dataset}"

    command = [
        "python", f"{code_dir}/utils/run_t5_trainer.py",
        "--epoch", str(epoch),
        "--per_gpu_batch_size", "128",
        "--learning_rate", "1e-3",
        "--save_path", save_dir,
        "--log_path", f"{log_dir}/{stage}.{model}.{top_or_rand}.{scale}.{encoding}.{all_data}.log",
        "--doc_file_path", doc_file_path,
        "--pretrain_model_path", "resources/transformer_models/t5-base",
        "--docid_path", docid_path,
        "--train_file_path", train_path,
        "--test_file_path", test_path,
        "--eval_file_path", eval_path,
        "--dataset_script_dir", "data/data_scripts",
        "--dataset_cache_dir", "negs_tutorial_cache",
        "--add_doc_num", str(add_doc_num),
        "--max_seq_length", str(max_seq_length),
        "--max_docid_length", str(max_docid_length),
        "--use_origin_head", use_origin_head,
        "--output_every_n_step", str((train_subset_size // 1000 or 1) if train_subset_size is not None else 500),
        "--save_every_n_epoch", str(save_every_n_epoch),
        "--operation", operation,
        "--wandb_run_name", wandb_run_name,
        "--seed", str(seed),
    ]

    if load_ckpt == "True":
        command.extend(["--load_ckpt", load_ckpt, "--load_ckpt_path", model_ckpt_path])
    if train_subset_size is not None:
        command.extend(["--train_subset_size", str(train_subset_size)])

    if train_subset_size is not None:
        command.extend(["--train_subset_size", str(train_subset_size)])
    # Add resume parameters
    if resume_checkpoint:
        command.extend(["--resume_from_checkpoint", resume_checkpoint])
    if resume_wandb_id:
        command.extend(["--resume_wandb_run_id", resume_wandb_id])
    if resume_epoch > 0:
        command.extend(["--resume_epoch", str(resume_epoch)])
    if "url" in encoding.lower() or "tu" in encoding.lower():
        print("URL encoding detected in encoding name, adding --url flag to command")   
        command.append("--url")

    print(f"Running stage: {stage_name}")
    subprocess.run(command, check=True)


def main():
    args = parse_arguments()
    encoding = args.encoding
    scale = args.scale
    top_or_rand, scale_val = scale.split("_")

    config, add_doc_num = load_config(encoding, scale, dataset=args.dataset)
    max_docid_length = config["max_docid_length"]
    use_origin_head = config["use_origin_head"]
    encoding = args.encoding_name_exact  # Use the exact encoding name for training (which may include additional info like IR weight and loss type)

    # Determine which stages to run based on resume_stage
    stages_to_run = ["pretrain", "search_pretrain", "finetune"]
    if args.resume_stage:
        if args.only_current_stage:
            # Run ONLY the specified stage
            stages_to_run = [args.resume_stage]
        else:
            # If resuming, run from the resume stage onwards
            stage_idx = stages_to_run.index(args.resume_stage)
            stages_to_run = stages_to_run[stage_idx:]

    # Stage 1: Content-to-DocID Pretraining
    if "pretrain" in stages_to_run:
        resume_params = {}
        if args.resume_stage == "pretrain":
            resume_params = {
                "resume_checkpoint": args.resume_from_checkpoint,
                "resume_wandb_id": args.resume_wandb_run_id,
                "resume_epoch": args.resume_epoch
            }
        
        run_stage(
            stage_name="Content-to-DocID Pretraining",
            model="t5_128_10",
            load_model="t5_128_10",
            all_data="pretrain",
            cur_data="pretrain",
            stage="pretrain",
            load_ckpt="False",
            operation="training",
            max_seq_length=128,
            epoch=10,
            encoding=encoding,
            dataset=args.dataset,
            add_doc_num=add_doc_num,
            max_docid_length=max_docid_length,
            use_origin_head=use_origin_head,
            save_every_n_epoch=5,
            top_or_rand=top_or_rand,
            scale=scale_val,
            seed=args.seed,
            dedup=args.dedup,
            train_subset_size=args.train_subset_size,
            **resume_params
        )

    # Stage 2: PseudoQuery-to-DocID Pretraining
    if "search_pretrain" in stages_to_run:
        resume_params = {}
        if args.resume_stage == "search_pretrain":
            resume_params = {
                "resume_checkpoint": args.resume_from_checkpoint,
                "resume_wandb_id": args.resume_wandb_run_id,
                "resume_epoch": args.resume_epoch
            }
        
        run_stage(
            stage_name="PseudoQuery-to-DocID Pretraining",
            model="t5_128_10",
            load_model="t5_128_10",
            all_data="pretrain_search",
            cur_data="search_pretrain",
            stage="search_pretrain",
            load_ckpt="True",
            operation="training",
            max_seq_length=64,
            epoch=20,
            encoding=encoding,
            dataset=args.dataset,
            add_doc_num=add_doc_num,
            max_docid_length=max_docid_length,
            use_origin_head=use_origin_head,
            save_every_n_epoch=10,
            top_or_rand=top_or_rand,
            scale=scale_val,
            seed=args.seed,
            dedup=args.dedup,
            train_subset_size=args.train_subset_size,
            **resume_params
        )

    # Stage 3: Query-to-DocID Finetuning
    if "finetune" in stages_to_run:
        resume_params = {}
        if args.resume_stage == "finetune":
            resume_params = {
                "resume_checkpoint": args.resume_from_checkpoint,
                "resume_wandb_id": args.resume_wandb_run_id,
                "resume_epoch": args.resume_epoch
            }
        
        run_stage(
            stage_name="Query-to-DocID Finetuning",
            model="t5_128_1",
            load_model="t5_128_10",
            all_data="pretrain_search_finetune",
            cur_data="finetune",
            stage="finetune",
            load_ckpt="True",
            operation="training",
            max_seq_length=64,
            epoch=10,
            encoding=encoding,
            dataset=args.dataset,
            add_doc_num=add_doc_num,
            max_docid_length=max_docid_length,
            use_origin_head=use_origin_head,
            save_every_n_epoch=1,
            top_or_rand=top_or_rand,
            scale=scale_val,
            seed=args.seed,
            dedup=args.dedup,
            train_subset_size=args.train_subset_size,
            **resume_params
        )

    print("Training pipeline completed.")


if __name__ == '__main__':
    main()