
import os
import time
import torch
import argparse
import wandb
import json

# Add src to Python path for imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from torch.utils.data import DataLoader
from torch.amp import autocast
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    T5Tokenizer,
    T5ForConditionalGeneration
)
try:
    from utils import set_seed, load_model
    from trie import Trie
    from evaluate import evaluator
    from pretrain.T5ForPretrain import T5ForPretrain
    from pretrain_dataset import PretrainDataForT5
except ImportError:
    from utils.utils import set_seed, load_model
    from utils.trie import Trie
    from utils.evaluate import evaluator
    from pretrain.T5ForPretrain import T5ForPretrain
    from utils.pretrain_dataset import PretrainDataForT5
from tqdm.auto import tqdm
import random

# print pytorch version
print(f"PyTorch version: {torch.__version__}")
torch.set_float32_matmul_precision('high')
# torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
# print(f"BF16 supported: {torch.cuda.is_bf16_supported()}")

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=2, type=int)
    parser.add_argument("--per_gpu_batch_size", default=25, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--warmup_ratio", default=0, type=float)
    parser.add_argument("--output_every_n_step", default=25, type=int)
    parser.add_argument("--save_every_n_epoch", default=1, type=int)
    parser.add_argument("--operation", default="training", type=str)
    parser.add_argument("--use_docid_rank", default="False", type=str)
    parser.add_argument("--load_ckpt", default="False", type=str)
    parser.add_argument("--save_path", default="./model/", type=str)
    parser.add_argument("--log_path", default="./log/", type=str)
    parser.add_argument("--doc_file_path", required=True, type=str)
    parser.add_argument("--docid_path", required=True, type=str)
    parser.add_argument("--train_file_path", type=str)
    parser.add_argument("--test_file_path", type=str)
    parser.add_argument("--eval_file_path", type=str, help="Path to evaluation file (query_dev.{ENCODING}.jsonl)")
    parser.add_argument("--pretrain_model_path", required=True, type=str)
    parser.add_argument("--load_ckpt_path", default="./model/", type=str)
    parser.add_argument("--dataset_script_dir", type=str)
    parser.add_argument("--dataset_cache_dir", type=str)
    parser.add_argument("--add_doc_num", type=int)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--max_docid_length", type=int, default=1)
    parser.add_argument("--use_origin_head", default="False", type=str)
    parser.add_argument("--num_beams", default=10, type=int)
    parser.add_argument("--wandb_project", type=str, default="ddro-training-alexia", help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="WandB run name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--resume_wandb_run_id", type=str, default=None, help="WandB run ID to resume")
    parser.add_argument("--resume_epoch", type=int, default=0, help="Epoch number to resume from")
    parser.add_argument("--train_subset_size", type=int, default=None, help="Number of training samples to use (for faster iteration). If None, use entire dataset.")
    parser.add_argument("--url", action="store_true", help="Whether to keep url encoding in predictions during evaluation")
    return parser.parse_args()


def load_encoded_docid(docid_path):
    encode_2_docid, encoded_docids = {}, []
    # encode_2_docid: {"32204,32412, ...": ["d108472", ...], ...}
    # encoded_docids: [[32204, 32412, ...], ...]
    with open(docid_path, "r") as fr:
        for line in fr:
            # [d108472]	32204,32412,...
            docid, encode = line.strip().split("\t")
            encode_list = [int(x) for x in encode.split(",")]
            encoded_docids.append(encode_list)
            encode_key = ','.join(map(str, encode_list))
            encode_2_docid.setdefault(encode_key, []).append(docid.lower())
    return encoded_docids, encode_2_docid


def prefix_allowed_tokens_fn_builder(trie, tokenizer):
    def fn(batch_id, sent):
        allowed = trie.get(sent.tolist())
        if not allowed:
            return [tokenizer.pad_token_id]
        return allowed
    return fn


def build_model_and_tokenizer(args):
    tokenizer = T5Tokenizer.from_pretrained(args.pretrain_model_path)
    base_model = T5ForConditionalGeneration.from_pretrained(args.pretrain_model_path)
    base_model.resize_token_embeddings(base_model.config.vocab_size + args.add_doc_num)
    model = T5ForPretrain(base_model, args)

    if args.resume_from_checkpoint:
        if not os.path.exists(args.resume_from_checkpoint):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume_from_checkpoint}")

        print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            print("Loaded model weights from checkpoint dict")
        else:
            state_dict = checkpoint
            print("Loaded model weights from raw state_dict")

        state_dict = strip_compile_prefix_from_state_dict(state_dict)
        model.load_state_dict(state_dict, strict=True)

    elif args.load_ckpt == "True":
        print(f"Loading checkpoint from: {args.load_ckpt_path}")
        state_dict = load_model(args.load_ckpt_path)
        state_dict = strip_compile_prefix_from_state_dict(state_dict)
        model.load_state_dict(state_dict, strict=True)
    return model, tokenizer


def unwrap_compiled_model(model):
    """Return the original nn.Module if model was wrapped by torch.compile."""
    return getattr(model, "_orig_mod", model)


def strip_compile_prefix_from_state_dict(state_dict):
    """
    torch.compile may save keys with '_orig_mod.' prefix.
    This lets old compiled checkpoints load into the normal uncompiled model.
    """
    prefix = "_orig_mod."
    if any(k.startswith(prefix) for k in state_dict.keys()):
        return {
            k[len(prefix):] if k.startswith(prefix) else k: v
            for k, v in state_dict.items()
        }
    return state_dict


def load_eval_data(eval_file_path):
    """Load evaluation data from jsonl file."""
    eval_data = []
    with open(eval_file_path, 'r') as f:
        for line in f:
            # {"input_ids": [11, 3822, ...], "query_id": "[d1650436]", "doc_id": "32353,32639,..."}
            data = json.loads(line.strip())
            eval_data.append(data)
    return eval_data


def run_eval_during_training(model, eval_data, tokenizer, encoded_docids, encode_2_docid, args, epoch, step, url=False):
    """Run evaluation during training and return metrics."""
    model.eval()
    
    # Build Trie for constrained generation
    trie = Trie([[0]+docid for docid in encoded_docids])
    prefix_fn = prefix_allowed_tokens_fn_builder(trie, tokenizer)
    
    test_dataset = PretrainDataForT5([args.eval_file_path], args.max_seq_length, args.max_docid_length, tokenizer, args.dataset_script_dir, args.dataset_cache_dir, args)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=8)
        
    truth, prediction = [], []
    eval_loss_sum = 0.0
    eval_loss_batches = 0
    
    with torch.no_grad():
        for testing_data in tqdm(test_dataloader, total=len(test_dataloader), desc="Evaluating"):
            for key in testing_data.keys():
                if key in ["query_id", "doc_id"]:
                    continue
                testing_data[key] = testing_data[key].cuda()

            model_inputs = {k: v for k, v in testing_data.items() if k not in ["query_id", "doc_id"]}
            outputs = model(**model_inputs)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs
            loss = loss.mean() if loss.dim() > 0 else loss
            eval_loss_sum += loss.item()
            eval_loss_batches += 1
            
            input_ids = model_inputs["input_ids"]
            labels = testing_data["query_id"]
            truth.extend([docid] for docid in labels)

            generated = model.generate(
                input_ids, 
                max_length=args.max_docid_length+1, 
                num_beams=args.num_beams, 
                num_return_sequences=args.num_beams, 
                prefix_allowed_tokens_fn=prefix_fn
            )

            for j in range(input_ids.shape[0]):
                doc_rank = []
                batch_output = generated[j*args.num_beams:(j+1)*args.num_beams].cpu().numpy().tolist()

                for docid in batch_output:
                    if url:
                        docid_string = ','.join(str(x) for x in docid if x != 0)
                    else:
                        docid_string = ','.join(str(x) for x in docid if x not in [0, 1])

                    # Check if this docid exists in our dictionary
                    if docid_string in encode_2_docid:
                        docid_list = encode_2_docid[docid_string]
                        if len(docid_list) > 1:
                            random.shuffle(docid_list)
                            doc_rank.extend(docid_list)
                        else:
                            doc_rank.extend(docid_list)
                
                prediction.append(doc_rank)
        
        # truth2, prediction2 = [], []
        # for sample in tqdm(eval_data, desc="Evaluating"):
        #     # Ground truth: the doc_id field contains comma-separated docids
        #     doc_ids_str = sample["doc_id"]
        #     truth2.append([doc_ids_str])  # Wrap in list to match original format
        #     # truth2.append([sample['query_id']])
            
        #     # Generate predictions
        #     input_ids = torch.tensor([sample["input_ids"]]).cuda()
        #     generated2 = model.generate(
        #         input_ids, 
        #         max_length=args.max_docid_length+1, 
        #         num_beams=args.num_beams, 
        #         num_return_sequences=args.num_beams, 
        #         prefix_allowed_tokens_fn=prefix_fn
        #     )
            
            # Convert predictions to docid format (removing special tokens)
            # preds2 = generated2.cpu().tolist()
            # doc_rank=[]
            # for doc_id in preds2:
            #     docid_string = ','.join(str(x) for x in doc_id if x not in [0, 1])
            #     if docid_string in encode_2_docid:
            #             docid_list = encode_2_docid[docid_string]
            #             if len(docid_list) > 1:
            #                 random.shuffle(docid_list)
            #                 doc_rank.extend(docid_list)
            #             else:
            #                 doc_rank.extend(docid_list)
            # prediction2.append(doc_rank)
            
            # prediction2.append([','.join(str(x) for x in pred if x not in [0, 1]) for pred in preds2])
    # print(f"Predictions2: {prediction2[:5]}")
    # print(f"Truth2: {truth2[:5]}")


    # Calculate metrics
    avg_eval_loss = eval_loss_sum / eval_loss_batches if eval_loss_batches > 0 else float("nan")

    eval_df = evaluator().evaluate_ranking(truth, prediction)
    avg_metrics = eval_df.mean()

    # eval_df2 = evaluator().evaluate_ranking(truth2, prediction2)
    # avg_metrics2 = eval_df2.mean()

    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Eval Loss: {avg_eval_loss:.4f}")
    print(f"MRR@10: {avg_metrics['MRR@10']:.4f} | MRR: {avg_metrics['MRR']:.4f}")
    print(f"P@1: {avg_metrics['P@1']:.4f} | P@10: {avg_metrics['P@10']:.4f} | P@20: {avg_metrics['P@20']:.4f}")
    print(f"R@1: {avg_metrics['R@1']:.4f} | R@10: {avg_metrics['R@10']:.4f} | R@100: {avg_metrics['R@100']:.4f} | R@1000: {avg_metrics['R@1000']:.4f}")
    print(f"Hit@1: {avg_metrics['Hit@1']:.4f} | Hit@5: {avg_metrics['Hit@5']:.4f} | Hit@10: {avg_metrics['Hit@10']:.4f} | Hit@100: {avg_metrics['Hit@100']:.4f}")
    print("="*80 + "\n")
    # print("2ND EVALUATOR RESULTS (on eval_data2)")
    # print("="*80)
    # print(f"MRR@10: {avg_metrics2['MRR@10']:.4f} | MRR: {avg_metrics2['MRR']:.4f}")
    # print(f"P@1: {avg_metrics2['P@1']:.4f} | P@10: {avg_metrics2['P@10']:.4f} | P@20: {avg_metrics2['P@20']:.4f}")
    # print(f"R@1: {avg_metrics2['R@1']:.4f} | R@10: {avg_metrics2['R@10']:.4f} | R@100: {avg_metrics2['R@100']:.4f} | R@1000: {avg_metrics2['R@1000']:.4f}")
    # print(f"Hit@1: {avg_metrics2['Hit@1']:.4f} | Hit@5: {avg_metrics2['Hit@5']:.4f} | Hit@10: {avg_metrics2['Hit@10']:.4f} | Hit@100: {avg_metrics2['Hit@100']:.4f}")
    # print("="*80 + "\n")

    
    # Log to wandb
    wandb_log = {
        "eval/epoch": epoch + 1,
        "eval/loss": avg_eval_loss,
    }
    
    # Add all metrics from evaluator
    for metric_name, metric_value in avg_metrics.items():
        wandb_log[f"eval/{metric_name}"] = metric_value
    
    wandb.log(wandb_log, step=step)
    
    model.train()
    return avg_metrics


def run_training(args, model, tokenizer):
    # Load eval dataset for evaluation during training
    if args.eval_file_path and os.path.exists(args.eval_file_path):
        print(f"Loading evaluation data from {args.eval_file_path}...")
        eval_data = load_eval_data(args.eval_file_path) # list of {"input_ids": [11, 3822, ...], "query_id": "[d1650436]", "doc_id": "32353,32639,..."}
        # Load encoded docids for evaluation
        encoded_docids, encode_2_docid = load_encoded_docid(args.docid_path) 
        # encode_2_docid: {"32204,32412, ...": ["d108472", ...], ...}
        # encoded_docids: [[32204, 32412, ...], ...]
        print(f"Loaded {len(eval_data)} evaluation samples")
    else:
        eval_data = None
        if args.eval_file_path:
            print(f"Warning: Evaluation file {args.eval_file_path} not found, skipping evaluation during training.")
        else:
            print("Warning: No eval file path provided, skipping evaluation during training.")
    

    print(f"Loading training data from {args.train_file_path}...")
    train_dataset = PretrainDataForT5([args.train_file_path], args.max_seq_length, args.max_docid_length, tokenizer, args.dataset_script_dir, args.dataset_cache_dir, args)
    # Sample subset if specified
    if args.train_subset_size is not None and args.train_subset_size > 0:
        print(f"Sampling {args.train_subset_size} training samples from the full dataset of size {len(train_dataset)}...")
        total_samples = len(train_dataset)
        if args.train_subset_size < total_samples:
            # if there is an eval dataset, we want to make sure the eval samples are included in the training subset for more meaningful evaluation during training
            if eval_data is not None:
                eval_query_ids = set(sample["query_id"] for sample in eval_data) # [d1650436], ...
                eval_indices = [i for i, sample in enumerate(train_dataset) if sample["query_id"] in eval_query_ids] # lines id to keep
                print(f"Found {len(eval_indices)} training samples that overlap with evaluation queries. Including all of them in the training subset.")
                if len(eval_indices) > args.train_subset_size:
                    print(f"Warning: Number of eval-overlapping samples ({len(eval_indices)}) exceeds the specified train_subset_size ({args.train_subset_size}).")
                    print(f"Randomly sampling {args.train_subset_size} samples from the eval-overlapping samples for training.")
                    random.shuffle(eval_indices)
                    final_indices = eval_indices[:args.train_subset_size]  # In case there are more eval-overlapping samples than train_subset_size, we take a subset of them
                remaining_indices = [i for i in range(total_samples) if i not in eval_indices] # lines id to sample from
                random.shuffle(remaining_indices)
                sampled_remaining_indices = remaining_indices[:max(0, args.train_subset_size - len(eval_indices))]
                final_indices = eval_indices + sampled_remaining_indices
                random.shuffle(final_indices)
            
            else:
                indices = list(range(total_samples))
                random.shuffle(indices)
                final_indices = indices[:args.train_subset_size]

            train_dataset.nlp_dataset = train_dataset.nlp_dataset.select(final_indices)
            train_dataset.total_len = len(train_dataset.nlp_dataset)
        else:
            print(f"Warning: train_subset_size ({args.train_subset_size}) >= total dataset size ({total_samples}). Using entire dataset.")
    
    dataloader = DataLoader(
        train_dataset, 
        batch_size=args.per_gpu_batch_size, 
        shuffle=True, 
        num_workers=8,
        # pin_memory=True,
        # prefetch_factor=4,
        # persistent_workers=True
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training script.")

    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU / CUDA setup does not support bf16 well enough for bf16 AMP.")

    model = model.to("cuda")  # Keep parameters in fp32.

    if hasattr(torch, "compile"):
        # print("skipping compilation on purpose")
        print("Compiling model with torch.compile()...")
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False, dynamic=True)
        print("Model compiled successfully!")


    optimizer=torch.optim.AdamW(model.parameters(), lr=args.learning_rate, fused=True)

    first_param = next(unwrap_compiled_model(model).parameters())
    print(f"First model param dtype: {first_param.dtype}")  # should be torch.float32
    print(f"First optimizer param dtype: {optimizer.param_groups[0]['params'][0].dtype}")  # should be torch.float32
    
    # Auto-extract epoch from checkpoint if resuming
    start_epoch = 0
    if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
        print(f"Loading training state from checkpoint: {args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")
        
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"]
            print(f"Resuming from epoch {start_epoch + 1}")
        else:
            start_epoch = args.resume_epoch
            print(f"No epoch info in checkpoint, using resume_epoch argument: {start_epoch}")
        
        # Load optimizer state
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("Loaded optimizer state")
        else:
            print("Warning: No optimizer state found in checkpoint")
    else:
        start_epoch = args.resume_epoch if args.resume_epoch > 0 else 0
    
    steps_done = start_epoch * len(dataloader)
    if steps_done > 0:
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        int(args.warmup_ratio * len(dataloader) * args.epochs),
        len(dataloader) * args.epochs,
        last_epoch=steps_done - 1 if steps_done > 0 else -1
    )
    
    # Load scheduler state if resuming from checkpoint
    if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
        checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            print("Loaded scheduler state")

    os.makedirs(args.save_path, exist_ok=True)
    logger = open(args.log_path, "a")
    
    # Initialize wandb with resume capability
    # First check if checkpoint has wandb run id
    checkpoint_wandb_id = None
    if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
        checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")
        checkpoint_wandb_id = checkpoint.get("wandb_run_id", None)
    
    # Priority: explicit arg > checkpoint > new run
    wandb_run_id_to_use = args.resume_wandb_run_id or checkpoint_wandb_id
    
    if wandb_run_id_to_use:
        print(f"Resuming wandb run: {wandb_run_id_to_use}")
        wandb.init(
            project=args.wandb_project,
            id=wandb_run_id_to_use,
            resume="must",
            config=vars(args)
        )
    else:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
            reinit=True
        )

    print(f"Starting training from epoch {start_epoch + 1} to {args.epochs}")
    print(f"Training dataset size: {len(train_dataset)}")

    count_nan_loss = 0
    nan_batch_saved = False  # Only save the first problematic batch
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(dataloader)):
            # if step >= 2:
            #     # to test faster, only run 2 steps per epoch for now - remove this later
            #     break
            batch = {
                k: v.to("cuda", non_blocking=True)
                for k, v in batch.items()
                if k not in ["query_id", "doc_id"]
            }
            optimizer.zero_grad(set_to_none=True)

            # bf16 mixed precision: fp32 weights, bf16 eligible ops.
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(**batch)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs
                loss = loss.mean() if loss.dim() > 0 else loss

            # Keep the scalar loss safe for backward/logging.
            loss = loss.float()

            if step < 5:
                first_param = next(unwrap_compiled_model(model).parameters())
                print(f"Loss dtype after autocast region: {loss.dtype}")
                print(f"Model parameter dtype: {first_param.dtype}")  # should be fp32

            if not torch.isfinite(loss):
                count_nan_loss += 1
                print(f"Skipping step due to non-finite loss: {loss.item()}, count_nan_loss: {count_nan_loss}")
                
                # Save only the first problematic batch for analysis
                if not nan_batch_saved:
                    nan_batch_path = os.path.join(args.save_path, f"nan_batch_loss_epoch{epoch+1}_step{step}.pt")
                    torch.save({
                        "batch": batch,
                        "epoch": epoch + 1,
                        "step": step,
                        "loss_value": loss.item(),
                        "reason": "non-finite loss",
                        "outputs": {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in outputs.items()} if isinstance(outputs, dict) else None
                    }, nan_batch_path)
                    nan_batch_saved = True
                    print(f"Saved first problematic batch to {nan_batch_path}")
                
                continue

            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                unwrap_compiled_model(model).parameters(),
                max_norm=1.0
            )

            if not torch.isfinite(grad_norm):
                count_nan_loss += 1
                print(f"Skipping step due to non-finite gradient: {grad_norm}, count_nan_loss: {count_nan_loss}")
                
                # Save only the first problematic batch for analysis
                if not nan_batch_saved:
                    nan_batch_path = os.path.join(args.save_path, f"nan_batch_grad_epoch{epoch+1}_step{step}.pt")
                    # Collect gradient statistics
                    grad_stats = {}
                    for name, param in unwrap_compiled_model(model).named_parameters():
                        if param.grad is not None:
                            grad_stats[name] = {
                                "mean": param.grad.abs().mean().item(),
                                "max": param.grad.abs().max().item(),
                                "min": param.grad.abs().min().item(),
                                "has_nan": torch.isnan(param.grad).any().item(),
                                "has_inf": torch.isinf(param.grad).any().item()
                            }
                    
                    torch.save({
                        "batch": batch,
                        "epoch": epoch + 1,
                        "step": step,
                        "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                        "reason": "non-finite gradient",
                        "grad_stats": grad_stats,
                        "loss_value": loss.item()
                    }, nan_batch_path)
                    nan_batch_saved = True
                    print(f"Saved first problematic batch to {nan_batch_path}")
                
                continue

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            # OLD WAY WITHOUT MIXED PRECISION:
            # batch = {k: v.cuda() for k, v in batch.items() if k not in ["query_id", "doc_id"]}
            # outputs = model(**batch)
            # loss = outputs["loss"] if isinstance(outputs, dict) else outputs
            # loss = loss.mean() if loss.dim() > 0 else loss
            # loss.backward()
            # optimizer.step()
            # scheduler.step()
            # model.zero_grad()
            # total_loss += loss.item()
            
            # Prepare wandb log dict at every steps
            log_dict = {
                "train/loss": loss.item(),
                "train/learning_rate": scheduler.get_last_lr()[0],
                "train/epoch": step / len(dataloader) + epoch,
                "train/step": step + epoch * len(dataloader),
                "train/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            }

            if step % args.output_every_n_step == 0:
                # gradient statistics only logged every output_every_n_step
                print(f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}, Grad Norm: {grad_norm:.4f}")
                
                # Log gradient statistics per layer
                raw_model = unwrap_compiled_model(model)
                
                # Organize gradients by layer
                layer_grads = {}  # e.g., {"encoder.block.0": [], "encoder.block.1": [], ...}
                all_grads = []
                
                for name, param in raw_model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        grad_mean = param.grad.abs().mean().item()
                        all_grads.append(grad_mean)
                        
                        # Extract layer name (e.g., "encoder.block.0" from "encoder.block.0.layer.0.SelfAttention.q.weight")
                        if "encoder.block." in name:
                            layer_key = "encoder.block." + name.split("encoder.block.")[1].split(".")[0]
                            layer_grads.setdefault(layer_key, []).append(grad_mean)
                        elif "decoder.block." in name:
                            layer_key = "decoder.block." + name.split("decoder.block.")[1].split(".")[0]
                            layer_grads.setdefault(layer_key, []).append(grad_mean)
                        elif "lm_head" in name:
                            layer_grads.setdefault("lm_head", []).append(grad_mean)
                        elif "shared" in name:
                            layer_grads.setdefault("shared", []).append(grad_mean)
                
                # Log mean gradient per layer
                for layer_name, grads in sorted(layer_grads.items()):
                    if grads:
                        log_dict[f"grads_per_layer/{layer_name}"] = sum(grads) / len(grads)
                
                # Log overall mean gradient across all layers
                if all_grads:
                    log_dict["grads_per_layer/mean_all_layers"] = sum(all_grads) / len(all_grads)
                
            wandb.log(log_dict, step=step + epoch * len(dataloader))

        print(f"Epoch {epoch+1} completed. Avg Loss: {total_loss / len(dataloader):.4f}")
        
        # Run evaluation after each epoch
        if eval_data is not None:
            print(f"\nRunning evaluation after epoch {epoch+1}...")
            run_eval_during_training(
                model, 
                eval_data, 
                tokenizer, 
                encoded_docids, 
                encode_2_docid, 
                args, 
                epoch, 
                (epoch+1) * len(dataloader),
                url=args.url if hasattr(args, "url") else False
            )
        
        if (epoch + 1) % args.save_every_n_epoch == 0:
            raw_model = unwrap_compiled_model(model)
            torch.save(raw_model.state_dict(), os.path.join(args.save_path, f"model_epoch{epoch+1}.pt"))
            # torch.save(model.state_dict(), os.path.join(args.save_path, f"model_epoch{epoch+1}.pt"))
        
        # add temporary checkpoint saving every epoch for easier resumption during training
        raw_model = unwrap_compiled_model(model)
        checkpoint_path = os.path.join(args.save_path, f"last_checkpoint.pt")
        checkpoint = {
            "epoch": epoch+1,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "wandb_run_id": wandb.run.id,
            "args": vars(args)
        }
        torch.save(checkpoint, checkpoint_path)

    # Save final model
    raw_model = unwrap_compiled_model(model)
    torch.save(raw_model.state_dict(), os.path.join(args.save_path, "model_final.pt"))
    # torch.save(model.state_dict(), os.path.join(args.save_path, "model_final.pt"))
    
    # Report on NaN issues encountered during training
    if count_nan_loss > 0:
        print(f"\n{'='*80}")
        print(f"Training completed with {count_nan_loss} NaN loss/gradient occurrences")
        if nan_batch_saved:
            print(f"First problematic batch saved to {args.save_path} for analysis")
        print(f"{'='*80}\n")
    
    logger.close()


def run_evaluation(args, model, tokenizer):
    model = model.to("cuda")
    model.eval()

    encoded_docids, encode_2_docid = load_encoded_docid(args.docid_path)
    trie = Trie([[0]+docid for docid in encoded_docids])
    prefix_fn = prefix_allowed_tokens_fn_builder(trie, tokenizer)

    test_dataset = PretrainDataForT5([args.test_file_path], args.max_seq_length, args.max_docid_length, tokenizer, args.dataset_script_dir, args.dataset_cache_dir, args)
    dataloader = DataLoader(test_dataset, batch_size=args.per_gpu_batch_size, shuffle=False, num_workers=8)

    truth, prediction = [], []
    for batch in tqdm(dataloader):
        batch = {k: v.cuda() for k, v in batch.items() if k not in ["query_id", "doc_id"]}
        labels = batch["docid_labels"]
        truth.extend([[','.join(str(x) for x in doc if x != 0 and x != 1)] for doc in labels.tolist()])
        input_ids = batch["input_ids"]
        outputs = model.generate(input_ids, max_length=args.max_docid_length+1, num_beams=args.num_beams, num_return_sequences=args.num_beams, prefix_allowed_tokens_fn=prefix_fn)
        for i in range(0, len(outputs), args.num_beams):
            preds = outputs[i:i+args.num_beams].cpu().tolist()
            prediction.append([','.join(str(x) for x in pred if x not in [0, 1]) for pred in preds])

    eval_df = evaluator().evaluate_ranking(truth, prediction)
    print(eval_df.mean())
    eval_df.to_csv(args.log_path.replace(".log", ".csv"), index=False)


if __name__ == '__main__':
    args = parse_arguments()
    print(f"Arguments: {args}")
    set_seed(args.seed)
    model, tokenizer = build_model_and_tokenizer(args)
    if args.operation == "training":
        run_training(args, model, tokenizer)
    elif args.operation == "testing":
        run_evaluation(args, model, tokenizer)