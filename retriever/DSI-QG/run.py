# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
#     "transformers",
#     "wandb",
#     "datasets",
#     "sentencepiece",
#     "accelerate>=0.26.0"
# ]
# ///

import os
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch._inductor')
warnings.filterwarnings('ignore', message='.*find_unused_parameters.*')


from data import IndexingTrainDataset, GenerateDataset, IndexingCollator, QueryEvalCollator
from transformers import (
    AutoTokenizer,
    T5Tokenizer,
    T5TokenizerFast,
    T5ForConditionalGeneration,
    TrainingArguments,
    TrainerCallback,
    MT5Tokenizer,
    MT5TokenizerFast,
    MT5ForConditionalGeneration,
    HfArgumentParser,
    set_seed,
)
from trainer import DSITrainer, DocTqueryTrainer
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from typing import Optional
import json
from tqdm import tqdm
import logging
set_seed(313)


@dataclass
class RunArguments:
    model_name: str = field(default=None)
    model_path: Optional[str] = field(default=None)
    max_length: Optional[int] = field(default=32)
    id_max_length: Optional[int] = field(default=20)
    remove_prompt: Optional[bool] = field(default=False)
    train_file: str = field(default=None)
    valid_file: str = field(default=None)
    task: str = field(default=None,  metadata={"help": "DSI, docTquery, generation"})
    top_k: Optional[int] = field(default=10)
    num_return_sequences: Optional[int] = field(default=10)
    q_max_length: Optional[int] = field(default=32)
    custom_docid_format: Optional[bool] = field(
        default=False, 
        metadata={"help": "If True, allows docids with dashes and spaces (e.g., '3 - 5 - 4'). If False, only integer docids."}
    )


def make_compute_metrics(tokenizer, valid_ids):

    def compute_metrics(eval_preds):
        hit_at_1 = 0
        hit_at_10 = 0
        
        # Log first 5 examples for sanity check
        logger = logging.getLogger(__name__)
        logger.info("\n" + "="*80)
        logger.info("SANITY CHECK: First 5 generated docids")
        logger.info(f"Total valid docids in training set: {len(valid_ids)}")
        logger.info("="*80)
        
        for idx, (beams, label) in enumerate(zip(eval_preds.predictions, eval_preds.label_ids)):
            rank_list = tokenizer.batch_decode(beams,
                                               skip_special_tokens=True)
            label_id = tokenizer.decode(label, skip_special_tokens=True)
            
            # Log first 5 examples
            if idx < 5:
                logger.info(f"\nExample {idx + 1}:")
                logger.info(f"  Target docid: {label_id}")
                logger.info(f"  Is target in valid_ids: {label_id in valid_ids}")
                logger.info(f"  Top 10 predictions: {rank_list[:10]}")
                logger.info(f"  Are predictions in valid_ids: {[p in valid_ids for p in rank_list[:10]]}")
            
            # filter out duplicates and invalid docids
            filtered_rank_list = []
            for docid in rank_list:
                if docid not in filtered_rank_list and docid in valid_ids:
                    filtered_rank_list.append(docid)

            hits = np.where(np.array(filtered_rank_list)[:10] == label_id)[0]
            if len(hits) != 0:
                hit_at_10 += 1
                if hits[0] == 0:
                    hit_at_1 += 1
        
        logger.info("="*80 + "\n")
        return {"Hits@1": hit_at_1 / len(eval_preds.predictions), "Hits@10": hit_at_10 / len(eval_preds.predictions)}
    return compute_metrics


def main():

    parser = HfArgumentParser((TrainingArguments, RunArguments))
    training_args, run_args = parser.parse_args_into_dataclasses()

    # Configure logging to only log from main process
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARNING,
    )
    logger = logging.getLogger(__name__)

    # We use wandb logger: https://wandb.ai/site.
    if training_args.local_rank == 0:  # only on main process
        # Initialize wandb run
        wandb.login()
        wandb.init(project="DSI-train-Alexia", name=training_args.run_name)

    if 'mt5' in run_args.model_name:
        # Use AutoTokenizer to avoid tokenizer class mismatch warnings
        tokenizer = AutoTokenizer.from_pretrained(run_args.model_name, cache_dir='cache', use_fast=False, legacy=True)
        if run_args.model_path:
            model = MT5ForConditionalGeneration.from_pretrained(run_args.model_path, cache_dir='cache')
        else:
            model = MT5ForConditionalGeneration.from_pretrained(run_args.model_name, cache_dir='cache')
    else:
        tokenizer = AutoTokenizer.from_pretrained(run_args.model_name, cache_dir='cache', use_fast=False, legacy=True)
        if run_args.model_path:
            model = T5ForConditionalGeneration.from_pretrained(run_args.model_path, cache_dir='cache')
        else:
            model = T5ForConditionalGeneration.from_pretrained(run_args.model_name, cache_dir='cache')

    if run_args.task == "docTquery":
        train_dataset = IndexingTrainDataset(path_to_data=run_args.train_file,
                                             max_length=run_args.max_length,
                                             cache_dir='cache',
                                             tokenizer=tokenizer)

        valid_dataset = IndexingTrainDataset(path_to_data=run_args.valid_file,
                                             max_length=run_args.max_length,
                                             cache_dir='cache',
                                             remove_prompt=run_args.remove_prompt,
                                             tokenizer=tokenizer)
        trainer = DocTqueryTrainer(
            do_generation=False,
            model=model,
            processing_class=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=IndexingCollator(
                tokenizer,
                padding='longest',
            ),
        )
        trainer.train()

    elif run_args.task == "DSI":
        train_dataset = IndexingTrainDataset(path_to_data=run_args.train_file,
                                             max_length=run_args.max_length,
                                             cache_dir='cache',
                                             tokenizer=tokenizer)

        valid_dataset = IndexingTrainDataset(path_to_data=run_args.valid_file,
                                             max_length=run_args.max_length,
                                             cache_dir='cache',
                                             remove_prompt=run_args.remove_prompt,
                                             tokenizer=tokenizer)
        
        # Debug: Check a few training examples
        logger.info("\n" + "="*80)
        logger.info("DEBUG: Checking training data format")
        logger.info(f"Tokenizer class: {type(tokenizer).__name__}")
        logger.info("="*80)
        collator = IndexingCollator(tokenizer, padding='longest')
        sample_batch = [train_dataset[i] for i in range(3)]
        batch = collator(sample_batch)
        logger.info(f"Input IDs shape: {batch['input_ids'].shape}")
        logger.info(f"Labels shape: {batch['labels'].shape}")
        for i in range(3):
            input_text = tokenizer.decode(batch['input_ids'][i], skip_special_tokens=True)
            label_text = tokenizer.decode(batch['labels'][i][batch['labels'][i] != -100], skip_special_tokens=True)
            logger.info(f"\nExample {i+1}:")
            logger.info(f"  Input (first 100 chars): {input_text[:100]}...")
            logger.info(f"  Label (target docid): {label_text}")
        logger.info("="*80 + "\n")
        logger.info("="*80 + "\n")
        ################################################################
        # docid generation constraint: configurable based on docid format
        SPIECE_UNDERLINE = "▁"
        ALLOWED_TOKEN_IDS = []
        
        if run_args.custom_docid_format:
            # Allow digits, spaces, and dashes for custom docids like "1 - 4 - 3"
            logger.info("Using custom docid format (with dashes and spaces)")
            for token, token_id in tokenizer.get_vocab().items():
                # Allow tokens that are digits
                if token[0] == SPIECE_UNDERLINE:
                    if token[1:].isdigit():
                        ALLOWED_TOKEN_IDS.append(token_id)
                    # Allow dash and space combinations
                    elif token[1:] in ['-', ' ', '- ', ' -', ' - ']:
                        ALLOWED_TOKEN_IDS.append(token_id)
                # Allow standalone tokens
                if token == SPIECE_UNDERLINE:
                    ALLOWED_TOKEN_IDS.append(token_id)
                elif token.isdigit():
                    ALLOWED_TOKEN_IDS.append(token_id)
                elif token in ['-', ' ', '▁-', '▁ ', '-▁', ' ▁', '▁-▁', '▁ ▁']:
                    ALLOWED_TOKEN_IDS.append(token_id)
        else:
            # Only allow integer tokens (backward compatible)
            logger.info("Using integer-only docid format")
            for token, token_id in tokenizer.get_vocab().items():
                if token[0] == SPIECE_UNDERLINE:
                    if token[1:].isdigit():
                        ALLOWED_TOKEN_IDS.append(token_id)
                if token == SPIECE_UNDERLINE:
                    ALLOWED_TOKEN_IDS.append(token_id)
                elif token.isdigit():
                    ALLOWED_TOKEN_IDS.append(token_id)
        
        ALLOWED_TOKEN_IDS.append(tokenizer.eos_token_id)
        
        logger.info(f"Allowed tokens for docid generation: {len(ALLOWED_TOKEN_IDS)} tokens")
        logger.info(f"Sample allowed tokens: {[tokenizer.decode([tid]) for tid in ALLOWED_TOKEN_IDS[:20]]}")

        def restrict_decode_vocab(batch_idx, prefix_beam):
            return ALLOWED_TOKEN_IDS
        ################################################################

        trainer = DSITrainer(
            model=model,
            processing_class=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=IndexingCollator(
                tokenizer,
                padding='longest',
            ),
            compute_metrics=make_compute_metrics(tokenizer, train_dataset.valid_ids),  # Changed from fast_tokenizer to tokenizer
            restrict_decode_vocab=restrict_decode_vocab,
            id_max_length=run_args.id_max_length
        )
        trainer.train()

    elif run_args.task == 'generation':
        generate_dataset = GenerateDataset(path_to_data=run_args.valid_file,
                                           max_length=run_args.max_length,
                                           cache_dir='cache',
                                           tokenizer=tokenizer)

        trainer = DocTqueryTrainer(
            do_generation=True,
            model=model,
            processing_class=tokenizer,
            args=training_args,
            data_collator=QueryEvalCollator(
                tokenizer,
                padding='longest',
            ),
        )
        predict_results = trainer.predict(generate_dataset,
                                          top_k=run_args.top_k,
                                          num_return_sequences=run_args.num_return_sequences,
                                          max_length=run_args.q_max_length)
        
        # Get docids from trainer (stored during prediction)
        docids = trainer.stored_docids
        predictions = predict_results.predictions.reshape(-1, predict_results.predictions.shape[-1])
        
        with open(f"{run_args.valid_file}.q{run_args.num_return_sequences}.docTquery", 'w') as f:
            for tokens, docid in tqdm(zip(predictions, docids), total=len(docids), desc="Writing file"):
                query = tokenizer.decode(tokens, skip_special_tokens=True)
                # Handle both string and tensor docids
                docid_value = docid.item() if hasattr(docid, 'item') else docid
                jitem = json.dumps({'text_id': docid_value, 'text': query})
                f.write(jitem + '\n')

    else:
        raise NotImplementedError("--task should be in 'DSI' or 'docTquery' or 'generation'")


if __name__ == "__main__":
    main()

