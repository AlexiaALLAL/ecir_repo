import argparse
import gzip
import os
import random
import pandas as pd
import numpy as np
import torch
import time
from datasets import Dataset
from transformers import (
    T5ForConditionalGeneration,
    T5TokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed as hf_set_seed,
)

def extract_query_doc_pairs(input_path: str, output_path: str):
    print(f"Extracting query-doc pairs from {input_path} and saving to {output_path}...", flush=True)
    with gzip.open(input_path, 'rt') as f:
        df = pd.read_csv(f, sep='\t', header=None, names=[
            'query', 'id', 'long_answer', 'short_answer', 'title',
            'abstract', 'content', 'document_url', 'doc_tac', 'language'
        ])
        print(df.head(), flush=True)
    df_extracted = df[['id', 'query', 'doc_tac']].fillna('')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_extracted.to_csv(output_path, sep='\t', index=False, header=False, compression='gzip')

def prepare_dataset(data_path: str, tokenizer, max_length: int = 512):
    df = pd.read_csv(data_path, sep='\t', names=['id', 'query', 'doc_tac'], compression='gzip')
    df['doc_tac'] = df['doc_tac'].fillna('').astype(str)

    def tokenize(example):
        inputs = tokenizer(
            example['doc_tac'], max_length=max_length,
            truncation=True, padding="max_length"
        )
        targets = tokenizer(
            example['query'], max_length=max_length,
            truncation=True, padding="max_length"
        )
        inputs['labels'] = targets['input_ids']
        return inputs

    dataset = Dataset.from_pandas(df[['query', 'doc_tac']])
    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    return dataset

def split_dataset(dataset, seed: int, test_size: float = 0.2):
    split = dataset.train_test_split(test_size=test_size, seed=seed)
    return split['train'], split['test']


def set_global_seed(seed: int, cpu_threads: int):
    cpu_threads = max(1, cpu_threads)

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "true" if cpu_threads > 1 else "false"
    os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
    # os.environ["MKL_NUM_THREADS"] = str(cpu_threads)
    # os.environ["RAYON_NUM_THREADS"] = str(cpu_threads)
    os.environ["OMP_SCHEDULE"] = "STATIC"
    os.environ["OMP_PROC_BIND"] = "CLOSE"
    os.environ["GOMP_CPU_AFFINITY"] = "N-M"

    hf_set_seed(seed)
    # if torch.cuda.is_available():
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default="cache")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--cpu_threads", type=int, default=64, help="CPU thread cap for tokenization and CPU kernels")
    args = parser.parse_args()

    set_global_seed(args.seed, cpu_threads=args.cpu_threads)

    start_time = time.time()
    extract_query_doc_pairs(args.dataset_path, args.output_file)
    print(f"Query-doc pairs extracted and saved in {time.time() - start_time:.2f} seconds", flush=True)

    tokenizer = T5TokenizerFast.from_pretrained(
        "castorini/doc2query-t5-large-msmarco",
        cache_dir=args.cache_dir,
        legacy=False
    )
    model = T5ForConditionalGeneration.from_pretrained(
        "castorini/doc2query-t5-large-msmarco",
        cache_dir=args.cache_dir
    )

    start_time = time.time()
    dataset = prepare_dataset(args.output_file, tokenizer)
    print(f"dataset prepared in {time.time() - start_time:.2f} seconds", flush=True)
    start_time = time.time()
    train_dataset, eval_dataset = split_dataset(dataset, seed=args.seed, test_size=0.1)
    print(f"dataset splitted in {time.time() - start_time:.2f} seconds", flush=True)
    print(f"train dataset size: {len(train_dataset)}, eval dataset size: {len(eval_dataset)}", flush=True)

    output_dir = f"resources/checkpoints/finetuned_docTTTTTquery_on_{args.dataset_name}"
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="steps",
        eval_steps=10000,
        per_device_train_batch_size=4, # TODO: increase batch size if GPU memory allows
        per_device_eval_batch_size=4,
        learning_rate=3e-4,
        num_train_epochs=2,
        weight_decay=0.01,
        save_total_limit=2,
        save_steps=10000,
        logging_dir="logs",
        logging_steps=500,
        load_best_model_at_end=True,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    trainer.train()
    trainer.save_model(output_dir)

if __name__ == "__main__":
    main()
