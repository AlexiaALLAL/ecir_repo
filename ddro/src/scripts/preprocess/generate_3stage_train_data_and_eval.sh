#!/bin/bash


# make sure to change the encoding to either "url_title" or "pq"
ENCODING_METHOD="pq_nc24_cs128"  # Options: atomic, pq, rq-kmeans, rq-module, url, summary
DATASET="msmarco"  # Options: msmarco, nq
DEDUPLICATE=false # Whether to remove documents with duplicate titles between train and dev (default: false)
SEED=42  # Random seed for reproducible data generation

echo "generating 3-stage training data"
python src/data/data_prep/build_t5_data/gen_train_data_pipline.py \
    --cur_data general_pretrain --encoding "$ENCODING_METHOD" \
    --dataset "$DATASET" --dedup "$DEDUPLICATE" \
    --seed "$SEED"
python src/data/data_prep/build_t5_data/gen_train_data_pipline.py \
    --cur_data search_pretrain --encoding "$ENCODING_METHOD" \
    --dataset "$DATASET" --dedup "$DEDUPLICATE" \
    --seed "$SEED"
python src/data/data_prep/build_t5_data/gen_train_data_pipline.py \
    --cur_data finetune --encoding "$ENCODING_METHOD" \
    --dataset "$DATASET" --dedup "$DEDUPLICATE" \
    --seed "$SEED"

echo "Generating evaluation data..."
python src/data/data_prep/build_t5_data/gen_eval_data_pipline.py \
    --encoding "$ENCODING_METHOD" --dataset "$DATASET" \
    --dedup "$DEDUPLICATE" --seed "$SEED"

