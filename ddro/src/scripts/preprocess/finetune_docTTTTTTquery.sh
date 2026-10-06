#!/bin/sh

# Args
DATASET_PATH="resources/datasets/processed/nq-data/nq_train.gz"  
DATASET_NAME="nq"
OUTPUT_FILE="resources/datasets/processed/${DATASET_NAME}-data.tsv.gz"

python src/pretrain/finetune_docTTTTTquery_NQ.py \
  --dataset_path "$DATASET_PATH" \
  --dataset_name "$DATASET_NAME" \
  --output_file "$OUTPUT_FILE"