#!/bin/bash

DEDUPLICATE=false # Whether to remove documents with duplicate titles between train and dev (default: false)

# Input NQ train/dev TSV.gz files (created from previous preprocessing)
# Output directory
if [ "$DEDUPLICATE" = true ]; then
  TRAIN_FILE="resources/datasets/processed/nq-data/nq_train_dedup.gz"
  DEV_FILE="resources/datasets/processed/nq-data/nq_val_dedup.gz"
  OUTPUT_DIR="resources/datasets/processed/nq-msmarco-dedup"
else
  TRAIN_FILE="resources/datasets/processed/nq-data/nq_train.gz"
  DEV_FILE="resources/datasets/processed/nq-data/nq_val.gz"
  OUTPUT_DIR="resources/datasets/processed/nq-msmarco"
fi

mkdir -p "$OUTPUT_DIR"

# Run conversion
python src/data/data_prep/nq/convert_nq_to_msmarco_format.py \
  --nq_train_file "$TRAIN_FILE" \
  --nq_dev_file "$DEV_FILE" \
  --output_dir "$OUTPUT_DIR"