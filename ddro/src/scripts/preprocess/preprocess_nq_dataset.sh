#!/bin/bash

# Input files
DEV_FILE="resources/datasets/raw/nq-data/nq-dev-all.jsonl.gz"
TRAIN_FILE="resources/datasets/raw/nq-data/simplified-nq-train.jsonl.gz"
DEDUPLICATE=false # Whether to remove documents with duplicate titles between train and dev (default: false)

# Output directory
OUTPUT_DIR="resources/datasets/processed/nq-data"
mkdir -p "$OUTPUT_DIR"

# Output file names (without extensions)
MERGED_FILE="${OUTPUT_DIR}/nq_merged"
if [ "$DEDUPLICATE" = true ]; then
  TRAIN_FILE_OUT="${OUTPUT_DIR}/nq_train_dedup"
  VAL_FILE_OUT="${OUTPUT_DIR}/nq_val_dedup"
else
  TRAIN_FILE_OUT="${OUTPUT_DIR}/nq_train"
  VAL_FILE_OUT="${OUTPUT_DIR}/nq_val"
fi

# Run the preprocessing script
CMD = "src/data/data_prep/nq/process_nq_dataset.py \
  --dev_file "$DEV_FILE" \
  --train_file "$TRAIN_FILE" \
  --output_merged_file "$MERGED_FILE" \
  --output_train_file "$TRAIN_FILE_OUT" \
  --output_val_file "$VAL_FILE_OUT" \
  --output_json_dir "$OUTPUT_DIR"
  "

if [ "$DEDUPLICATE" = true ]; then
  CMD="$CMD --deduplicate"
fi

python $CMD
