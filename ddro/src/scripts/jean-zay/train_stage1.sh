#!/bin/bash

# Script 2: Stage 1 Training - Content-to-DocID Pretraining

### PARSE COMMAND LINE ARGUMENTS ###
# Default values
DATASET="msmarco" # can be msmarco or nq
ENCODING="pq"  # Options: atomic, pq, rq-kmeans, rq-module, rq-ir, url, summary
SUB_SPACE=16 # number of codebooks
CLUSTER_NUM=512 # codebook size
SEED=42

# Parse named arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --encoding)
      ENCODING="$2"
      shift 2
      ;;
    --sub_space)
      SUB_SPACE="$2"
      shift 2
      ;;
    --cluster_num)
      CLUSTER_NUM="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--dataset msmarco|nq] [--encoding pq|rq-kmeans|rq-module|rq-ir|atomic|url|summary] [--sub_space NUM] [--cluster_num NUM] [--seed NUM]"
      exit 1
      ;;
  esac
done

echo "Running Stage 1 Training with parameters:"
echo "  DATASET: $DATASET"
echo "  ENCODING: $ENCODING"
echo "  SUB_SPACE: $SUB_SPACE"
echo "  CLUSTER_NUM: $CLUSTER_NUM"
echo "  SEED: $SEED"
echo ""

ENCODING_NAME="${ENCODING}_nc${SUB_SPACE}_cs${CLUSTER_NUM}"
ENCODING_CONFIG="nc${SUB_SPACE}_cs${CLUSTER_NUM}"
DEDUPLICATE=false

echo "=== Step 2/4: Stage 1 Training ==="
python src/utils/run_training_pipeline.py \
    --encoding "$ENCODING_CONFIG" \
    --encoding_name_exact "$ENCODING_NAME" \
    --dataset "$DATASET" \
    --dedup "$DEDUPLICATE" \
    --scale top_300k \
    --seed "$SEED" \
    --resume_stage pretrain \
    --only_current_stage

echo ""
echo "=== Step 2/4: Stage 1 Training Completed! ==="
