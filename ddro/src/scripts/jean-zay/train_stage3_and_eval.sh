#!/bin/bash

# Script 4: Stage 3 Training + Evaluation - Query-to-DocID Finetuning

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

echo "Running Stage 3 Training + Evaluation with parameters:"
echo "  DATASET: $DATASET"
echo "  ENCODING: $ENCODING"
echo "  SUB_SPACE: $SUB_SPACE"
echo "  CLUSTER_NUM: $CLUSTER_NUM"
echo "  SEED: $SEED"
echo ""

ENCODING_NAME="${ENCODING}_nc${SUB_SPACE}_cs${CLUSTER_NUM}"
ENCODING_CONFIG="nc${SUB_SPACE}_cs${CLUSTER_NUM}"
DEDUPLICATE=false

echo "=== Step 4/4: Stage 3 Training and Final Evaluation ==="
python src/utils/run_training_pipeline.py \
    --encoding "$ENCODING_CONFIG" \
    --encoding_name_exact "$ENCODING_NAME" \
    --dataset "$DATASET" \
    --dedup "$DEDUPLICATE" \
    --scale top_300k \
    --seed "$SEED" \
    --resume_stage finetune \
    --only_current_stage

echo ""
echo "=== Evaluation ==="
python src/pretrain/launch_ddro_eval_from_config.py \
  --dataset "$DATASET" \
  --encoding "$ENCODING_CONFIG" \
  --encoding_exact_name "$ENCODING_NAME" \
  --scale top_300k

echo ""
echo "=== Step 4/4: Stage 3 Training and Evaluation Completed! ==="
