#!/bin/sh

# ----------------------------
# Configuration
# ----------------------------
DATASET="nq"      # dataset: 'msmarco' or 'nq'
ENCODING_METHOD="pq"
L="8"
V="128"
L="1"


# Parse named arguments
while [ $# -gt 0 ]; do
  case $1 in
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --encoding_method)
      ENCODING_METHOD="$2"
      shift 2
      ;;
    --L)
      L="$2"
      shift 2
      ;;
    --C)
      C="$2"
      shift 2
      ;;
    --V)
      V="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--encoding_method pq|rq-kmeans|rq-module|rq-ir|atomic] [--L NUM] [--C NUM] [--V NUM] [--seed NUM] [--results_file FILE]"
      exit 1
      ;;
  esac
done

if [ "$ENCODING_METHOD" = "pq" ]; then
    ENCODING_CONFIG="nc${C}_cs${V}"
    ENCODING_EXACT_NAME="${ENCODING_METHOD}_nc${C}_cs${V}"
elif [ "$ENCODING_METHOD" = "rq-kmeans" ] || [ "$ENCODING_METHOD" = "rq-module" ]; then
    ENCODING_CONFIG="nc${L}_cs${V}"
    ENCODING_EXACT_NAME="${ENCODING_METHOD}_nc${L}_cs${V}"
elif [ "$ENCODING_METHOD" = "pq-rq" ]; then
    ENCODING_CONFIG="nc$((C*L))_cs${V}"
    ENCODING_EXACT_NAME="${ENCODING_METHOD}_nc${C}_cs${V}_rqsteps${L}"
else
    echo "Unsupported encoding method: $ENCODING_METHOD"
    exit 1
fi

SCALE="top_300k"       # scale: 'top_300k'
NUM_BEAMS=10           # number of beams for evaluation
EPOCH=                 # epoch to evaluate (leave empty for latest checkpoint)
DEDUPLICATE=true     # Whether to remove documents with duplicate titles between train and dev (default: false)

echo "=== Evaluation Configuration ==="
echo "Dataset: $DATASET"

echo "Encoding: $ENCODING_CONFIG"
echo "Encoding (Exact Name): $ENCODING_EXACT_NAME"
echo "Scale: $SCALE"
if [ -n "$EPOCH" ]; then
    echo "Epoch: $EPOCH"
else
    echo "Epoch: Latest Checkpoint"
fi
echo "================================"

# ----------------------------
# Run Evaluation
# ----------------------------
CMD="src/pretrain/launch_ddro_eval_from_config.py \
  --dataset $DATASET \
  --encoding $ENCODING_CONFIG \
  --encoding_exact_name $ENCODING_EXACT_NAME \
  --scale $SCALE \
  --num_beams $NUM_BEAMS"

if [ -n "$EPOCH" ]; then
    CMD="$CMD --epoch $EPOCH"
fi

if [ "$DEDUPLICATE" = true ]; then
  CMD="$CMD --deduplicate"
  echo "Deduplication: Enabled (removing documents with duplicate titles between train and dev)"
fi

python $CMD
echo "=== Evaluation Completed ==="