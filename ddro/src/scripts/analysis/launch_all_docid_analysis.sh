#!/bin/bash

# Param grid
DATASET=("msmarco" "nq")
ENCODING_METHODS=("pq" "rq-kmeans" "rq-module" "pq-rq")
SEED=42
RESULTS_FILE="src/analysis/analysis_results_test"

# Loop through all combinations of parameters
for DATASET in "${DATASET[@]}"; do
  RESULTS_FILE="${RESULTS_FILE}_${DATASET}.csv"
  for ENCODING_METHOD in "${ENCODING_METHODS[@]}"; do
    echo "Running analysis for encoding method: $ENCODING_METHOD"
    if [ "$ENCODING_METHOD" == "pq" ] || [ "$ENCODING_METHOD" == "rq-kmeans" ] || [ "$ENCODING_METHOD" == "rq-module" ]; then
    # We vary on a whole grid of (C,V) for pq and (L,V) for rq-kmeans and rq-module
      if [ "$ENCODING_METHOD" == "pq" ]; then
        CLUSTER_NUMS=(512 256 128)
        SUB_SPACES=(24 16 8 4)
        RQ_STEPS=(1)
      else
        CLUSTER_NUMS=(512 256 128)
        SUB_SPACES=(1)
        RQ_STEPS=(24 16 8 4)
      fi
      for SUB_SPACE in "${SUB_SPACES[@]}"; do
      for RQ_STEP in "${RQ_STEPS[@]}"; do
      for CLUSTER_NUM in "${CLUSTER_NUMS[@]}"; do
        # Call analyze_docids.sh with the current combination of parameters
        echo "Running analysis for encoding: $ENCODING_METHOD, sub_space: $SUB_SPACE, rq_step: $RQ_STEP, cluster_num: $CLUSTER_NUM"
        bash ./src/scripts/analysis/analyze_docids.sh \
              --dataset "$DATASET" \
              --encoding_method "$ENCODING_METHOD" --L "$RQ_STEP" \
              --C "$SUB_SPACE" --V "$CLUSTER_NUM" --seed "$SEED" \
              --results_file "$RESULTS_FILE"
      done
      done
      done
    elif [ "$ENCODING_METHOD" == "pq-rq" ]; then
    # For pq-rq, we vary on a list of (C, L) pairs with a fixed V
      SUB_SPACES_RQ_STEPS=(
        "12 2" "6 4" "4 6" "2 12" 
        "8 2" "4 4" "2 8" 
        "4 2" "2 4"
        )
      CLUSTER_NUM=256
      for SUB_SPACE_RQ_STEP in "${SUB_SPACES_RQ_STEPS[@]}"; do
        SUB_SPACE=$(echo "$SUB_SPACE_RQ_STEP" | cut -d ' ' -f 1)
        RQ_STEP=$(echo "$SUB_SPACE_RQ_STEP" | cut -d ' ' -f 2)
        echo "Running analysis for encoding: $ENCODING_METHOD, sub_space: $SUB_SPACE, rq_step: $RQ_STEP, cluster_num: $CLUSTER_NUM"
        bash ./src/scripts/analysis/analyze_docids.sh \
              --dataset "$DATASET" \
              --encoding_method "$ENCODING_METHOD" --L "$RQ_STEP" \
              --C "$SUB_SPACE" --V "$CLUSTER_NUM" --seed "$SEED" \
              --results_file "$RESULTS_FILE"
      done
    else 
      echo "Unsupported encoding method: $ENCODING_METHOD"
    fi
  done
done

echo "All analyses complete! Results saved to $RESULTS_FILE"