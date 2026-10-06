#!/bin/bash

# Script: Prepare data + Full 3-Stage Training Pipeline + Evaluation
# Runs all 3 training stages in sequence: prepare_data -> pretrain -> search_pretrain -> finetune -> evaluation
# Supports automatic restart from checkpoint if job times out

### PARSE COMMAND LINE ARGUMENTS ###
# Default values
DATASET="msmarco" # can be msmarco or nq
ENCODING="pq"  # Options: atomic, pq, rq-kmeans, rq-module, pq-rq, prq-module, url, summary
SUB_SPACE=16 # number of codebooks
CLUSTER_NUM=512 # codebook size
SUBSPACE_RQ=1 # Number of RQ steps (only for pq-rq)
SEED=42
RESUME_FROM_CHECKPOINT=""  # Path to checkpoint to resume from
RESUME_STAGE=""  # Which stage to resume from (pretrain, search_pretrain, or finetune)
RESUME_WANDB_ID=""  # WandB run ID to resume
RESUME_EPOCH=""  # Epoch number to resume from
RESULTS_CSV_FILE="automatic_ir_results.csv"  # CSV file to append results to (optional)

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
    --subspace_rq)
      SUBSPACE_RQ="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --resume_from_checkpoint)
      RESUME_FROM_CHECKPOINT="$2"
      shift 2
      ;;
    --resume_stage)
      RESUME_STAGE="$2"
      shift 2
      ;;
    --resume_wandb_id)
      RESUME_WANDB_ID="$2"
      shift 2
      ;;
    --resume_epoch)
      RESUME_EPOCH="$2"
      shift 2
      ;;
    --results_csv_file)
      RESULTS_CSV_FILE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--dataset msmarco|nq] [--encoding pq|rq-kmeans|rq-module|pq-rq|atomic|url|summary] [--sub_space NUM] [--cluster_num NUM] [--subspace_rq NUM] [--seed NUM] [--resume_from_checkpoint PATH] [--resume_stage STAGE] [--resume_wandb_id ID] [--resume_epoch NUM] [--results_csv_file FILE]"
      exit 1
      ;;
  esac
done

echo "Running Full Training Pipeline with parameters:"
echo "  DATASET: $DATASET"
echo "  ENCODING: $ENCODING"
echo "  SUB_SPACE: $SUB_SPACE"
echo "  CLUSTER_NUM: $CLUSTER_NUM"
echo "  SUBSPACE_RQ: $SUBSPACE_RQ"
echo "  SEED: $SEED"
if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
  echo "  RESUME_FROM_CHECKPOINT: $RESUME_FROM_CHECKPOINT"
  echo "  RESUME_STAGE: $RESUME_STAGE"
  echo "  RESUME_WANDB_ID: $RESUME_WANDB_ID"
  echo "  RESUME_EPOCH: $RESUME_EPOCH"
fi
echo ""

# Determine encoding name based on method
if [ "$ENCODING" == "pq" ] || [ "$ENCODING" == "rq-kmeans" ] || [ "$ENCODING" == "rq-module" ]; then
  ENCODING_NAME="${ENCODING}_nc$(($SUB_SPACE * $SUBSPACE_RQ))_cs${CLUSTER_NUM}"
  ENCODING_CONFIG="nc$(($SUB_SPACE * $SUBSPACE_RQ))_cs${CLUSTER_NUM}"
elif [ "$ENCODING" == "pq-rq" ] || [ "$ENCODING" == "prq-module" ]; then
  ENCODING_NAME="${ENCODING}_nc${SUB_SPACE}_cs${CLUSTER_NUM}_rqsteps${SUBSPACE_RQ}"
  ENCODING_CONFIG="nc$(($SUB_SPACE * $SUBSPACE_RQ))_cs${CLUSTER_NUM}"
else
  ENCODING_NAME="${ENCODING}"
  ENCODING_CONFIG="${ENCODING}"
fi

DEDUPLICATE=true


if [ -z "$RESUME_FROM_CHECKPOINT" ]; then

### DOCID GENERATION ###
echo "Generating encoded docids..."

RECON_WEIGHT=1.0  # Only relevant for rq-module encoding method, set to 0 to disable reconstruction loss
VQ_WEIGHT=1.0  # Only relevant for rq-module encoding method, set to 0 to disable VQ loss


# Model parameters for product quantization (if applicable)
BATCH_SIZE=1024
COMMITMENT_WEIGHT=0.25  # Weight for commitment loss in VQ

# Set paths to input files and output directory
if [ "$DATASET" == "msmarco" ]; then
  INPUT_DOC_PATH=resources/datasets/processed/msmarco-docs-sents.top.300k.json.gz
  INPUT_EMBED_PATH=resources/datasets/processed/msmarco-data/ms_doc_embeddings/doc_embeddings.json.gz
  QUERY_EMBED_PATH=resources/datasets/processed/msmarco-data/ms_query_embeddings/query_embeddings.json.gz
  QREL_PATH=resources/datasets/raw/msmarco-data/msmarco-doctrain-qrels.tsv
  OUTPUT_PATH="resources/datasets/processed/msmarco-data/encoded_docid/"
else
  INPUT_DOC_PATH=resources/datasets/processed/nq-msmarco/nq-merged-json/nq-docs-sents.json # to check if need dedup version
  INPUT_EMBED_PATH=resources/datasets/processed/nq-msmarco/nq_doc_embeddings/doc_embeddings.json.gz
  QUERY_EMBED_PATH=resources/datasets/processed/nq-msmarco/nq_query_embeddings/query_embeddings.json.gz # to check
  QREL_PATH=resources/datasets/processed/nq-msmarco/nq_qrels.tsv # to check
  OUTPUT_PATH="resources/datasets/processed/nq-data/encoded_docid/"
fi

OUTPUT_PATH="${OUTPUT_PATH}${ENCODING_NAME}_docid.txt"

PRETRAIN_MODEL_PATH="resources/transformer_models/t5-base"
SUMMARY_PATH="data/summaries.json"

# Run the Python script
CMD="python src/data/data_prep/generate_encoded_docids.py \
  --encoding \"$ENCODING\" \
  --input_doc_path \"$INPUT_DOC_PATH\" \
  --input_embed_path \"$INPUT_EMBED_PATH\" \
  --input_qrel_path \"$QREL_PATH\" \
  --recon_weight \"$RECON_WEIGHT\" \
  --vq_weight \"$VQ_WEIGHT\" \
  --output_path \"$OUTPUT_PATH\" \
  --pretrain_model_path \"$PRETRAIN_MODEL_PATH\" \
  --summary_path \"$SUMMARY_PATH\" \
  --sub_space \"$SUB_SPACE\" \
  --subspace_rq \"$SUBSPACE_RQ\" \
  --cluster_num \"$CLUSTER_NUM\" \
  --batch_size \"$BATCH_SIZE\" \
  --commitment_weight \"$COMMITMENT_WEIGHT\" \
  --seed \"$SEED\" \
  --input_query_path \"$QUERY_EMBED_PATH\""
echo "Running command: $CMD"
eval $CMD


### 3-STAGE TRAINING DATA GENERATION AND EVALUATION DATA GENERATION ###
echo "generating 3-stage training data"
python src/data/data_prep/build_t5_data/gen_train_data_pipline.py \
    --cur_data general_pretrain --encoding "$ENCODING_NAME" \
    --dataset "$DATASET" --dedup "$DEDUPLICATE" \
    --seed "$SEED"
python src/data/data_prep/build_t5_data/gen_train_data_pipline.py \
    --cur_data search_pretrain --encoding "$ENCODING_NAME" \
    --dataset "$DATASET" --dedup "$DEDUPLICATE" \
    --seed "$SEED"
python src/data/data_prep/build_t5_data/gen_train_data_pipline.py \
    --cur_data finetune --encoding "$ENCODING_NAME" \
    --dataset "$DATASET" --dedup "$DEDUPLICATE" \
    --seed "$SEED"

echo "Generating evaluation data..."
python src/data/data_prep/build_t5_data/gen_eval_data_pipline.py \
    --encoding "$ENCODING_NAME" --dataset "$DATASET" \
    --dedup "$DEDUPLICATE" --seed "$SEED"

else
  echo "Skipping data preprocessing (resuming from checkpoint: $RESUME_FROM_CHECKPOINT)"
fi

echo "=== Full 3-Stage Training Pipeline ==="
echo "Stage 1: Content-to-DocID Pretraining"
echo "Stage 2: PseudoQuery-to-DocID Pretraining"
echo "Stage 3: Query-to-DocID Finetuning"
echo ""

# Build the command with optional resume parameters
CMD="python src/utils/run_training_pipeline.py \
    --encoding \"$ENCODING_CONFIG\" \
    --encoding_name_exact \"$ENCODING_NAME\" \
    --dataset \"$DATASET\" \
    --dedup \"$DEDUPLICATE\" \
    --scale top_300k \
    --seed \"$SEED\""

# Add resume parameters if provided
if [ -n "$RESUME_STAGE" ]; then
  CMD="$CMD --resume_stage \"$RESUME_STAGE\""
fi

if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
  CMD="$CMD --resume_from_checkpoint \"$RESUME_FROM_CHECKPOINT\""
fi

if [ -n "$RESUME_WANDB_ID" ]; then
  CMD="$CMD --resume_wandb_run_id \"$RESUME_WANDB_ID\""
fi

if [ -n "$RESUME_EPOCH" ]; then
  CMD="$CMD --resume_epoch \"$RESUME_EPOCH\""
fi

echo "Running command: $CMD"
echo ""

# Run training pipeline (all 3 stages)
eval $CMD

# Check if training completed successfully
if [ $? -eq 0 ]; then
  echo ""
  echo "=== Training Completed Successfully ==="
  echo ""
  
  # Run evaluation
  echo "=== Running Final Evaluation ==="
  
  echo "Starting evaluation with beam_size=10..."
  python src/pretrain/launch_ddro_eval_from_config.py \
    --dataset "$DATASET" \
    --encoding "$ENCODING_CONFIG" \
    --encoding_exact_name "$ENCODING_NAME" \
    --scale top_300k \
    --num_beams 10 \
    --encoding_method "$ENCODING_METHOD" \
    --num_codebooks "$SUBSPACE_RQ" \
    --nb_subspaces "$SUB_SPACE" \
    --codebook_size "$CLUSTER_NUM" \
    --results_csv_file "$RESULTS_CSV_FILE"

  echo "Starting evaluation with beam_size=100..."
  python src/pretrain/launch_ddro_eval_from_config.py \
    --dataset "$DATASET" \
    --encoding "$ENCODING_CONFIG" \
    --encoding_exact_name "$ENCODING_NAME" \
    --scale top_300k \
    --num_beams 100 \
    --encoding_method "$ENCODING_METHOD" \
    --num_codebooks "$SUBSPACE_RQ" \
    --nb_subspaces "$SUB_SPACE" \
    --codebook_size "$CLUSTER_NUM" \
    --results_csv_file "$RESULTS_CSV_FILE"
  
  
  if [ $? -eq 0 ]; then
    echo ""
    echo "=== Full Training and Evaluation Completed Successfully! ==="
    
    # Create completion marker file in the finetune directory
    # Find the finetune directory (may have subset suffix like _10000)
    FINETUNE_DIR=$(find "outputs/${DATASET}" -maxdepth 1 -type d -name "t5_128_1_top_300k_${ENCODING_NAME}_pretrain_search_finetune*" 2>/dev/null | head -1)
    if [ -n "$FINETUNE_DIR" ] && [ -d "$FINETUNE_DIR" ]; then
      touch "$FINETUNE_DIR/.training_completed"
      echo "Created completion marker at $FINETUNE_DIR/.training_completed"
    else
      echo "WARNING: Could not find finetune directory to create completion marker"
    fi
  else
    echo "ERROR: Evaluation failed"
    exit 1
  fi
else
  echo "ERROR: Training pipeline failed or was interrupted"
  exit 1
fi
