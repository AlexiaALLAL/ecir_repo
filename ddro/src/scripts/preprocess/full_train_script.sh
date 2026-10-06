#!/bin/bash

# This runs successively the scripts to generate encoded docids, then the 3-stage training data and train data.

### PARSE COMMAND LINE ARGUMENTS ###
# Default values
DATASET="msmarco" # can be msmarco or nq
ENCODING_METHOD="pq"  # Options: atomic, pq, rq-kmeans, rq-module, rq-ir, pq-rq, prq-module, url, summary
SUB_SPACE=1 # number of codebooks
CLUSTER_NUM=256 # codebook size
SUBSPACE_RQ=1 # Number of RQ steps
SEED=42
RESULTS_CSV_FILE="automatic_ir_results.csv" # CSV file to append results to (optional)

# Parse named arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --encoding)
      ENCODING_METHOD="$2"
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
    --results_csv_file)
      RESULTS_CSV_FILE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--dataset msmarco|nq] [--encoding pq|rq-kmeans|rq-module|rq-ir|pq-rq|prq-module|atomic|url|summary] [--sub_space NUM] [--cluster_num NUM] [--subspace_rq NUM] [--seed NUM] [--results_csv_file FILE]"
      exit 1
      ;;
  esac
done

echo "Running with parameters:"
echo "  DATASET: $DATASET"
echo "  ENCODING_METHOD: $ENCODING_METHOD"
echo "  SUB_SPACE: $SUB_SPACE"
echo "  CLUSTER_NUM: $CLUSTER_NUM"
echo "  SUBSPACE_RQ: $SUBSPACE_RQ"
echo "  SEED: $SEED"
echo ""


### DOCID GENERATION ###
echo "Generating encoded docids..."

RECON_WEIGHT=1.0  # Only relevant for rq-module encoding method, set to 0 to disable reconstruction loss
VQ_WEIGHT=1.0  # Only relevant for rq-module encoding method, set to 0 to disable VQ loss
IR_WEIGHT=3e-3  # Only relevant for rq-module encoding method, set to 0 to disable IR loss
IR_QUERY_PROCESSING=true  # Whether to apply recon and vq loss on query embeddings in RQ-IR
IR_LOSS_ON_LATENT=false  # Whether to apply IR loss on latent embeddings instead of recon output in RQ-IR
IR_LOSS_TYPE="infonce"  # Type of IR loss: "infonce" or "triplet"
IR_LOSS_MARGIN=0.5  # Margin (epsilon) for triplet loss

# Model parameters for product quantization (if applicable)
BATCH_SIZE=1024
USE_ENCODER_DECODER=false  # Use trainable encoder/decoder in RQVAE
ENCODER_HIDDEN_DIMS="512 256 128"  # Hidden dimensions for encoder (e.g., 512 256 128)
CODEBOOK_EMBEDDING_DIM=32  # Dimension of codebook vectors (after encoder, if used)
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

ENCODING_NAME="${ENCODING_METHOD}" # will be set based on encoding method and parameters

if [ "$ENCODING_METHOD" == "pq" ] || [ "$ENCODING_METHOD" == "rq-kmeans" ] || [ "$ENCODING_METHOD" == "rq-module" ] ; then
  ENCODING_NAME="${ENCODING_NAME}_nc$(($SUB_SPACE * $SUBSPACE_RQ))_cs${CLUSTER_NUM}"
elif [ "$ENCODING_METHOD" == "pq-rq" ] || [ "$ENCODING_METHOD" == "prq-module" ]; then
  ENCODING_NAME="${ENCODING_NAME}_nc${SUB_SPACE}_cs${CLUSTER_NUM}_rqsteps${SUBSPACE_RQ}"
elif [ "$ENCODING_METHOD" == "rq-ir" ]; then
  echo "RQ-IR encoding method not currently supported for full training pipeline."
  exit 1
fi

# Build output path based on encoder/decoder usage
if [ "$ENCODING_METHOD" == "rq-ir" ] || [ "$ENCODING_METHOD" == "rq-module" ]; then
  if [ "$USE_ENCODER_DECODER" = true ]; then
    ENCODING_NAME="${ENCODING_NAME}_e${USE_ENCODER_DECODER}_w${COMMITMENT_WEIGHT}_d${CODEBOOK_EMBEDDING_DIM}"
  fi
fi

# output path for RQ-IR
if [ "$ENCODING_METHOD" == "rq-ir" ]; then
  ENCODING_NAME="${ENCODING_NAME}_recon_${RECON_WEIGHT}_vq_${VQ_WEIGHT}_ir_${IR_WEIGHT}"
  ENCODING_NAME="${ENCODING_NAME}_queryproc_${IR_QUERY_PROCESSING}_losslatent_${IR_LOSS_ON_LATENT}"
  ENCODING_NAME="${ENCODING_NAME}_${IR_LOSS_TYPE}"
fi

echo "Final encoding name for this run: $ENCODING_NAME"

OUTPUT_PATH="${OUTPUT_PATH}${ENCODING_NAME}_docid.txt"

PRETRAIN_MODEL_PATH="resources/transformer_models/t5-base"
SUMMARY_PATH="data/summaries.json"


# Run the Python script
CMD="python src/data/data_prep/generate_encoded_docids.py \
  --encoding \"$ENCODING_METHOD\" \
  --input_doc_path \"$INPUT_DOC_PATH\" \
  --input_embed_path \"$INPUT_EMBED_PATH\" \
  --input_qrel_path \"$QREL_PATH\" \
  --recon_weight \"$RECON_WEIGHT\" \
  --vq_weight \"$VQ_WEIGHT\" \
  --ir_weight \"$IR_WEIGHT\" \
  --output_path \"$OUTPUT_PATH\" \
  --pretrain_model_path \"$PRETRAIN_MODEL_PATH\" \
  --summary_path \"$SUMMARY_PATH\" \
  --sub_space \"$SUB_SPACE\" \
  --subspace_rq \"$SUBSPACE_RQ\" \
  --cluster_num \"$CLUSTER_NUM\" \
  --batch_size \"$BATCH_SIZE\""

if [ "$USE_ENCODER_DECODER" = true ]; then
  CMD="$CMD \
  --use_encoder_decoder \
  --encoder_hidden_dims $ENCODER_HIDDEN_DIMS \
  --codebook_embedding_dim \"$CODEBOOK_EMBEDDING_DIM\""
fi
if [ "$IR_QUERY_PROCESSING" = true ]; then
  CMD="$CMD --ir_query_processing"
fi
if [ "$IR_LOSS_ON_LATENT" = true ]; then
  CMD="$CMD --ir_loss_on_latent"
fi

CMD="$CMD \
  --commitment_weight \"$COMMITMENT_WEIGHT\" \
  --ir_loss_type \"$IR_LOSS_TYPE\" \
  --ir_loss_margin \"$IR_LOSS_MARGIN\" \
  --seed \"$SEED\" \
  --input_query_path \"$QUERY_EMBED_PATH\""
echo "Running command: $CMD"
eval $CMD


### 3-STAGE TRAINING DATA GENERATION AND EVALUATION DATA GENERATION ###
DEDUPLICATE=true # Whether to remove documents with duplicate titles between train and dev (default: false)

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



### TRAINING ###
echo "Starting training..."

if [ "$ENCODING_METHOD" == "pq" ] || [ "$ENCODING_METHOD" == "rq-kmeans" ] || [ "$ENCODING_METHOD" == "rq-module" ] || [ "$ENCODING_METHOD" == "pq-rq" ] || [ "$ENCODING_METHOD" == "prq-module" ]; then
  ENCODING_CONFIG="nc$(($SUB_SPACE * $SUBSPACE_RQ))_cs${CLUSTER_NUM}"
elif [ "$ENCODING_METHOD" == "rq-ir" ]; then
  echo "RQ-IR encoding method not currently supported for full training pipeline."
  exit 1
fi

python src/utils/run_training_pipeline.py \
    --encoding "$ENCODING_CONFIG" \
    --encoding_name_exact "$ENCODING_NAME" \
    --dataset "$DATASET" \
    --dedup "$DEDUPLICATE" \
    --scale top_300k \
    --seed "$SEED" \



### EVALUATION ###
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

if [ -n "$RESULTS_CSV_FILE" ]; then
  echo "Results have been appended to $RESULTS_CSV_FILE"
fi

echo "=== Full Training and Evaluation Completed!!! ==="