#!/bin/bash

# Script 1: DocID Generation + Training Data Generation + Eval Data Generation

### PARSE COMMAND LINE ARGUMENTS ###
# Default values
DATASET="msmarco" # can be msmarco or nq
ENCODING="pq"  # Options: atomic, pq, rq-kmeans, rq-module, rq-ir, pq-rq, prq-module, url, summary
SUB_SPACE=16 # number of codebooks
CLUSTER_NUM=512 # codebook size
SUBSPACE_RQ=4  # Number of RQ steps for each subspace in PQ-RQ (only relevant if using pq-rq encoding method)
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
    --subspace_rq)
      SUBSPACE_RQ="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--dataset msmarco|nq] [--encoding pq|rq-kmeans|rq-module|rq-ir|pq-rq|atomic|url|summary] [--sub_space NUM] [--cluster_num NUM] [--subspace_rq NUM] [--seed NUM]"
      exit 1
      ;;
  esac
done

echo "Running Data Preparation with parameters:"
echo "  DATASET: $DATASET"
echo "  ENCODING: $ENCODING"
echo "  SUB_SPACE: $SUB_SPACE"
echo "  CLUSTER_NUM: $CLUSTER_NUM"
echo "  SUBSPACE_RQ: $SUBSPACE_RQ"
echo "  SEED: $SEED"
echo ""


### DOCID GENERATION ###
echo "=== Step 1/4: Generating encoded docids ==="

ENCODING_METHOD="${ENCODING}"
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
  OUTPUT_PATH="resources/datasets/processed/msmarco-data/encoded_docid/${ENCODING_METHOD}"
else
  INPUT_DOC_PATH=resources/datasets/processed/nq-msmarco/nq-merged-json/nq-docs-sents.json # to check if need dedup version
  INPUT_EMBED_PATH=resources/datasets/processed/nq-msmarco/nq_doc_embeddings/doc_embeddings.json.gz
  QUERY_EMBED_PATH=resources/datasets/processed/nq-msmarco/nq_query_embeddings/query_embeddings.json.gz # to check
  QREL_PATH=resources/datasets/processed/nq-msmarco/nq_qrels.tsv # to check
  OUTPUT_PATH="resources/datasets/processed/nq-data/encoded_docid/${ENCODING_METHOD}"
fi

# add nc and cs if using pq, rq-kmeans, rq-module or rq-ir encoding
if [ "$ENCODING_METHOD" == "pq" ] || [ "$ENCODING_METHOD" == "rq-kmeans" ] || [ "$ENCODING_METHOD" == "rq-module" ] || [ "$ENCODING_METHOD" == "rq-ir" ]; then
  OUTPUT_PATH="${OUTPUT_PATH}_nc$((SUB_SPACE * SUBSPACE_RQ))_cs${CLUSTER_NUM}" # nc = number codebooks, cs = codebook size
fi
elif [ "$ENCODING_METHOD" == "pq-rq" ] || [ "$ENCODING_METHOD" == "prq-module" ]; then
  OUTPUT_PATH="${OUTPUT_PATH}_nc${SUB_SPACE}_cs${CLUSTER_NUM}_rqsteps${SUBSPACE_RQ}"
fi


# Build output path based on encoder/decoder usage
if [ "$ENCODING_METHOD" == "rq-ir" ] || [ "$ENCODING_METHOD" == "rq-module" ]; then
  if [ "$USE_ENCODER_DECODER" = true ]; then
    OUTPUT_PATH="${OUTPUT_PATH}e${USE_ENCODER_DECODER}_w${COMMITMENT_WEIGHT}_d${CODEBOOK_EMBEDDING_DIM}"
  fi
fi

# output path for RQ-IR
if [ "$ENCODING_METHOD" == "rq-ir" ]; then
  OUTPUT_PATH="${OUTPUT_PATH}_recon_${RECON_WEIGHT}_vq_${VQ_WEIGHT}_ir_${IR_WEIGHT}"
  OUTPUT_PATH="${OUTPUT_PATH}_queryproc_${IR_QUERY_PROCESSING}_losslatent_${IR_LOSS_ON_LATENT}"
  OUTPUT_PATH="${OUTPUT_PATH}_${IR_LOSS_TYPE}"
fi
OUTPUT_PATH="${OUTPUT_PATH}_docid.txt"

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
echo "Generating 3-stage training and evaluation data"

if [ "$ENCODING_METHOD" == "pq" ] || [ "$ENCODING_METHOD" == "rq-kmeans" ] || [ "$ENCODING_METHOD" == "rq-module" ]; then
  ENCODING_NAME="${ENCODING}_nc$((SUB_SPACE * SUBSPACE_RQ))_cs${CLUSTER_NUM}"
elif [ "$ENCODING_METHOD" == "pq-rq" ] || [ "$ENCODING_METHOD" == "prq-module" ]; then
  ENCODING_NAME="${ENCODING_NAME}_nc${SUB_SPACE}_cs${CLUSTER_NUM}_rqsteps${SUBSPACE_RQ}"
elif [ "$ENCODING_METHOD" == "rq-ir" ]; then
  raise NotImplementedError "RQ-IR encoding method not currently supported for full training pipeline."
fi
DEDUPLICATE=true # Whether to remove documents with duplicate titles between train and dev (default: false)

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

python src/data/data_prep/build_t5_data/gen_eval_data_pipline.py \
    --encoding "$ENCODING_NAME" --dataset "$DATASET" \
    --dedup "$DEDUPLICATE" --seed "$SEED"

echo "=== Data Preparation Completed! ==="
