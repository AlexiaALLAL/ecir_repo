#!/bin/bash


# Set default values - MODIFY THESE PATHS FOR YOUR SETUP
DATASET="msmarco" # can be msmarco or nq
ENCODING_METHOD="rq-module"  # Options: atomic/pq/rq-kmeans/rq-module/rq-ir/url/summary
L="1"  # e.g., 4 for RQ with 4 codebooks
C="1"  # e.g., 256 for codebooks of size 256
V="128"  # e.g., 512 for codebooks of size 512
SEED=42
RESULTS_FILE="src/analysis/analysis_results_temporary.csv"

# Parse named arguments
while [[ $# -gt 0 ]]; do
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
    --seed)
      SEED="$2"
      shift 2
      ;;
    --results_file)
      RESULTS_FILE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--encoding_method pq|rq-kmeans|rq-module|rq-ir|atomic] [--L NUM] [--C NUM] [--V NUM] [--seed NUM] [--results_file FILE]"
      exit 1
      ;;
  esac
done

# if pq, rq-kmeans or rq-module:
if [[ "$ENCODING_METHOD" == "pq" ]]; then
    ENCODING_NAME="${ENCODING_METHOD}_nc${C}_cs${V}"
elif [[ "$ENCODING_METHOD" == "rq-kmeans" || "$ENCODING_METHOD" == "rq-module" ]]; then
    ENCODING_NAME="${ENCODING_METHOD}_nc${L}_cs${V}"
elif [[ "$ENCODING_METHOD" == "pq-rq" ]]; then
    ENCODING_NAME="${ENCODING_METHOD}_nc${C}_cs${V}_rqsteps${L}"
else
    echo "Unsupported encoding method: $ENCODING_METHOD"
    exit 1
fi


# Paths
if [[ "$DATASET" == "msmarco" ]]; then
    DOC_EMBED_PATH="resources/datasets/processed/msmarco-data/ms_doc_embeddings/doc_embeddings.json.gz"
    QUERY_EMBED_PATH="resources/datasets/processed/msmarco-data/ms_query_embeddings/query_embeddings.json.gz"
    QREL_PATH="resources/datasets/raw/msmarco-data/msmarco-doctrain-qrels.tsv.gz"
    CORPUS_PATH="resources/datasets/processed/msmarco-docs-sents.top.300k.json"
    DOCID_PATH="resources/datasets/processed/msmarco-data/encoded_docid/${ENCODING_NAME}_docid.txt"
else
    DOC_EMBED_PATH="resources/datasets/processed/nq-msmarco/nq_doc_embeddings/doc_embeddings.json.gz"
    QUERY_EMBED_PATH="resources/datasets/processed/nq-msmarco-dedup/nq_query_embeddings/query_embeddings.json.gz"
    QREL_PATH="resources/datasets/processed/nq-msmarco-dedup/nq_qrels_train.tsv.gz"
    CORPUS_PATH="resources/datasets/processed/nq-msmarco/nq-merged-json/nq-docs-sents.json"
    DOCID_PATH="resources/datasets/processed/nq-data/encoded_docid/${ENCODING_NAME}_docid.txt"
fi


if [[ "$ENCODING_METHOD" == "rq-module" ]]; then
  if [[ "$DATASET" == "msmarco" ]]; then
    MODEL_PATH="resources/datasets/processed/msmarco-data/encoded_docid/checkpoints/${ENCODING_NAME}_docid/best_model.pth"
  else
    MODEL_PATH="resources/datasets/processed/nq-data/encoded_docid/checkpoints/${ENCODING_NAME}_docid/best_model.pth"
  fi
else
  if [[ "$DATASET" == "msmarco" ]]; then
    MODEL_PATH="resources/datasets/processed/msmarco-data/encoded_docid/${ENCODING_NAME}_docid_model.pkl"
  else
    MODEL_PATH="resources/datasets/processed/nq-data/encoded_docid/${ENCODING_NAME}_docid_model.pkl"
  fi
fi


# Metrics to compute - comment out metrics you don't want to run
METRICS=(
    "uniqueness"                # Uniqueness ratio of docids
    "entropy_gini"              # Entropy and Gini index per token level
    # "correlation"               # Correlation between embedding and docid similarity (requires doc_embed_path)
    # "shared_tokens"             # Embedding similarity for pairs with shared tokens, mean over k (requires doc_embed_path)
    "shared_tokens_new"        # Embedding similarity for pairs with shared tokens, mean over (k, doc) (requires doc_embed_path)
    # "ranking_preservation"      # Ranking preservation for random triplets (requires doc_embed_path)
    # "query_ranking"             # Query-doc ranking preservation (requires model_path, query_embed_path, qrel_path)
    # "query_closeness"           # Query-doc closeness (requires model_path, query_embed_path, qrel_path)
    "robustness"                # Robustness to text reformulation (requires corpus_path and model_path)
    "robustness_ranking_preservation" # Ranking preservation for reformulated vs random docs (requires corpus_path and model_path)
    # "sample_docs"               # Save sample documents by first token (requires corpus_path)
    "plot_distributions"        # Plot code-level distributions
)

# Analysis parameters
NB_PAIRS=10000  # Number of pairs for correlation calculation
NB_TRIPLETS=1000  # Number of triplets for ranking preservation calculation
NB_QUERIES_RANKING_PRESERVATION=10000 # Number of queries from train set to sample for ranking preservation analysis
NUM_SAMPLES_PER_TOKEN=10  # Number of documents to sample per first token
NUM_SAMPLES_FOR_ROBUSTNESS=500  # Number of documents to sample for robustness analysis
BATCH_SIZE=1024  # Batch size for encoding (reduce if you get memory/segmentation errors)


# Build the command
CMD="python src/analysis/analyze_docids.py --encoding $ENCODING_NAME --seed $SEED --results_file $RESULTS_FILE --dataset $DATASET"

# Add metrics to compute
CMD="$CMD --metrics ${METRICS[@]}"

# Add standard analysis arguments
CMD="$CMD --doc_embed_path $DOC_EMBED_PATH --corpus_path $CORPUS_PATH --docid_path $DOCID_PATH"
CMD="$CMD --nb_pairs $NB_PAIRS --nb_triplets $NB_TRIPLETS --nb_queries_ranking_preservation $NB_QUERIES_RANKING_PRESERVATION --num_samples_per_token $NUM_SAMPLES_PER_TOKEN --num_samples_for_robustness $NUM_SAMPLES_FOR_ROBUSTNESS"

CMD="$CMD --batch_size $BATCH_SIZE"

CMD="$CMD --model_path $MODEL_PATH --query_embed_path $QUERY_EMBED_PATH --qrel_path $QREL_PATH"
CMD="$CMD --num_codebooks $L --nb_subspaces $C --codebook_size $V"


echo "=========================================="
echo "Document ID Analysis Script"
echo "=========================================="
echo "Metrics to compute:    ${METRICS[@]}"
echo "Dataset:               $DATASET"
echo "Encoding:              $ENCODING_NAME"
echo "Random seed:           $SEED"
echo "Batch size:            $BATCH_SIZE"
echo "Doc embeddings:        $DOC_EMBED_PATH"
echo "Corpus:                $CORPUS_PATH"
echo "Num codebooks (L):     $L"
echo "Num subspaces (C):     $C"
echo "Codebook size (V):     $V"
echo "Nb pairs:              $NB_PAIRS"
echo "Nb triplets:           $NB_TRIPLETS"
echo "Nb queries for ranking preservation: $NB_QUERIES_RANKING_PRESERVATION"
echo "Samples robustness:    $NUM_SAMPLES_FOR_ROBUSTNESS"
echo "Results file:          $RESULTS_FILE"
echo "Model path:            $MODEL_PATH"
echo "Query embeddings:      $QUERY_EMBED_PATH"
echo "Qrels:                 $QREL_PATH"
echo "DocID path:            $DOCID_PATH"
echo "=========================================="
echo ""
echo "Running command:"
echo "$CMD"
echo ""
echo "=========================================="
echo ""


eval $CMD