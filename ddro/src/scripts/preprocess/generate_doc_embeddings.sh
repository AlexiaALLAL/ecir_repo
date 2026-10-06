#!/bin/sh

DATASET="nq-queries"   # Specify the dataset type (e.g., "msmarco", "msmarco-queries", "nq", "nq-queries")

if [ "$DATASET" == "msmarco" ]; then
  INPUT_PATH="resources/datasets/processed/msmarco-docs-sents.top.300k.json.gz" 
  OUTPUT_PATH="resources/datasets/processed/msmarco-data/ms_doc_embeddings/doc_embeddings.json.gz"
elif [ "$DATASET" == "msmarco-queries" ]; then
  INPUT_PATH="resources/datasets/raw/msmarco-data/msmarco-doctrain-queries.tsv.gz"
  OUTPUT_PATH="resources/datasets/processed/msmarco-data/ms_query_embeddings/query_embeddings.json.gz"
elif [ "$DATASET" == "nq" ]; then
  INPUT_PATH="resources/datasets/processed/nq-data/nq_merged.gz"
  OUTPUT_PATH="resources/datasets/processed/nq-msmarco/nq_doc_embeddings/doc_embeddings.json.gz"
elif [ "$DATASET" == "nq-queries" ]; then
  INPUT_PATH="resources/datasets/processed/nq-msmarco-dedup/nq_queries_train.tsv.gz"
  OUTPUT_PATH="resources/datasets/processed/nq-msmarco-dedup/nq_query_embeddings/query_embeddings.json.gz"
else
  echo "Invalid dataset specified. Please choose from: msmarco, msmarco-queries, nq, nq-queries."
  exit 1
fi

MODEL_NAME="sentence-transformers/gtr-t5-base"
BATCH_SIZE=256

# Run the Python script
python src/data/data_prep/generate_doc_embeddings.py \
  --input_path "$INPUT_PATH" \
  --output_path "$OUTPUT_PATH" \
  --model_name "$MODEL_NAME" \
  --batch_size "$BATCH_SIZE" \
  --dataset "$DATASET" 
  
# Print completion message
echo "Document embeddings generation completed. Output saved to $OUTPUT_PATH."