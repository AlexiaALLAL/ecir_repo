#!/bin/sh

dataset_name="nq"

python src/data/data_prep/generate_pseudo_queries.py \
  --input_file resources/datasets/processed/nq-msmarco/nq-merged-json/nq-docs-sents.json \
  --checkpoint_path resources/checkpoints/finetuned_docTTTTTquery_on_${dataset_name} \
  --output_path resources/datasets/processed/${dataset_name}-data/${dataset_name}_pseudo_queries.jsonl \
  --batch_size 64 \
  --max_docs None \
  --num_return_sequences 10


  # --input_file data/temporal_splits_${dataset_name}/D0_train_filtered_with_sents.jsonl \
