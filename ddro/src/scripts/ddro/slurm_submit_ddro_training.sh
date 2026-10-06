#!/bin/sh

# Move to project directory
# cd /path/to/DDRO-Direct-Document-Relevance-Optimization/ddro

# Set dataset configuration
FILE_PATH=resources/datasets/processed/msmarco-data
ENCODING=rq-module  # options: pq | url_title | rq-kmeans | rq-module | rq-ir

# Run training script
python src/pretrain/train_ddro_encoder_decoder.py \
  --train_file $FILE_PATH/pyserini_data/msmarco_train_triples \
  --dev_file $FILE_PATH/pyserini_data/msmarco_dev_triples \
  --train_queries_file resources/datasets/raw/msmarco-data/msmarco-doctrain-queries.tsv.gz \
  --dev_queries_file resources/datasets/raw/msmarco-data/msmarco-docdev-queries.tsv.gz \
  --docid_path $FILE_PATH/encoded_docid/${ENCODING}_docid.txt \
  --output_dir outputs-sft-msmarco/ddro/ddro_ckp_${ENCODING}_5epoch_lr1e-5_BETA_049 \
  --pretrain_model_path resources/transformer_models/t5-base \
  --use_origin_head False \
  --max_prompt_length 128 \
  --checkpoint_path outputs/t5_128_1_top_300k_${ENCODING}_pretrain_search_finetune/model_final.pt

# --train_file $FILE_PATH/hard_negatives_DL19/msmarco_train_triples \
# --dev_file $FILE_PATH/hard_negatives_DL19/msmarco_dev_triples \
# --doc_lookup_path resources/datasets/raw/msmarco-data/msmarco-docs-lookup.tsv.gz \
