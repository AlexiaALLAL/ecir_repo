#!/bin/sh

SEED=42  # Random seed for reproducible training

#  The --encoding flag supports formats like pq, url, atomic, summary.

# Run DDRO training pipeline
python src/utils/run_training_pipeline.py \
    --encoding nc8_cs256 \
    --encoding_name_exact rq-module_nc8_cs256 \
    --dataset msmarco \
    --dedup true \
    --scale top_300k \
    --seed "$SEED" \
    # --resume_stage "search_pretrain" \
    # --train_subset_size 10000 \

    # --resume_epoch 14 \
    # --resume_from_checkpoint "outputs/nq/t5_128_10_top_300k_pq_pretrain_search_/model_epoch14.pt" \
    # --resume_wandb_run_id "thc5gfri" \

