#!/bin/bash

# Detect and source conda environment setup
source $(conda info --base)/etc/profile.d/conda.sh
conda activate neubig  # Replace with your actual environment name

# Set CUDA device
export CUDA_VISIBLE_DEVICES=6

# Define model and datastore path
MODEL=neulab/gpt2-finetuned-wikitext103
MODEL_ID=gpt2_ft
DSTORE_DIR=/data/nuri/knn-transformers/dstore_gpt2_finetuned_wiki
DSTORE_SIZE=247289

# Create logs directory if it doesn't exist
mkdir -p logs

# Run kNN-LM evaluation and save logs
nohup python -u run_clm.py \
  --model_name_or_path ${MODEL} \
  --dataset_name wikitext \
  --dataset_config_name wikitext-103-raw-v1 \
  --output_dir checkpoints/${MODEL_ID} \
  --do_eval \
  --eval_subset validation \
  --dstore_dir ${DSTORE_DIR} \
  --dstore_size ${DSTORE_SIZE} \
  --knn \
  > logs/eval_knnlm_${MODEL_ID}.log 2>&1 &