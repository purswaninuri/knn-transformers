#!/bin/bash

# Source conda setup and activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate neubig  # Change to your actual environment name

# Set GPU device (optional: remove or modify if not needed)
export CUDA_VISIBLE_DEVICES=6

# Create logs and datastore directory if they don’t exist
mkdir -p logs
mkdir -p dstore_gpt2_base

# Run evaluation and save kNN-LM datastore
nohup python run_clm.py \
  --model_name_or_path gpt2 \
  --dataset_name wikitext \
  --dataset_config_name wikitext-103-raw-v1 \
  --do_eval \
  --save_knnlm_dstore \
  --dstore_dir ./dstore_gpt2_base \
  --dstore_size 247289 \
  > logs/save_knn_gpt2_base.log 2>&1 &
