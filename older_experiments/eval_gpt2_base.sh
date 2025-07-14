#!/bin/bash

# Detect and source conda environment setup
source $(conda info --base)/etc/profile.d/conda.sh
conda activate neubig  # replace with your actual environment name

# Create logs directory if it doesn't exist
mkdir -p logs

# Run the evaluation and save logs
nohup python run_clm.py \
  --model_name_or_path gpt2 \
  --dataset_name wikitext \
  --dataset_config_name wikitext-103-raw-v1 \
  --do_eval \
  > logs/eval_gpt2_base.log 2>&1 &
