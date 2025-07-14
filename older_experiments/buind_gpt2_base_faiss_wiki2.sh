#!/bin/bash

# Set the GPU device (use GPU 6)
export CUDA_VISIBLE_DEVICES=6

# Activate your conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate neubig  # change if your env name is different

# Define the model and output path
MODEL=gpt2
MODEL_ID=gpt2_base
DSTORE_DIR=/data/nuri/knn-transformers/dstore_gpt2_base

# Create necessary directories
mkdir -p checkpoints/${MODEL_ID}
mkdir -p logs

# Run FAISS index builder
nohup python -u run_clm.py \
  --model_name_or_path ${MODEL} \
  --dataset_name wikitext \
  --dataset_config_name wikitext-103-raw-v1 \
  --output_dir checkpoints/${MODEL_ID} \
  --dstore_dir ${DSTORE_DIR} \
  --build_index \
  --dstore_size 247289 \
  > logs/build_index_${MODEL_ID}.log 2>&1 &