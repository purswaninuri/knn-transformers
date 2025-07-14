#!/bin/bash

# 🧠 Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate neubig

# 🎯 Set CUDA device
export CUDA_VISIBLE_DEVICES=7

# 📁 Model and output setup
MODEL=meta-llama/Llama-3.2-1B
MODEL_ID=llama3.2_1B
DSTORE_DIR=./dstore_${MODEL_ID}
CHECKPOINT_DIR=./checkpoints/${MODEL_ID}
LOG_DIR=./logs

# 🧹 Remove old .npy and index files to avoid read-only errors
echo "[Cleanup] Removing old datastore and index files..."
rm -f ${DSTORE_DIR}/dstore_*_keys.npy
rm -f ${DSTORE_DIR}/dstore_*_vals.npy
rm -f ${DSTORE_DIR}/*.index

# 📂 Create necessary directories
mkdir -p ${DSTORE_DIR}
mkdir -p ${CHECKPOINT_DIR}
mkdir -p ${LOG_DIR}

# 🚀 Step 1: Save kNN datastore over a subset of training data
echo "[Step 1] Saving kNN datastore (subset of training set)..."
nohup python -u /data/nuri/knn-transformers/run_clm.py \
  --model_name_or_path ${MODEL} \
  --dataset_name wikitext \
  --dataset_config_name wikitext-103-raw-v1 \
  --eval_subset train \
  --max_eval_samples 10000 \
  --block_size 512 \
  --dstore_dir ${DSTORE_DIR} \
  --output_dir ${CHECKPOINT_DIR} \
  --save_knnlm_dstore \
  --fp16 \
  > ${LOG_DIR}/save_knn_${MODEL_ID}.log 2>&1

# ✅ Wait for completion
wait
echo "[Done] Datastore saved."
