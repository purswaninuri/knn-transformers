#!/bin/bash

# 🧠 Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate neubig

# 🎯 Set CUDA device
export CUDA_VISIBLE_DEVICES=6

# 📁 Model and output setup
MODEL=neulab/gpt2-finetuned-wikitext103
MODEL_ID=gpt2_finetuned_wiki
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

# 🚀 Step 1: Save kNN datastore (keys and vals)
echo "[Step 1] Saving kNN datastore..."
nohup python -u /data/nuri/knn-transformers/run_clm.py \
  --model_name_or_path ${MODEL} \
  --dataset_name wikitext \
  --dataset_config_name wikitext-103-raw-v1 \
  --do_eval \
  --eval_subset train \
  --output_dir ${CHECKPOINT_DIR} \
  --dstore_dir ${DSTORE_DIR} \
  --save_knnlm_dstore \
  > ${LOG_DIR}/save_knn_${MODEL_ID}.log 2>&1

# ✅ Wait for completion
wait
echo "[Done] Datastore saved."