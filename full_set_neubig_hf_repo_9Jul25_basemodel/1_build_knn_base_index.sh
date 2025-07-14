#!/bin/bash

# 🧠 Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate neubig

# 🎯 Set CUDA device
export CUDA_VISIBLE_DEVICES=6

# 📁 Model and output setup
MODEL=gpt2
MODEL_ID=gpt2_base
DSTORE_DIR=./dstore_${MODEL_ID}
CHECKPOINT_DIR=./checkpoints/${MODEL_ID}
LOG_DIR=./logs

# 📐 Auto-detect dstore size (number of rows)
echo "[Info] Detecting datastore size..."
DSTORE_SIZE=116988150
echo "[Info] Detected dstore size: ${DSTORE_SIZE}"

# 🧠 Step 2: Build FAISS index
echo "[Step 2] Building FAISS index..."
nohup python -u /data/nuri/knn-transformers/run_clm.py \
--model_name_or_path ${MODEL} \
--dataset_name wikitext \
--dataset_config_name wikitext-103-raw-v1 \
--output_dir ${CHECKPOINT_DIR} \
--dstore_dir ${DSTORE_DIR} \
--build_index \
--dstore_size ${DSTORE_SIZE} \
> ${LOG_DIR}/build_index_${MODEL_ID}.log 2>&1

echo "[All done] Check logs in ${LOG_DIR}."
