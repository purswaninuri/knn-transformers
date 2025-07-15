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

# 📐 Detect or set dstore size (this must match number of rows saved earlier)
echo "[Info] Detecting datastore size..."
DSTORE_SIZE=121198980  # <-- Update this to match actual rows in your .npy
echo "[Info] Detected dstore size: ${DSTORE_SIZE}"

# 📦 Build FAISS index
echo "[Step 2] Building FAISS index..."
nohup python -u /data/nuri/knn-transformers/run_clm.py \
  --model_name_or_path ${MODEL} \
  --dataset_name wikitext \
  --dataset_config_name wikitext-103-raw-v1 \
  --output_dir ${CHECKPOINT_DIR} \
  --dstore_dir ${DSTORE_DIR} \
  --build_index \
  --dstore_size ${DSTORE_SIZE} \
  --ncentroids 4096 \
  --code_size 64 \
  --probe 32 \
  > ${LOG_DIR}/build_index_${MODEL_ID}.log 2>&1

echo "[All done] FAISS index build complete. Check logs in ${LOG_DIR}."
