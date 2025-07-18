#!/bin/bash

# 🧠 Activate environment and set GPU
source $(conda info --base)/etc/profile.d/conda.sh
conda activate neubig
export CUDA_VISIBLE_DEVICES=7

# 📁 Paths and config
MODEL=meta-llama/Llama-3.2-1B
MODEL_ID=llama3.2_1B
CHECKPOINT_DIR=checkpoints/${MODEL_ID}
DSTORE_DIR=/data/nuri/knn-transformers/dstore_${MODEL_ID}
LOG_DIR=logs

mkdir -p ${CHECKPOINT_DIR}
mkdir -p ${LOG_DIR}

####################################
# 🧠 1. Evaluate with kNN-LM (GPU + IVF+PQ)
####################################
echo "[Eval 1] kNN-LM using pre-built IVF+PQ FAISS index on GPU..."

nohup python -u /data/nuri/knn-transformers/run_clm.py \
  --model_name_or_path ${MODEL} \
  --dataset_name wikitext \
  --dataset_config_name wikitext-103-raw-v1 \
  --output_dir ${CHECKPOINT_DIR} \
  --do_eval \
  --eval_subset validation \
  --knn \
  --dstore_dir ${DSTORE_DIR} \
  --dstore_size 121198980 \
  --knn_gpu True \
  --ncentroids 16384 \
  --code_size 64 \
  --probe 32 \
  > ${LOG_DIR}/eval_knn_${MODEL_ID}_ivfpq.log 2>&1

wait
echo "[✅ Done] kNN-LM evaluation."

######################################
# 🤖 2. Evaluate with RetoMaton (GPU + IVF+PQ)
######################################
echo "[Eval 2] RetoMaton using pre-built IVF+PQ FAISS index on GPU..."

nohup python -u /data/nuri/knn-transformers/run_clm.py \
  --model_name_or_path ${MODEL} \
  --dataset_name wikitext \
  --dataset_config_name wikitext-103-raw-v1 \
  --output_dir ${CHECKPOINT_DIR} \
  --do_eval \
  --eval_subset validation \
  --retomaton \
  --dstore_dir ${DSTORE_DIR} \
  --dstore_size 121198980 \
  --knn_gpu True \
  --ncentroids 16384 \
  --code_size 64 \
  --probe 32 \
  --min_knns 9999999 \
  > ${LOG_DIR}/eval_retomaton_${MODEL_ID}_ivfpq.log 2>&1

echo "[✅ All Evaluations Complete] Check logs in '${LOG_DIR}'"
