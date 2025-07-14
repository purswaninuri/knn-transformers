#!/bin/bash

# 🧠 Activate environment and set GPU
source $(conda info --base)/etc/profile.d/conda.sh
conda activate neubig
export CUDA_VISIBLE_DEVICES=6

# 📁 Paths and config
MODEL=neulab/gpt2-finetuned-wikitext103
MODEL_ID=gpt2_finetuned_wiki
CHECKPOINT_DIR=checkpoints/${MODEL_ID}
DSTORE_DIR=/data/nuri/knn-transformers/full_set_neubig_hf_repo_7Jul25/dstore_gpt2_finetuned_wiki
LOG_DIR=logs

mkdir -p ${CHECKPOINT_DIR}
mkdir -p ${LOG_DIR}

########################################
# 🔍 1. Evaluate Base Model (no kNN/Reto)
########################################
echo "[Eval 1] Base LM only..."

nohup python -u /data/nuri/knn-transformers/run_clm.py \
  --model_name_or_path ${MODEL} \
  --dataset_name wikitext \
  --dataset_config_name wikitext-103-raw-v1 \
  --output_dir ${CHECKPOINT_DIR} \
  --do_eval \
  --eval_subset validation \
  > ${LOG_DIR}/eval_base_${MODEL_ID}.log 2>&1

wait
echo "[✅ Done] Base model evaluation."

####################################
# 🧠 2. Evaluate with kNN-LM
####################################
echo "[Eval 2] kNN-LM..."

nohup python -u /data/nuri/knn-transformers/run_clm.py \
  --model_name_or_path ${MODEL} \
  --dataset_name wikitext \
  --dataset_config_name wikitext-103-raw-v1 \
  --output_dir ${CHECKPOINT_DIR} \
  --do_eval \
  --eval_subset validation \
  --dstore_dir ${DSTORE_DIR} \
  --dstore_size 116988150 \
  --knn \
  --k 1024 \
  --lmbda 0.5 \
  --knn_temp 1.0 \
  > ${LOG_DIR}/eval_knn_${MODEL_ID}.log 2>&1

wait
echo "[✅ Done] kNN-LM evaluation."

######################################
# 🤖 3. Evaluate with RetoMaton
######################################
echo "[Eval 3] RetoMaton..."

nohup python -u /data/nuri/knn-transformers/run_clm.py \
  --model_name_or_path ${MODEL} \
  --dataset_name wikitext \
  --dataset_config_name wikitext-103-raw-v1 \
  --output_dir ${CHECKPOINT_DIR} \
  --do_eval \
  --eval_subset validation \
  --dstore_dir ${DSTORE_DIR} \
  --dstore_size 116988150 \
  --retomaton \
  --k 1024 \
  --lmbda 0.5 \
  --knn_temp 1.0 \
  --min_knns 9999999 \
  > ${LOG_DIR}/eval_retomaton_${MODEL_ID}.log 2>&1

echo "[✅ All Evaluations Complete] Check logs in '${LOG_DIR}'"
