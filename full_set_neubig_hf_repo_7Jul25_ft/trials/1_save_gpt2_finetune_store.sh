# Source conda setup and activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate neubig  # Change to your actual environment name

# Set GPU device (optional: remove or modify if not needed)
export CUDA_VISIBLE_DEVICES=6

MODEL=neulab/gpt2-finetuned-wikitext103

mkdir -p checkpoints
mkdir -p logs

nohup python -u /data/nuri/knn-transformers/run_clm.py\
  --model_name_or_path ${MODEL} \
  --dataset_name wikitext --dataset_config_name wikitext-103-raw-v1 \
  --do_eval --eval_subset train \
  --output_dir checkpoints/${MODEL} \
  --dstore_dir checkpoints/${MODEL} \
  --save_knnlm_dstore \
  > logs/save_knn_gpt2_finetuned_wiki.log 2>&1 &