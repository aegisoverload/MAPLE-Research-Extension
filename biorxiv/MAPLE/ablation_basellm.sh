GPU=$1

# This one may have OOM issue... but it is what it is :(
CUDA_VISIBLE_DEVICES=${GPU} python MAPLE.py \
    --target_dir "./results/MAPLE/epsinf_5shot_50_10examples_qwen14b" \
    --priv_file_path ../biorxiv_train_metadata_t5emb.csv \
    --train_file_path '../biorxiv_train_metadata_t5emb.csv' \
    --val_file_path ../biorxiv_valid_metadata.csv \
    --model_id Qwen/Qwen2.5-14B-Instruct \
    --num_incontext 50 \
    --n_sample 10 \
    --PE_T 4 \
    --PE_noise_multiplier 0.0

CUDA_VISIBLE_DEVICES=${GPU} python MAPLE.py \
    --target_dir "../results/MAPLE/epsinf_5shot_50_10examples_qwen3b" \
    --priv_file_path ../biorxiv_train_metadata_t5emb.csv \
    --train_file_path '../biorxiv_train_metadata_t5emb.csv' \
    --val_file_path ../biorxiv_valid_metadata.csv \
    --model_id Qwen/Qwen2.5-3B-Instruct \
    --num_incontext 50 \
    --n_sample 10 \
    --PE_T 4 \
    --PE_noise_multiplier 0.0

CUDA_VISIBLE_DEVICES=${GPU} python MAPLE.py \
    --target_dir "./results/MAPLE/epsinf_5shot_50_10examples_qwen1b" \
    --priv_file_path ../biorxiv_train_metadata_t5emb.csv \
    --train_file_path '../biorxiv_train_metadata_t5emb.csv' \
    --val_file_path ../biorxiv_valid_metadata.csv \
    --model_id Qwen/Qwen2.5-1B-Instruct \
    --num_incontext 50 \
    --n_sample 10 \
    --PE_T 4 \
    --PE_noise_multiplier 0.0

CUDA_VISIBLE_DEVICES=${GPU} python MAPLE.py \
    --target_dir "./results/MAPLE/epsinf_5shot_50_10examples_gemma1b" \
    --priv_file_path ../biorxiv_train_metadata_t5emb.csv \
    --train_file_path '../biorxiv_train_metadata_t5emb.csv' \
    --val_file_path ../biorxiv_valid_metadata.csv \
    --model_id google/gemma-3-1b-it \
    --num_incontext 50 \
    --n_sample 10 \
    --PE_T 4 \
    --PE_noise_multiplier 0.0

CUDA_VISIBLE_DEVICES=${GPU} python MAPLE.py \
    --target_dir "./results/MAPLE/epsinf_5shot_50_10examples_gemma4b" \
    --priv_file_path ../biorxiv_train_metadata_t5emb.csv \
    --train_file_path '../biorxiv_train_metadata_t5emb.csv' \
    --val_file_path ../biorxiv_valid_metadata.csv \
    --model_id google/gemma-3-4b-it \
    --num_incontext 50 \
    --n_sample 10 \
    --PE_T 4 \
    --PE_noise_multiplier 0.0

CUDA_VISIBLE_DEVICES=${GPU} python MAPLE.py \
    --target_dir "./results/MAPLE/epsinf_5shot_50_10examples_gemma12b" \
    --priv_file_path ../biorxiv_train_metadata_t5emb.csv \
    --train_file_path '../biorxiv_train_metadata_t5emb.csv' \
    --val_file_path ../biorxiv_valid_metadata.csv \
    --model_id google/gemma-3-12b-it \
    --num_incontext 50 \
    --n_sample 10 \
    --PE_T 4 \
    --PE_noise_multiplier 0.0
