GPU=$1

CUDA_VISIBLE_DEVICES=${GPU} python MAPLE.py \
    --target_dir "./results/MAPLE/epsinf_5shot_50_10examples_synsize5k" \
    --priv_file_path ../biorxiv_train_metadata_t5emb.csv \
    --train_file_path '../biorxiv_train_metadata_t5emb.csv' \
    --val_file_path ../biorxiv_valid_metadata.csv \
    --num_incontext 50 \
    --n_sample 10 \
    --PE_T 4 \
    --PE_num_syn 5000 \
    --PE_noise_multiplier 0.0
    
CUDA_VISIBLE_DEVICES=${GPU} python MAPLE.py \
    --target_dir "./results/MAPLE/epsinf_5shot_50_10examples_synsize10k" \
    --priv_file_path ../biorxiv_train_metadata_t5emb.csv \
    --train_file_path '../biorxiv_train_metadata_t5emb.csv' \
    --val_file_path ../biorxiv_valid_metadata.csv \
    --num_incontext 50 \
    --n_sample 10 \
    --PE_T 4 \
    --PE_num_syn 10000 \
    --PE_noise_multiplier 0.0
    
# Can add DP noise to test the DP case.