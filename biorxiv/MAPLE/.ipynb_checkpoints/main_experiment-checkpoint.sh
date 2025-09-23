GPU=$1


###### Main Experiments #######

# MAPLE, eps=\infty
CUDA_VISIBLE_DEVICES=${GPU} python MAPLE.py \
    --target_dir "./results/MAPLE/epsinf_10shot_50_10examples" \
    --priv_file_path ../biorxiv_train_metadata_t5emb.csv \
    --train_file_path '../biorxiv_train_metadata_t5emb.csv' \
    --val_file_path ../biorxiv_valid_metadata.csv \
    --num_incontext 50 \
    --n_sample 10 \
    --PE_T 9 \
    --PE_noise_multiplier 0.0
    
# AugPE, eps=\infty
CUDA_VISIBLE_DEVICES=${GPU} python PE.py \
    --target_dir "./results/AugPE/epsinf_10shot" \
    --priv_file_path ../biorxiv_train_metadata_t5emb.csv \
    --train_file_path '../biorxiv_train_metadata_t5emb.csv' \
    --val_file_path ../biorxiv_valid_metadata.csv \
    --PE_T 9 \
    --PE_noise_multiplier 0.0
    
# MAPLE, eps=4.0, noise calibrated by zCDP (suboptimal).
CUDA_VISIBLE_DEVICES=${GPU} python MAPLE.py \
    --target_dir "./results/MAPLE/eps4.0_5shot_50_10examples" \
    --priv_file_path ../biorxiv_train_metadata_t5emb.csv \
    --train_file_path '../biorxiv_train_metadata_aim_eps4.0_(1,9).csv' \
    --val_file_path ../biorxiv_valid_metadata.csv \
    --num_incontext 50 \
    --n_sample 10 \
    --PE_T 4 \
    --PE_noise_multiplier 2.5787580697142025
    
# AugPE, eps=4.0, noise calibrated by GDP directly.
CUDA_VISIBLE_DEVICES=${GPU} python PE.py \
    --target_dir "./results/AugPE/eps4.0_10shot" \
    --priv_file_path ../biorxiv_train_metadata_t5emb.csv \
    --train_file_path '../biorxiv_train_metadata_t5emb.csv' \
    --val_file_path ../biorxiv_valid_metadata.csv \
    --PE_T 9 \
    --PE_noise_multiplier 3.214362874574832
    
# MAPLE, eps=2.0, noise calibrated by zCDP (suboptimal).
CUDA_VISIBLE_DEVICES=${GPU} python MAPLE.py \
    --target_dir "./results/MAPLE/eps2.0_5shot_50_10examples" \
    --priv_file_path ../biorxiv_train_metadata_t5emb.csv \
    --train_file_path '../biorxiv_train_metadata_aim_eps2.0_(1,9).csv' \
    --val_file_path ../biorxiv_valid_metadata.csv \
    --num_incontext 50 \
    --n_sample 10 \
    --PE_T 4 \
    --PE_noise_multiplier 4.7482491352519505
    
# AugPE, eps=2.0, noise calibrated by GDP directly.
CUDA_VISIBLE_DEVICES=${GPU} python PE.py \
    --target_dir "./results/AugPE/eps2.0_10shot" \
    --priv_file_path ../biorxiv_train_metadata_t5emb.csv \
    --train_file_path '../biorxiv_train_metadata_t5emb.csv' \
    --val_file_path ../biorxiv_valid_metadata.csv \
    --PE_T 9 \
    --PE_noise_multiplier 5.872135064401613
    
# MAPLE, eps=1.0, noise calibrated by zCDP (suboptimal).
CUDA_VISIBLE_DEVICES=${GPU} python MAPLE.py \
    --target_dir "./results/MAPLE/eps1.0_3shot_50_10examples" \
    --priv_file_path ../biorxiv_train_metadata_t5emb.csv \
    --train_file_path '../biorxiv_train_metadata_aim_eps1.0_(1,9).csv' \
    --val_file_path ../biorxiv_valid_metadata.csv \
    --num_incontext 50 \
    --n_sample 10 \
    --PE_T 2 \
    --PE_noise_multiplier 6.869194837509232
    
# AugPE, eps=1.0, noise calibrated by GDP directly.
CUDA_VISIBLE_DEVICES=${GPU} python PE.py \
    --target_dir "./results/AugPE/eps1.0_10shot" \
    --priv_file_path ../biorxiv_train_metadata_t5emb.csv \
    --train_file_path '../biorxiv_train_metadata_t5emb.csv' \
    --val_file_path ../biorxiv_valid_metadata.csv \
    --PE_T 9 \
    --PE_noise_multiplier 10.889585799108582
    
###### Ablation study #######
# eps = inf

# AugPE+E
CUDA_VISIBLE_DEVICES=${GPU} python PE_nometadata.py \
    --target_dir "./results/AugPE+E/epsinf_10shot" \
    --priv_file_path ../biorxiv_train_metadata_t5emb.csv \
    --train_file_path '../biorxiv_train_metadata_t5emb.csv' \
    --val_file_path ../biorxiv_valid_metadata.csv \
    --PE_T 9 \
    --PE_noise_multiplier 0.0
    
# AugPE+M
CUDA_VISIBLE_DEVICES=${GPU} python PE_noexamples.py \
    --target_dir "./results/AugPE+M/epsinf_10shot" \
    --priv_file_path ../biorxiv_train_metadata_t5emb.csv \
    --train_file_path '../biorxiv_train_metadata_t5emb.csv' \
    --val_file_path ../biorxiv_valid_metadata.csv \
    --PE_T 9 \
    --PE_noise_multiplier 0.0