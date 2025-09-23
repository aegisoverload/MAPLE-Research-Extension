GPU=$1

# ==============MAPLE=========================
# eps=inf,
CUDA_VISIBLE_DEVICES=${GPU} python PE_M+E.py \
    --target_dir "./results/MAPLE/epsinf_10shot_50_10examples" \
    --priv_file_path ../PMC_V2_train_paired_t5emb.csv \
    --train_file_path '../PMC_V2_train_paired.csv' \
    --val_file_path ../PMC_V2_valid_paired.csv \
    --num_incontext 50 \
    --n_sample 10 \
    --PE_T 9 \
    --PE_noise_multiplier 0.0

# eps=4.0,
CUDA_VISIBLE_DEVICES=${GPU} python PE_M+E.py \
    --target_dir "./results/MAPLE/eps4.0_5shot_50_10examples" \
    --priv_file_path ../PMC_V2_train_paired_t5emb.csv \
    --train_file_path '../pmc_train_metadata_aim_eps4.0_(1,9).csv' \
    --val_file_path ../PMC_V2_valid_paired.csv \
    --num_incontext 50 \
    --n_sample 10 \
    --PE_T 4 \
    --PE_noise_multiplier 2.8298179991433448

# eps=inf,
CUDA_VISIBLE_DEVICES=${GPU} python PE_M+E.py \
    --target_dir "./results/MAPLE/eps2.0_5shot_50_10examples" \
    --priv_file_path ../PMC_V2_train_paired_t5emb.csv \
    --train_file_path '../pmc_train_metadata_aim_eps2.0_(1,9).csv' \
    --val_file_path ../PMC_V2_valid_paired.csv \
    --num_incontext 50 \
    --n_sample 10 \
    --PE_T 4 \
    --PE_noise_multiplier 5.279735441365229

# eps=inf,
CUDA_VISIBLE_DEVICES=${GPU} python PE_M+E.py \
    --target_dir "./results/MAPLE/eps1.0_5shot_50_10examples" \
    --priv_file_path ../PMC_V2_train_paired_t5emb.csv \
    --train_file_path '../pmc_train_metadata_aim_eps1.0_(1,9).csv' \
    --val_file_path ../PMC_V2_valid_paired.csv \
    --num_incontext 50 \
    --n_sample 10 \
    --PE_T 4 \
    --PE_noise_multiplier 9.982961895455777

# ==============AugPE=========================
# eps=inf,
CUDA_VISIBLE_DEVICES=${GPU} python PE.py \
    --target_dir "./results/AugPE/epsinf_10shot_50_10examples" \
    --priv_file_path ../PMC_V2_train_paired_t5emb.csv \
    --train_file_path '../PMC_V2_train_paired.csv' \
    --val_file_path ../PMC_V2_valid_paired.csv \
    --num_incontext 50 \
    --n_sample 10 \
    --PE_T 9 \
    --PE_noise_multiplier 0.0


# eps=4.0,
CUDA_VISIBLE_DEVICES=${GPU} python PE.py \
    --target_dir "./results/AugPE/eps4.0_10shot_50_10examples" \
    --priv_file_path ../PMC_V2_train_paired_t5emb.csv \
    --train_file_path '../PMC_V2_train_paired.csv' \
    --val_file_path ../PMC_V2_valid_paired.csv \
    --num_incontext 50 \
    --n_sample 10 \
    --PE_T 9 \
    --PE_noise_multiplier 3.5575676151673346


# eps=2.0,
CUDA_VISIBLE_DEVICES=${GPU} python PE.py \
    --target_dir "./results/AugPE/eps2.0_10shot_50_10examples" \
    --priv_file_path ../PMC_V2_train_paired_t5emb.csv \
    --train_file_path '../PMC_V2_train_paired.csv' \
    --val_file_path ../PMC_V2_valid_paired.csv \
    --num_incontext 50 \
    --n_sample 10 \
    --PE_T 9 \
    --PE_noise_multiplier 6.597507733016656


# eps=1.0,
CUDA_VISIBLE_DEVICES=${GPU} python PE.py \
    --target_dir "./results/AugPE/eps1.0_10shot_50_10examples" \
    --priv_file_path ../PMC_V2_train_paired_t5emb.csv \
    --train_file_path '../PMC_V2_train_paired.csv' \
    --val_file_path ../PMC_V2_valid_paired.csv \
    --num_incontext 50 \
    --n_sample 10 \
    --PE_T 9 \
    --PE_noise_multiplier 12.408984498128014
