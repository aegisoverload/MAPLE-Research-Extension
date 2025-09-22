GPU=$1

### MAPLE part ###

# eps = inf
CUDA_VISIBLE_DEVICES=${GPU} python MAPLE.py \
    --target_dir "./results/MAPLE/epsinf_10shot_50_10examples" \
    --priv_file_path ../train_dataset_with_metadata_v4_t5emb.csv \
    --train_file_path '../train_dataset_with_metadata_v4_t5emb.csv' \
    --val_file_path ../val_dataset_with_metadata_v4.csv \
    --num_incontext 50 \
    --n_sample 10 \
    --PE_T 9 \
    --PE_noise_multiplier 0.0

# eps = 4.0
CUDA_VISIBLE_DEVICES=${GPU} python MAPLE.py \
    --target_dir "./results/MAPLE/epsinf_3shot_50_10examples" \
    --priv_file_path ../train_dataset_with_metadata_v4_t5emb.csv \
    --train_file_path '../openreview_train_metadata_aim_eps4.0_(1,9).csv' \
    --val_file_path ../val_dataset_with_metadata_v4.csv \
    --num_incontext 50 \
    --n_sample 10 \
    --PE_T 2 \
    --PE_noise_multiplier 1.876414555237503
    
# eps = 2.0
CUDA_VISIBLE_DEVICES=${GPU} python MAPLE.py \
    --target_dir "./results/MAPLE/epsinf_3shot_50_10examples" \
    --priv_file_path ../train_dataset_with_metadata_v4_t5emb.csv \
    --train_file_path '../openreview_train_metadata_aim_eps2.0_(1,9).csv' \
    --val_file_path ../val_dataset_with_metadata_v4.csv \
    --num_incontext 50 \
    --n_sample 10 \
    --PE_T 2 \
    --PE_noise_multiplier 3.4202124076206792
    
# eps = 1.0
CUDA_VISIBLE_DEVICES=${GPU} python MAPLE.py \
    --target_dir "./results/MAPLE/epsinf_3shot_50_10examples" \
    --priv_file_path ../train_dataset_with_metadata_v4_t5emb.csv \
    --train_file_path '../openreview_train_metadata_aim_eps1.0_(1,9).csv' \
    --val_file_path ../val_dataset_with_metadata_v4.csv \
    --num_incontext 50 \
    --n_sample 10 \
    --PE_T 2 \
    --PE_noise_multiplier 6.325387702318415


### PE part ###
# # eps=inf,
CUDA_VISIBLE_DEVICES=${GPU} python PE.py \
    --target_dir "./results_epsinf_10shot_PE" \
    --priv_file_path ../train_dataset_with_metadata_v4_t5emb.csv \
    --train_file_path '../train_dataset_with_metadata_v4_t5emb.csv' \
    --val_file_path ../val_dataset_with_metadata_v4.csv \
    --PE_T 9 \
    --PE_noise_multiplier 0.0

# eps=4.0,
CUDA_VISIBLE_DEVICES=${GPU} python PE.py \
    --target_dir "./results_eps4.0_10shot_PE" \
    --priv_file_path ../train_dataset_with_metadata_v4_t5emb.csv \
    --train_file_path '../train_dataset_with_metadata_v4_t5emb.csv' \
    --val_file_path ../val_dataset_with_metadata_v4.csv \
    --PE_T 9 \
    --PE_noise_multiplier 3.000778920378073


# eps=2.0,
CUDA_VISIBLE_DEVICES=${GPU} python PE.py \
    --target_dir "./results_eps2.0_10shot_PE" \
    --priv_file_path ../train_dataset_with_metadata_v4_t5emb.csv \
    --train_file_path '../train_dataset_with_metadata_v4_t5emb.csv' \
    --val_file_path ../val_dataset_with_metadata_v4.csv \
    --PE_T 9 \
    --PE_noise_multiplier 5.418702035826035


# eps=1.0,
CUDA_VISIBLE_DEVICES=${GPU} python PE.py \
    --target_dir "./results_eps1.0_10shot_PE" \
    --priv_file_path ../train_dataset_with_metadata_v4_t5emb.csv \
    --train_file_path '../train_dataset_with_metadata_v4_t5emb.csv' \
    --val_file_path ../val_dataset_with_metadata_v4.csv \
    --PE_T 9 \
    --PE_noise_multiplier 9.935410662820507

