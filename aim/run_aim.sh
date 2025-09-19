#### biorxiv ####

CUDA_VISIBLE_DEVICES=$1 python workplace_biorxiv.py --data_path ../biorxiv/biorxiv_train_metadata_t5emb.csv --save_path [SAVE_PATH] --rho [RHO FOR AIM, CHECK THE PRIVACY_ACCOUNTING PART]

#### openreview ####

CUDA_VISIBLE_DEVICES=$1 python workplace_openreview.py --data_path ../openreview/train_dataset_with_metadata_v4_t5emb.csv --save_path [SAVE_PATH] --rho [RHO FOR AIM, CHECK THE PRIVACY_ACCOUNTING PART]