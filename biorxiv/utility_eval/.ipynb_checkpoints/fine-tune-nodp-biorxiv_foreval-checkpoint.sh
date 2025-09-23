#!/bin/bash
model=prajjwal1/bert-small


# Fixed batch size configuration
# actual batch size = GRAD_ACCUM * #GPU * per_device_train_batch_size
# GRAD_ACCUM=2 #nproc_per_node=4
# GRAD_ACCUM=4 #nproc_per_node=2
GRAD_ACCUM=8 #nproc_per_node=1

# Base output directory 
BASE_OUTPUT_DIR="../MAPLE/results/MAPLE/epsinf_5shot_50_10examples/bert-small"
TRAIN_DATA_PATH="../MAPLE/results/MAPLE/epsinf_5shot_50_10examples/PE_iter_3.csv"
VALID_DATA_PATH="../biorxiv_valid_metadata.csv"
TEST_DATA_PATH="../biorxiv_test_metadata.csv"
# We follow Yue et al. 2023 similar to Aug-PE for epochs and lr

# This is non-dp case
CUDA_VISIBLE_DEVICES=$1 python -m torch.distributed.run --nproc_per_node 1 fine-tune-nodp-biorxiv_foreval.py \
--train_data_path ${TRAIN_DATA_PATH} \
--valid_data_path ${VALID_DATA_PATH} \
--output_dir ${BASE_OUTPUT_DIR} \
--model_name ${model} \
--sequence_len 512 \
--per_device_train_batch_size 64 \
--gradient_accumulation_steps ${GRAD_ACCUM} \
--eval_steps 10 \
--log_level info \
--per_device_eval_batch_size 64 \
--eval_accumulation_steps 1 \
--seed 42 \
--prediction_loss_only \
--weight_decay 0.01 \
--remove_unused_columns False \
--num_train_epochs 50 \
--logging_steps 10 \
--max_grad_norm 0 \
--lr_scheduler_type constant \
--learning_rate 3e-4 \
--disable_tqdm True \
--dataloader_num_workers 8 \
--label_names labels 

# compute NTP
CUDA_VISIBLE_DEVICES=$1 python ntp_eval.py \
--target_dir $BASE_OUTPUT_DIR \
--tst_file_path $TEST_DATA_PATH \
--save_path $BASE_OUTPUT_DIR


