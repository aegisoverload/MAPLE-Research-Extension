GPU=$1

cls_batch_size=32

iteration=$2
target_folder=$3

python ./utility_eval/modify_label.py --target_dir ${target_folder} --iteration ${iteration}

result_folder="${target_folder}/${iteration}"
train_file="${item}/PE_iter_${iteration}_augpe.csv"

num_train_epochs=10
max_seq_length=512
model="roberta-base"
min_token_threshold=100
item=${result_folder}

# remember to change label part properly.

## calculate acc 
for seed in 42
do
for label in "label1" "label2"
do
echo $train_file
if [ -e "$train_file" ]; then
    echo "$train_file does exist."
    output_dir=${item}/${label}_ep${num_train_epochs}_${model}_seq${max_seq_length}_seed${seed}_emb1024/
    if [ -e "${output_dir}test_${num_train_epochs}.0_results.json" ]; then
        echo "${output_dir}test_${num_train_epochs}.0_results.json  does exist. -- SKIP running classification"
    else
        echo "${output_dir}test_${num_train_epochs}.0_results.json  does not exist. -- RUN running classification"
        CUDA_VISIBLE_DEVICES=${GPU} python utility_eval/run_classification.py \
            --report_to none --clean_dataset  --min_token_threshold ${min_token_threshold} \
            --model_name_or_path  ${model} \
            --output_dir ${output_dir} \
            --train_file ${train_file} \
            --validation_file data/openreview/iclr23_reviews_val.csv \
            --test_file data/openreview/iclr23_reviews_test.csv \
            --do_train --do_eval --do_predict --max_seq_length ${max_seq_length} --per_device_train_batch_size ${cls_batch_size} --per_device_eval_batch_size ${cls_batch_size} \
            --learning_rate 3e-5 --num_train_epochs ${num_train_epochs} \
            --overwrite_output_dir --overwrite_cache \
            --save_strategy epoch --save_total_limit 1 --load_best_model_at_end \
            --logging_strategy epoch \
            --seed ${seed} \
            --metric_for_best_model accuracy_all --greater_is_better True \
            --eval_strategy epoch --label_column_name ${label}
    fi
else
    echo "$train_file does not exist."
fi
done
done



