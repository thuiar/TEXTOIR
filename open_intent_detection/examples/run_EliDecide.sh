#!/usr/bin bash

for dataset in 'banking' 'oos' 'stackoverflow'
do
    for known_cls_ratio  in 0.25 0.5 0.75
    do
        for labeled_ratio in 1.0
        do
            for seed in 0 1 2 3 4
            do
                python run.py \
                --dataset $dataset \
                --method 'EliDecide' \
                --known_cls_ratio $known_cls_ratio \
                --labeled_ratio $labeled_ratio \
                --seed $seed \
                --backbone 'bert_con' \
                --config_file_name 'EliDecide' \
                --loss_fct 'CrossEntropyLoss' \
                --gpu_id '1' \
                --train \
                --results_file_name 'results.csv' \
                --output_dir '/home/sharing/disk1/disk1/zouyuetian/TEXTOIR' \
                --save_results \
                --save_model \
                --pretrain
            done
        done
    done
done
