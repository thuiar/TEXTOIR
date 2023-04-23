#!/usr/bin bash

for dataset in 'banking' 'oos' 'stackoverflow'
do
    for known_cls_ratio in 0.25 0.5 0.75
    do
        for labeled_ratio in 1.0
        do
            for seed in 0 1 2 3 4 5 6 7 8 9
            do 
                python run.py \
                --dataset $dataset \
                --method 'KCL' \
                --known_cls_ratio $known_cls_ratio \
                --labeled_ratio $labeled_ratio \
                --seed $seed \
                --backbone 'bert_kcl' \
                --config_file_name 'KCL' \
                --loss_fct 'KCLLoss' \
                --gpu_id '0' \
                --save_results \
                --train \
                --results_file_name 'KCL_freeze_no_earlystop.csv' 
            done
        done
    done
done
