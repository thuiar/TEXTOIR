#!/usr/bin bash

for dataset in 'banking' 
do
    for known_cls_ratio in 0.25
    do
        for labeled_ratio in 1.0
        do 
            for seed in 0
            do
                python run.py \
                --dataset $dataset \
                --method 'DeepUnk' \
                --known_cls_ratio $known_cls_ratio \
                --labeled_ratio $labeled_ratio \
                --seed $seed \
                --backbone 'bert_norm' \
                --config_file_name 'DeepUnk' \
                --loss_fct 'CosineFaceLoss' \
                --gpu_id '0' \
                --train \
                --save_results \
                --results_file_name 'results_DeepUnk.csv'
            done
        done
    done
done
