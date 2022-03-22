#!/usr/bin bash

for dataset in 'banking' 'oos' 'stackoverflow'
do
    for known_cls_ratio in 0.25 0.5 0.75
    do
        for labeled_ratio in 0.2 0.4 0.6 0.8 1.0
        do
            for seed in 0 1 2 3 4 5 6 7 8 9
            do 
                python run.py \
                --dataset $dataset \
                --method 'MixUp' \
                --known_cls_ratio $known_cls_ratio \
                --labeled_ratio $labeled_ratio \
                --seed $seed \
                --backbone 'bert_mixup' \
                --config_file_name 'K+1-way' \
                --loss_fct 'CrossEntropyLoss' \
                --gpu_id '0' \
                --train \
                --save_results \
                --results_file_name 'results_K+1-way.csv'
            done
        done
    done
done
