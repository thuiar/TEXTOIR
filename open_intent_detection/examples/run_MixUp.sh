#!/usr/bin bash
for dataset in 'banking'
do
    for known_cls_ratio in 0.25
    do
        for labeled_ratio in 0.2
        do
            for seed in 0
            do
                python run.py \
                --dataset $dataset \
                --method 'MixUp' \
                --known_cls_ratio $known_cls_ratio \
                --labeled_ratio $labeled_ratio \
                --seed $seed \
                --backbone 'bert_mixup' \
                --config_file_name 'MixUp' \
                --loss_fct 'CrossEntropyLoss' \
                --gpu_id '0' \
                --train \
                --save_results \
                --results_file_name 'results_MixUp.csv'
            done
        done
    done
done
