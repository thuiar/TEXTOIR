#!/usr/bin bash

for dataset in 'clinc'
do
    for known_cls_ratio in 0.75
    do
        for seed in 1
        do 
            python run.py \
            --dataset $dataset \
            --method 'CDACPlus' \
            --known_cls_ratio $known_cls_ratio \
            --seed $seed \
            --backbone 'bert_CDAC' \
            --config_file_name CDACPlus \
            --gpu_id '0' \
            --train \
            --save_results \
            --results_file_name 'results_CDACPlus.csv'
        done
    done
done

for dataset in 'clinc'
do
    for known_cls_ratio in 0.75
    do
        for seed in 6
        do 
            python run.py \
            --dataset $dataset \
            --method 'DeepAligned' \
            --setting 'semi_supervised' \
            --known_cls_ratio $known_cls_ratio \
            --seed $seed \
            --backbone 'bert' \
            --config_file_name 'DeepAligned' \
            --gpu_id '0' \
            --train \
            --save_results \
            --results_file_name 'results_DeepAligned.csv' 
        done
    done
done

for dataset in  'banking'  'clinc'
do
    for known_cls_ratio in 0.75
    do
        for seed in 8
        do 
            python run.py \
            --dataset $dataset \
            --method 'DTC_BERT' \
            --setting 'semi_supervised' \
            --known_cls_ratio $known_cls_ratio \
            --seed $seed \
            --backbone 'bert_DTC' \
            --config_file_name 'DTC_BERT' \
            --gpu_id '0' \
            --train \
            --save_results \
            --results_file_name 'results_DTC_BERT.csv' 
        done
    done
done
