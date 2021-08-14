#!/usr/bin bash


for dataset in 'stackoverflow'
do
    for known_cls_ratio in 0.75
    do
        for seed in 0 1 2 3 4 5 6 7 8 9
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
