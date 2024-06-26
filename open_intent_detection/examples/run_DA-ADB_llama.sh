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
                --method 'DA-ADB_llama' \
                --known_cls_ratio $known_cls_ratio \
                --labeled_ratio $labeled_ratio \
                --seed $seed \
                --backbone 'llama_disaware' \
                --config_file_name 'DA-ADB_llama' \
                --loss_fct 'CrossEntropyLoss' \
                --gpu_id '0' \
                --train \
                --pretrain \
                --results_file_name 'results_DA-ADB_llama.csv' \
                --save_results
            done
        done
    done
done

