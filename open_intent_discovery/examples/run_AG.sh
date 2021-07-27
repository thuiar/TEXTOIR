#!/usr/bin bash

for dataset in 'banking' 'clinc' 
do
    for seed in 0 1 2 3 4 5 6 7 8 9
    do
        python run.py \
        --dataset $dataset \
        --method 'AG' \
        --setting 'unsupervised' \
        --seed $seed \
        --backbone 'glove' \
        --config_file_name 'AG' \
        --gpu_id '0' \
        --train \
        --save_results \
        --results_file_name 'results_AG.csv'
    done
done
