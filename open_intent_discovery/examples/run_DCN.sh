#!/usr/bin bash

for dataset in 'banking' 'clinc' 
do
    for seed in 0 1 2 3 4 5 6 7 8 9
    do
        python run.py \
        --dataset $dataset \
        --method 'DCN' \
        --setting 'unsupervised' \
        --seed $seed \
        --backbone 'sae' \
        --config_file_name 'DCN' \
        --gpu_id '0' \
        --save_results \
        --save_model \
        --train \
        --results_file_name 'results_DCN.csv'
    done
done
