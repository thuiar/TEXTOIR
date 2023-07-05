#!/usr/bin bash

for dataset in 'banking' 'clinc' 'stackoverflow'
do  
    for cluster_num_factor in 1.0 2.0 3.0 4.0
    do
        for seed in  0 1 2 3 4 5 6 7 8 9
        do
            python run.py \
            --dataset $dataset \
            --method 'CC' \
            --setting 'unsupervised' \
            --seed $seed \
            --cluster_num_factor $cluster_num_factor \
            --known_cls_ratio 0 \
            --backbone 'bert_CC' \
            --config_file_name 'CC' \
            --gpu_id '1' \
            --save_results \
            --train \
            --save_model \
            --results_file_name 'results_CC.csv'
        done
    done
done