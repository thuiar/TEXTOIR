#!/usr/bin bash

for dataset in   'banking' 'clinc' 'stackoverflow'
do
    for cluster_num_factor in 1.0 2.0 3.0 4.0
    do
        for seed in 0 1 2 3 4 5 6 7 8 9
        do
            python run.py \
            --dataset $dataset \
            --method 'SAE' \
            --setting 'unsupervised' \
            --seed $seed \
            --backbone 'sae' \
            --config_file_name 'SAE' \
            --cluster_num_factor $cluster_num_factor \
            --known_cls_ratio 0 \
            --gpu_id '1' \
            --save_results \
            --train \
            --save_model \
            --results_file_name 'results_SAE.csv'\
            --output_dir '/home/sharing/disk2/zhl/TEXTOIR/baseline_outputs'
        done
    done
done
