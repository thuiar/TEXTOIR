#!/usr/bin bash

for dataset in   'banking' 'clinc' 'stackoverflow'
do
    for seed in 0 1 2 3 4 5 6 7 8 9
    do
        for cluster_num_factor in 1.0 2.0 3.0 4.0
        do
            python run.py \
            --dataset $dataset \
            --method 'DEC' \
            --setting 'unsupervised' \
            --cluster_num_factor $cluster_num_factor \
            --known_cls_ratio 0 \
            --seed $seed \
            --backbone 'sae' \
            --config_file_name 'DEC' \
            --gpu_id '1' \
            --save_results \
            --save_model \
            --train \
            --results_file_name 'results_DEC_test.csv' \
            --output_dir '/home/sharing/disk2/zhl/TEXTOIR/baseline_outputs_test'
        done
    done
done
