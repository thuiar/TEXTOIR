#!/usr/bin bash
 

for seed in 0 1 2 3 4 5 6 7 8 9
do
    for dataset in  'banking' 'clinc' 'stackoverflow'
    do  
        for cluster_num_factor in 1.0 2.0 3.0 4.0
        do
            python run.py \
            --dataset $dataset \
            --method 'SCCL' \
            --setting 'unsupervised' \
            --seed $seed \
            --backbone 'bert_SCCL' \
            --config_file_name 'SCCL' \
            --gpu_id '0' \
            --cluster_num_factor $cluster_num_factor \
            --known_cls_ratio 0 \
            --save_results \
            --train \
            --results_file_name 'results_SCCL.csv'
        done
    done
done