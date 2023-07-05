#!/usr/bin bash

for dataset in 'banking' 'clinc' 'stackoverflow'
do
    for known_cls_ratio in    0.25 0.5 0.75
    do
        for cluster_num_factor in   1.0 2.0 3.0 4.0
        do
            for seed in   0 1 2 3 4 5 6 7 8 9
            do 
                python run.py \
                --dataset $dataset \
                --method 'KCL_BERT' \
                --setting 'semi_supervised' \
                --known_cls_ratio $known_cls_ratio \
                --cluster_num_factor $cluster_num_factor \
                --seed $seed \
                --backbone 'bert_KCL' \
                --config_file_name 'KCL_BERT' \
                --gpu_id '1' \
                --train \
                --save_results \
                --results_file_name 'results_KCL_BERT_new.csv' 
            done
        done
    done
done
 
 