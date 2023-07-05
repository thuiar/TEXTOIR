#!/usr/bin bash

for dataset in  'banking' 'clinc' 'stackoverflow'
do
    for known_cls_ratio in    0.25 0.5 0.75
    do
        for cluster_num_factor in   1.0 2.0 3.0 4.0
        do
            for seed in   0 1 2 3 4 5 6 7 8 9
            do 
                python run.py \
                --dataset $dataset \
                --method 'DeepAligned' \
                --train \
                --setting 'semi_supervised' \
                --labeled_ratio 0.1 \
                --known_cls_ratio $known_cls_ratio \
                --seed $seed \
                --cluster_num_factor $cluster_num_factor \
                --backbone 'bert_DeepAligned' \
                --config_file_name 'DeepAligned' \
                --gpu_id '0' \
                --results_file_name 'DeepAligned.csv' \
                --save_results 
            done
        done
    done
done

