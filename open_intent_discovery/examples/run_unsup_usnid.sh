#!/usr/bin bash

for dataset in  'banking' 'clinc'  'stackoverflow'
do
    for cluster_num_factor in 1.0 2.0 3.0 4.0
    do
        for seed in 0 1 2 3 4 5 6 7 8 9 
        do 
            python run.py \
            --dataset $dataset \
            --method 'UnsupUSNID' \
            --setting 'unsupervised' \
            --known_cls_ratio 0 \
            --seed $seed \
            --train \
            --config_file_name 'UnsupUSNID' \
            --cluster_num_factor $cluster_num_factor \
            --backbone 'bert_USNID_Unsup' \
            --gpu_id '1' \
            --results_file_name 'results_unsupervised.csv' \
            --save_results \
            --output_dir '/home/sharing/disk2/zhl/TEXTOIR/outputs_un_pre/' 
        done
    done
done

 