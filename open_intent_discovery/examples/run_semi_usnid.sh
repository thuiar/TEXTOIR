#!/usr/bin bash
for seed in    0 1 2 3 4 5 6 7 8 9
do
    for dataset in  'banking' 'clinc' 'stackoverflow'
    do  
        for known_cls_ratio in  0.25  0.5 0.25  
        do
            for cluster_num_factor in 1.0 2.0 3.0 4.0
            do
                python run.py \
                --dataset $dataset \
                --method 'USNID' \
                --setting 'semi_supervised' \
                --known_cls_ratio $known_cls_ratio \
                --seed $seed \
                --train \
                --tune \
                --cluster_num_factor $cluster_num_factor \
                --backbone 'bert_USNID' \
                --config_file_name 'SemiUSNID' \
                --gpu_id '0' \
                --results_file_name 'results_semisupervised.csv' \
                --save_results \
                --output_dir '/home/sharing/disk2/zhl/TEXTOIR/outputs_un_pre/' 
            done
        done
    done
done