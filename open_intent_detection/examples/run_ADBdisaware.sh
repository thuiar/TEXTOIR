#!/usr/bin bash

for dataset in 'banking' 'oos' 'stackoverflow'
do
    for labeled_ratio in 0.2 0.4 0.6 0.8 1.0
    do
        for seed in 0 1 2 3 4 5 6 7 8 9
        do
            for scale in 4
            do
                for known_cls_ratio in 0.25 0.5 0.75
                do
                    python run.py \
                    --dataset $dataset \
                    --method 'DA-ADB' \
                    --known_cls_ratio $known_cls_ratio \
                    --labeled_ratio $labeled_ratio \
                    --seed $seed \
                    --scale $scale \
                    --backbone 'bert_disaware' \
                    --config_file_name 'ADBdisaware' \
                    --loss_fct 'CrossEntropyLoss' \
                    --gpu_id '0' \
                    --pretrain \
                    --train \
                    --results_file_name 'results_DA-ADB.csv' \
                    --save_results \
                    --save_model
                done
            done
        done
    done
done
