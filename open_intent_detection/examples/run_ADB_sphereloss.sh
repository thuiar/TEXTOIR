#!/usr/bin bash

for dataset in 'banking' 'oos' 'stackoverflow'
do 
    for known_cls_ratio in  0.75
    do
        for labeled_ratio in 1.0
        do
            for seed in 1 2 3 4 5 6 7 8 9
            do
                for scale in 5.0
                do  
                    for alpha in 0.5
                    do
                        for rampup_epoch in 40
                        do
                            python run.py \
                            --dataset $dataset \
                            --method 'ADB' \
                            --known_cls_ratio $known_cls_ratio \
                            --labeled_ratio $labeled_ratio \
                            --seed $seed \
                            --backbone 'bert_norm' \
                            --config_file_name 'ADB' \
                            --loss_fct 'SphereFaceLoss' \
                            --gpu_id '0' \
                            --pretrain \
                            --train \
                            --results_file_name 'results_ADB_sphereloss.csv' \
                            --save_results \
                            --save_model
                        done
                    done
                done
            done
        done
    done
done

