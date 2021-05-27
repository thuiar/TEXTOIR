#!/usr/bin bash

for seed in 0
do
    for dataset in 'banking'
    do
        for known_cls_ratio in 0.5 0.75
        do
            for labeled_ratio in 0.2 0.4 0.6 0.8 1.0
            do 
                python run.py \
                --dataset $dataset \
                --method 'DeepUnk' \
                --known_cls_ratio $known_cls_ratio \
                --labeled_ratio $labeled_ratio \
                --seed $seed \
                --num_train_epochs 100 \
                --gpu_id '1' \
                --train \
                --save_model

            done
        done
    done
done

for seed in 0
do
    for dataset in 'oos' 'stackoverflow' 'snips'
    do
        for known_cls_ratio in 0.25 0.5 0.75
        do
            for labeled_ratio in 0.2 0.4 0.6 0.8 1.0
            do 
                python run.py \
                --dataset $dataset \
                --method 'DeepUnk' \
                --known_cls_ratio $known_cls_ratio \
                --labeled_ratio $labeled_ratio \
                --seed $seed \
                --num_train_epochs 100 \
                --gpu_id '1' \
                --train \
                --save_model

            done
        done
    done
done

for seed in 0 1 2 3 4 5 6 7 8 9
do
    for dataset in 'banking' 'oos' 'stackoverflow' 'snips'
    do
        for known_cls_ratio in 0.25 0.5 0.75
        do
            for labeled_ratio in 0.2 0.4 0.6 0.8 1.0
            do 
                python run.py \
                --dataset $dataset \
                --method 'DeepUnk' \
                --known_cls_ratio $known_cls_ratio \
                --labeled_ratio $labeled_ratio \
                --seed $seed \
                --num_train_epochs 100 \
                --gpu_id '1' \
                --train \
                --save_model

            done
        done
    done
done
