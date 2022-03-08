#!/usr/bin bash
for dataset in 'banking'
do
    for known_cls_ratio in 0.25
    do
        for labeled_ratio in 1.0
        do
            for seed in 0 
            do
                python run.py \
                --dataset $dataset \
                --method SEG \
                --known_cls_ratio $known_cls_ratio \
                --labeled_ratio $labeled_ratio \
                --seed $seed \
                --gpu_id 0 \
                --train \
                --save_results \
                --config_file_name 'SEG' \
                --loss_fct 'SEGLoss' \
                --results_file_name 'results_SEG.csv' \
                --backbone 'bert_seg'

            done
        done
    done
done

# for dataset in 'banking'
# do
#     for known_cls_ratio in 0.75
#     do
#         for labeled_ratio in 0.2 0.4 0.6 0.8 1.0
#         do
#             for seed in 0 1 2 3 4 5 6 7 8 9 
#             do
#                 python run.py \
#                 --dataset $dataset \
#                 --method SEG \
#                 --known_cls_ratio $known_cls_ratio \
#                 --labeled_ratio $labeled_ratio \
#                 --seed $seed \
#                 --gpu_id 0 \
#                 --train \
#                 --save_results \
#                 --config_file_name 'SEG' \
#                 --loss_fct 'SEGLoss' \
#                 --results_file_name 'results_SEG_all.csv' \
#                 --backbone 'bert_seg'

#             done
#         done
#     done
# done

# for dataset in 'oos' 'stackoverflow'
# do
#     for known_cls_ratio in 0.25 0.5 0.75
#     do
#         for labeled_ratio in 0.2 0.4 0.6 0.8 1.0
#         do
#             for seed in 0 1 2 3 4 5 6 7 8 9 
#             do
#                 python run.py \
#                 --dataset $dataset \
#                 --method SEG \
#                 --known_cls_ratio $known_cls_ratio \
#                 --labeled_ratio $labeled_ratio \
#                 --seed $seed \
#                 --gpu_id 0 \
#                 --train \
#                 --save_results \
#                 --config_file_name 'SEG' \
#                 --loss_fct 'SEGLoss' \
#                 --results_file_name 'results_SEG_all.csv' \
#                 --backbone 'bert_seg'

#             done
#         done
#     done
# done
