for dataset in 'banking'
do
    for labeled_ratio in 1.0
    do
        for seed in 0 1 2 3 4 5 6 7 8 9
        do
            for scale in 8
            do
                for alpha in 0.1
                do
                    for known_cls_ratio in 0.75
                    do
                        python run.py \
                        --dataset $dataset \
                        --method 'ADB' \
                        --known_cls_ratio $known_cls_ratio \
                        --labeled_ratio $labeled_ratio \
                        --seed $seed \
                        --scale $scale \
                        --alpha $alpha \
                        --backbone 'bert_disaware' \
                        --config_file_name 'ADBdisaware' \
                        --loss_fct 'CrossEntropyLoss' \
                        --gpu_id '0' \
                        --pretrain \
                        --train \
                        --results_file_name 'results_ADBdisaware.csv' \
                        --save_results
                    done
                done
            done
        done
    done
done
