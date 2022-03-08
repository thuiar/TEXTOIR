for dataset in 'banking'
do
    for labeled_ratio in 0.2
    do
        for seed in 0 
        do
            for scale in 4
            do
                for known_cls_ratio in 0.25
                do
                    python run.py \
                    --dataset $dataset \
                    --method 'ADB' \
                    --known_cls_ratio $known_cls_ratio \
                    --labeled_ratio $labeled_ratio \
                    --seed $seed \
                    --scale $scale \
                    --backbone 'bert_disaware' \
                    --config_file_name 'ADBdisaware' \
                    --loss_fct 'CrossEntropyLoss' \
                    --gpu_id '1' \
                    --pretrain \
                    --train \
                    --results_file_name 'results_ADBdisaware.csv' \
                    --save_results
                done
            done
        done
    done
done
