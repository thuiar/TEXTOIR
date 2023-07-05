
for dataset in  'banking' 'clinc' 'stackoverflow'
do
    for known_cls_ratio in    0.25 0.5 0.75
    do
        for cluster_num_factor in   1.0 2.0 3.0 4.0
        do
            for seed in    0 1 2 3 4 5 6 7 8 9
            do 
                python run.py \
                --dataset $dataset \
                --method 'GCD' \
                --train \
                --setting 'semi_supervised' \
                --labeled_ratio 0.1 \
                --known_cls_ratio $known_cls_ratio \
                --cluster_num_factor $cluster_num_factor \
                --seed $seed \
                --backbone 'bert_GCD' \
                --config_file_name 'GCD' \
                --gpu_id '1' \
                --results_file_name 'GCD.csv.csv' \
                --save_results 
                --save_model
            done
        done
    done
done