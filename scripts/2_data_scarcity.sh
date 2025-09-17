#!/bin/bash

# SBATCH 스크립트를 직접 실행하는 함수
submit_job() {
    local learning_type=$1
    local sampling_ratio=$2
    local seed=$3

    CMD=$(cat <<EOT
#!/bin/bash
#SBATCH --job-name='resnet50_data_scarcity'
#SBATCH --mem=30G
#SBATCH --output='/vast/AI_team/heon/repositories/camelyon/logs/resnet50/$(date +%Y%m%d)/${learning_type}_${run_prefix}_${seed}_${lambda}_${use_threshold}_out.txt'
#SBATCH --error='/vast/AI_team/heon/repositories/camelyon/logs/resnet50/$(date +%Y%m%d)/${learning_type}_${run_prefix}_${seed}_${lambda}_${use_threshold}_err.txt'
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --tres-per-task=gres/shard=40,cpu=24

export CUDA_VISIBLE_DEVICES=\$SLURM_JOB_GPUS # to be escaped

/vast/AI_team/heon/repositories/camelyon/.env/bin/python3.11 \
    /vast/AI_team/heon/repositories/camelyon/experiments/attention_induction/train_downstream.py \
    --data_dir /vast/AI_team/dataset/CAMELYON16/feature/resnet50_custom_20x_h5 \
    --slide_dir /vast/AI_team/dataset/CAMELYON16/slide \
    --learning '$learning_type' \
    --run_name 'resnet50_${learning_type}_${sampling_ratio}' \
    --in_features 1024 \
    --mpp 0.5 \
    --random_state 2023 \
    --minimal_earlystop_epoch 20 \
    --max_patiences 7 \
    --num_workers 16 \
    --prefetch_factor 8 \
    --num_neg 1024 \
    --num_pos 1024 \
    --verbose \
    --threshold 1 \
    --sampling_ratio $sampling_ratio \
    --device 'cuda'
EOT
    )
    
    echo "$CMD" | sbatch
}

seeds=(2021 2022 2023 2024 2025)
learning_types=("attention_induction" "ltr")
sampling_ratios=(0.20 0.4 0.6 0.8)
for seed in "${seeds[@]}"; do
    for learning_type in "${learning_types[@]}"; do
        for sampling_ratio in "${sampling_ratios[@]}"; do
            submit_job $learning_type $sampling_ratio $seed
        done
    done
done
