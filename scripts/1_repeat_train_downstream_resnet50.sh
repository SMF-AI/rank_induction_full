#!/bin/bash

# SBATCH 스크립트를 직접 실행하는 함수
submit_job() {
    local learning_type=$1
    local run_prefix=$2
    local seed=$3
    local lambda=$4
    local use_threshold=$5

    CMD=$(cat <<EOT
#!/bin/bash
#SBATCH --job-name='resnet50'
#SBATCH --mem=50G
#SBATCH --output='/vast/AI_team/heon/repositories/camelyon/logs/resnet50/$(date +%Y%m%d)/${learning_type}_${run_prefix}_${seed}_${lambda}_${use_threshold}_out.txt'
#SBATCH --error='/vast/AI_team/heon/repositories/camelyon/logs/resnet50/$(date +%Y%m%d)/${learning_type}_${run_prefix}_${seed}_${lambda}_${use_threshold}_err.txt'
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --tres-per-task=gres/shard=40,cpu=24

export CUDA_VISIBLE_DEVICES=\$SLURM_JOB_GPUS # to be escaped

/vast/AI_team/heon/repositories/camelyon/.env/bin/python3.11 \
    /vast/AI_team/heon/repositories/camelyon/experiments/attention_induction/train_downstream.py \
    --data_dir /vast/AI_team/dataset/CAMELYON16/feature/resnet50_3rd_20x_h5 \
    --slide_dir /vast/AI_team/dataset/CAMELYON16/slide \
    --learning '$learning_type' \
    --run_name '${run_prefix}_${seed}' \
    --random_state $seed \
    --in_features 1024 \
    --mpp 0.5 \
    --minimal_earlystop_epoch 20 \
    --max_patiences 7 \
    --num_workers 16 \
    --prefetch_factor 8 \
    --num_neg 1024 \
    --num_pos 1024 \
    $(if [ "$use_threshold" = true ]; then echo "--threshold 1"; fi) \
    --_lambda $lambda \
    --device 'cuda'
EOT
    )
    
    echo "$CMD" | sbatch
}

learning_types=("base" "attention_induction" "ltr")
lambdas=(1 0.1 0.01 0.001)
use_thresholds=(true false)

for seed in {2018..2025}; do
    for lambda in "${lambdas[@]}"; do
        for learning_type in "${learning_types[@]}"; do
            for use_threshold in "${use_thresholds[@]}"; do
                submit_job "${learning_type}" "resnet50_20x_${learning_type}" ${seed} ${lambda} ${use_threshold}
            done
        done
    done
done
