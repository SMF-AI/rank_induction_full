##SBATCH --nodelist=gpusvr05

SLURM_SCRIPT_TEMPLATE = """#!/bin/bash
#SBATCH --job-name='featurize_{filename}'
#SBATCH --mem=20G
#SBATCH --output='{log_dir}/slurm_out_{filename}.txt'
#SBATCH --error='{log_dir}/slurm_err_{filename}.txt'
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --tres-per-task=gres/shard=8,cpu=1

export CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS

heon/repositories/camelyon/.env/bin/python3.11 \
    {root_dir}/camelyon/feature/featurize_w_single_wsi.py \
    -i "{input_path}" \
    -o "{output_path}" \
    -m "{model}" \
    -d "cuda"
"""
