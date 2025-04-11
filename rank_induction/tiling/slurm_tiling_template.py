SLURM_SCRIPT_TEMPLATE = """#!/bin/bash
#SBATCH --job-name='tiling_{filename}'
#SBATCH --priority=100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --output='{log_dir}/slurm_out_{filename}.txt'
#SBATCH --error='{log_dir}/slurm_err_{filename}.txt'
#SBATCH --time=1:30:00
#SBATCH --nodelist=gpusvr05  # 특정 서버에서만 실행

/home/heon/anaconda3/envs/camelyon/bin/python3 {root_dir}/camelyon/tiling/create_patches_w_single_wsi.py \
    -i "{input_path}" \
    -o "{output_dir}" \
    --n_workers 4 \
    --mpp {mpp} \
    --tile_size {tile_size} \
    --overlap {overlap} \
    --format {format}
    {otsu_flag} 
"""
