#!/bin/bash

run_tiling() {
    local input_dir=$1
    local output_dir=$2

    python3 seedp/tiling/create_patches_w_slurm.py \
        -i "$input_dir" \
        -o "$output_dir" \
        -t 224 \
        --use_otsu \
        --format h5 \
        --ext '.tif' \
        --mpp 1.0
}

# 데이터셋 경로 정의
BASE_INPUT_DIR="/vast/AI_team/dataset/CAMELYON16/slide"
BASE_OUTPUT_DIR="/vast/AI_team/dataset/CAMELYON16/patch/10x_h5"

# Train
run_tiling "$BASE_INPUT_DIR/train/N" "$BASE_OUTPUT_DIR/train/N"
run_tiling "$BASE_INPUT_DIR/train/M" "$BASE_OUTPUT_DIR/train/M"

# Test
run_tiling "$BASE_INPUT_DIR/test/N" "$BASE_OUTPUT_DIR/test/N"
run_tiling "$BASE_INPUT_DIR/test/M" "$BASE_OUTPUT_DIR/test/M"
