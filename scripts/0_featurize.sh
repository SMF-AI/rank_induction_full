#!/bin/bash

# 모델 설정
MODEL="resnet50_custom"

# 데이터 경로
INPUT_BASE="/vast/AI_team/dataset/CAMELYON16/patch/20x_h5"
OUTPUT_BASE="/vast/AI_team/dataset/CAMELYON16/feature/resnet50_custom_20x_h5"

# 루프 실행
for SPLIT in train test; do
  for LABEL in M N; do
    INPUT_PATH="${INPUT_BASE}/${SPLIT}/${LABEL}"
    OUTPUT_PATH="${OUTPUT_BASE}/${SPLIT}/${LABEL}"
    
    CMD="python3 camelyon/feature/featurize_w_slurm.py -i \"${INPUT_PATH}\" -o \"${OUTPUT_PATH}\" -m \"${MODEL}\""
    echo $CMD
    python3 camelyon/feature/featurize_w_slurm.py -i "${INPUT_PATH}" -o "${OUTPUT_PATH}" -m "${MODEL}"
  done
done
