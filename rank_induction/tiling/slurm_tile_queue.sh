#!/bin/bash

HELP_MSG="Usage: $0 -i <input_dir> -o <output_dir>

Description:
This script processes whole slide images (WSI) by tiling them into smaller patches.
It takes input and output directories as arguments and applies the tiling process to images.

Parameters:
  -i <input_dir>   : Specify the base input directory containing subdirectories for datasets (e.g., 'train', 'test').
  -o <output_dir>  : Specify the base output directory where the patches will be saved.

Example structure of input_dir:
<input_dir>
  ├── test
  │   ├── M
  │   └── N
  └── train
      ├── M
      └── N
<test_dir>
  ├── test
  │   ├── M
  │   └── N
  └── train
      ├── M
      └── N
$ /bin/bash camelyon/tiling/submit_tile_queue.sh -i <input_dir>/train -o <output_dir>/train

Example usage:
$0 -i /path/to/input -o /path/to/output
"


BASE_INPUT_DIR=""
BASE_OUTPUT_DIR=""

parse_args() {
    while getopts "hi:o:" opt; do
        case ${opt} in
            h)
                echo "$HELP_MSG"
                exit 0
                ;;
            i)
                BASE_INPUT_DIR="${OPTARG}"
                ;;
            o)
                BASE_OUTPUT_DIR="${OPTARG}"
                ;;
            *)
                echo "Invalid option: -$OPTARG"
                echo "Usage: $0 -i <input_dir> -o <output_dir>"
                exit 1
                ;;
        esac
    done
}


parse_args "$@"
if [ -z "$BASE_INPUT_DIR" ] || [ -z "$BASE_OUTPUT_DIR" ]; then
    echo "Both input and output directories are required."
    echo "Usage: $0 -i <input_dir> -o <output_dir>"
    exit 1
fi

PHASES=("train" "test")
SUBDIRS=("N" "M")
for phase in "${PHASES[@]}"; do
    for subdir in "${SUBDIRS[@]}"; do
        input_dir="${BASE_INPUT_DIR}/${phase}/${subdir}"
        output_dir="${BASE_OUTPUT_DIR}/${phase}/${subdir}"
        
        echo "Processing: ${input_dir} -> ${output_dir}"

        # TODO: 인자추가
        mkdir -p "${output_dir}"
        chmod -R 777 "${output_dir}"
        cmd="/home/heon/anaconda3/envs/camelyon/bin/python3 camelyon/tiling/create_patches_w_slurm.py -i ${input_dir} -o ${output_dir} -t 256 --use_otsu --ext '.tif' --mpp 0.5"
        echo "실행할 명령어: ${cmd}"
        eval "${cmd}"
        echo "Queue submitted: ${input_dir}/"

    done
done