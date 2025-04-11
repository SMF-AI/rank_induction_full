"""[summary] SLURM을 이용한 HPC WSI Featurization 작업

Notes:
    필요한 정보와 구조는 다음과 같음:
        - input_dir: 작업을 할 WSI 들의 패치들이 저장되어 있는 위치
        - output_dir: 특징을 .npy로 저장할 모든 경로
"""

import os
import sys
import glob
import argparse
import subprocess
from datetime import datetime


FEATURE_DIR = os.path.dirname(os.path.abspath(__file__))
CAMELYON_DIR = os.path.dirname(FEATURE_DIR)
ROOT_DIR = os.path.dirname(CAMELYON_DIR)
LOG_DIR = os.path.join(ROOT_DIR, "logs")
LOG_SLURM_DIR = os.path.join(LOG_DIR, "feature_slrum_job")
sys.path.append(ROOT_DIR)

from rank_induction.log_ops import get_logger
from slurm_featurize_template import SLURM_SCRIPT_TEMPLATE


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Script for featurizing mutiple WSIs with SLURM.",
        epilog=(
            "Example usage:\n"
            "  python3 camelyon/feature/featurize_w_slurm.py \\\n"
            "      --input </path/to/input> \\\n"
            "      --output </path/to/output> \\\n"
            "      --model histai/hibou-B"
        ),
    )
    parser.add_argument("-i", "--input_dir", type=str, required=True)
    parser.add_argument("-o", "--output_dir", type=str, required=True)
    parser.add_argument(
        "-m", "--model", type=str, required=False, default="histai/hibou-B"
    )
    return parser.parse_args()


def get_slurm_cmd(
    input_path: str,
    output_path: str,
    model: str,
    log_dir: str,
) -> str:

    return SLURM_SCRIPT_TEMPLATE.format(
        root_dir=ROOT_DIR,
        input_path=input_path,
        output_path=output_path,
        filename=os.path.basename(input_path),
        model=model,
        log_dir=log_dir,
    )


if __name__ == "__main__":
    args = get_args()
    logger = get_logger(module_name="feature_slrum_submit")

    log_today_dir = os.path.join(LOG_SLURM_DIR, datetime.now().strftime("%Y-%m-%d"))
    os.makedirs(log_today_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    for input_path in glob.glob(f"{args.input_dir}/*"):
        logger.info(f"Processing H5: {input_path}")
        output_path = os.path.join(args.output_dir, os.path.basename(input_path))
        slurm_cmd = get_slurm_cmd(
            input_path=input_path,
            output_path=output_path,
            model=args.model,
            log_dir=log_today_dir,
        )
        logger.debug("SLURM CMD:")
        logger.debug(slurm_cmd)

        process = subprocess.Popen(["echo", slurm_cmd], stdout=subprocess.PIPE)
        submit = subprocess.Popen(["sbatch"], stdin=process.stdout, text=True)
        process.stdout.close()
        output = submit.communicate()[0]
