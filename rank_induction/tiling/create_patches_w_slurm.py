"""[summary] SLURM을 이용한 HPC WSI tiling 작업

Notes:
    필요한 정보와 구조는 다음과 같음:
        - input_dir: 작업을 할 모든 WSI 들이 저장되어 있는 위치
        - output_dir: patch를 저장할 모든 경로
        - (있을 시에) annotation_dir: annotation json 파일이 저장되어 있는 폴더 위치
        - (annotation 있을 시에) label: annotation의 label

    SLURM을 이용한 WSI tiling 작업은 크게 두 가지 케이스로 나뉨:
        1. annotation 데이터 없이 전경만으로 진행
            - 이 경우에는 기본 slurm script로 진행한다
        2. annotation 데이터 고려하여 patch 따기
            - annotation 데이터가 있을 경우에는 wsi-annotation 파일 짝을 호출 해야함
            - 그리고 annotation 데이터 정보도 slurm script에 추가해서 실행한다
"""

import os
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

TILING_DIR = os.path.dirname(os.path.abspath(__file__))
CAMELYON_DIR = os.path.dirname(TILING_DIR)
ROOT_DIR = os.path.dirname(CAMELYON_DIR)
LOG_DIR = os.path.join(ROOT_DIR, "logs")
LOG_SLURM_DIR = os.path.join(LOG_DIR, "slurm_tile")

from rank_induction.log_ops import get_logger
from rank_induction.data_models import Labels, BinaryLabels
from rank_induction.tiling.slurm_tiling_template import SLURM_SCRIPT_TEMPLATE


def get_args():
    parser = argparse.ArgumentParser(
        epilog=(
            "Example usage:\n"
            "  python3 seedp/tiling/create_patches_w_slurm.py \\\n"
            "      -i </path/to/input> \\\n"
            "      -o </path/to/output> \\\n"
            "      -t 224 \\\n"
            "      --use_otsu \\\n"
            "      --limit_bounds \\\n"
            "      --format h5 \\\n"
            "      --ext '.svs' \\\n"
            "      --mpp 1.0"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        required=True,
        help="Root directory with all the WSIs to be tiled",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=True,
        help="Output directory where patches will be saved.",
    )
    parser.add_argument(
        "-t",
        "--tile_size",
        type=int,
        help="Tile size (px). Assumes square tiles, so only one dimension (width or height) needs to be specified.",
        required=True,
    )
    parser.add_argument(
        "--use_otsu",
        action="store_true",
        help="Use Otsu thresholding",
    )
    parser.add_argument("--limit_bounds", default=True, type=bool, required=False)
    parser.add_argument("--format", choices=["png", "h5"], default="h5")
    parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Overlap between tiles (default: 0)",
    )
    parser.add_argument(
        "-a", "--annotation_dir", help="The directory of annotation data, if available"
    )
    parser.add_argument(
        "--ext", type=str, default=".svs", help="File extension of the WSIs with '.' "
    )
    parser.add_argument(
        "--mpp", type=float, default=1.0, help="MPP of the WSIs (default: 1.0)"
    )

    return parser.parse_args()


def get_wsi_annotation_path_pair(wsi_dir: Path, annotation_dir: Path, ext: str) -> dict:
    """[summary] annotation 파일이 있는 WSI-annotation 파일 경로 짝 반환

    Notes:
        argparse에 제공한 input_dir (WSI 파일들이 모여있는 경로)에 있는 파일들,
        그리고 annotation_dir (annotation json 파일들이 모여 있는 경로)에 있는 파일들의
        파일명이 같을 경우에 annotation이 있다 간주하여 짝을 맞추어서 추후에 사용한다.

    Args:
        wsi_dir (Path): WSI image들이 위치되어 있는 경로
        annotation_dir (Path): annotation json 파일들이 위치되어 있는 경로
        ext (dir): WSI 파일의 extension

    Returns:
        wsi2annotation_path_pairs (dict): WSI, annotation 파일 경로 짝
    """
    wsi2annotation_path_pairs = dict()

    for annotation_path in annotation_dir.glob("*.json"):
        wsi_path = wsi_dir / annotation_path.with_suffix(ext).name
        if wsi_path.exists():
            wsi2annotation_path_pairs[wsi_path] = annotation_path

    return wsi2annotation_path_pairs


def get_slurm_cmd(
    wsi_path: str,
    output_dir: str,
    mpp: float,
    use_otsu: bool,
    tile_size: int,
    limit_bounds: bool,
    overlap: int,
    format: str,
    log_dir: str,
) -> str:
    """[summary] 조건에 따라 실행할 SLURM 명령어 반환

    Notes:
        slurm_tiling_template 파일에 있는 template에 기본적인 내용을 추가해준다.
        나중에 annotation file이 있을 경우에는 다른 코드에서 해당 cmd 뒤에
        parameter를 추가해줄 것임.

        create_patches_w_single_wsi는 output directory를 WSI 기준으로 하기 때문에
        output_dir를 파일명과 합쳐진 형태로 지정해준다.
    """
    filename = Path(wsi_path).stem
    otsu_flag = "--use_otsu" if use_otsu else ""
    limit_bounds_flag = "--limit_bounds" if limit_bounds else ""
    output_path = (
        str(Path(output_dir, filename + ".h5"))
        if format == "h5"
        else str(Path(output_dir, filename))
    )
    slurm_cmd = SLURM_SCRIPT_TEMPLATE.format(
        filename=filename,
        root_dir=ROOT_DIR,
        input_path=wsi_path,
        output_dir=output_path,
        mpp=mpp,
        otsu_flag=otsu_flag,
        limit_bounds_flag=limit_bounds_flag,
        tile_size=tile_size,
        overlap=overlap,
        format=format,
        log_dir=log_dir,
    )

    return slurm_cmd


if __name__ == "__main__":
    args = get_args()
    logger = get_logger(module_name="tiling_slurm")

    log_today_dir = os.path.join(LOG_SLURM_DIR, datetime.now().strftime("%Y-%m-%d"))
    os.makedirs(log_today_dir, exist_ok=True)

    wsi2annotation_path_pairs = dict()
    wsi_dir = Path(args.input_dir)
    if args.annotation_dir:
        annotation_dir = Path(args.annotation_dir)
        wsi2annotation_path_pairs = get_wsi_annotation_path_pair(
            wsi_dir, annotation_dir, args.ext
        )

    os.makedirs(args.output_dir, exist_ok=True)
    for wsi_path in wsi_dir.glob(f"*{args.ext}"):
        slurm_cmd: str = get_slurm_cmd(
            wsi_path=wsi_path,
            output_dir=args.output_dir,
            mpp=args.mpp,
            use_otsu=args.use_otsu,
            tile_size=args.tile_size,
            limit_bounds=args.limit_bounds,
            overlap=args.overlap,
            log_dir=log_today_dir,
            format=args.format,
        )

        if wsi_path in wsi2annotation_path_pairs:
            additional_params = (
                f' -a "{wsi2annotation_path_pairs[wsi_path]}"' f' -l "{args.label}"'
            )
            slurm_cmd += additional_params

        process = subprocess.Popen(["echo", slurm_cmd], stdout=subprocess.PIPE)
        submit = subprocess.Popen(["sbatch"], stdin=process.stdout, text=True)
        process.stdout.close()
        output = submit.communicate()[0]
