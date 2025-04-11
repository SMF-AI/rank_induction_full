"""
하나의 WSI 파일을 타일링하는 코드
    - input의 단위는 WSI파일의 경로이다 (폴더 아님 X)
    - annotation 파일도 있다면 annotation_path에 json파일 경로를 명시해야함
    - output dir은 svs (slide 별 기준으로) 관리/저장을 하는 것을 원칙으로 함

기본적으로 tiling은 otsu binarization 기법으로 진행을 한다.

annotation이 있을 시에는 annotation 정보를 기준으로 필터해서 저장함
"""

import os
import argparse
from openslide import OpenSlide

from rank_induction.misc import get_deepzoom_level
from rank_induction.data_models import (
    WSI,
    Patches,
    DigestPath,
    Polygons,
    Labels,
    BinaryLabels,
)
from rank_induction.log_ops import get_logger
from rank_induction.patch_filter import check_small_compressed_format


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input_filepath", type=str, required=True, help="Filepath of the WSI"
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=True,
        help="Directory of where the patches will be saved",
    )
    parser.add_argument("--n_workers", type=int, required=True)
    parser.add_argument("--mpp", type=float, default=1.0)
    parser.add_argument("--tile_size", type=int, default=224)
    parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Overlap between tiles (default: 0)",
    )
    parser.add_argument(
        "--limit_bounds",
        default=True,
        type=bool,
        help="Limit the patch to the bound of the WSI",
        required=False,
    )

    parser.add_argument(
        "--use_otsu",
        action="store_true",
        help="Use Otsu thresholding",
    )
    parser.add_argument("--format", choices=["png", "h5"], default="h5")

    parser.add_argument("--input_type", choices=["wsi", "jpg"], default="jpg")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    logger = get_logger(module_name="tiling_wsi")
    logger.info("-" * 10 + "Parameters used" + "-" * 10)
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")

    if args.input_type == "wsi":
        deepzoom_level = get_deepzoom_level(args.mpp, OpenSlide(args.input_filepath))
        wsi = WSI(args.input_filepath)

        if args.use_otsu:
            patches: Patches = wsi.do_tile_otsu(
                tile_size=args.tile_size,
                overlap=args.overlap,
                limit_bounds=args.limit_bounds,
                patch_filter=check_small_compressed_format,
                deepzoom_level=deepzoom_level,
                n_workers=args.n_workers,
                logger=logger,
            )
        else:
            patches: Patches = wsi.do_tile(
                tile_size=args.tile_size,
                overlap=args.overlap,
                limit_bounds=args.limit_bounds,
                deepzoom_level=deepzoom_level,
                n_workers=args.n_workers,
                logger=logger,
            )

    else:
        jpg = DigestPath(args.input_filepath)

        if args.use_otsu:
            patches: Patches = jpg.do_tile_otsu(
                tile_size=args.tile_size,
                overlap=args.overlap,
                patch_filter=check_small_compressed_format,
            )
        else:
            patches: Patches = jpg.do_tile(
                tile_size=args.tile_size,
                overlap=args.overlap,
            )

    if len(patches) == 0:
        logger.info("No patches found")
    else:
        logger.info(f"Total {len(patches)} patches")
        logger.info(f"Saving patches to {args.output_dir}")
        patches.save(args.output_dir, format=args.format)
        logger.info(f"Saved patches to {args.output_dir}")
