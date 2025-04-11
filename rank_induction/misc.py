import os
import json
import random
from typing import List, Tuple

import pickle
import numpy as np
import torch
import torchvision
from torchvision.datasets import ImageFolder
from openslide import OpenSlide
from openslide.deepzoom import DeepZoomGenerator


def read_json(file_path) -> dict:
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_deepzoom_level(desired_mpp: float, slide: OpenSlide) -> int:
    """return deepzoom level

    Args:
        desired_mpp (float): 타깃 MPP(microns per pixels)
        slide (OpenSlide): openslide object

    Returns:
        int: DeepZoom level

    Example:
        1) desired_mpp=0.24, slide_mpp=0.24, max_deepzoom_level=17
        => return 17
        2) desired_mpp=0.5, slide_mpp=0.24, max_deepzoom_level=17
        => return 16

    Raise:
        ValueError: desired_mpp is too small
    """

    try:
        slide_mpp: str = slide.properties["openslide.mpp-x"]
        slide_mpp = float(slide_mpp)
    except KeyError:
        raise KeyError("slide_mpp is not found")
    except ValueError:
        raise ValueError(
            f"slide_mpp is not float, passed: {slide_mpp}, type({type(slide_mpp)})"
        )

    up_level = round(desired_mpp / slide_mpp)

    if up_level < 1:
        raise ValueError(
            "desired_mpp is too small: "
            f"desired_mpp({desired_mpp}), slide_mpp({slide_mpp})"
        )

    dzg = DeepZoomGenerator(slide)
    return dzg.level_count - up_level


def save_pickle(object, path):
    with open(path, "wb") as f:
        pickle.dump(object, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def collate_center(
    data_dir: str,
    test_center_names: List[str],
    transform: callable,
):
    test_datasets = [
        ImageFolder(os.path.join(data_dir, center), transform)
        for center in test_center_names
    ]

    test_dataset = test_datasets[0]
    for ds in test_datasets[1:]:
        test_dataset.samples.extend(ds.samples)
        test_dataset.targets.extend(ds.targets)
    return test_dataset


def build_model():
    model = torchvision.models.resnet50(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)

    return model


def get_slide_mpp(slide) -> float:
    """슬라이드의 MPP"""
    try:
        slide_mpp: str = slide.properties["openslide.mpp-x"]
        return float(slide_mpp)
    except KeyError:
        raise KeyError("slide_mpp is not found")
    except ValueError:
        raise ValueError(
            f"slide_mpp is not float, passed: {slide_mpp}, type({type(slide_mpp)})"
        )


def get_deepzoom_level(desired_mpp: float, slide: OpenSlide) -> int:
    """
    Args:
        desired_mpp (float): 타깃 MPP(microns per pixels)
        slide (OpenSlide): openslide object

    Returns:
        int: DeepZoom level

    Example:
        1) desired_mpp=0.24, slide_mpp=0.24, max_deepzoom_level=17
        => return 17
        2) desired_mpp=0.5, slide_mpp=0.24, max_deepzoom_level=17
        => return 16

    Raise:
        ValueError: desired_mpp is too small
    """

    try:
        slide_mpp: str = slide.properties["openslide.mpp-x"]
        slide_mpp = float(slide_mpp)
    except KeyError:
        raise KeyError("slide_mpp is not found")
    except ValueError:
        raise ValueError(
            f"slide_mpp is not float, passed: {slide_mpp}, type({type(slide_mpp)})"
        )

    up_level = int(np.log2(round(desired_mpp / slide_mpp)))

    if up_level < 0:
        raise ValueError(
            "desired_mpp is too small: "
            f"desired_mpp({desired_mpp}), slide_mpp({slide_mpp})"
        )

    dzg = DeepZoomGenerator(slide)
    max_deepzoom_level = dzg.level_count - 1

    return max_deepzoom_level - up_level


def get_foreground_tile_address(
    dgz: DeepZoomGenerator,
    deepzoom_level: int,
    binary_image: np.ndarray,
    foreground_threshold: float = 0.05,
    downsample: int = 16,
) -> Tuple[Tuple[int, int]]:

    mask_image = binary_image / 255

    # 총 18단계(mpp=0.24)에서 mpp=0.5로 하려면 17인데, 인덱스로는 0~17(mpp=0.25)
    n_cols, n_row = dgz.level_tiles[deepzoom_level]

    # level 0부터 길이 축소
    max_deepzoom_level = dgz.level_count - 1
    tile_size = dgz._z_t_downsample
    level0_interval = tile_size * (2 ** (max_deepzoom_level - deepzoom_level))
    tile_size_thumnail = int(level0_interval / downsample)

    candidates = list()
    for row in range(n_row):
        for col in range(n_cols):
            (x1, y1), *_ = dgz.get_tile_coordinates(deepzoom_level, (col, row))
            x1_thumnail, y1_thumnail = (
                int(x1 / downsample),
                int(y1 / downsample),
            )

            x2_thumnail = x1_thumnail + tile_size_thumnail
            y2_thumnail = y1_thumnail + tile_size_thumnail
            window = mask_image[y1_thumnail:y2_thumnail, x1_thumnail:x2_thumnail]
            if 0 in window.shape:
                continue

            background_percent = window.sum() / np.prod(window.shape)
            foreground_ratio = 1 - background_percent
            if foreground_ratio > foreground_threshold:
                candidates.append((col, row))

    return candidates




def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # GPU 연산의 결정성을 높이기 위한 설정
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)