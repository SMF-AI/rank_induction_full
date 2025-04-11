import os
import sys
import glob
import random
import tqdm
import colorsys
from collections import defaultdict
from typing import Tuple, Dict
from functools import partial

import cv2
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from torchvision.transforms import (
    Compose,
    Resize,
    Normalize,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    ToTensor,
)
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset

from seestaina.augmentation import Augmentor
from seestaina.structure_preversing import Augmentor as StructureAugmentor

CAMELYON_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CAMELYON_DIR)
EXP_DIR = os.path.join(ROOT_DIR, "experiments")
RANDSTAINNA_PATH = "/home/heon/dev/RandStainNA"

data_transforms = {
    "train": Compose(
        [
            Resize((224, 224)),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            ToTensor(),
        ]
    ),
    "test": Compose(
        [
            Resize((224, 224)),
            ToTensor(),
        ]
    ),
}


def cal_image_stats(image_dir: str, fraction=0.1) -> Tuple[tuple, tuple]:
    means = list()
    stds = list()

    for dirname in os.listdir(image_dir):
        image_paths = glob.glob(os.path.join(image_dir, dirname, "*.png"))
        sample_size = int(fraction * len(image_paths))
        for filename in tqdm.tqdm(random.sample(image_paths, sample_size)):
            image = Image.open(filename)
            image_array = np.array(image) / 255.0
            means.append(image_array.mean(axis=(0, 1)))
            stds.append(image_array.std(axis=(0, 1)))

    mean = np.stack(means, axis=0).mean(axis=0)
    std = np.stack(stds, axis=0).mean(axis=0)

    return tuple(mean), tuple(std)


def add_stain_vec_aug(
    train_transform: Compose,
    aug_density: bool = False,
    aug_saturation: bool = False,
) -> Compose:
    augmentor = Augmentor()
    aug_fn = partial(
        augmentor.image_augmentation_with_stain_vector,
        aug_saturation=aug_saturation,
        aug_density=aug_density,
    )

    train_transform.transforms.insert(1, aug_fn)

    return train_transform


def add_norm(data_transforms: Dict[str, Compose], means, stds) -> Dict[str, Compose]:
    for phase, compose in data_transforms.items():
        compose.transforms.append(Normalize(means, stds))

    return data_transforms


def get_dist_params(image_dir: str) -> dict:

    l_mean = list()
    l_sd = list()
    a_mean = list()
    a_sd = list()
    b_mean = list()
    b_sd = list()

    for class_dir in os.listdir(image_dir):
        image_paths = glob.glob(os.path.join(image_dir, class_dir, "*.png"))
        for filename in image_paths:
            rgb_image = cv2.imread(filename)
            lab_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
            L_channel, A_channel, B_channel = cv2.split(lab_image)
            l_mean.append(L_channel.mean())
            a_mean.append(A_channel.mean())
            b_mean.append(B_channel.mean())

            l_sd.append(L_channel.std())
            a_sd.append(A_channel.std())
            b_sd.append(B_channel.std())

    params = {
        "L": {
            "avg": {
                "mean": float(np.mean(l_mean)),
                "std": float(np.std(l_mean)),
            },
            "std": {
                "mean": float(np.mean(l_sd)),
                "std": float(np.std(l_sd)),
            },
        },
        "A": {
            "avg": {
                "mean": float(np.mean(a_mean)),
                "std": float(np.std(a_mean)),
            },
            "std": {
                "mean": float(np.mean(a_sd)),
                "std": float(np.std(a_sd)),
            },
        },
        "B": {
            "avg": {
                "mean": float(np.mean(b_mean)),
                "std": float(np.std(b_mean)),
            },
            "std": {
                "mean": float(np.mean(b_sd)),
                "std": float(np.std(b_sd)),
            },
        },
    }
    return params


class RandStainNAPIL:
    def __init__(self, template_path: str, randstainna_path: str = None):
        self.template_path = template_path
        self.randstainna_path = randstainna_path

    def __call__(self, image_array: Image.Image) -> Image.Image:
        image_array = self.randstainna_fn(image_array)
        image_array = cv2.cvtColor(np.array(image_array), cv2.COLOR_BGR2RGB)
        return Image.fromarray(image_array)

    def fit(self, subset_folder: Subset):
        if self.randstainna_path:
            sys.path.append(self.randstainna_path)
        else:
            sys.path.append(RANDSTAINNA_PATH)
        from randstainna import RandStainNA

        dist_params = get_dist_params(subset_folder)
        config_template = OmegaConf.load(self.template_path)
        config_template.update(dist_params)
        OmegaConf.save(config_template, self.template_path)

        self.randstainna_fn: callable = RandStainNA(
            yaml_file=self.template_path,
            std_hyper=-0.3,
            probability=0.5,
            distribution="normal",
            is_train=True,
        )

        return


class StainSepPIL:
    def __init__(
        self,
        aug_saturation: bool = True,
        aug_density: bool = True,
    ):
        self.aug_saturation = aug_saturation
        self.aug_density = aug_density
        self.distribution = "normal"

    def __call__(self, image: Image.Image) -> Image.Image:
        return self.augmentor.image_augmentation_with_stain_vector(
            image,
            aug_saturation=self.aug_saturation,
            aug_density=self.aug_density,
        )

    def fit(self, stain_cache_dir):
        self.augmentor = StructureAugmentor(dist=self.distribution, od_threshold=0.01)
        self.augmentor.load_stain_cache(stain_cache_dir)

        return


class MixAugPIL:
    def __init__(
        self,
        template_path: str,
        aug_saturation: bool = True,
        aug_density: bool = True,
        distribution="normal",
    ):
        self.template_path = template_path
        self.aug_saturation = aug_saturation
        self.aug_density = aug_density
        self.distribution = distribution

    def __call__(self, image: Image.Image) -> Image.Image:
        p = random.random()
        if p >= 0.5:
            return image

        if p >= 0.25:
            return self.randstainna_fn(image)

        return self.stain_sep(image)

    def fit(self, subset_folder, stain_cache_dir):
        self.stain_sep = StainSepPIL(self.aug_saturation, self.aug_density)
        self.stain_sep.fit(stain_cache_dir)

        self.randstainna_fn = RandStainNAPIL(self.template_path)
        self.randstainna_fn.fit(subset_folder)
        return


def add_sa(method, transform: Compose, args, subset_folder, center_name):

    if method == "spcn":
        aug_fn = StainSepPIL(
            aug_saturation=args.aug_saturation,
            aug_density=args.aug_density,
        )
        aug_fn.fit(os.path.join(args.stain_cache_dir, center_name))

    elif method == "randstainna":
        aug_fn = RandStainNAPIL(os.path.join(EXP_DIR, "randstainna_template.yaml"))
        aug_fn.fit(subset_folder)

    elif method == "mix":
        aug_fn = MixAugPIL(
            os.path.join(EXP_DIR, "randstainna_template.yaml"),
            aug_saturation=args.aug_saturation,
            aug_density=args.aug_density,
        )
        aug_fn.fit(subset_folder, os.path.join(args.stain_cache_dir, center_name))

    transform.transforms.insert(1, aug_fn)

    return


def cal_image_stats(
    image_folder: ImageFolder,
) -> Tuple[np.ndarray, np.ndarray]:
    sum_per_channel = np.zeros(3)
    sum_squared_per_channel = np.zeros(3)
    total_pixels = 0

    for image, label in tqdm.tqdm(image_folder):
        img_array = np.array(image) / 255.0
        sum_per_channel += np.sum(img_array, axis=(0, 1))
        sum_squared_per_channel += np.sum(img_array**2, axis=(0, 1))
        total_pixels += img_array.shape[0] * img_array.shape[1]

    mean = sum_per_channel / total_pixels
    std = np.sqrt((sum_squared_per_channel / total_pixels) - (mean**2))

    return mean, std


def get_dist_params(image_folder) -> dict:

    l_mean = list()
    l_sd = list()
    a_mean = list()
    a_sd = list()
    b_mean = list()
    b_sd = list()

    for image, label in image_folder:
        lab_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2LAB)
        L_channel, A_channel, B_channel = cv2.split(lab_image)
        l_mean.append(L_channel.mean())
        a_mean.append(A_channel.mean())
        b_mean.append(B_channel.mean())

        l_sd.append(L_channel.std())
        a_sd.append(A_channel.std())
        b_sd.append(B_channel.std())

    params = {
        "L": {
            "avg": {
                "mean": float(np.mean(l_mean)),
                "std": float(np.std(l_mean)),
            },
            "std": {
                "mean": float(np.mean(l_sd)),
                "std": float(np.std(l_sd)),
            },
        },
        "A": {
            "avg": {
                "mean": float(np.mean(a_mean)),
                "std": float(np.std(a_mean)),
            },
            "std": {
                "mean": float(np.mean(a_sd)),
                "std": float(np.std(a_sd)),
            },
        },
        "B": {
            "avg": {
                "mean": float(np.mean(b_mean)),
                "std": float(np.std(b_mean)),
            },
            "std": {
                "mean": float(np.mean(b_sd)),
                "std": float(np.std(b_sd)),
            },
        },
    }
    return params


def get_qq_indices_lab(imagefolder, lower, upper) -> np.ndarray:
    labs = defaultdict(list)
    for image, label in tqdm.tqdm(imagefolder):
        lab_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2LAB)
        mean_px = np.array(lab_image).mean(axis=(0, 1))

        labs["l"].append(mean_px[0])
        labs["a"].append(mean_px[1])
        labs["b"].append(mean_px[2])

    lab_distribution = np.stack(
        [np.array(labs["l"]), np.array(labs["a"]), np.array(labs["b"])], axis=1
    )
    mean_lab = np.mean(lab_distribution, axis=0)
    distances = np.abs(lab_distribution - mean_lab).sum(axis=1)
    lower_dist = np.quantile(distances, lower)
    upper_dist = np.quantile(distances, upper)

    target_indices = np.where((lower_dist <= distances) & (distances <= upper_dist))
    return target_indices[0]


def get_qq_indices_hsv(imagefolder, lower, upper) -> np.ndarray:
    hues = list()
    for image, label in tqdm.tqdm(imagefolder):
        mean_px = np.array(image).mean(axis=(0, 1))
        hsv = colorsys.rgb_to_hsv(*(mean_px / 255))
        hues.append(hsv[0])

    hues = np.array(hues)
    lower_dist = np.quantile(hues, lower)
    upper_dist = np.quantile(hues, upper)

    target_indices = np.where((lower_dist <= hues) & (hues <= upper_dist))
    return target_indices[0]
