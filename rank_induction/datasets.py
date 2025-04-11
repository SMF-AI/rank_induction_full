from __future__ import annotations

import os
import glob
import math
import random
from typing import List, Tuple
from collections import defaultdict, Counter

import cv2
import torch
import tqdm
import numpy as np
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from shapely import Polygon
from openslide import OpenSlide
from openslide.deepzoom import DeepZoomGenerator
from openslide.lowlevel import OpenSlideUnsupportedFormatError


from rank_induction.data_models import Patch, Patches, BinaryLabels, Polygons
from rank_induction.patch_filter import AnnotationFilter
from rank_induction.misc import get_deepzoom_level

CAMELYON_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CAMELYON_DIR)


class PatchDataSet(Dataset):
    """패치(Pathces)을 입력받는 데이터셋"""

    def __init__(self, patches: Patches, transform=None):
        self.patches = patches
        self.transform = transform

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, Patch]:
        patch: Patch = self.patches[idx]
        res = Image.fromarray(patch.image_array)

        if self.transform:
            res = self.transform(res)

        return res


class PathImageDataSet(Dataset):
    """패치의 이미지 경로(Path), 이미지(torch.Tensor)를 반환하는 데이터셋"""

    def __init__(
        self,
        patch_dir: str,
        transform=None,
        ext: str = "*.png",
    ):
        self.patch_dir = patch_dir
        self.transform = transform
        self.ext = ext

        self.patch_paths = glob.glob(os.path.join(self.patch_dir, ext))

    def __len__(self):
        return len(self.patch_paths)

    def __getitem__(self, idx) -> Tuple[str, torch.Tensor]:
        res = Image.open(self.patch_paths[idx])
        if self.transform:
            res = self.transform(res)

        return self.patch_paths[idx], res


class MILDataset(Dataset):
    """Multiple Instance Learning을 위한 데이터셋 클래스

    Multiple Instance Learning에서는 bag이라 불리는 인스턴스들의 집합을 다룹니다.
    각 bag은 여러 개의 이미지(인스턴스)로 구성되며, bag 단위로 레이블이 할당됩니다.

    Args:
        root_dir (str, optional): 데이터셋의 루트 디렉토리 경로.
            None인 경우 빈 데이터셋이 생성됩니다. Defaults to None.
        transform (callable, optional): 이미지에 적용할 변환 함수.
            None인 경우 변환을 적용하지 않습니다. Defaults to None.
        device (str, optional): 데이터를 로드할 디바이스. Defaults to "cuda".
        dry_run (bool, optional): True인 경우 클래스당 10개의 샘플만 로드합니다.
            테스트 목적으로 사용됩니다. Defaults to False.

    Attributes:
        bag_paths (List[List[str]]): 각 bag에 포함된 이미지들의 경로 리스트
        bag_labels (List[str]): 각 bag의 레이블 리스트

    Example:
        root_dir
        ├──M/
        │   ├── 10_11_150619.h5
        ...
        ├── M
        │   ├── 10_11_150619.h5

    """

    def __init__(
        self,
        root_dir: str = None,
        device: str = "cuda",
        dry_run: bool = False,
    ):
        self.root_dir = root_dir
        self.device = device
        self.bag_h5_paths: List[str] = list()
        self.bag_labels: torch.Tensor = torch.Tensor()
        self._temp_bag_labels = list()
        if root_dir is not None:
            self._update_bag_paths(dry_run)

    def _update_bag_paths(self, dry_run: bool = False) -> None:
        label_cnt = Counter()
        max_samples = (
            4 if dry_run else float("inf")
        )  # dry_run이면 4개, 아니면 제한 없음
        for dir_path, _, filenames in os.walk(self.root_dir):
            label = os.path.basename(dir_path)
            try:
                BinaryLabels[label].value
            except KeyError:
                continue

            label_value = BinaryLabels[label].value
            if label_cnt[label_value] >= max_samples:
                continue

            h5_files = [f for f in filenames if f.endswith(".h5")]
            for filename in h5_files:
                if label_cnt[label_value] >= max_samples:
                    break

                self.bag_h5_paths.append(os.path.join(dir_path, filename))
                self._temp_bag_labels.append(torch.tensor([label_value]))
                label_cnt[label_value] += 1

            if all(
                label_cnt[l] >= max_samples
                for l in [BinaryLabels["M"].value, BinaryLabels["N"].value]
            ):
                self.bag_labels = (
                    torch.stack(self._temp_bag_labels, dim=0)
                    if self._temp_bag_labels
                    else torch.tensor([])
                )
                return

        self.bag_labels = (
            torch.stack(self._temp_bag_labels, dim=0)
            if self._temp_bag_labels
            else torch.tensor([])
        )

        return

    def __len__(self) -> int:
        return len(self.bag_labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            idx (int)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: instance, bag_label
                1) instances: (N, D)
                2) bag_label: (1, )
        """

        h5_path = self.bag_h5_paths[idx]
        features: torch.Tensor = Patches.from_feature_h5(h5_path).features
        label: torch.Tensor = self.bag_labels[idx].float()

        return features.to(self.device), label.to(self.device)

    @classmethod
    def build_from_instance(
        cls, bag_h5_paths, bag_labels, device: str = "cuda"
    ) -> MILDataset:
        mil_feature_dataset = cls(root_dir=None, device=device)
        mil_feature_dataset.bag_h5_paths = bag_h5_paths
        mil_feature_dataset.bag_labels = bag_labels

        return mil_feature_dataset


class AttentionInductionDataset(MILDataset):
    """

    Example:
    >>> from camelyon.datasets import MixedSupervisionDataset
    >>> dataset = MixedSupervisionDataset(
            root_dir="/home/heon/heon_vast/datasets/camelyon17_otsu/feature/test",
            slide_dir="/home/heon/heon_vast/datasets/camelyon17_otsu/slide/test",
            annotation_dir="/CAMELYON16/annotations"
        )
    >>> dataset.collect_patch_labels()

    >>> instances, bag_label, patch_label = dataset[0]
    """

    def __init__(
        self,
        root_dir: str,
        slide_dir: str,
        image_dir: str,
        annotation_dir: str,
        mpp: float = 1.0,
        tile_size: int = 224,
        overlap: int = 0,
        device: str = "cuda",
        overlap_ratio: float = 0.05,
        morphology_value: int = 0,
        dry_run: bool = False,
    ):
        self.root_dir = root_dir
        self.slide_dir = slide_dir
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.mpp = mpp
        self.tile_size = tile_size
        self.overlap = overlap
        self.overlap_ratio = overlap_ratio
        self.morphology_value = morphology_value
        self.device = device

        self.bag_h5_paths: List[str] = list()
        self.bag_labels = torch.Tensor
        self._temp_bag_labels = list()
        self.patch_labels: List[List[bool]] = list()
        if root_dir is not None:
            self._update_bag_paths(dry_run)
            self.collect_patch_labels()

    def collect_patch_labels(self) -> None:
        self.annotations = self._collect_annotation()

        self.patch_labels: List[np.ndarray] = list()
        for bag_h5_path in tqdm.tqdm(
            self.bag_h5_paths,
            desc="Collecting patch labels from annotations",
            ncols=100,
        ):
            image_h5_path = bag_h5_path.replace(self.root_dir, self.image_dir)
            n_tissue = self._collect_patch_labels_for_bag(bag_h5_path, image_h5_path)
            if n_tissue.sum() == 0:
                self.patch_labels.append(np.array(n_tissue))

            else:
                n_tissue = np.array(n_tissue)
                n_tissue = n_tissue / n_tissue.sum()
                self.patch_labels.append(n_tissue)

        return

    def _collect_annotation(self) -> dict:
        """Annotation을 수집하여 Annotation filter을 저장"""
        if self.annotation_dir is None:
            return

        elif not os.path.exists(self.annotation_dir):
            raise FileNotFoundError(f"{self.annotation_dir} does not exist")

        self.annotations = dict()
        for root, dirs, files in os.walk(self.annotation_dir, followlinks=True):
            for file in files:
                if not file.endswith(".xml"):
                    continue

                annotation_path = os.path.join(root, file)
                annotation_polygon = Polygons.from_xml(annotation_path)
                polygons = annotation_polygon.data
                if self.morphology_value != 0:
                    morphology_polys = [
                        p.buffer(self.morphology_value) for p in polygons
                    ]
                    polygons = morphology_polys

                annotation_filter = AnnotationFilter(polygons, self.overlap_ratio)

                slide_name = os.path.basename(file).replace(".xml", "")
                self.annotations[slide_name] = annotation_filter

        return self.annotations

    def _check_is_masked(self, query_polygon: Polygon, slide_name: str) -> bool:
        """해당패치영역이 마스킹되어있는지 확인

        Return:
            bool: True 인경우 겹침을 의미

        - Annotation_filter가 True인 경우 겸침 없음.

        """
        if slide_name not in self.annotations:
            return False

        annotation_filter = self.annotations[slide_name]
        return not annotation_filter(query_polygon)

    def _check_cache(self, slide_name: str) -> str:
        """슬라이드 이름을 기반으로 캐시 파일 경로를 반환 (존재하면 경로, 없으면 None)"""
        if self.morphology_value != 0:
            cache_dir = os.path.join(
                ROOT_DIR,
                f"data/att_cache_labels/{self.mpp}_expanded_{self.morphology_value}",
            )
        else:
            cache_dir = os.path.join(ROOT_DIR, f"data/att_cache_labels/{self.mpp}")

        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{slide_name}.npy")

        return cache_path

    def _collect_patch_labels_for_bag(
        self, feature_h5_path: str, image_h5_path: str
    ) -> np.ndarray:
        """하나의 WSI내의 패치에 대한 라벨을 획득"""

        slide_name = os.path.basename(feature_h5_path).replace(".h5", "")
        cache_path = self._check_cache(slide_name)
        if os.path.exists(cache_path):
            return np.load(cache_path)

        feature_patches = Patches.from_feature_h5(feature_h5_path)
        image_patches = Patches.from_patch_h5(image_h5_path)

        slide_name = os.path.basename(feature_h5_path.rstrip(".h5"))
        label = os.path.dirname(feature_h5_path).split("/")[-1]
        slide_path = os.path.join(self.slide_dir, label, slide_name + ".tif")

        if slide_name not in self.annotations:
            empty_array = np.zeros(len(feature_patches.features), dtype=np.float32)
            np.save(cache_path, empty_array)
            return empty_array

        osr = OpenSlide(slide_path)
        w, h = osr.dimensions
        thumnail = np.array(osr.get_thumbnail((int(w / 16), int(h / 16))))
        gray_image = cv2.cvtColor(thumnail, cv2.COLOR_RGB2GRAY)
        threshold, binary_image = cv2.threshold(
            gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        dgz = DeepZoomGenerator(
            osr, tile_size=self.tile_size, overlap=self.overlap, limit_bounds=True
        )
        deepzoom_level = get_deepzoom_level(self.mpp, osr)

        wsi_patch_labels = []
        for i, patch in enumerate(feature_patches):
            location, level, size = dgz.get_tile_coordinates(
                deepzoom_level, patch.address
            )
            x1, y1 = location
            w, h = size
            x2 = x1 + w * osr.level_downsamples[level]
            y2 = y1 + h * osr.level_downsamples[level]
            polygon = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
            if not self._check_is_masked(polygon, slide_name):
                wsi_patch_labels.append(0)
                continue

            gray_patch = cv2.cvtColor(image_patches[i].image_array, cv2.COLOR_BGR2GRAY)
            n_tissue = (gray_patch <= threshold).sum()
            wsi_patch_labels.append(n_tissue)

        osr.close()
        wsi_patch_labels = np.array(wsi_patch_labels, dtype=np.float32)
        np.save(cache_path, wsi_patch_labels)
        return wsi_patch_labels

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            idx (int)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: instance, bag_label
                1) instances: (n, c, h, w)
                2) bag_label: (2, )
        """

        patches = Patches.from_feature_h5(self.bag_h5_paths[idx])
        features = patches.features.float()
        patch_label: torch.Tensor = torch.tensor(self.patch_labels[idx]).float()
        bag_label: torch.Tensor = self.bag_labels[idx].float()

        return (
            features.to(self.device),
            bag_label.to(self.device),
            patch_label.to(self.device),
        )

    @classmethod
    def build_from_instance(
        cls, bag_h5_paths, bag_labels, patch_labels, device: str = "cuda"
    ):
        dataset = cls(
            root_dir=None,
            slide_dir=None,
            image_dir=None,
            annotation_dir=None,
            device=device,
        )
        dataset.bag_h5_paths = bag_h5_paths
        dataset.bag_labels = bag_labels
        dataset.patch_labels = patch_labels

        return dataset


class RankNetInductionDataset(AttentionInductionDataset):
    """

    Example:
    >>> from camelyon.datasets import MixedSupervisionDataset
    >>> dataset = MixedSupervisionDataset(
            root_dir="/home/heon/heon_vast/datasets/camelyon17_otsu/feature/test",
            slide_dir="/home/heon/heon_vast/datasets/camelyon17_otsu/slide/test",
            annotation_dir="/CAMELYON16/annotations"
        )
    >>> dataset.collect_patch_labels()

    """

    def __init__(
        self,
        root_dir: str,
        slide_dir: str,
        annotation_dir: str,
        mpp: float = 1.0,
        tile_size: int = 224,
        overlap: int = 0,
        device: str = "cuda",
        overlap_ratio: float = 0.05,
        morphology_value: int = 0,
        dry_run: bool = False,
    ):

        self.root_dir = root_dir
        self.slide_dir = slide_dir
        self.annotation_dir = annotation_dir
        self.mpp = mpp
        self.tile_size = tile_size
        self.overlap = overlap
        self.device = device
        self.overlap_ratio = overlap_ratio
        self.morphology_value = morphology_value
        self.dry_run = dry_run

        self.bag_h5_paths: List[str] = list()
        self.bag_labels = torch.Tensor
        self._temp_bag_labels = list()
        self.patch_labels: List[List[bool]] = list()
        if root_dir is not None:
            self._update_bag_paths(dry_run)
            self.collect_patch_labels()

    def _check_cache(self, slide_name: str) -> str:
        """슬라이드 이름을 기반으로 캐시 파일 경로를 반환 (존재하면 경로, 없으면 None)"""
        if self.morphology_value != 0:
            cache_dir = os.path.join(
                ROOT_DIR,
                f"data/ltr_cache_labels/{self.mpp}_expanded_{self.morphology_value}",
            )
        else:
            cache_dir = os.path.join(ROOT_DIR, f"data/ltr_cache_labels/{self.mpp}")

        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{slide_name}.npy")

        return cache_path

    def _collect_patch_labels_for_bag(self, feature_h5_path: str) -> List[float]:
        """하나의 WSI내의 패치에 대한 라벨을 획득"""

        slide_name = os.path.basename(feature_h5_path).replace(".h5", "")
        cache_path = self._check_cache(slide_name)
        if os.path.exists(cache_path):
            return np.load(cache_path)

        feature_patches = Patches.from_feature_h5(feature_h5_path)

        slide_name = os.path.basename(feature_h5_path.rstrip(".h5"))
        label = os.path.dirname(feature_h5_path).split("/")[-1]
        slide_path = os.path.join(self.slide_dir, label, slide_name + ".tif")

        if slide_name not in self.annotations:
            empty_array = np.zeros(len(feature_patches.features), dtype=np.float32)
            np.save(cache_path, empty_array)
            return empty_array

        osr = OpenSlide(slide_path)
        w, h = osr.dimensions
        dgz = DeepZoomGenerator(
            osr,
            tile_size=self.tile_size,
            overlap=self.overlap,
        )
        deepzoom_level = get_deepzoom_level(self.mpp, osr)

        wsi_patch_labels = []
        for patch in feature_patches:
            location, level, size = dgz.get_tile_coordinates(
                deepzoom_level, patch.address
            )
            x1, y1 = location
            w, h = size
            x2 = x1 + w * osr.level_downsamples[level]
            y2 = y1 + h * osr.level_downsamples[level]
            polygon = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
            if not self._check_is_masked(polygon, slide_name):
                wsi_patch_labels.append(0)
            else:
                wsi_patch_labels.append(1)

        osr.close()
        wsi_patch_labels = np.array(wsi_patch_labels, dtype=np.float32)
        np.save(cache_path, wsi_patch_labels)
        return wsi_patch_labels

    def collect_patch_labels(self) -> None:
        self.annotations = self._collect_annotation()
        for feature_h5_path in tqdm.tqdm(
            self.bag_h5_paths,
            desc="Collecting patch labels from annotations",
            ncols=100,
        ):
            ranknet_label = self._collect_patch_labels_for_bag(feature_h5_path)
            self.patch_labels.append(ranknet_label)

        return

    @classmethod
    def build_from_instance(
        cls, bag_h5_paths, bag_labels, patch_labels, device: str = "cuda"
    ):
        dataset = cls(
            root_dir=None,
            slide_dir=None,
            annotation_dir=None,
            device=device,
        )
        dataset.bag_h5_paths = bag_h5_paths
        dataset.bag_labels = bag_labels
        dataset.patch_labels = patch_labels

        return dataset


def get_balanced_weight_sequence(dataset: MILDataset) -> torch.Tensor:
    """Compute a balanced weight sequence for MIL dataset.

    Args:
        dataset (MILDataset): The dataset containing bag labels.

    Returns:
        torch.Tensor: A tensor of weights for each bag, balancing the class distribution.

    Example:
        >>> dataset = MILDataset(...)
        >>> weights = get_balanced_weight_sequence(dataset)
        >>> print(weights.shape)  # torch.Size([num_samples])
    """

    bag_labels: torch.Tensor = dataset.bag_labels
    n_pos = bag_labels.sum().item()
    n_total = len(bag_labels)

    weight_per_class = [n_total / (n_total - n_pos), n_total / (n_pos)]

    balanced_weight = [weight_per_class[int(bag_label)] for bag_label in bag_labels]

    return torch.tensor(balanced_weight, dtype=torch.float32)


class DigestPathAttentionInductionDataset(AttentionInductionDataset):
    """

    Example:
    >>> from camelyon.datasets import DigestPathAttentionInductionDataset
    >>> dataset = DigestPathAttentionInductionDataset(
            root_dir="/home/heon/heon_vast/datasets/camelyon17_otsu/feature/test",
            slide_dir="/home/heon/heon_vast/datasets/camelyon17_otsu/slide/test",
            annotation_dir="/CAMELYON16/annotations"
        )
    >>> dataset.collect_patch_labels()

    >>> instances, bag_label, patch_label = dataset[0]
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _check_cache(self, slide_name: str) -> str:
        """슬라이드 이름을 기반으로 캐시 파일 경로를 반환 (존재하면 경로, 없으면 None)"""
        if self.morphology_value != 0:
            cache_dir = os.path.join(
                ROOT_DIR,
                f"data/att_cache_Digest_labels/{self.tile_size}_expanded_{self.morphology_value}",
            )
        else:
            cache_dir = os.path.join(
                ROOT_DIR, f"data/att_cache_Digest_labels/{self.tile_size}"
            )

        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{slide_name}.npy")

        return cache_path

    def _collect_patch_labels_for_bag(
        self, feature_h5_path: str, image_h5_path: str
    ) -> np.ndarray:
        """하나의 WSI내의 패치에 대한 라벨을 획득"""

        slide_name = os.path.basename(feature_h5_path).replace(".h5", "")
        cache_path = self._check_cache(slide_name)
        if os.path.exists(cache_path):
            return np.load(cache_path)

        feature_patches = Patches.from_feature_h5(feature_h5_path)
        image_patches = Patches.from_patch_h5(image_h5_path)

        slide_name = os.path.basename(feature_h5_path.rstrip(".h5"))
        label = os.path.dirname(feature_h5_path).split("/")[-1]
        slide_path = os.path.join(self.slide_dir, label, slide_name + ".jpg")

        if slide_name not in self.annotations:
            empty_array = np.zeros(len(feature_patches.features), dtype=np.float32)
            np.save(cache_path, empty_array)
            return empty_array

        img = Image.open(slide_path).convert("RGB")
        w, h = img.size
        thumnail = np.array(img.resize((int(w / 16), int(h / 16)), Image.BILINEAR))
        gray_image = cv2.cvtColor(thumnail, cv2.COLOR_RGB2GRAY)

        threshold, binary_image = cv2.threshold(
            gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        wsi_patch_labels = []
        for i, patch in enumerate(feature_patches):
            x1, y1 = patch.address
            x2 = x1 + self.tile_size
            y2 = y1 + self.tile_size
            polygon = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
            if not self._check_is_masked(polygon, slide_name):
                wsi_patch_labels.append(0)
                continue

            gray_patch = cv2.cvtColor(image_patches[i].image_array, cv2.COLOR_BGR2GRAY)
            n_tissue = (gray_patch <= threshold).sum()
            wsi_patch_labels.append(n_tissue)

        wsi_patch_labels = np.array(wsi_patch_labels, dtype=np.float32)
        np.save(cache_path, wsi_patch_labels)
        return wsi_patch_labels


class DigestPathRankNetInductionDataset(RankNetInductionDataset):
    """

    Example:
    >>> from camelyon.datasets import DigestPathRankNetInductionDataset
    >>> dataset = DigestPathRankNetInductionDataset(
            root_dir="/home/heon/heon_vast/datasets/camelyon17_otsu/feature/test",
            slide_dir="/home/heon/heon_vast/datasets/camelyon17_otsu/slide/test",
            annotation_dir="/CAMELYON16/annotations"
        )
    >>> dataset.collect_patch_labels()

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _check_cache(self, slide_name: str) -> str:
        """슬라이드 이름을 기반으로 캐시 파일 경로를 반환 (존재하면 경로, 없으면 None)"""
        if self.morphology_value != 0:
            cache_dir = os.path.join(
                ROOT_DIR,
                f"data/ltr_cache_Digest_labels/{self.tile_size}_expanded_{self.morphology_value}",
            )
        else:
            cache_dir = os.path.join(
                ROOT_DIR, f"data/ltr_cache_Digest_labels/{self.tile_size}"
            )

        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{slide_name}.npy")

        return cache_path

    def _collect_patch_labels_for_bag(self, feature_h5_path: str) -> List[float]:
        """하나의 WSI내의 패치에 대한 라벨을 획득"""

        slide_name = os.path.basename(feature_h5_path).replace(".h5", "")
        cache_path = self._check_cache(slide_name)
        if os.path.exists(cache_path):
            return np.load(cache_path)

        feature_patches = Patches.from_feature_h5(feature_h5_path)

        slide_name = os.path.basename(feature_h5_path.rstrip(".h5"))
        label = os.path.dirname(feature_h5_path).split("/")[-1]
        slide_path = os.path.join(self.slide_dir, label, slide_name + ".jpg")

        if slide_name not in self.annotations:
            empty_array = np.zeros(len(feature_patches.features), dtype=np.float32)
            np.save(cache_path, empty_array)
            return empty_array

        wsi_patch_labels = []
        for patch in feature_patches:
            x1, y1 = patch.address
            x2 = x1 + self.tile_size
            y2 = y1 + self.tile_size
            polygon = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
            if not self._check_is_masked(polygon, slide_name):
                wsi_patch_labels.append(0)
            else:
                wsi_patch_labels.append(1)

        wsi_patch_labels = np.array(wsi_patch_labels, dtype=np.float32)
        np.save(cache_path, wsi_patch_labels)
        return wsi_patch_labels
