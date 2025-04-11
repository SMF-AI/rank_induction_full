import glob
import random
from typing import Set, List, Tuple
from dataclasses import dataclass, field

import torch
import pandas as pd
import numpy as np
from PIL import Image
from torchvision.transforms import Compose

from .data_model import Labels
import h5py


def read_hd5_file(filepath, key) -> np.ndarray:
    with h5py.File(filepath, "r") as f:
        return np.array(f[key])


@dataclass
class BinarySet:
    benign: list = field(default_factory=list)
    malignant: list = field(default_factory=list)

    @property
    def index(self) -> List[int]:
        return self.benign + self.malignant


@dataclass
class CenterDataset:
    name: str
    train: BinarySet = field(default_factory=BinarySet)
    valid: BinarySet = field(default_factory=BinarySet)
    test: BinarySet = field(default_factory=BinarySet)


@dataclass
class PCAMSet:
    pcam_dir: str
    rumc: CenterDataset = CenterDataset("rumc")
    umcu: CenterDataset = CenterDataset("umcu")

    def fill_center_data(
        self,
        metadata: pd.DataFrame,
        rumc_slide_names: Set[str],
        umcu_slide_names: Set[str],
        phase: str,
    ) -> None:
        metadata["slide_name"] = (
            metadata["wsi"].str.split("_", n=1).apply(lambda x: x[1])
        )

        for i, row in metadata.iterrows():
            label = Labels.malignant.name if row["tumor_patch"] else Labels.benign.name
            slide_name = row["slide_name"]

            if slide_name in rumc_slide_names:
                binaryset: BinarySet = getattr(self.rumc, phase)
                getattr(binaryset, label).append(i)
            elif slide_name in umcu_slide_names:
                binaryset: BinarySet = getattr(self.umcu, phase)
                getattr(binaryset, label).append(i)

        return

    def split_center(self, rumc_slide_names: Set[str], umcu_slide_names: Set[str]):
        for file_path in glob.glob(self.pcam_dir + "/*"):
            for phase in ["train", "valid", "test"]:
                if "meta" in file_path:
                    metadata = pd.read_csv(file_path)
                    self.fill_center_data(
                        metadata, rumc_slide_names, umcu_slide_names, phase
                    )

        return self


class PcamTorchDataSet(torch.utils.data.Dataset):
    """_summary_


    Example:

    """

    def __init__(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        transfom: Compose,
        augmentations: List[callable] = list(),
        device="cuda",
    ):
        self.xs = xs
        self.ys = ys
        self.transform = transfom
        self.augmentations = augmentations
        self.device = device

    def __len__(self):
        return len(self.xs)

    def _stochastic_aug(self, image: Image.Image) -> Image.Image:
        if self.augmentations and random.random() < 0.5:
            aug = random.sample(self.augmentations, k=1)[0]
            try:
                image: Image.Image = aug(image)
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image)
            except np.linalg.LinAlgError:
                return image

        return image

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self.xs[idx]
        label = self.ys[idx]
        aug_image = self._stochastic_aug(Image.fromarray(image))

        return self.transform(aug_image).to(self.device), torch.tensor([label]).to(
            self.device
        ).view(-1)
