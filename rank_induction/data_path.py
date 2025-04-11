import os
import glob
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .data_models import Labels, Centers


@dataclass
class CenterData:
    """Individual data in a specific center

    >>> center_data = CenterData("...path")

    # Access to malignant data
    >>> image_paths, labels = center_data.malignant

    # Access to all data
    >>> image_paths, labels = center_data.data

    """

    center_path: str

    def __post_init__(self):
        negative_paths = glob.glob(
            os.path.join(self.center_path, Labels.benign.name, "*.png")
        )
        negative_labels = np.zeros(len(negative_paths))
        setattr(self, Labels.benign.name, (negative_paths, negative_labels))

        positive_paths = glob.glob(
            os.path.join(self.center_path, Labels.malignant.name, "*.png")
        )
        positive_labels = np.ones(len(positive_paths))
        setattr(
            self,
            Labels.malignant.name,
            (positive_paths, positive_labels),
        )

    @property
    def data(self) -> Tuple[list, np.ndarray]:
        """Access tuple of (x, y)

        Returns:
            Tuple[list, np.ndarray]: list of image_paths, label array

        Examples:
            >>> center_data = CenterData("...path")
            >>> center_data.data
        """
        malignant_paths, malignant_labels = self.malignant
        benign_paths, benign_labels = self.benign

        return malignant_paths + benign_paths, np.concatenate(
            [malignant_labels, benign_labels]
        )


@dataclass
class ImagePaths:
    data_root: str

    def __post_init__(self):
        for center in Centers:
            setattr(
                self,
                center.name.lower(),
                CenterData(os.path.join(self.data_root, center.name)),
            )


CAMELYON16 = "/CAMELYON16"
