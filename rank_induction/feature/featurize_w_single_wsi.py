import os
import argparse
from logging import Logger

import h5py
import torch
import numpy as np
import torchvision
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.transforms import *
from transformers.modeling_outputs import BaseModelOutputWithPooling

from rank_induction.log_ops import get_logger
from rank_induction.data_models import Patches
from rank_induction.datasets import PathImageDataSet, PatchDataSet


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract features from a WSI",
        epilog=(
            "Example usage:\n"
            "  python3 camelyon/feature/featurize_w_single_wsi.py\\\n"
            "      --input_dir </path/to/input> \\\n"
            "      --output_dir </path/to/output> \\\n"
            "      --model_name resnet50 \\\n"
            "      --device cuda:0"
        ),
    )
    parser.add_argument(
        "-i",
        "--input_path",
        required=True,
        help=".h5 path",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        required=True,
        help="output 5h path",
    )
    parser.add_argument("-d", "--device", default="cuda:0")
    parser.add_argument("-m", "--model_name", default="histai/hibou-B")

    return parser.parse_args()


def load_model(model_name: str) -> torch.nn.Module:
    if model_name == "resnet50_custom":
        from rank_induction.networks.resnet50 import ResNetBackbone

        return ResNetBackbone().eval()

    if model_name not in models._api.list_models():
        raise ValueError(
            f"model_name({model_name}) must be one of {models._api.list_models()}"
        )
    else:
        encoder = getattr(models, model_name)(pretrained=True)

    if hasattr(encoder, "fc"):
        encoder.fc = nn.Identity()
    elif hasattr(encoder, "classifier"):
        encoder.classifier = nn.Identity()

    return encoder


def forward_h5(
    input_path: str,
    output_path: str,
    model: torch.nn.Module,
    transform: callable,
    logger: Logger,
    batch_size=32,
    device: str = "cuda",
):
    """입력인자로 전달되 디렉토리를 순회하면서, h5로 특징을 추출합니다"""

    logger.info(f"Processing directory: {input_path}")

    patches = Patches.from_patch_h5(input_path)
    dataset = PatchDataSet(patches, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    res = list()
    for transformed_images in dataloader:
        xs: torch.Tensor = transformed_images.to(device)
        output: torch.Tensor = model(xs)
        for vector in output:
            res.append(vector.detach().cpu().numpy())  # (D,)

    vectors = np.stack(res, axis=0)  # (N, D)

    with h5py.File(output_path, "w") as fh:
        for vector, patch in zip(vectors, patches):
            col, row = patch.address
            dataset = fh.create_dataset(f"{col}_{row}", data=vector, compression="gzip")
            dataset.attrs["col"] = col
            dataset.attrs["row"] = row
            dataset.attrs["x_min"] = patch.coordinates.x_min
            dataset.attrs["y_min"] = patch.coordinates.y_min
            dataset.attrs["x_max"] = patch.coordinates.x_max
            dataset.attrs["y_max"] = patch.coordinates.y_max

    logger.info(f"Saved features in {output_path}")

    return


def main() -> int:

    logger = get_logger(module_name="Featurizer")

    args = get_args()
    for k, v in vars(args).items():
        logger.info(f"{k}: {v}")

    logger.info("Load model")
    model = load_model(args.model_name)
    model.to(args.device).eval()

    transform = Compose(
        [
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    forward_h5(
        input_path=args.input_path,
        output_path=args.output_path,
        model=model,
        transform=transform,
        logger=logger,
        device=args.device,
    )


if __name__ == "__main__":
    main()
