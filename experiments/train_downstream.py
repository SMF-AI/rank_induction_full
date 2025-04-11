import os
import random
import argparse

import mlflow
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

from rank_induction.datasets import (
    MILDataset,
    AttentionInductionDataset,
    RankNetInductionDataset,
    DigestPathAttentionInductionDataset,
    DigestPathRankNetInductionDataset,
    get_balanced_weight_sequence,
)
from rank_induction.networks.mil import AttentionBasedFeatureMIL
from rank_induction.trainer import BinaryClassifierTrainer, MixedSupervisionTrainer
from rank_induction.losses import AttentionInductionLoss, RankNetLoss, get_pos_weight
from rank_induction.log_ops import TRACKING_URI, get_experiment
from rank_induction.misc import seed_everything, worker_init_fn


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        epilog=(
            "Example usage:\n"
            """python3 experiments/attention_induction/train_downstream.py \\
            --data_dir /CAMELYON16/feature/resnet18 \\ 
            --slide_dir /CAMELYON16/slide \\ 
            --learning 'attention_induction' \\ 
            --run_name linear_prob_atten_resnet18 \\
            --in_features 1024
            """
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/CAMELYON16/patch",
    )
    parser.add_argument(
        "--slide_dir",
        type=str,
        default="/CAMELYON16/slide",
    )
    parser.add_argument(
        "--annotation_dir",
        type=str,
        default="/CAMELYON16",
    )
    parser.add_argument(
        "-r",
        "--run_name",
        type=str,
        default="train_from_feature",
    )
    parser.add_argument("--in_features", type=int, default=512, required=True)

    # Required arguments
    parser.add_argument("--mpp", type=float, default=0.25, required=True)
    parser.add_argument(
        "--learning",
        type=str,
        help="training strategy",
        choices=["base", "attention_induction", "ltr"],
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help=(
            "For development purposes: "
            "runs with only 10 samples per class "
            "to verify code execution without runtime errors"
        ),
    )
    # 데이터셋 종류
    parser.add_argument(
        "--dataset",
        choices=["camelyon", "digest"],
        default="camelyon",
        help="which WSI collection to use",
    )

    # Optional arguments
    parser.add_argument(
        "--n_classes",
        type=int,
        required=False,
        default=1,
    )
    parser.add_argument("--sampling_ratio", type=float, default=1.0)
    parser.add_argument("--ignore_equal", action="store_true")
    parser.add_argument("--margin", default=1.0, type=float, help="Ranknet margin")
    parser.add_argument("--overlap_ratio", default=0.05, type=float)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--experiment_name", type=str, default="attention_induction")
    parser.add_argument("-a", "--accumulation_steps", type=int, default=8)
    parser.add_argument("--use_balanced_weight", action="store_true")
    parser.add_argument("--_lambda", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--max_patiences", type=int, default=20)
    parser.add_argument("--minimal_earlystop_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--random_state", type=int, default=2025)
    parser.add_argument(
        "--num_pos",
        type=int,
        default=None,
        help="Number of positive instances for RankNet loss calculation",
    )
    parser.add_argument(
        "--num_neg",
        type=int,
        default=None,
        help="Number of negative instances for RankNet loss calculation",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Threshold value to apply to the attention weight",
    )
    parser.add_argument(
        "--morphology_value",
        type=int,
        default=0,
        help="Morphology value for test polygon extensoin",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def get_encoder(model_name) -> torch.nn.Module:
    import torchvision.models as models

    if model_name not in models._api.list_models():
        raise ValueError(
            f"model_name({model_name}) must be one of {models._api.list_models()}"
        )

    encoder = getattr(models, model_name)(pretrained=True)
    if hasattr(encoder, "fc"):
        encoder.fc = nn.Identity()
    elif hasattr(encoder, "classifier"):
        encoder.classifier = nn.Identity()

    return encoder


def get_train_val_test_dataset_base(
    data_dir,
    device,
    dry_run: bool = False,
    random_state=2025,
    sampling_ratio: float = 1.0,
):
    train_dataset: Dataset = MILDataset(
        root_dir=os.path.join(data_dir, "train"),
        device=device,
        dry_run=dry_run,
    )

    train_bag_h5_paths = train_dataset.bag_h5_paths
    train_bag_labels = train_dataset.bag_labels
    if sampling_ratio != 1.0:
        total_n = len(train_bag_h5_paths)
        k = int(total_n * sampling_ratio)
        sampling_indices = random.sample(range(total_n), k)
        train_bag_h5_paths = [train_bag_h5_paths[i] for i in sampling_indices]
        train_bag_labels = torch.stack(
            [train_bag_labels[i] for i in sampling_indices], dim=0
        )

    train_bag_h5_paths, val_bag_h5_paths, train_bag_labels, val_bag_labels = (
        train_test_split(
            train_bag_h5_paths,
            train_bag_labels,
            test_size=0.2,
            stratify=train_bag_labels,
            random_state=random_state,
        )
    )

    train_dataset = MILDataset.build_from_instance(
        train_bag_h5_paths, train_bag_labels, device=device
    )
    val_dataset = MILDataset.build_from_instance(
        val_bag_h5_paths, val_bag_labels, device=device
    )

    test_dataset: Dataset = MILDataset(
        os.path.join(data_dir, "test"),
        device=device,
        dry_run=dry_run,
    )

    return train_dataset, val_dataset, test_dataset


def get_train_val_test_dataset(
    data_dir,
    device,
    dry_run: bool = False,
    random_state=2025,
    overlap_ratio: float = 0.05,
    mpp=0.25,
    sampling_ratio: float = 1.0,
    morphology_value: int = 0,
    dataset: str = "camelyon",
):
    if dataset == "camelyon":
        ds_cls = AttentionInductionDataset
        slide_dir = "/CAMELYON16/slide"
        annotation_dir = os.path.join("/CAMELYON16", "annotations")
        image_dir = "/CAMELYON16/patch/20x_h5"
    elif dataset == "digest":
        ds_cls = DigestPathAttentionInductionDataset
        slide_dir = "/DigestPath2019/task2_classification/slide"
        annotation_dir = os.path.join(
            "/DigestPath2019/task2_classification", "annotations"
        )
        image_dir = (
            "/DigestPath2019/task2_classification/patch/224"
        )

    train_dataset: Dataset = ds_cls(
        root_dir=os.path.join(data_dir, "train"),
        slide_dir=os.path.join(slide_dir, "train"),
        annotation_dir=annotation_dir,
        image_dir=os.path.join(image_dir, "train"),
        mpp=mpp,
        device=device,
        overlap_ratio=overlap_ratio,
        morphology_value=morphology_value,
        dry_run=dry_run,
    )
    train_bag_paths = train_dataset.bag_h5_paths
    train_bag_labels = train_dataset.bag_labels
    train_patch_labels = train_dataset.patch_labels
    if sampling_ratio != 1.0:
        total_n = len(train_bag_paths)
        k = int(total_n * sampling_ratio)
        sampling_indices = random.sample(range(total_n), k)
        train_bag_paths = [train_bag_paths[i] for i in sampling_indices]
        train_bag_labels = torch.stack(
            [train_bag_labels[i] for i in sampling_indices], dim=0
        )
        train_patch_labels = [train_patch_labels[i] for i in sampling_indices]

    (
        train_bag_paths,
        val_bag_paths,
        train_bag_labels,
        val_bag_labels,
        train_patch_labels,
        val_patch_labels,
    ) = train_test_split(
        train_bag_paths,
        train_bag_labels,
        train_patch_labels,
        test_size=0.2,
        stratify=train_bag_labels,
        random_state=random_state,
    )
    train_dataset = ds_cls.build_from_instance(
        train_bag_paths,
        train_bag_labels,
        train_patch_labels,
        device=device,
    )

    val_dataset = ds_cls.build_from_instance(
        val_bag_paths,
        val_bag_labels,
        val_patch_labels,
        device=device,
    )
    test_dataset: Dataset = ds_cls(
        root_dir=os.path.join(data_dir, "test"),
        slide_dir=os.path.join(slide_dir, "test"),
        annotation_dir=annotation_dir,
        image_dir=os.path.join(image_dir, "test"),
        mpp=mpp,
        device=device,
        dry_run=dry_run,
        overlap_ratio=overlap_ratio,
    )
    return train_dataset, val_dataset, test_dataset


def get_train_val_test_dataset_ranknet(
    data_dir,
    device,
    dry_run: bool = False,
    random_state=2025,
    mpp=0.25,
    overlap_ratio=0.05,
    sampling_ratio: float = 1.0,
    morphology_value: int = 0,
    dataset: str = "camelyon",
):

    if dataset == "camelyon":
        ds_cls = RankNetInductionDataset
        slide_dir = "/CAMELYON16/slide"
        annotation_dir = os.path.join("/CAMELYON16", "annotations")
    elif dataset == "digest":
        ds_cls = DigestPathRankNetInductionDataset
        slide_dir = "/DigestPath2019/task2_classification/slide"
        annotation_dir = os.path.join(
            "/DigestPath2019/task2_classification", "annotations"
        )

    train_dataset: Dataset = ds_cls(
        root_dir=os.path.join(data_dir, "train"),
        slide_dir=os.path.join(slide_dir, "train"),
        annotation_dir=annotation_dir,
        mpp=mpp,
        device=device,
        dry_run=dry_run,
        morphology_value=morphology_value,
        overlap_ratio=overlap_ratio,
    )
    train_bag_paths = train_dataset.bag_h5_paths
    train_bag_labels = train_dataset.bag_labels
    train_patch_labels = train_dataset.patch_labels

    if sampling_ratio != 1.0:
        total_n = len(train_bag_paths)
        k = int(total_n * sampling_ratio)
        sampling_indices = random.sample(range(total_n), k)
        train_bag_paths = [train_bag_paths[i] for i in sampling_indices]
        train_bag_labels = torch.stack(
            [train_bag_labels[i] for i in sampling_indices], dim=0
        )
        train_patch_labels = [train_patch_labels[i] for i in sampling_indices]

    (
        train_bag_paths,
        val_bag_paths,
        train_bag_labels,
        val_bag_labels,
        train_patch_labels,
        val_patch_labels,
    ) = train_test_split(
        train_bag_paths,
        train_bag_labels,
        train_patch_labels,
        test_size=0.2,
        stratify=train_bag_labels,
        random_state=random_state,
    )

    train_dataset = ds_cls.build_from_instance(
        train_bag_paths,
        train_bag_labels,
        train_patch_labels,
        device=device,
    )
    val_dataset = ds_cls.build_from_instance(
        val_bag_paths,
        val_bag_labels,
        val_patch_labels,
        device=device,
    )
    test_dataset: Dataset = ds_cls(
        root_dir=os.path.join(data_dir, "test"),
        slide_dir=os.path.join(slide_dir, "test"),
        annotation_dir=annotation_dir,
        mpp=mpp,
        device=device,
        dry_run=dry_run,
        overlap_ratio=overlap_ratio,
    )
    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    mp.set_start_method("spawn")

    args = get_args()
    seed_everything(args.random_state)

    if args.learning == "base":
        train_dataset, val_dataset, test_dataset = get_train_val_test_dataset_base(
            args.data_dir,
            args.device,
            dry_run=args.dry_run,
            random_state=args.random_state,
            sampling_ratio=args.sampling_ratio,
        )
    elif args.learning == "attention_induction":
        train_dataset, val_dataset, test_dataset = get_train_val_test_dataset(
            args.data_dir,
            args.device,
            dry_run=args.dry_run,
            random_state=args.random_state,
            mpp=args.mpp,
            sampling_ratio=args.sampling_ratio,
            morphology_value=args.morphology_value,
            dataset=args.dataset,
        )
    elif args.learning == "ltr":
        train_dataset, val_dataset, test_dataset = get_train_val_test_dataset_ranknet(
            args.data_dir,
            args.device,
            dry_run=args.dry_run,
            random_state=args.random_state,
            mpp=args.mpp,
            overlap_ratio=args.overlap_ratio,
            sampling_ratio=args.sampling_ratio,
            morphology_value=args.morphology_value,
            dataset=args.dataset,
        )

    weights = get_balanced_weight_sequence(train_dataset)
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))
    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        worker_init_fn=worker_init_fn,
    )
    val_dataloader = DataLoader(
        val_dataset,
        num_workers=args.num_workers,
        shuffle=True,
        prefetch_factor=args.prefetch_factor,
    )
    test_dataloader = DataLoader(
        test_dataset, num_workers=args.num_workers, shuffle=True
    )

    if args.learning == "attention_induction":
        return_with = "attention_weight"
    elif args.learning == "ltr":
        return_with = "attention_score"
    elif args.learning == "base":
        return_with = "contribution"
    else:
        raise ValueError(
            f"learning({args.learning}) must be one of 'attention_induction' or 'ltr'"
        )

    model = AttentionBasedFeatureMIL(
        in_features=args.in_features,
        adaptor_dim=256,
        num_classes=args.n_classes,
        threshold=args.threshold,
        return_with=return_with,
    ).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    weight = None
    if args.use_balanced_weight:
        weight = get_pos_weight(train_dataset.bag_labels).to(args.device)

    if args.learning == "base":
        trainer = BinaryClassifierTrainer(
            model=model,
            loss=torch.nn.BCEWithLogitsLoss(pos_weight=weight),
            optimizer=optimizer,
        )
    elif args.learning == "attention_induction":
        trainer = MixedSupervisionTrainer(
            model=model,
            loss=AttentionInductionLoss(_lambda=args._lambda, pos_weight=weight),
            optimizer=optimizer,
        )
    elif args.learning == "ltr":
        trainer = MixedSupervisionTrainer(
            model=model,
            loss=RankNetLoss(
                _lambda=args._lambda,
                sigma=args.sigma,
                ignore_equal=args.ignore_equal,
                num_pos=args.num_pos,
                num_neg=args.num_neg,
                device=args.device,
            ),
            optimizer=optimizer,
        )

    mlflow.set_tracking_uri(TRACKING_URI)
    exp = get_experiment(args.experiment_name)
    with mlflow.start_run(experiment_id=exp.experiment_id, run_name=args.run_name):
        mlflow.log_params(vars(args))
        mlflow.log_artifact(os.path.abspath(__file__))

        trainer.train(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            max_patiences=args.max_patiences,
            minimal_earlystop_epoch=args.minimal_earlystop_epoch,
            n_epochs=args.epochs,
            accumulation_steps=args.accumulation_steps,
            use_mlflow=True,
            verbose=args.verbose,
        )
        trainer.test(test_dataloader, use_mlflow=True, verbose=args.verbose)
        mlflow.pytorch.log_model(model, artifact_path="model")
