"""Evaluate generated images with the classifier-feature FashionMNIST-FID."""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm

from fashion_mnist_fid import (
    FEATURE_DIM,
    OutOfRangePixelStats,
    RunningFeatureStats,
    atomic_json_dump,
    calculate_frechet_distance,
    get_real_stats,
    load_classifier_checkpoint,
    sha256_file,
)


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class ImageDirectoryDataset(Dataset):
    def __init__(self, root):
        self.paths = sorted(
            path for path in Path(root).rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not self.paths:
            raise ValueError(f"No supported images found under {root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            image = image.convert("L")
            image = TF.resize(image, [32, 32], antialias=True)
            return TF.to_tensor(image)


def load_tensor_file(path):
    path = Path(path)
    if path.suffix.lower() == ".npz":
        archive = np.load(path)
        if "images" not in archive:
            raise ValueError("NPZ input must contain an 'images' array")
        images = torch.from_numpy(archive["images"])
    else:
        value = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(value, dict):
            if "images" not in value:
                raise ValueError("Tensor checkpoint must contain an 'images' key")
            value = value["images"]
        images = torch.as_tensor(value)
    if images.ndim == 3:
        images = images.unsqueeze(1)
    if images.ndim != 4 or images.shape[1] != 1:
        raise ValueError(
            f"Expected generated images [N, 1, H, W], received {tuple(images.shape)}"
        )
    if tuple(images.shape[-2:]) != (32, 32):
        images = TF.resize(images, [32, 32], antialias=True)
    return images.to(dtype=torch.float32)


def build_dataset(path):
    path = Path(path)
    if path.is_dir():
        return ImageDirectoryDataset(path), False
    if path.suffix.lower() not in {".pt", ".pth", ".npz"}:
        raise ValueError("Input must be an image directory or .pt/.pth/.npz file")
    return TensorDataset(load_tensor_file(path)), True


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--classifier", required=True)
    parser.add_argument("--real-stats", default=None)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", default="fashion_mnist_fid_result.json")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--rebuild-real-stats", action="store_true")
    parser.add_argument(
        "--clamp",
        action="store_true",
        help="Record out-of-range pixels and clamp tensor inputs to [0, 1]",
    )
    return parser.parse_args()


@torch.inference_mode()
def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier, classifier_metadata, classifier_hash = (
        load_classifier_checkpoint(args.classifier, device)
    )
    real_stats, real_stats_path, real_stats_hash = get_real_stats(
        path=args.real_stats,
        model=classifier,
        classifier_metadata=classifier_metadata,
        classifier_hash=classifier_hash,
        data_root=args.data_root,
        device=device,
        rebuild=args.rebuild_real_stats,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    dataset, raw_tensor_input = build_dataset(args.input)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    features = RunningFeatureStats(FEATURE_DIM)
    pixels = OutOfRangePixelStats()
    for batch in tqdm(loader, desc="Generated features", unit="batch"):
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        images = images.to(device=device, dtype=torch.float32, non_blocking=True)
        pixels.update(images)
        if raw_tensor_input and args.clamp:
            images = images.clamp(0, 1)
        elif images.min().item() < 0 or images.max().item() > 1:
            raise ValueError("Input contains values outside [0, 1]; pass --clamp explicitly")
        features.update(classifier.extract_features(images))
    fake_mu, fake_sigma = features.compute()
    score = calculate_frechet_distance(
        real_stats["mu"], real_stats["sigma"], fake_mu, fake_sigma
    )
    result = {
        "metric": "FashionMNIST-FID",
        "FashionMNIST-FID": score,
        "num_samples": features.num_samples,
        "input": os.path.abspath(args.input),
        "classifier_checkpoint": os.path.abspath(args.classifier),
        "classifier_checkpoint_sha256": classifier_hash,
        "classifier_test10k_accuracy": classifier_metadata["test_accuracy"],
        "real_stats": os.path.abspath(real_stats_path),
        "real_stats_sha256": real_stats_hash,
        "feature_dim": FEATURE_DIM,
        **pixels.compute(),
    }
    atomic_json_dump(result, args.output)
    print(f"FashionMNIST-FID: {score:.6f}")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
