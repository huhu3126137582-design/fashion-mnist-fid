import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
import torchvision
from scipy import linalg
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from tqdm.auto import tqdm

from models.fashion_mnist_classifier import (
    ARCHITECTURE_VERSION,
    FEATURE_DIM,
    FEATURE_VERSION,
    INPUT_RANGE,
    INPUT_SHAPE,
    FashionMNISTClassifier,
)


MIN_TEST_ACCURACY = 0.93
REAL_STATS_VERSION = "fashion-test10k-stats-v1"
TEST_IMAGE_FILE = "t10k-images-idx3-ubyte"
TEST_LABEL_FILE = "t10k-labels-idx1-ubyte"


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_torch_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(fd)
    try:
        torch.save(value, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def atomic_json_dump(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def fashion_mnist_eval_transform():
    return T.Compose([
        T.Resize(
            (INPUT_SHAPE[1], INPUT_SHAPE[2]),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        ),
        T.ToTensor(),
    ])


def fashion_mnist_test_dataset(data_root):
    return torchvision.datasets.FashionMNIST(
        root=data_root,
        train=False,
        download=True,
        transform=fashion_mnist_eval_transform(),
    )


def dataset_test_fingerprint(data_root):
    raw_dir = Path(data_root) / "FashionMNIST" / "raw"
    fingerprints = {}
    for filename in (TEST_IMAGE_FILE, TEST_LABEL_FILE):
        path = raw_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing FashionMNIST test file: {path}")
        fingerprints[filename] = {
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return fingerprints


def classifier_checkpoint_metadata(
    mean,
    std,
    seed,
    split_metadata,
    best_validation_accuracy,
    test_accuracy,
):
    return {
        "checkpoint_version": 1,
        "architecture_version": ARCHITECTURE_VERSION,
        "feature_version": FEATURE_VERSION,
        "feature_dim": FEATURE_DIM,
        "input_shape": list(INPUT_SHAPE),
        "input_range": list(INPUT_RANGE),
        "normalization": {
            "mean": [float(mean)],
            "std": [float(std)],
        },
        "split": split_metadata,
        "seed": int(seed),
        "best_validation_accuracy": float(best_validation_accuracy),
        "test_accuracy": float(test_accuracy),
    }


def validate_classifier_metadata(metadata):
    if metadata.get("checkpoint_version") != 1:
        raise ValueError("Unsupported classifier checkpoint version")
    expected = {
        "architecture_version": ARCHITECTURE_VERSION,
        "feature_version": FEATURE_VERSION,
        "feature_dim": FEATURE_DIM,
        "input_shape": list(INPUT_SHAPE),
        "input_range": list(INPUT_RANGE),
    }
    for key, expected_value in expected.items():
        if metadata.get(key) != expected_value:
            raise ValueError(
                f"Classifier checkpoint {key} mismatch: "
                f"expected {expected_value!r}, received {metadata.get(key)!r}"
            )
    normalization = metadata.get("normalization")
    if not isinstance(normalization, dict):
        raise ValueError("Classifier checkpoint is missing normalization metadata")
    mean = normalization.get("mean")
    std = normalization.get("std")
    if not isinstance(mean, list) or len(mean) != 1:
        raise ValueError("Classifier checkpoint mean must contain one value")
    if not math.isfinite(float(mean[0])):
        raise ValueError("Classifier checkpoint mean must be finite")
    if (
        not isinstance(std, list)
        or len(std) != 1
        or not math.isfinite(float(std[0]))
        or float(std[0]) <= 0
    ):
        raise ValueError("Classifier checkpoint std must contain one positive value")
    test_accuracy = metadata.get("test_accuracy")
    if (
        test_accuracy is None
        or not math.isfinite(float(test_accuracy))
        or not MIN_TEST_ACCURACY <= float(test_accuracy) <= 1.0
    ):
        raise ValueError(
            "Classifier checkpoint test accuracy must be at least "
            f"{MIN_TEST_ACCURACY:.0%}; received {test_accuracy!r}"
        )
    best_validation_accuracy = metadata.get("best_validation_accuracy")
    if (
        best_validation_accuracy is None
        or not math.isfinite(float(best_validation_accuracy))
        or not 0 <= float(best_validation_accuracy) <= 1.0
    ):
        raise ValueError(
            "Classifier checkpoint best validation accuracy is invalid"
        )
    if not isinstance(metadata.get("seed"), int):
        raise ValueError("Classifier checkpoint seed must be an integer")
    split = metadata.get("split")
    if not isinstance(split, dict):
        raise ValueError("Classifier checkpoint is missing split metadata")
    expected_split = {
        "official_train_size": 60000,
        "train_size": 55000,
        "validation_size": 5000,
        "validation_per_class": 500,
    }
    for key, expected_value in expected_split.items():
        if split.get(key) != expected_value:
            raise ValueError(
                f"Classifier checkpoint split {key} must be {expected_value}"
            )


def save_classifier_checkpoint(state_dict, metadata, path):
    validate_classifier_metadata(metadata)
    atomic_torch_save(
        {"metadata": metadata, "state_dict": state_dict},
        path,
    )


def load_classifier_checkpoint(path, device):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Classifier checkpoint does not exist: {path}")
    checkpoint_hash = sha256_file(path)
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ValueError(f"Failed to load classifier checkpoint {path}: {exc}") from exc
    if not isinstance(checkpoint, dict):
        raise ValueError("Classifier checkpoint must be a dictionary")
    metadata = checkpoint.get("metadata")
    state_dict = checkpoint.get("state_dict")
    if not isinstance(metadata, dict) or not isinstance(state_dict, dict):
        raise ValueError(
            "Classifier checkpoint must contain metadata and state_dict"
        )
    validate_classifier_metadata(metadata)
    model = FashionMNISTClassifier(
        mean=metadata["normalization"]["mean"],
        std=metadata["normalization"]["std"],
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device=device, dtype=torch.float32)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, metadata, checkpoint_hash


class RunningFeatureStats:
    def __init__(self, feature_dim=FEATURE_DIM, device="cpu"):
        self.feature_dim = int(feature_dim)
        self.device = torch.device(device)
        self.sum = torch.zeros(
            self.feature_dim, dtype=torch.float64, device=self.device
        )
        self.sum_outer = torch.zeros(
            self.feature_dim, self.feature_dim,
            dtype=torch.float64, device=self.device
        )
        self.num_samples = 0

    def update(self, features):
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected features [B, {self.feature_dim}], "
                f"received {tuple(features.shape)}"
            )
        features = features.detach().to(device=self.device, dtype=torch.float64)
        if not torch.isfinite(features).all():
            raise ValueError("Features contain non-finite values")
        self.sum += features.sum(dim=0)
        self.sum_outer += features.T @ features
        self.num_samples += features.shape[0]

    def compute(self):
        if self.num_samples < 2:
            raise ValueError("At least two feature vectors are required")
        mean = self.sum / self.num_samples
        covariance_numerator = (
            self.sum_outer
            - self.num_samples * torch.outer(mean, mean)
        )
        covariance = covariance_numerator / (self.num_samples - 1)
        covariance = (covariance + covariance.T) * 0.5
        return mean.cpu(), covariance.cpu()


class OutOfRangePixelStats:
    def __init__(self):
        self.below_zero = 0
        self.above_one = 0
        self.total = 0

    def update(self, images):
        if not isinstance(images, torch.Tensor):
            raise TypeError("Images must be a torch.Tensor")
        self.below_zero += int((images < 0).sum().item())
        self.above_one += int((images > 1).sum().item())
        self.total += images.numel()

    def compute(self):
        if self.total == 0:
            raise ValueError("No pixels were accumulated")
        below_ratio = self.below_zero / self.total
        above_ratio = self.above_one / self.total
        return {
            "pre_clamp_below_zero_pixel_ratio": below_ratio,
            "pre_clamp_above_one_pixel_ratio": above_ratio,
            "pre_clamp_out_of_range_pixel_ratio": below_ratio + above_ratio,
        }


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(np.asarray(mu1, dtype=np.float64))
    mu2 = np.atleast_1d(np.asarray(mu2, dtype=np.float64))
    sigma1 = np.atleast_2d(np.asarray(sigma1, dtype=np.float64))
    sigma2 = np.atleast_2d(np.asarray(sigma2, dtype=np.float64))
    if mu1.shape != mu2.shape:
        raise ValueError(f"Mean shape mismatch: {mu1.shape} vs {mu2.shape}")
    if sigma1.shape != sigma2.shape:
        raise ValueError(
            f"Covariance shape mismatch: {sigma1.shape} vs {sigma2.shape}"
        )
    if sigma1.shape != (mu1.size, mu1.size):
        raise ValueError("Covariance shape does not match feature dimension")

    diff = mu1 - mu2
    covariance_mean = linalg.sqrtm(sigma1 @ sigma2)
    if not np.isfinite(covariance_mean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covariance_mean = linalg.sqrtm(
            (sigma1 + offset) @ (sigma2 + offset)
        )
    if np.iscomplexobj(covariance_mean):
        imaginary_diagonal = np.diagonal(covariance_mean).imag
        if not np.allclose(imaginary_diagonal, 0, atol=1e-3):
            raise ValueError(
                "Frechet distance produced a non-negligible imaginary component"
            )
        covariance_mean = covariance_mean.real

    value = (
        diff @ diff
        + np.trace(sigma1)
        + np.trace(sigma2)
        - 2 * np.trace(covariance_mean)
    )
    if not math.isfinite(float(value)):
        raise ValueError("FashionMNIST-FID is not finite")
    if value < -1e-6:
        raise ValueError(f"FashionMNIST-FID is negative: {value}")
    return max(float(value), 0.0)


def expected_real_stats_metadata(
    classifier_metadata,
    classifier_hash,
    data_root,
):
    return {
        "stats_version": REAL_STATS_VERSION,
        "dataset": "FashionMNIST",
        "split": "test",
        "num_samples": 10000,
        "feature_dim": FEATURE_DIM,
        "classifier_sha256": classifier_hash,
        "architecture_version": classifier_metadata["architecture_version"],
        "feature_version": classifier_metadata["feature_version"],
        "classifier_test_accuracy": float(
            classifier_metadata["test_accuracy"]
        ),
        "input_shape": list(INPUT_SHAPE),
        "input_range": list(INPUT_RANGE),
        "resize": {
            "source_size": [28, 28],
            "target_size": [32, 32],
            "interpolation": "bilinear",
            "antialias": True,
        },
        "normalization": classifier_metadata["normalization"],
        "test_files": dataset_test_fingerprint(data_root),
    }


def default_real_stats_path(classifier_hash):
    return Path("logs") / "fashion_mnist_fid" / (
        f"real_stats_test10k_{classifier_hash[:16]}.pt"
    )


def validate_real_stats_path(path, classifier_hash):
    classifier_prefix = classifier_hash[:16]
    if classifier_prefix not in Path(path).name:
        raise ValueError(
            "Real stats filename must contain classifier hash prefix "
            f"{classifier_prefix}"
        )


def validate_real_stats_payload(payload, expected_metadata, path):
    if not isinstance(payload, dict):
        raise ValueError("Real stats cache must be a dictionary")
    metadata = payload.get("metadata")
    mu = payload.get("mu")
    sigma = payload.get("sigma")
    if not isinstance(metadata, dict):
        raise ValueError("Real stats cache is missing metadata")
    if metadata != expected_metadata:
        mismatches = [
            key for key in sorted(set(metadata) | set(expected_metadata))
            if metadata.get(key) != expected_metadata.get(key)
        ]
        raise ValueError(
            "Real stats metadata mismatch for: " + ", ".join(mismatches)
        )
    if not isinstance(mu, torch.Tensor) or not isinstance(sigma, torch.Tensor):
        raise ValueError("Real stats mu and sigma must be tensors")
    if mu.dtype != torch.float64 or sigma.dtype != torch.float64:
        raise ValueError("Real stats mu and sigma must use float64")
    if tuple(mu.shape) != (FEATURE_DIM,):
        raise ValueError(f"Real stats mu has invalid shape {tuple(mu.shape)}")
    if tuple(sigma.shape) != (FEATURE_DIM, FEATURE_DIM):
        raise ValueError(
            f"Real stats sigma has invalid shape {tuple(sigma.shape)}"
        )
    if not torch.isfinite(mu).all() or not torch.isfinite(sigma).all():
        raise ValueError("Real stats contain non-finite values")
    validate_real_stats_path(path, expected_metadata["classifier_sha256"])


def load_real_stats(path, expected_metadata):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Real stats cache does not exist: {path}. "
            "Pass --rebuild_real_stats explicitly to create it."
        )
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ValueError(f"Failed to load real stats cache {path}: {exc}") from exc
    validate_real_stats_payload(payload, expected_metadata, path)
    return payload, sha256_file(path)


@torch.inference_mode()
def rebuild_real_stats(
    path,
    model,
    expected_metadata,
    data_root,
    device,
    batch_size=256,
    num_workers=4,
):
    dataset = fashion_mnist_test_dataset(data_root)
    if len(dataset) != expected_metadata["num_samples"]:
        raise ValueError(
            f"Expected 10000 FashionMNIST test samples, received {len(dataset)}"
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.device(device).type == "cuda",
    )
    accumulator = RunningFeatureStats(FEATURE_DIM, device="cpu")
    progress = tqdm(
        loader,
        desc="Real stats test10k",
        unit="batch",
        dynamic_ncols=True,
    )
    for images, _ in progress:
        images = images.to(device=device, dtype=torch.float32, non_blocking=True)
        features = model.extract_features(images)
        accumulator.update(features)
    mu, sigma = accumulator.compute()
    if accumulator.num_samples != expected_metadata["num_samples"]:
        raise ValueError(
            "Real stats sample count mismatch: "
            f"{accumulator.num_samples} vs {expected_metadata['num_samples']}"
        )
    atomic_torch_save(
        {
            "metadata": expected_metadata,
            "mu": mu,
            "sigma": sigma,
        },
        path,
    )


def get_real_stats(
    path,
    model,
    classifier_metadata,
    classifier_hash,
    data_root,
    device,
    rebuild=False,
    batch_size=256,
    num_workers=4,
):
    expected_metadata = expected_real_stats_metadata(
        classifier_metadata, classifier_hash, data_root
    )
    if path is None:
        path = default_real_stats_path(classifier_hash)
    path = Path(path)
    validate_real_stats_path(path, classifier_hash)
    if rebuild:
        rebuild_real_stats(
            path=path,
            model=model,
            expected_metadata=expected_metadata,
            data_root=data_root,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    payload, stats_hash = load_real_stats(path, expected_metadata)
    return payload, path, stats_hash
