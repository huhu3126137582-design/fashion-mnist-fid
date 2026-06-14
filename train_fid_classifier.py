import argparse
import hashlib
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from tqdm.auto import tqdm

from fashion_mnist_fid import (
    MIN_TEST_ACCURACY,
    classifier_checkpoint_metadata,
    fashion_mnist_eval_transform,
    save_classifier_checkpoint,
    sha256_file,
)
from models.fashion_mnist_classifier import FashionMNISTClassifier, INPUT_SHAPE


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the fixed FashionMNIST-FID feature classifier"
    )
    parser.add_argument("--data_root", default="data")
    parser.add_argument(
        "--output",
        default="logs/fashion_mnist_fid/classifier.pt",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def stratified_split(targets, seed, validation_per_class=500):
    targets = torch.as_tensor(targets, dtype=torch.long)
    generator = torch.Generator().manual_seed(seed)
    train_indices = []
    validation_indices = []
    train_counts = {}
    validation_counts = {}
    for class_id in range(10):
        class_indices = torch.where(targets == class_id)[0]
        permutation = torch.randperm(class_indices.numel(), generator=generator)
        class_indices = class_indices[permutation]
        validation = class_indices[:validation_per_class].tolist()
        train = class_indices[validation_per_class:].tolist()
        validation_indices.extend(validation)
        train_indices.extend(train)
        train_counts[str(class_id)] = len(train)
        validation_counts[str(class_id)] = len(validation)
    return train_indices, validation_indices, train_counts, validation_counts


def indices_sha256(indices):
    payload = ",".join(str(index) for index in indices).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def build_datasets(data_root, seed):
    eval_transform = fashion_mnist_eval_transform()
    train_transform = T.Compose([
        T.Resize(
            (INPUT_SHAPE[1], INPUT_SHAPE[2]),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        ),
        T.RandomCrop((32, 32), padding=4, padding_mode="reflect"),
        T.RandomHorizontalFlip(),
        T.RandomRotation(
            degrees=8,
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        ),
        T.ToTensor(),
        T.RandomErasing(
            p=0.15,
            scale=(0.02, 0.12),
            ratio=(0.5, 2.0),
            value=0,
        ),
    ])
    base = torchvision.datasets.FashionMNIST(
        root=data_root, train=True, download=True, transform=eval_transform
    )
    train_indices, validation_indices, train_counts, validation_counts = (
        stratified_split(base.targets, seed)
    )
    augmented = torchvision.datasets.FashionMNIST(
        root=data_root, train=True, download=True, transform=train_transform
    )
    fixed = torchvision.datasets.FashionMNIST(
        root=data_root, train=True, download=True, transform=eval_transform
    )
    test = torchvision.datasets.FashionMNIST(
        root=data_root, train=False, download=True, transform=eval_transform
    )
    split_metadata = {
        "name": "stratified-55k-5k-v1",
        "official_train_size": len(base),
        "train_size": len(train_indices),
        "validation_size": len(validation_indices),
        "validation_per_class": 500,
        "train_class_counts": train_counts,
        "validation_class_counts": validation_counts,
        "train_indices_sha256": indices_sha256(train_indices),
        "validation_indices_sha256": indices_sha256(validation_indices),
    }
    return (
        Subset(augmented, train_indices),
        Subset(fixed, validation_indices),
        test,
        fixed,
        split_metadata,
    )


@torch.inference_mode()
def compute_mean_std(dataset, batch_size, num_workers):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    pixel_sum = torch.zeros((), dtype=torch.float64)
    pixel_squared_sum = torch.zeros((), dtype=torch.float64)
    pixel_count = 0
    for images, _ in tqdm(
        loader,
        desc="Dataset mean/std",
        unit="batch",
        dynamic_ncols=True,
    ):
        images = images.to(dtype=torch.float64)
        pixel_sum += images.sum()
        pixel_squared_sum += (images * images).sum()
        pixel_count += images.numel()
    mean = pixel_sum / pixel_count
    variance = pixel_squared_sum / pixel_count - mean.square()
    return mean.item(), variance.clamp_min(0).sqrt().item()


def cosine_with_warmup(optimizer, total_steps, warmup_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return max(step, 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.inference_mode()
def evaluate_accuracy(model, loader, device, description):
    model.eval()
    correct = 0
    total = 0
    progress = tqdm(
        loader,
        desc=description,
        unit="batch",
        dynamic_ncols=True,
        leave=False,
    )
    for images, targets in progress:
        images = images.to(device=device, dtype=torch.float32, non_blocking=True)
        targets = targets.to(device=device, non_blocking=True)
        logits = model(images)
        correct += int((logits.argmax(dim=1) == targets).sum().item())
        total += targets.numel()
        progress.set_postfix(accuracy=f"{correct / total:.4%}")
    return correct / total


def main():
    args = parse_args()
    output = os.path.abspath(args.output)
    if os.path.exists(output) and not args.overwrite:
        raise FileExistsError(
            f"Classifier checkpoint already exists: {output}. "
            "Pass --overwrite to replace it."
        )
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> device: {device}")
    print(f">>> output: {output}")

    train_set, validation_set, test_set, statistics_set, split_metadata = (
        build_datasets(args.data_root, args.seed)
    )
    mean, std = compute_mean_std(
        statistics_set, args.eval_batch_size, args.num_workers
    )
    print(f">>> normalization: mean={mean:.9f}, std={std:.9f}")

    generator = torch.Generator().manual_seed(args.seed)
    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        generator=generator,
        **loader_kwargs,
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=args.eval_batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.eval_batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    model = FashionMNISTClassifier(mean=[mean], std=[std]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    total_steps = args.epochs * len(train_loader)
    scheduler = cosine_with_warmup(
        optimizer,
        total_steps=total_steps,
        warmup_steps=max(len(train_loader), total_steps // 20),
    )

    best_validation_accuracy = -1.0
    best_state_dict = None
    epoch_progress = tqdm(
        range(1, args.epochs + 1),
        desc="Classifier epochs",
        unit="epoch",
        dynamic_ncols=True,
    )
    for epoch in epoch_progress:
        model.train()
        correct = 0
        total = 0
        loss_sum = 0.0
        batch_progress = tqdm(
            train_loader,
            desc=f"Train {epoch:02d}/{args.epochs:02d}",
            unit="batch",
            dynamic_ncols=True,
            leave=False,
        )
        for images, targets in batch_progress:
            images = images.to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            targets = targets.to(device=device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits = model(images)
                loss = F.cross_entropy(
                    logits,
                    targets,
                    label_smoothing=args.label_smoothing,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()

            batch_size = targets.numel()
            loss_sum += loss.item() * batch_size
            correct += int((logits.argmax(dim=1) == targets).sum().item())
            total += batch_size
            batch_progress.set_postfix(
                loss=f"{loss_sum / total:.4f}",
                accuracy=f"{correct / total:.2%}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

        validation_accuracy = evaluate_accuracy(
            model,
            validation_loader,
            device,
            description=f"Validation {epoch:02d}",
        )
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        epoch_progress.set_postfix(
            train_accuracy=f"{correct / total:.2%}",
            validation_accuracy=f"{validation_accuracy:.2%}",
            best=f"{best_validation_accuracy:.2%}",
        )

    if best_state_dict is None:
        raise RuntimeError("Training did not produce a classifier state")
    model.load_state_dict(best_state_dict, strict=True)
    model.to(device)
    test_accuracy = evaluate_accuracy(
        model, test_loader, device, description="Test test10k"
    )
    print(f">>> best validation accuracy: {best_validation_accuracy:.4%}")
    print(f">>> test10k accuracy:         {test_accuracy:.4%}")

    if test_accuracy < MIN_TEST_ACCURACY:
        raise RuntimeError(
            f"Classifier test accuracy {test_accuracy:.4%} is below the "
            f"required {MIN_TEST_ACCURACY:.0%}; no formal checkpoint was written"
        )

    metadata = classifier_checkpoint_metadata(
        mean=mean,
        std=std,
        seed=args.seed,
        split_metadata=split_metadata,
        best_validation_accuracy=best_validation_accuracy,
        test_accuracy=test_accuracy,
    )
    save_classifier_checkpoint(best_state_dict, metadata, output)
    print(f">>> classifier checkpoint: {output}")
    print(f">>> classifier SHA256:     {sha256_file(output)}")


if __name__ == "__main__":
    main()
