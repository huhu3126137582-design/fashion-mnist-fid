"""Build a measured FashionMNIST-FID degradation comparison for the README."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from fashion_mnist_fid import (
    FEATURE_DIM,
    RunningFeatureStats,
    calculate_frechet_distance,
    fashion_mnist_test_dataset,
    load_classifier_checkpoint,
)


@torch.inference_mode()
def main():
    root = Path(__file__).resolve().parents[1]
    classifier_path = root / "logs/fashion_mnist_fid/classifier.pt"
    stats_path = next(
        (root / "logs/fashion_mnist_fid").glob("real_stats_test10k_*.pt")
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier, _, _ = load_classifier_checkpoint(classifier_path, device)
    real_stats = torch.load(stats_path, map_location="cpu", weights_only=False)
    dataset = fashion_mnist_test_dataset(root / "data")
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)
    generator = torch.Generator().manual_seed(2026)
    variants = {
        "Original test images": lambda x, n: x,
        "Light noise": lambda x, n: (x + 0.10 * n).clamp(0, 1),
        "Strong noise": lambda x, n: (x + 0.30 * n).clamp(0, 1),
        "Random noise": lambda x, n: n.mul(0.5).add(0.5).clamp(0, 1),
    }
    accumulators = {name: RunningFeatureStats(FEATURE_DIM) for name in variants}
    previews = {}
    for images, _ in loader:
        noise = torch.randn(images.shape, generator=generator)
        for name, transform in variants.items():
            changed = transform(images, noise)
            if name not in previews:
                previews[name] = changed[:20]
            features = classifier.extract_features(
                changed.to(device=device, dtype=torch.float32)
            )
            accumulators[name].update(features)

    scores = {}
    for name, accumulator in accumulators.items():
        mu, sigma = accumulator.compute()
        scores[name] = calculate_frechet_distance(
            real_stats["mu"], real_stats["sigma"], mu, sigma
        )

    figure, axes = plt.subplots(4, 1, figsize=(12, 9), constrained_layout=True)
    for axis, (name, images) in zip(axes, previews.items()):
        grid = make_grid(images, nrow=20, padding=1, pad_value=1)
        axis.imshow(grid.permute(1, 2, 0), vmin=0, vmax=1)
        axis.set_title(
            f"{name} | measured FashionMNIST-FID: {scores[name]:.2f}",
            loc="left",
            fontsize=13,
            fontweight="bold",
        )
        axis.axis("off")
    figure.suptitle(
        "FashionMNIST-FID responds to increasing distribution corruption",
        fontsize=16,
        fontweight="bold",
    )
    output = root / "assets/fid_quality_comparison.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, facecolor="white")
    print(output)
    for name, score in scores.items():
        print(f"{name}: {score:.6f}")


if __name__ == "__main__":
    main()
