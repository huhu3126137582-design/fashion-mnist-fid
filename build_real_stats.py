"""Build or validate classifier-specific FashionMNIST test10k statistics."""

import argparse

import torch

from fashion_mnist_fid import get_real_stats, load_classifier_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classifier", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier, metadata, classifier_hash = load_classifier_checkpoint(
        args.classifier, device
    )
    _, path, stats_hash = get_real_stats(
        path=args.output,
        model=classifier,
        classifier_metadata=metadata,
        classifier_hash=classifier_hash,
        data_root=args.data_root,
        device=device,
        rebuild=args.rebuild,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"Real stats: {path}")
    print(f"SHA256:    {stats_hash}")


if __name__ == "__main__":
    main()
