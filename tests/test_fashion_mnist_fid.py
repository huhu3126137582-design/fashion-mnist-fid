import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from fashion_mnist_fid import (
    FEATURE_DIM,
    MIN_TEST_ACCURACY,
    OutOfRangePixelStats,
    RunningFeatureStats,
    calculate_frechet_distance,
    default_real_stats_path,
    load_real_stats,
    validate_real_stats_payload,
    validate_classifier_metadata,
)
from models.fashion_mnist_classifier import (
    ARCHITECTURE_VERSION,
    FEATURE_VERSION,
    FashionMNISTClassifier,
)


def valid_classifier_metadata():
    return {
        "checkpoint_version": 1,
        "architecture_version": ARCHITECTURE_VERSION,
        "feature_version": FEATURE_VERSION,
        "feature_dim": FEATURE_DIM,
        "input_shape": [1, 32, 32],
        "input_range": [0.0, 1.0],
        "normalization": {"mean": [0.3], "std": [0.35]},
        "split": {
            "official_train_size": 60000,
            "train_size": 55000,
            "validation_size": 5000,
            "validation_per_class": 500,
        },
        "seed": 2026,
        "best_validation_accuracy": 0.94,
        "test_accuracy": MIN_TEST_ACCURACY,
    }


class ClassifierContractTests(unittest.TestCase):
    def setUp(self):
        self.model = FashionMNISTClassifier(mean=[0.3], std=[0.35]).eval()

    def test_features_are_direct_classifier_input(self):
        images = torch.rand(4, 1, 32, 32, dtype=torch.float32)
        captured = {}

        def capture_input(_module, inputs):
            captured["features"] = inputs[0].detach().clone()

        handle = self.model.classifier.register_forward_pre_hook(capture_input)
        try:
            logits = self.model(images)
        finally:
            handle.remove()
        features = self.model.extract_features(images)
        self.assertEqual(tuple(logits.shape), (4, 10))
        self.assertEqual(tuple(features.shape), (4, 128))
        torch.testing.assert_close(features, captured["features"])
        self.assertFalse(torch.allclose(features.norm(dim=1), torch.ones(4)))

    def test_input_contract_is_strict(self):
        with self.assertRaises(TypeError):
            self.model(torch.rand(2, 1, 32, 32, dtype=torch.float64))
        with self.assertRaises(ValueError):
            self.model(torch.rand(2, 3, 32, 32))
        invalid = torch.zeros(2, 1, 32, 32)
        invalid[0, 0, 0, 0] = 1.01
        with self.assertRaises(ValueError):
            self.model(invalid)

    def test_accuracy_threshold_is_enforced(self):
        metadata = valid_classifier_metadata()
        metadata["test_accuracy"] = MIN_TEST_ACCURACY - 1e-6
        with self.assertRaises(ValueError):
            validate_classifier_metadata(metadata)

    def test_nonfinite_normalization_is_rejected(self):
        metadata = valid_classifier_metadata()
        metadata["normalization"]["mean"] = [float("nan")]
        with self.assertRaises(ValueError):
            validate_classifier_metadata(metadata)


class StatisticsTests(unittest.TestCase):
    def test_running_stats_match_direct_computation(self):
        generator = torch.Generator().manual_seed(7)
        features = torch.randn(31, FEATURE_DIM, generator=generator)
        stats = RunningFeatureStats()
        stats.update(features[:10])
        stats.update(features[10:])
        mean, covariance = stats.compute()
        expected = features.double()
        torch.testing.assert_close(mean, expected.mean(dim=0))
        torch.testing.assert_close(
            covariance,
            torch.cov(expected.T),
            rtol=1e-10,
            atol=1e-10,
        )
        self.assertEqual(mean.dtype, torch.float64)
        self.assertEqual(covariance.dtype, torch.float64)

    def test_identical_distributions_have_zero_fid(self):
        generator = np.random.default_rng(11)
        matrix = generator.normal(size=(FEATURE_DIM, FEATURE_DIM))
        covariance = matrix @ matrix.T + np.eye(FEATURE_DIM) * 1e-3
        mean = generator.normal(size=FEATURE_DIM)
        value = calculate_frechet_distance(
            mean, covariance, mean.copy(), covariance.copy()
        )
        self.assertAlmostEqual(value, 0.0, places=5)

    def test_out_of_range_ratios_use_global_pixel_counts(self):
        stats = OutOfRangePixelStats()
        stats.update(torch.tensor([[[[-1.0, 0.0], [1.0, 2.0]]]]))
        stats.update(torch.tensor([[[[0.5, 0.5], [0.5, -0.1]]]]))
        result = stats.compute()
        self.assertEqual(result["pre_clamp_below_zero_pixel_ratio"], 2 / 8)
        self.assertEqual(result["pre_clamp_above_one_pixel_ratio"], 1 / 8)
        self.assertEqual(
            result["pre_clamp_out_of_range_pixel_ratio"], 3 / 8
        )

    def test_missing_real_stats_fails_without_rebuild(self):
        classifier_hash = "a" * 64
        expected_metadata = {
            "classifier_sha256": classifier_hash,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = (
                Path(directory)
                / f"real_stats_test10k_{classifier_hash[:16]}.pt"
            )
            with self.assertRaises(FileNotFoundError):
                load_real_stats(path, expected_metadata)

    def test_default_stats_path_contains_classifier_hash(self):
        classifier_hash = "0123456789abcdef" + "0" * 48
        self.assertIn(
            classifier_hash[:16],
            default_real_stats_path(classifier_hash).name,
        )

    def test_real_stats_metadata_mismatch_fails(self):
        classifier_hash = "b" * 64
        expected_metadata = {
            "classifier_sha256": classifier_hash,
            "feature_dim": FEATURE_DIM,
        }
        payload = {
            "metadata": {
                "classifier_sha256": classifier_hash,
                "feature_dim": FEATURE_DIM + 1,
            },
            "mu": torch.zeros(FEATURE_DIM, dtype=torch.float64),
            "sigma": torch.eye(FEATURE_DIM, dtype=torch.float64),
        }
        path = f"real_stats_test10k_{classifier_hash[:16]}.pt"
        with self.assertRaises(ValueError):
            validate_real_stats_payload(payload, expected_metadata, path)

if __name__ == "__main__":
    unittest.main()
