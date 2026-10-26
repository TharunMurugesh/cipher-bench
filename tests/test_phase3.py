from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits

from data.benchmark import load_digit_pair
from data.splits import build_split_indices
from metrics.compute import compute_metrics
from models.tinycnn import TinyCNN, range_penalty_value_and_gradient, train_tinycnn
from scripts.run_phase3_ablation import write_json


class TestPhase3Ablations(unittest.TestCase):
    def test_range_penalty_gradient_matches_finite_difference(self):
        values = np.asarray([-3.1, -0.5, 2.7], dtype=np.float64)
        bound, coefficient, epsilon = 2.0, 0.3, 1e-6
        value, gradient = range_penalty_value_and_gradient(values, bound=bound, coefficient=coefficient)
        numeric = np.zeros_like(values)
        for i in range(values.size):
            plus, minus = values.copy(), values.copy()
            plus[i] += epsilon
            minus[i] -= epsilon
            lp = range_penalty_value_and_gradient(plus, bound=bound, coefficient=coefficient)[0]
            lm = range_penalty_value_and_gradient(minus, bound=bound, coefficient=coefficient)[0]
            numeric[i] = (lp - lm) / (2 * epsilon)
        self.assertGreater(value, 0)
        np.testing.assert_allclose(gradient, numeric, rtol=1e-5, atol=1e-7)

    def test_range_penalty_training_is_finite_and_keeps_model_shapes(self):
        rng = np.random.default_rng(71)
        x = rng.normal(0, 0.2, (16, 64))
        y = np.asarray([0, 1] * 8, dtype=np.int64)
        model, details = train_tinycnn(
            x[:12], y[:12], x[12:], y[12:], seed=71, max_epochs=2, patience=2,
            range_bound=0.25, range_penalty=0.1,
        )
        outputs = model.intermediates(x[12:])
        self.assertEqual(outputs["pre_square"].shape, (4, 1, 6, 6))
        self.assertEqual(outputs["squared"].shape, (4, 1, 6, 6))
        self.assertEqual(outputs["pooled"].shape, (4, 1, 3, 3))
        self.assertTrue(np.isfinite(model.decision_function(x[12:])).all())
        self.assertEqual(details["range_bound"], 0.25)
        self.assertEqual(details["range_penalty"], 0.1)

    def test_secondary_benchmark_mapping_and_split_are_reproducible(self):
        data_a, data_b = load_digit_pair(3, 8), load_digit_pair(3, 8)
        np.testing.assert_array_equal(data_a.labels, data_b.labels)
        self.assertEqual(set(np.unique(data_a.labels)), {0, 1})
        source = load_digits()
        np.testing.assert_array_equal(data_a.labels, (source.target[data_a.source_indices] == 8).astype(np.int64))
        split_a = build_split_indices(data_a.labels, seed=3038)
        split_b = build_split_indices(data_b.labels, seed=3038)
        self.assertEqual(split_a.split_id, split_b.split_id)
        np.testing.assert_array_equal(split_a.train_indices, split_b.train_indices)
        self.assertFalse(np.intersect1d(split_a.train_indices, split_a.test_indices).size)
        self.assertEqual(len(split_a.test_indices), int(np.ceil(len(data_a.labels) * 0.30)))

    def test_metrics_and_result_persistence(self):
        result = compute_metrics(np.asarray([0, 0, 1, 1]), np.asarray([0, 1, 1, 1]))
        self.assertEqual(result["accuracy"], 0.75)
        self.assertEqual(result["positive_class_metrics"]["recall"], 1.0)
        self.assertEqual(result["confusion_matrix_labels_0_1"], [[1, 1], [0, 2]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "roundtrip.json"
            write_json(path, {"labels": [0, 1], "predictions": [0, 1], "accuracy": 1.0})
            import json
            restored = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(restored["predictions"], [0, 1])


if __name__ == "__main__":
    unittest.main()
