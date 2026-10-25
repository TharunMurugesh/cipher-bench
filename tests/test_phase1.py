from __future__ import annotations

import unittest

import numpy as np

from data.dataset import CLASS_MAPPING, LABEL_ORDER, POSITIVE_CLASS, load_binary_digits
from data.diagnostics import fit_standardizer
from data.splits import build_split_indices
from metrics.compute import compute_metrics
from models.polynomial_graph import PolynomialGraph
from models.tinycnn import TinyCNN, train_tinycnn


class DatasetAndSplitTests(unittest.TestCase):
    def test_binary_dataset_mapping_and_original_shapes(self):
        data = load_binary_digits()
        self.assertEqual(CLASS_MAPPING, {"digit_0": 0, "digit_1": 1})
        self.assertEqual(POSITIVE_CLASS, 1)
        self.assertEqual(LABEL_ORDER, [0, 1])
        self.assertEqual(data.images.shape, (360, 8, 8))
        self.assertEqual(data.features.shape, (360, 64))
        self.assertTrue(np.array_equal(data.features, data.images.reshape(-1, 64)))
        self.assertEqual(set(np.unique(data.labels)), {0, 1})
        self.assertEqual(data.source_indices.shape, (360,))

    def test_split_is_reproducible_stratified_and_disjoint(self):
        labels = load_binary_digits().labels
        first = build_split_indices(labels, seed=2026)
        second = build_split_indices(labels, seed=2026)
        self.assertEqual(first.split_id, second.split_id)
        np.testing.assert_array_equal(first.train_indices, second.train_indices)
        np.testing.assert_array_equal(first.test_indices, second.test_indices)
        self.assertEqual(len(first.test_indices), 108)
        self.assertEqual(len(np.intersect1d(first.train_indices, first.test_indices)), 0)
        self.assertEqual(set(first.fit_indices) | set(first.validation_indices), set(first.train_indices))
        self.assertEqual(len(np.intersect1d(first.fit_indices, first.validation_indices)), 0)
        self.assertLessEqual(abs(np.mean(labels[first.test_indices]) - np.mean(labels)), 0.01)

    def test_standardizer_uses_only_supplied_training_rows(self):
        train = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
        test = np.array([[1e9, -1e9]])
        scaler = fit_standardizer(train)
        scaled_test = scaler.transform(test)
        np.testing.assert_allclose(scaler.mean_, train.mean(axis=0))
        self.assertFalse(np.allclose(scaler.mean_, np.concatenate([train, test]).mean(axis=0)))
        self.assertTrue(np.isfinite(scaled_test).all())


class ModelAndMetricTests(unittest.TestCase):
    def test_tinycnn_and_polynomial_graph_output_shapes_and_parity(self):
        rng = np.random.default_rng(9)
        x = rng.normal(size=(5, 64))
        model = TinyCNN.initialize(kernel_count=1, seed=7)
        graph = PolynomialGraph(model.parameters())
        self.assertEqual(model.intermediates(x)["pre_square"].shape, (5, 1, 6, 6))
        self.assertEqual(model.intermediates(x)["squared"].shape, (5, 1, 6, 6))
        self.assertEqual(model.intermediates(x)["pooled"].shape, (5, 1, 3, 3))
        self.assertEqual(model.decision_function(x).shape, (5,))
        self.assertEqual(model.predict(x).shape, (5,))
        self.assertEqual(graph.logits(x).shape, (5,))
        np.testing.assert_allclose(graph.logits(x), model.decision_function(x), rtol=0, atol=1e-12)
        np.testing.assert_array_equal(graph.predict(x), model.predict(x))

    def test_tinycnn_training_returns_finite_model_and_validation_selection(self):
        rng = np.random.default_rng(10)
        x = rng.normal(size=(16, 64))
        y = np.array([0, 1] * 8)
        model, selection = train_tinycnn(
            x[:12], y[:12], x[12:], y[12:], seed=4, max_epochs=3, patience=4, batch_size=4
        )
        self.assertGreaterEqual(selection["best_epoch"], 1)
        self.assertLessEqual(selection["best_epoch"], 3)
        self.assertTrue(np.isfinite(model.decision_function(x)).all())

    def test_metrics_against_hand_calculated_example(self):
        result = compute_metrics(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]))
        self.assertEqual(result["confusion_matrix_labels_0_1"], [[1, 1], [0, 2]])
        self.assertAlmostEqual(result["accuracy"], 0.75)
        self.assertAlmostEqual(result["positive_class_metrics"]["precision"], 2 / 3)
        self.assertAlmostEqual(result["positive_class_metrics"]["recall"], 1.0)
        self.assertAlmostEqual(result["positive_class_metrics"]["f1"], 0.8)
        self.assertEqual(result["positive_class_metrics"]["support"], 2)

    def test_undefined_class_metrics_are_explicit_nulls(self):
        result = compute_metrics(np.array([0, 0]), np.array([0, 0]))
        self.assertIsNone(result["per_class"]["1"]["precision"])
        self.assertIsNone(result["per_class"]["1"]["recall"])
        self.assertIsNone(result["positive_class_metrics"]["f1"])
        self.assertIn("JSON null", result["undefined_metric_policy"])


if __name__ == "__main__":
    unittest.main()
