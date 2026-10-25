from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import tenseal as ts

from data.dataset import load_binary_digits
from he.ckks_inference import (
    EncryptedPathError,
    encrypted_features,
    make_context,
    plaintext_stages,
)


ROOT = Path(__file__).resolve().parents[1]


class TestCKKSFeatureExtraction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        parameter_doc = json.loads((ROOT / "artifacts/model_parameters.json").read_text(encoding="utf-8"))
        split_doc = json.loads((ROOT / "artifacts/splits.json").read_text(encoding="utf-8"))
        cls.parameters = parameter_doc["tinycnn"]["parameters"]
        cls.mean = np.asarray(parameter_doc["tinycnn"]["scaler_mean"], dtype=np.float64)
        cls.scale = np.asarray(parameter_doc["tinycnn"]["scaler_scale"], dtype=np.float64)
        cls.data = load_binary_digits()
        cls.sample_index = int(split_doc["indices"]["fit"][0])
        cls.raw = cls.data.features[cls.sample_index]
        cls.x = (cls.raw - cls.mean) / cls.scale
        cls.context = make_context("depth3_40bit_exploratory")

    def test_encrypted_addition(self):
        left = ts.ckks_vector(self.context, [1.25, -2.0])
        right = ts.ckks_vector(self.context, [0.5, 3.0])
        np.testing.assert_allclose((left + right).decrypt(), [1.75, 1.0], rtol=0.0, atol=1e-5)

    def test_ciphertext_times_plaintext_weights(self):
        encrypted = ts.ckks_vector(self.context, [0.5, -2.0, 3.0])
        product = encrypted * [2.0, 0.25, -1.0]
        np.testing.assert_allclose(product.decrypt(), [1.0, -0.5, -3.0], rtol=0.0, atol=1e-4)
        dot = encrypted.dot([2.0, 0.25, -1.0])
        self.assertAlmostEqual(dot.decrypt()[0], -2.5, delta=1e-4)

    def test_square_activation(self):
        encrypted = ts.ckks_vector(self.context, [-1.25, 0.5, 2.0])
        squared = encrypted.square()
        np.testing.assert_allclose(squared.decrypt(), [1.5625, 0.25, 4.0], rtol=0.0, atol=2e-4)

    def test_encrypted_average_pooling(self):
        encrypted = ts.ckks_vector(self.context, [1.0, 2.0, 3.0, 4.0])
        squared = encrypted.square()
        total = squared.sum()
        average = total * 0.25
        self.assertAlmostEqual(average.decrypt()[0], 7.5, delta=2e-4)

    def test_complete_feature_extraction_matches_float64_graph(self):
        result = encrypted_features(self.x, self.parameters, self.context)
        expected = plaintext_stages(self.x, self.parameters)
        self.assertEqual(len(result.convolution), 36)
        self.assertEqual(len(result.squared), 36)
        self.assertEqual(len(result.pooled), 9)
        np.testing.assert_allclose(result.convolution, expected["convolution"], rtol=0.0, atol=0.02)
        np.testing.assert_allclose(result.squared, expected["squared"], rtol=0.0, atol=0.05)
        np.testing.assert_allclose(result.pooled, expected["pooled"], rtol=0.0, atol=0.05)
        self.assertEqual(result.states["input"]["coeff_modulus_size"], 4)
        self.assertEqual(result.states["convolution"][0]["coeff_modulus_size"], 3)
        self.assertEqual(result.states["square"][0]["coeff_modulus_size"], 2)
        self.assertEqual(result.states["pooled"][0]["coeff_modulus_size"], 1)

    def test_requested_paper_parameters_stop_at_square(self):
        context = make_context("paper_literal")
        with self.assertRaises(EncryptedPathError) as caught:
            encrypted_features(self.x, self.parameters, context)
        self.assertEqual(caught.exception.stage, "square")
        self.assertIn("scale out of bounds", str(caught.exception))
        self.assertEqual(caught.exception.partial["states"]["convolution"][0]["coeff_modulus_size"], 2)


if __name__ == "__main__":
    unittest.main()
