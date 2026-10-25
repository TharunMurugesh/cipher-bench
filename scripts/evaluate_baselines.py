from __future__ import annotations

import csv
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
import sklearn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.dataset import CLASS_MAPPING, LABEL_ORDER, POSITIVE_CLASS, load_binary_digits
from data.diagnostics import (
    activation_diagnostics,
    fit_standardizer,
    pixel_feature_diagnostics,
    preprocessing_diagnostics,
)
from data.splits import build_split_indices
from metrics.compute import compute_metrics
from models.baselines import MajorityClassifier, fit_logistic_regression
from models.polynomial_graph import PolynomialGraph
from models.tinycnn import TinyCNN, train_tinycnn


ARTIFACTS = ROOT / "artifacts"
SEED = 2026


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default, allow_nan=False) + "\n", encoding="utf-8")


def _timed_predictions(model, features: np.ndarray) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    predictions = model.predict(features)
    elapsed = time.perf_counter() - start
    return np.asarray(predictions, dtype=np.int64), elapsed


def _model_entry(name, predictions, labels, train_seconds, predict_seconds, extra=None):
    entry = {
        "metrics": compute_metrics(labels, predictions),
        "training_seconds": float(train_seconds),
        "training_timing_scope": "model fit only; excludes preprocessing; TinyCNN includes validation selection and outer-train refit",
        "test_batch_model_inference_seconds": float(predict_seconds),
        "test_batch_model_inference_ms_per_sample": float(1000 * predict_seconds / len(labels)),
        "timing_scope": "single full test-batch model call; excludes preprocessing; no warmup or repetition",
        "predictions": predictions.tolist(),
    }
    if extra:
        entry.update(extra)
    return entry


def _save_predictions_csv(path: Path, source_indices, labels, predictions_by_model):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["test_position", "filtered_dataset_index", "source_dataset_index", "y_true", *predictions_by_model])
        for position, (filtered_idx, source_idx, truth) in enumerate(zip(source_indices["filtered"], source_indices["source"], labels)):
            writer.writerow([position, int(filtered_idx), int(source_idx), int(truth), *[int(predictions_by_model[key][position]) for key in predictions_by_model]])


def run() -> dict:
    ARTIFACTS.mkdir(exist_ok=True)
    digits = load_binary_digits()
    splits = build_split_indices(digits.labels, seed=SEED)
    train_idx = splits.train_indices
    test_idx = splits.test_indices
    fit_idx = splits.fit_indices
    val_idx = splits.validation_indices
    x, y, images = digits.features, digits.labels, digits.images
    x_train, y_train = x[train_idx], y[train_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    outer_scaler_start = time.perf_counter()
    scaler = fit_standardizer(x_train)
    outer_scaler_fit_seconds = time.perf_counter() - outer_scaler_start
    outer_train_transform_start = time.perf_counter()
    x_train_scaled = scaler.transform(x_train)
    outer_train_transform_seconds = time.perf_counter() - outer_train_transform_start
    outer_test_transform_start = time.perf_counter()
    x_test_scaled = scaler.transform(x_test)
    outer_test_transform_seconds = time.perf_counter() - outer_test_transform_start

    dataset_summary = {
        "dataset": "sklearn.datasets.load_digits",
        "task": "digits 0 vs 1",
        "mapping": {"digit 0": CLASS_MAPPING["digit_0"], "digit 1": CLASS_MAPPING["digit_1"]},
        "positive_class": POSITIVE_CLASS,
        "label_order": LABEL_ORDER,
        "image_shape": list(images.shape[1:]),
        "flattened_feature_count": int(x.shape[1]),
        "flatten_order": "C row-major from original 8x8 image",
        "pixel_range": [float(x.min()), float(x.max())],
        "sample_count": int(len(y)),
        "class_counts_all": {str(c): int((y == c).sum()) for c in LABEL_ORDER},
        "class_counts_train": {str(c): int((y_train == c).sum()) for c in LABEL_ORDER},
        "split_id": splits.split_id,
        "input_diagnostics": pixel_feature_diagnostics(x, images, train_idx),
        "paper_preprocessing": preprocessing_diagnostics(scaler, x_train, x_test),
        "alternative_preprocessing": "Not run in Phase 1; paper-style StandardScaler remains a separate control.",
    }
    _write_json(ARTIFACTS / "dataset_summary.json", dataset_summary)
    _write_json(ARTIFACTS / "splits.json", splits.to_dict(digits.source_indices))

    # A: deterministic majority-class control.
    start = time.perf_counter()
    majority = MajorityClassifier().fit(y_train)
    majority_training_seconds = time.perf_counter() - start
    majority_predictions, majority_predict_seconds = _timed_predictions(majority, x_test)

    # B: paper-style per-feature standardization, fitted only on the outer train partition.
    start = time.perf_counter()
    logistic = fit_logistic_regression(x_train_scaled, y_train, seed=SEED)
    logistic_training_seconds = time.perf_counter() - start
    logistic_predictions, logistic_predict_seconds = _timed_predictions(logistic, x_test_scaled)

    # C: validation-only epoch selection, then refit from scratch for the selected epoch count
    # on all outer training examples. The scaler used during selection is fit on inner-fit only.
    selection_preprocessing_start = time.perf_counter()
    selection_scaler = fit_standardizer(x[fit_idx])
    x_fit_scaled = selection_scaler.transform(x[fit_idx])
    x_val_scaled = selection_scaler.transform(x[val_idx])
    selection_preprocessing_seconds = time.perf_counter() - selection_preprocessing_start
    start = time.perf_counter()
    _selected_model, selection = train_tinycnn(
        x_fit_scaled,
        y[fit_idx],
        x_val_scaled,
        y[val_idx],
        seed=SEED,
        kernel_count=1,
        max_epochs=1000,
        patience=30,
    )
    # Training for a fixed, validation-selected number of epochs on all outer-train data.
    final_model, refit_info = train_tinycnn_fixed_epochs(
        x_train_scaled,
        y_train,
        epochs=selection["best_epoch"],
        seed=SEED,
        kernel_count=1,
    )
    cnn_training_seconds = time.perf_counter() - start
    cnn_predictions, cnn_predict_seconds = _timed_predictions(final_model, x_test_scaled)

    # D: separately exposed float64 polynomial graph, instantiated with the fitted CNN weights.
    graph = PolynomialGraph(final_model.parameters())
    graph_start = time.perf_counter()
    graph_predictions = graph.predict(x_test_scaled)
    graph_predict_seconds = time.perf_counter() - graph_start
    graph_logits = graph.logits(x_test_scaled)
    cnn_logits = final_model.decision_function(x_test_scaled)
    graph_parity = {
        "dtype": str(graph_logits.dtype),
        "max_abs_logit_difference_from_tinycnn": float(np.max(np.abs(graph_logits - cnn_logits))),
        "mean_abs_logit_difference_from_tinycnn": float(np.mean(np.abs(graph_logits - cnn_logits))),
        "prediction_flip_count": int(np.count_nonzero(graph_predictions != cnn_predictions)),
        "identical_arithmetic_implementation": bool(np.array_equal(graph_logits, cnn_logits)),
    }

    results = {
        "run_id": f"baseline-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{splits.split_id[-8:]}",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "dataset": "sklearn.datasets.load_digits",
        "task": "digit 0 vs digit 1",
        "seed": SEED,
        "split_id": splits.split_id,
        "split_protocol": {
            "outer": "70/30 stratified train/test; chosen reconstruction protocol, not attributed to the paper",
            "inner": "15% of outer training reserved for stratified TinyCNN validation/epoch selection",
            "test_samples": int(len(test_idx)),
        },
        "test_labels": y_test.tolist(),
        "test_filtered_dataset_indices": test_idx.tolist(),
        "test_source_dataset_indices": digits.source_indices[test_idx].tolist(),
        "class_mapping": {"digit 0": CLASS_MAPPING["digit_0"], "digit 1": CLASS_MAPPING["digit_1"]},
        "positive_class": POSITIVE_CLASS,
        "label_order": LABEL_ORDER,
        "preprocessing": "Per-feature StandardScaler. Logistic scaler fit on outer train. CNN selection scaler fit on inner fit; final CNN refit scaler fit on outer train. No test labels used for fitting or model selection.",
        "dependency_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "models": {},
        "preprocessing_timings": {
            "outer_scaler_fit_seconds": float(outer_scaler_fit_seconds),
            "outer_train_transform_seconds": float(outer_train_transform_seconds),
            "outer_test_transform_seconds": float(outer_test_transform_seconds),
            "cnn_inner_fit_and_validation_transform_seconds": float(selection_preprocessing_seconds),
        },
        "polynomial_graph_parity": graph_parity,
        "diagnostics": {
            "tinycnn_training_partition": activation_diagnostics(final_model, x_train_scaled, y_train),
            "tinycnn_test_partition": activation_diagnostics(final_model, x_test_scaled, y_test),
            "validation_epoch_selection": {
                "best_epoch": selection["best_epoch"],
                "selection_epochs_run": selection["epochs_run"],
                "best_validation_loss": selection["best_validation_loss"],
                "epoch_limit_reached": selection["epochs_run"] == 1000,
                "validation_size": int(len(val_idx)),
                "final_refit_epochs": refit_info["epochs_run"],
            },
            "nonfinite_or_large_values": "See dataset_summary.json and activation range fields; values above 1e6 are counted explicitly.",
        },
        "limitations": [
            "Exact reproduction is not claimed; the paper does not report the split seed/ratio or a complete TinyCNN training recipe.",
            "A single trainable 3x3 convolution filter is an explicit reconstruction choice because the paper does not report kernel count.",
            "2x2 average pooling uses stride 2, yielding 3x3 pooled maps from the 6x6 valid-convolution map; stride was not specified in the paper.",
            "The TinyCNN is trained end-to-end with the square activation and a logistic loss head so plaintext and polynomial graph arithmetic match; this training choice is not fully specified by the paper.",
            "Inference timings are one full batch call without warmup or repeated trials and exclude preprocessing; interpret them as a local diagnostic, not as directly comparable to paper timings.",
        ],
    }
    results["models"]["majority"] = _model_entry(
        "majority", majority_predictions, y_test, majority_training_seconds, majority_predict_seconds,
        {"predicted_class": majority.label_, "threshold": None},
    )
    results["models"]["logistic_regression"] = _model_entry(
        "logistic_regression", logistic_predictions, y_test, logistic_training_seconds, logistic_predict_seconds,
        {"threshold": 0.5, "solver": "lbfgs", "C": 1.0, "max_iter": 2000,
         "test_preprocessing_ms_per_sample": float(1000 * outer_test_transform_seconds / len(test_idx))},
    )
    results["models"]["tinycnn"] = _model_entry(
        "tinycnn", cnn_predictions, y_test, cnn_training_seconds, cnn_predict_seconds,
        {"threshold": 0.5, "architecture": "1x 3x3 valid convolution -> square -> 2x2 average pool stride 2 -> logistic head",
         "kernel_count": 1, "selection": selection},
    )
    results["models"]["plaintext_equivalent_polynomial_graph"] = _model_entry(
        "plaintext_equivalent_polynomial_graph", graph_predictions, y_test, 0.0, graph_predict_seconds,
        {"threshold": 0.0, "architecture": "same fitted TinyCNN weights and float64 conv -> square -> average pool -> linear-logit head",
         "not_encrypted": True, "training_seconds": None},
    )

    _write_json(ARTIFACTS / "baseline_results.json", results)
    _write_json(ARTIFACTS / "model_parameters.json", {
        "split_id": splits.split_id,
        "logistic_regression": {
            "classes": logistic.classes_.tolist(),
            "coefficients": logistic.coef_.tolist(),
            "intercept": logistic.intercept_.tolist(),
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
        },
        "tinycnn": {
            "preprocessing": "StandardScaler fit on outer train only",
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "parameters": final_model.parameters(),
        },
    })

    predictions_by_model = {
        "majority_prediction": majority_predictions,
        "logistic_regression_prediction": logistic_predictions,
        "tinycnn_prediction": cnn_predictions,
        "polynomial_graph_prediction": graph_predictions,
    }
    _save_predictions_csv(
        ARTIFACTS / "baseline_predictions.csv",
        {"filtered": test_idx, "source": digits.source_indices[test_idx]},
        y_test,
        predictions_by_model,
    )
    print(json.dumps({
        "run_id": results["run_id"],
        "split_id": splits.split_id,
        "counts": dataset_summary["class_counts_all"],
        "train_size": len(train_idx),
        "test_size": len(test_idx),
        "metrics": {name: value["metrics"] for name, value in results["models"].items()},
        "graph_parity": graph_parity,
        "artifacts": [
            "artifacts/dataset_summary.json", "artifacts/splits.json", "artifacts/baseline_results.json",
            "artifacts/baseline_predictions.csv", "artifacts/model_parameters.json",
        ],
    }, indent=2, allow_nan=False))
    return results


def train_tinycnn_fixed_epochs(train_x, train_y, *, epochs: int, seed: int, kernel_count: int):
    """Refit on the complete outer train set for the validation-selected epoch count."""
    # Re-use the model's training loop without a validation-driven stop by passing the
    # same training data as its monitor. The exact epoch count is fixed by validation.
    from models.tinycnn import _fit_fixed_epochs

    return _fit_fixed_epochs(train_x, train_y, epochs=epochs, seed=seed, kernel_count=kernel_count), {
        "epochs_run": epochs,
    }


if __name__ == "__main__":
    run()

