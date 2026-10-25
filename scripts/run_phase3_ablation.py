from __future__ import annotations

import csv
import json
import platform
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
import sklearn
import tenseal as ts
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.benchmark import load_digit_pair
from data.dataset import load_binary_digits
from data.splits import build_split_indices
from he.ckks_inference import (
    PARAMETER_SETS,
    encrypted_features,
    encrypted_im2col_partial_inference,
    error_summary,
    make_context,
    plaintext_stages,
)
from metrics.compute import compute_metrics
from metrics.statistics import classification_wilson_intervals, mcnemar_exact
from models.polynomial_graph import PolynomialGraph
from models.tinycnn import TinyCNN, _fit_fixed_epochs, train_tinycnn


SOURCE_ARTIFACTS = ROOT / "artifacts"
ARTIFACTS = Path(os.environ.get("CIPHERBENCH_ARTIFACT_DIR", SOURCE_ARTIFACTS)).resolve()
SEED_PRIMARY = 2026
SEED_SECONDARY = 3038
MAX_EPOCHS = 1000
PATIENCE = 30
L2 = 1e-4
RANGE_GRID = {
    "standard": {"bounds": [2.0, 4.0, 8.0], "penalties": [1e-3, 1e-2, 1e-1]},
    "pixel_0_1": {"bounds": [0.25, 0.5, 1.0], "penalties": [1e-3, 1e-2, 1e-1]},
}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _transform(raw: np.ndarray, preprocessing: str, scaler: StandardScaler | None) -> np.ndarray:
    values = np.asarray(raw, dtype=np.float64)
    if preprocessing == "standard":
        if scaler is None:
            raise ValueError("StandardScaler must be fit on the supplied training partition")
        return scaler.transform(values)
    if preprocessing == "pixel_0_1":
        return values / 16.0
    raise ValueError(f"Unknown preprocessing: {preprocessing}")


def _fit_preprocessor(raw_train: np.ndarray, preprocessing: str) -> tuple[StandardScaler | None, dict]:
    if preprocessing == "standard":
        scaler = StandardScaler().fit(raw_train)
        return scaler, {"kind": "per_feature_StandardScaler", "fit_rows": int(len(raw_train)),
                        "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(), "ddof": 0}
    return None, {"kind": "fixed_pixel_scale", "mapping": "raw_pixel / 16", "fit_rows": 0,
                  "reason": "The source images have fixed integer intensity domain [0,16]; no learned transform."}


def _metric_entry(y_true: np.ndarray, logits: np.ndarray, threshold: float = 0.0) -> dict:
    predictions = (np.asarray(logits) >= threshold).astype(np.int64)
    metrics = compute_metrics(y_true, predictions)
    metrics["wilson_95_intervals"] = classification_wilson_intervals(y_true, predictions)
    metrics["threshold_logit"] = float(threshold)
    return {"metrics": metrics, "predictions": predictions.tolist(), "logits": np.asarray(logits, dtype=float).tolist()}


def _log_loss(y: np.ndarray, logits: np.ndarray) -> float:
    return float(np.mean(np.logaddexp(0.0, logits) - y * logits))


def _activation_diagnostics(model: TinyCNN, x: np.ndarray, bound: float | None) -> dict:
    stages = model.intermediates(x)
    z = np.asarray(stages["pre_square"], dtype=np.float64)
    square = np.asarray(stages["squared"], dtype=np.float64)
    pooled = np.asarray(stages["pooled"], dtype=np.float64)
    return {
        "pre_square": {"min": float(z.min()), "max": float(z.max()), "mean": float(z.mean()),
                       "std": float(z.std()), "max_abs": float(np.abs(z).max()),
                       "bound": None if bound is None else float(bound),
                       "bound_violation_rate": None if bound is None else float(np.mean(np.abs(z) > bound)),
                       "bound_violation_fraction_of_elements": None if bound is None else float(np.mean(np.abs(z) > bound))},
        "square": {"min": float(square.min()), "max": float(square.max()), "max_abs": float(np.abs(square).max())},
        "pooled": {"min": float(pooled.min()), "max": float(pooled.max()), "max_abs": float(np.abs(pooled).max())},
        "nonfinite_count": int(sum(np.size(v) - np.isfinite(v).sum() for v in (z, square, pooled))),
        "count_abs_over_1e6": int(sum(np.sum(np.abs(v) > 1e6) for v in (z, square, pooled))),
    }


def _fit_validation_candidate(
    raw_x: np.ndarray,
    y: np.ndarray,
    fit_indices: np.ndarray,
    validation_indices: np.ndarray,
    *,
    preprocessing: str,
    range_bound: float | None,
    range_penalty: float,
    seed: int,
) -> dict:
    scaler, scaler_doc = _fit_preprocessor(raw_x[fit_indices], preprocessing)
    x_fit = _transform(raw_x[fit_indices], preprocessing, scaler)
    x_val = _transform(raw_x[validation_indices], preprocessing, scaler)
    start = time.perf_counter()
    selected_model, selection = train_tinycnn(
        x_fit, y[fit_indices], x_val, y[validation_indices], seed=seed,
        kernel_count=1, max_epochs=MAX_EPOCHS, patience=PATIENCE, l2=L2,
        range_bound=range_bound, range_penalty=range_penalty,
    )
    training_seconds = time.perf_counter() - start
    val_logits = selected_model.decision_function(x_val)
    metric_result = _metric_entry(y[validation_indices], val_logits)
    loss = _log_loss(y[validation_indices], val_logits)
    diagnostics = _activation_diagnostics(selected_model, x_val, range_bound)
    return {
        "selected_model": selected_model,
        "selection": selection,
        "validation": metric_result,
        "validation_log_loss": loss,
        "validation_activation_diagnostics": diagnostics,
        "preprocessor": scaler,
        "preprocessing_record": scaler_doc,
        "selection_training_seconds": float(training_seconds),
        "ranking": [
            metric_result["metrics"]["accuracy"],
            metric_result["metrics"]["positive_class_metrics"]["f1"],
            -loss,
            -float(diagnostics["pre_square"]["bound_violation_rate"] or 0.0),
        ],
    }


def _config_summary(candidate: dict) -> dict:
    return {key: candidate[key] for key in (
        "selection", "validation", "validation_log_loss", "validation_activation_diagnostics",
        "preprocessing_record", "selection_training_seconds", "ranking",
    )}


def _search_range_candidate(
    raw_x: np.ndarray,
    y: np.ndarray,
    fit_indices: np.ndarray,
    validation_indices: np.ndarray,
    preprocessing: str,
    seed: int,
) -> tuple[float, float, dict, list[dict]]:
    trials = []
    best = None
    grid = RANGE_GRID[preprocessing]
    for bound in grid["bounds"]:
        for penalty in grid["penalties"]:
            candidate = _fit_validation_candidate(
                raw_x, y, fit_indices, validation_indices, preprocessing=preprocessing,
                range_bound=bound, range_penalty=penalty, seed=seed,
            )
            trial = {"range_bound": bound, "range_penalty": penalty} | _config_summary(candidate)
            trials.append(trial)
            if best is None or tuple(candidate["ranking"]) > tuple(best["candidate"]["ranking"]):
                best = {"bound": bound, "penalty": penalty, "candidate": candidate}
    if best is None:
        raise RuntimeError("Range-penalty grid was empty")
    return best["bound"], best["penalty"], best["candidate"], trials


def _fit_full_training(
    raw_x: np.ndarray,
    y: np.ndarray,
    train_indices: np.ndarray,
    *,
    preprocessing: str,
    range_bound: float | None,
    range_penalty: float,
    epochs: int,
    seed: int,
) -> tuple[TinyCNN, StandardScaler | None, dict, float]:
    scaler, scaler_doc = _fit_preprocessor(raw_x[train_indices], preprocessing)
    x_train = _transform(raw_x[train_indices], preprocessing, scaler)
    start = time.perf_counter()
    model = _fit_fixed_epochs(
        x_train, y[train_indices], epochs=epochs, seed=seed, kernel_count=1,
        range_bound=range_bound, range_penalty=range_penalty,
    )
    return model, scaler, scaler_doc, float(time.perf_counter() - start)


def _select_threshold(y: np.ndarray, logits: np.ndarray) -> dict:
    unique = np.unique(np.asarray(logits, dtype=np.float64))
    candidates = {0.0}
    candidates.update(float(v) for v in unique)
    candidates.update(float((a + b) / 2.0) for a, b in zip(unique[:-1], unique[1:]))
    ranked = []
    for threshold in sorted(candidates):
        result = _metric_entry(y, logits, threshold)
        m = result["metrics"]
        ranked.append((m["positive_class_metrics"]["f1"], m["accuracy"], -abs(threshold), threshold, result))
    selected = max(ranked, key=lambda entry: entry[:3])
    return {"selection_rule": "maximize validation positive-class F1, then accuracy, then prefer threshold nearest zero",
            "threshold_logit": float(selected[3]), "validation_metrics": selected[4]["metrics"]}


def _secondary_split(data) -> dict:
    split = build_split_indices(data.labels, seed=SEED_SECONDARY, test_fraction=0.30,
                                validation_fraction_of_train=0.15)
    return split.to_dict(data.source_indices)


def _benchmark_definitions() -> dict:
    primary = load_binary_digits()
    split_primary = json.loads((SOURCE_ARTIFACTS / "splits.json").read_text(encoding="utf-8"))
    secondary = load_digit_pair(3, 8)
    split_secondary = _secondary_split(secondary)
    splits = {"digits_0_vs_1": split_primary, "digits_3_vs_8": split_secondary}
    write_json(ARTIFACTS / "phase3_splits.json", splits)
    return {
        "digits_0_vs_1": (primary, split_primary, "digit 0 -> 0, digit 1 -> 1"),
        "digits_3_vs_8": (secondary, split_secondary, "digit 3 -> 0, digit 8 -> 1"),
    }


def _fit_task(task_name: str, data, split_doc: dict, mapping: str, *, seed: int) -> tuple[dict, dict]:
    idx = split_doc["indices"]
    fit = np.asarray(idx["fit"], dtype=np.int64)
    val = np.asarray(idx["validation"], dtype=np.int64)
    train = np.asarray(idx["train"], dtype=np.int64)
    x, y = data.features, data.labels
    if set(np.unique(y)) != {0, 1}:
        raise ValueError(f"{task_name} labels do not follow the positive class convention")

    definitions = {
        "A0_standard_scaler_control": ("standard", None, 0.0),
        "A1_pixel_scale_only": ("pixel_0_1", None, 0.0),
    }
    selected_candidates = {}
    trial_records = {}
    for name, (preprocessing, bound, penalty) in definitions.items():
        candidate = _fit_validation_candidate(x, y, fit, val, preprocessing=preprocessing,
                                              range_bound=bound, range_penalty=penalty, seed=seed)
        selected_candidates[name] = candidate
        trial_records[name] = [_config_summary(candidate)]

    a2_bound, a2_penalty, a2, a2_trials = _search_range_candidate(x, y, fit, val, "standard", seed)
    a3_bound, a3_penalty, a3, a3_trials = _search_range_candidate(x, y, fit, val, "pixel_0_1", seed)
    selected_candidates["A2_standard_scaler_plus_range_penalty"] = a2
    selected_candidates["A3_pixel_scale_plus_range_penalty_RC_TinyCNN"] = a3
    trial_records["A2_standard_scaler_plus_range_penalty"] = a2_trials
    trial_records["A3_pixel_scale_plus_range_penalty_RC_TinyCNN"] = a3_trials

    configurations = {
        "A0_standard_scaler_control": ("standard", None, 0.0),
        "A1_pixel_scale_only": ("pixel_0_1", None, 0.0),
        "A2_standard_scaler_plus_range_penalty": ("standard", a2_bound, a2_penalty),
        "A3_pixel_scale_plus_range_penalty_RC_TinyCNN": ("pixel_0_1", a3_bound, a3_penalty),
    }
    model_results = {}
    model_records = {}
    train_metrics_y = y[train]
    val_y = y[val]
    for name, (preprocessing, bound, penalty) in configurations.items():
        candidate = selected_candidates[name]
        final_model, final_scaler, final_preproc_record, refit_seconds = _fit_full_training(
            x, y, train, preprocessing=preprocessing, range_bound=bound, range_penalty=penalty,
            epochs=int(candidate["selection"]["best_epoch"]), seed=seed,
        )
        x_train_final = _transform(x[train], preprocessing, final_scaler)
        train_logits = final_model.decision_function(x_train_final)
        x_val_selection = _transform(x[val], preprocessing, candidate["preprocessor"])
        validation_logits = candidate["selected_model"].decision_function(x_val_selection)
        val_thresholded = (validation_logits >= 0.0).astype(np.int64)
        record = {
            "configuration": {"preprocessing": preprocessing, "range_bound": bound,
                              "range_penalty_lambda": penalty, "kernel_count": 1,
                              "architecture": "3x3 valid convolution -> square -> 2x2 average pool stride 2 -> linear logistic-loss head",
                              "seed": seed, "positive_class": 1, "threshold_logit": 0.0},
            "selection": {"best_epoch": int(candidate["selection"]["best_epoch"]),
                          "epochs_run": int(candidate["selection"]["epochs_run"]),
                          "validation_objective": "binary logistic loss + head L2 + configured preactivation range penalty"},
            "training_metrics": _metric_entry(train_metrics_y, train_logits)["metrics"],
            "validation_metrics": candidate["validation"]["metrics"],
            "validation_log_loss": candidate["validation_log_loss"],
            "training_activation_diagnostics": _activation_diagnostics(final_model, x_train_final, bound),
            "validation_activation_diagnostics": candidate["validation_activation_diagnostics"],
            "preprocessing_selection_fit_rows": int(len(fit)),
            "preprocessing_final_fit_rows": int(len(train)),
            "preprocessing_final": final_preproc_record,
            "selection_search_training_seconds": float(candidate["selection_training_seconds"]),
            "full_train_refit_seconds": float(refit_seconds),
            "total_model_training_seconds": float(candidate["selection_training_seconds"] + refit_seconds),
            "validation_logits_from_selection_model": np.asarray(validation_logits).tolist(),
            "validation_predictions_from_selection_model": val_thresholded.tolist(),
        }
        model_results[name] = record
        model_records[name] = {"parameters": final_model.parameters(),
                               "preprocessing": final_preproc_record,
                               "selection_preprocessing": candidate["preprocessing_record"],
                               "configuration": record["configuration"],
                               "selection": record["selection"]}

    # Validation-only threshold adjustment is separate from architecture selection.
    a3_candidate = selected_candidates["A3_pixel_scale_plus_range_penalty_RC_TinyCNN"]
    a3_val_logits = np.asarray(a3_candidate["validation"]["logits"])
    threshold_record = _select_threshold(val_y, a3_val_logits)

    # Validation variability over fixed hyperparameters and repeated training seeds.
    variability = {}
    for name in ("A0_standard_scaler_control", "A3_pixel_scale_plus_range_penalty_RC_TinyCNN"):
        preprocessing, bound, penalty = configurations[name]
        seed_results = []
        for repeat_seed in (seed, seed + 1, seed + 2):
            if repeat_seed == seed:
                repeat_candidate = selected_candidates[name]
            else:
                repeat_candidate = _fit_validation_candidate(
                    x, y, fit, val, preprocessing=preprocessing, range_bound=bound,
                    range_penalty=penalty, seed=repeat_seed,
                )
            seed_results.append({"seed": repeat_seed, "metrics": repeat_candidate["validation"]["metrics"],
                                 "activation_diagnostics": repeat_candidate["validation_activation_diagnostics"]})
        accuracies = [entry["metrics"]["accuracy"] for entry in seed_results]
        variability[name] = {
            "protocol": "Same fixed training/validation rows and selected configuration; vary initialization/shuffle seed only.",
            "runs": seed_results,
            "validation_accuracy_mean": float(np.mean(accuracies)),
            "validation_accuracy_std_population": float(np.std(accuracies)),
        }

    task_result = {
        "task": task_name,
        "dataset": "sklearn.datasets.load_digits",
        "class_mapping": mapping,
        "positive_class": 1,
        "label_order": [0, 1],
        "split_id": split_doc.get("split_id"),
        "split_sizes": {name: len(idx[name]) for name in ("fit", "validation", "train", "test")},
        "development_only_model_selection": True,
        "models": model_results,
        "range_penalty_grid_search": {"standard": {"selected_bound": a2_bound, "selected_lambda": a2_penalty,
                                                       "selection_key": "validation accuracy, positive F1, negative unpenalized log loss, fewer bound violations",
                                                       "trials": a2_trials},
                                      "pixel_0_1": {"selected_bound": a3_bound, "selected_lambda": a3_penalty,
                                                    "selection_key": "validation accuracy, positive F1, negative unpenalized log loss, fewer bound violations",
                                                    "trials": a3_trials}},
        "A4_validation_selected_threshold": threshold_record,
        "validation_seed_variability": variability,
        "test_evaluation_status": "not yet evaluated; configurations and threshold are frozen before any test rows are read",
    }
    fit_quality = {"models": model_records, "threshold": threshold_record["threshold_logit"],
                   "selected_candidates": selected_candidates}
    return task_result, fit_quality


def _ckks_validation(task_data, split_doc, model_info: dict, model_name: str, preprocessing: str,
                     scaler: StandardScaler | None, context: ts.Context, method: str) -> dict:
    parameters = model_info["parameters"]
    head_weights = np.asarray(parameters["head_weights"], dtype=np.float64)
    idx = np.asarray(split_doc["indices"]["validation"], dtype=np.int64)
    labels = task_data.labels[idx]
    records = []
    for dataset_idx in idx:
        raw = task_data.features[dataset_idx]
        x = _transform(raw.reshape(1, -1), preprocessing, scaler)[0]
        reference = plaintext_stages(x, parameters)
        expected_logit = float(reference["pooled"] @ head_weights + float(parameters["head_bias"]))
        if method == "mask_dot":
            encrypted = encrypted_features(x, parameters, context)
            actual_conv, actual_square, actual_pooled = encrypted.convolution, encrypted.squared, encrypted.pooled
            head_start = time.perf_counter()
            actual_logit = float(np.asarray(actual_pooled) @ head_weights + float(parameters["head_bias"]))
            plaintext_head_seconds = time.perf_counter() - head_start
            actual_prediction = int(actual_logit >= 0.0)
            timing = encrypted.timings_seconds | {"total_feature_extraction_with_diagnostic_decrypts":
                                                   sum(encrypted.timings_seconds.values()),
                                                   "plaintext_head": plaintext_head_seconds,
                                                   "total_until_plaintext_prediction":
                                                   sum(encrypted.timings_seconds.values()) + plaintext_head_seconds}
            states = encrypted.states
            privacy = "Pooled encrypted features are decrypted before the plaintext linear head."
        else:
            encrypted = encrypted_im2col_partial_inference(
                x, parameters, head_weights, float(parameters["head_bias"]), context
            )
            actual_conv = encrypted["convolution"]
            actual_square = encrypted["square"]
            actual_pooled = encrypted["pooled_plaintext_after_early_decryption"]
            actual_logit = encrypted["classification_logit_plaintext_head"]
            actual_prediction = encrypted["prediction"]
            timing = encrypted["timings_seconds"]
            states = encrypted["states"]
            privacy = encrypted["privacy_boundary"]
        records.append({
            "filtered_index": int(dataset_idx), "label": int(task_data.labels[dataset_idx]),
            "plaintext_logit": expected_logit, "encrypted_path_logit": actual_logit,
            "plaintext_prediction": int(expected_logit >= 0.0), "encrypted_path_prediction": actual_prediction,
            "errors": {"convolution": error_summary(actual_conv, reference["convolution"]),
                       "square": error_summary(actual_square, reference["squared"]),
                       "pooled": error_summary(actual_pooled, reference["pooled"]),
                       "logit": error_summary([actual_logit], [expected_logit])},
            "timings_seconds": timing, "ciphertext_states": states,
            "serialized_ciphertext_bytes": (encrypted.serialized_ciphertext_bytes if method == "mask_dot"
                                             else encrypted.get("serialized_ciphertext_bytes")),
        })
    labels_all = np.asarray([r["label"] for r in records], dtype=np.int64)
    predictions = np.asarray([r["encrypted_path_prediction"] for r in records], dtype=np.int64)
    plain = np.asarray([r["plaintext_prediction"] for r in records], dtype=np.int64)
    stage_summary = {}
    for stage in ("convolution", "square", "pooled", "logit"):
        maes = [r["errors"][stage]["mean_absolute_error"] for r in records]
        maxima = [r["errors"][stage]["max_absolute_error"] for r in records]
        stage_summary[stage] = {"mean_of_sample_mae": float(np.mean(maes)),
                                "worst_sample_max_abs_error": float(np.max(maxima))}
    time_summary = {}
    timing_keys = sorted(set().union(*(record["timings_seconds"].keys() for record in records)))
    for key in timing_keys:
        times = np.asarray([record["timings_seconds"].get(key, np.nan) for record in records], dtype=float)
        time_summary[key] = {"mean_seconds": float(np.nanmean(times)), "median_seconds": float(np.nanmedian(times)),
                             "p95_seconds": float(np.nanquantile(times, .95))}
    return {
        "method": method, "model": model_name, "parameter_set": "depth3_40bit_exploratory",
        "parameters": PARAMETER_SETS["depth3_40bit_exploratory"], "privacy_boundary": privacy,
        "metrics": compute_metrics(labels_all, predictions),
        "prediction_agreement_with_plaintext_graph": {"agreement_count": int(np.sum(predictions == plain)),
                                                       "sample_count": len(records),
                                                       "flipped_predictions": int(np.sum(predictions != plain))},
        "numerical_error_summary": stage_summary, "timing_summary": time_summary,
        "per_sample_results": records,
    }


def _final_test(task_data, split_doc, task_result: dict, fit_quality: dict) -> None:
    test_idx = np.asarray(split_doc["indices"]["test"], dtype=np.int64)
    y_test = task_data.labels[test_idx]
    threshold = fit_quality["threshold"]
    for name, result in task_result["models"].items():
        info = fit_quality["models"][name]
        preprocessing = result["configuration"]["preprocessing"]
        scaler = None
        preproc = info["preprocessing"]
        if preprocessing == "standard":
            scaler = StandardScaler()
            scaler.mean_ = np.asarray(preproc["mean"], dtype=np.float64)
            scaler.scale_ = np.asarray(preproc["scale"], dtype=np.float64)
            scaler.var_ = np.square(scaler.scale_)
            scaler.n_features_in_ = 64
            scaler.n_samples_seen_ = int(info["preprocessing"]["fit_rows"])
        model = TinyCNN.from_parameters(info["parameters"])
        x_test = _transform(task_data.features[test_idx], preprocessing, scaler)
        logits = model.decision_function(x_test)
        default_test = _metric_entry(y_test, logits, 0.0)
        result["test_metrics_threshold_zero"] = default_test["metrics"]
        result["test_predictions_threshold_zero"] = default_test["predictions"]
        result["test_logits"] = default_test["logits"]
        result["test_activation_diagnostics"] = _activation_diagnostics(model, x_test,
                                                                          result["configuration"]["range_bound"])
        if name == "A3_pixel_scale_plus_range_penalty_RC_TinyCNN":
            adjusted = _metric_entry(y_test, logits, threshold)
            result["A4_test_metrics_validation_threshold"] = adjusted["metrics"]
            result["A4_test_predictions_validation_threshold"] = adjusted["predictions"]
    baseline = task_result["models"]["A0_standard_scaler_control"]["test_predictions_threshold_zero"]
    candidate = task_result["models"]["A3_pixel_scale_plus_range_penalty_RC_TinyCNN"]["test_predictions_threshold_zero"]
    task_result["A0_vs_A3_paired_test_comparison"] = mcnemar_exact(y_test, baseline, candidate)
    task_result["final_test_evaluation"] = {
        "rows": int(len(test_idx)), "performed_once_after_configuration_freeze": True,
        "split_was_used_in_prior_Phase1_and_Phase2_diagnostics": task_result["task"] == "digits_0_vs_1",
        "test_filtered_indices": test_idx.tolist(), "test_labels": y_test.tolist(),
    }
    task_result["test_evaluation_status"] = "completed once after freezing all model configurations and threshold"


def main() -> None:
    print("Loading fixed source split and pretrained Phase 1 baseline metadata...", flush=True)
    primary = load_binary_digits()
    prior_baseline = json.loads((SOURCE_ARTIFACTS / "baseline_results.json").read_text(encoding="utf-8"))
    phase1_model = json.loads((SOURCE_ARTIFACTS / "model_parameters.json").read_text(encoding="utf-8"))
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    if prior_baseline["split_id"] != phase1_model["split_id"]:
        raise RuntimeError("Phase 1 baseline and saved weights disagree on split identity")
    benchmarks = _benchmark_definitions()
    split_docs = {name: item[1] for name, item in benchmarks.items()}
    results = {
        "run_id": "phase3-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    "numpy": np.__version__, "scipy": scipy.__version__,
                    "scikit_learn": sklearn.__version__, "tenseal": ts.__version__},
        "protocol": {
            "training_budget": {"max_epochs": MAX_EPOCHS, "patience": PATIENCE,
                                 "batch_size": 32, "learning_rate": 1e-3, "l2_head": L2,
                                 "seed_primary": SEED_PRIMARY, "seed_secondary": SEED_SECONDARY},
            "model_selection": "Fit partition only for gradients; validation partition for epoch/config/threshold selection; refit on outer train for frozen test evaluation.",
            "primary_mapping": "digit 0 -> 0, digit 1 -> 1; positive class 1",
            "secondary_mapping": "digit 3 -> 0, digit 8 -> 1; positive class 1",
            "pixel_preprocessing": "raw pixel / 16, fixed source intensity domain, no learned test-informed scaling",
            "range_objective": "lambda * mean(max(abs(pre_square)-B,0)^2), analytically differentiated and applied during training",
            "hyperparameter_selection": "validation accuracy, then positive F1, lower unpenalized validation log loss, then fewer bound violations",
            "final_metric_priorities_frozen_before_test": ["accuracy", "positive-class precision", "positive-class recall", "positive-class F1", "macro F1"],
            "no_heldout_test_access_during_search": True,
        },
        "benchmarks": {},
        "saved_models": {},
    }
    fit_quality_by_task = {}
    # Training and selection complete for both benchmarks before the final test evaluation.
    for task_name, (data, split_doc, mapping) in benchmarks.items():
        print(f"Training and selecting configurations: {task_name}", flush=True)
        task_result, fit_quality = _fit_task(task_name, data, split_doc, mapping,
                                             seed=SEED_PRIMARY if task_name == "digits_0_vs_1" else SEED_SECONDARY)
        results["benchmarks"][task_name] = task_result
        results["saved_models"][task_name] = fit_quality["models"]
        fit_quality_by_task[task_name] = fit_quality
    results["status"] = "development_models_and_threshold_frozen"
    write_json(ARTIFACTS / "phase3_ablation.json", results)
    write_json(ARTIFACTS / "phase3_models.json", results["saved_models"])

    # CKKS numerical comparison is performed on primary-task validation rows only.
    # It uses the Phase 2 supported chain; the plaintext head remains explicitly plaintext.
    context_start = time.perf_counter()
    print("Running CKKS numerical validation on the primary validation split...", flush=True)
    context = make_context("depth3_40bit_exploratory")
    keygen_seconds = time.perf_counter() - context_start
    ckks_results = {"parameter_set": "depth3_40bit_exploratory",
                    "one_time_context_and_evaluation_key_generation_seconds": keygen_seconds,
                    "thread_count": 1, "batch_size": 1, "warmup_count": 0,
                    "primary_task_validation_only": True, "models": {}}
    for model_name in ("A0_standard_scaler_control", "A3_pixel_scale_plus_range_penalty_RC_TinyCNN"):
        task_model_info = fit_quality_by_task["digits_0_vs_1"]["models"][model_name]
        config = results["benchmarks"]["digits_0_vs_1"]["models"][model_name]["configuration"]
        scaler = None
        if config["preprocessing"] == "standard":
            preproc = task_model_info["preprocessing"]
            scaler = StandardScaler()
            scaler.mean_ = np.asarray(preproc["mean"], dtype=np.float64)
            scaler.scale_ = np.asarray(preproc["scale"], dtype=np.float64)
            scaler.var_ = np.square(scaler.scale_)
            scaler.n_features_in_ = 64
            scaler.n_samples_seen_ = int(preproc["fit_rows"])
        for method in ("mask_dot", "im2col_partial"):
            key = f"{model_name}:{method}"
            ckks_results["models"][key] = _ckks_validation(
                primary, split_docs["digits_0_vs_1"], task_model_info, model_name,
                config["preprocessing"], scaler, context, method,
            )
    write_json(ARTIFACTS / "phase3_ckks_validation.json", ckks_results)

    # Single final evaluation per task, after architecture, grid values, and threshold are frozen.
    for task_name, (data, split_doc, _mapping) in benchmarks.items():
        print(f"Evaluating frozen configurations on held-out test rows: {task_name}", flush=True)
        _final_test(data, split_doc, results["benchmarks"][task_name], fit_quality_by_task[task_name])
    results["status"] = "completed"
    results["completed_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(ARTIFACTS / "phase3_ablation.json", results)

    prediction_rows = []
    for task_name, (data, split_doc, _mapping) in benchmarks.items():
        test_indices = split_doc["indices"]["test"]
        task_result = results["benchmarks"][task_name]
        model_names = list(task_result["models"])
        for pos, filtered_idx in enumerate(test_indices):
            row = {"task": task_name, "test_position": pos, "filtered_index": int(filtered_idx),
                   "y_true": int(data.labels[filtered_idx])}
            for name in model_names:
                row[name] = task_result["models"][name]["test_predictions_threshold_zero"][pos]
            if "A4_test_predictions_validation_threshold" in task_result["models"]["A3_pixel_scale_plus_range_penalty_RC_TinyCNN"]:
                row["A4_RC_validation_threshold"] = task_result["models"]["A3_pixel_scale_plus_range_penalty_RC_TinyCNN"]["A4_test_predictions_validation_threshold"][pos]
            prediction_rows.append(row)
    with (ARTIFACTS / "phase3_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(prediction_rows[0]))
        writer.writeheader(); writer.writerows(prediction_rows)

    summary = {}
    for task_name, task in results["benchmarks"].items():
        summary[task_name] = {name: model.get("test_metrics_threshold_zero", {}).get("accuracy")
                              for name, model in task["models"].items()}
    print(json.dumps({"status": results["status"], "test_accuracy": summary,
                      "artifacts": ["artifacts/phase3_ablation.json", "artifacts/phase3_models.json",
                                    "artifacts/phase3_splits.json", "artifacts/phase3_predictions.csv",
                                    "artifacts/phase3_ckks_validation.json"]}, indent=2))


if __name__ == "__main__":
    main()
