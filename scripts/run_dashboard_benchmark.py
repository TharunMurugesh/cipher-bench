from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.benchmark import load_digit_pair
from data.dataset import load_binary_digits
from metrics.compute import compute_metrics
from models.tinycnn import train_tinycnn


def as_json_number(value: float) -> float | None:
    result = float(value)
    return result if np.isfinite(result) else None


def evaluate(y_true: np.ndarray, logits: np.ndarray) -> dict:
    predictions = (logits >= 0).astype(np.int64)
    result = compute_metrics(y_true, predictions)
    result["roc_auc"] = as_json_number(roc_auc_score(y_true, logits))
    result["confusion_matrix"] = result["confusion_matrix_labels_0_1"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("digits_0_vs_1", "digits_2_vs_7", "digits_3_vs_8", "digits_4_vs_9"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pair_by_task = {"digits_2_vs_7": (2, 7), "digits_3_vs_8": (3, 8), "digits_4_vs_9": (4, 9)}
    data = load_binary_digits() if args.task == "digits_0_vs_1" else load_digit_pair(*pair_by_task[args.task])
    x, y = np.asarray(data.features, dtype=np.float64), np.asarray(data.labels, dtype=np.int64)
    all_indices = np.arange(len(y))
    seeds = {"digits_0_vs_1": 2026, "digits_2_vs_7": 2027, "digits_3_vs_8": 3038, "digits_4_vs_9": 4049}
    task_seed = seeds[args.task]
    train_indices, test_indices = train_test_split(
        all_indices, test_size=0.30, random_state=task_seed,
        stratify=y,
    )
    fit_indices, validation_indices = train_test_split(
        train_indices, test_size=0.15, random_state=task_seed + 1,
        stratify=y[train_indices],
    )

    model_definitions = [
        {"key": "paper_model", "name": "Paper Model", "description": "Paper-style convolution → square → pooling → logistic head, retrained locally.", "preprocessing": "standard", "bound": None, "penalty": 0.0},
        {"key": "cipherbench_model", "name": "CipherBench Model", "description": "Optimized pixel scaling with validation-selected activation range penalty.", "preprocessing": "pixel_0_1", "bound": None, "penalty": None},
    ]
    output_models = {}

    for definition in model_definitions:
        configs = [(None, 0.0)] if definition["key"] == "paper_model" else [
            (bound, penalty) for bound in (0.25, 0.5, 1.0) for penalty in (1e-3, 1e-2, 1e-1)
        ]
        best = None
        total_fit_seconds = 0.0
        for bound, penalty in configs:
            if definition["preprocessing"] == "standard":
                scaler = StandardScaler().fit(x[fit_indices])
                x_fit, x_validation, x_test = scaler.transform(x[fit_indices]), scaler.transform(x[validation_indices]), scaler.transform(x[test_indices])
            else:
                x_fit, x_validation, x_test = x[fit_indices] / 16.0, x[validation_indices] / 16.0, x[test_indices] / 16.0
            started = time.perf_counter()
            model, training = train_tinycnn(
                x_fit, y[fit_indices], x_validation, y[validation_indices],
                seed=task_seed, kernel_count=1, max_epochs=300, patience=25,
                l2=1e-4, range_bound=bound, range_penalty=penalty,
            )
            total_fit_seconds += time.perf_counter() - started
            val_logits = model.decision_function(x_validation)
            validation_metrics = evaluate(y[validation_indices], val_logits)
            selection_key = (validation_metrics["accuracy"], validation_metrics["positive_class_metrics"]["f1"] or 0, -training["best_validation_loss"])
            if best is None or selection_key > best["selection_key"]:
                best = {"model": model, "training": training, "selection_key": selection_key,
                        "validation_metrics": validation_metrics, "bound": bound, "penalty": penalty,
                        "x_test": x_test}
        assert best is not None
        test_logits = best["model"].decision_function(best["x_test"])
        output_models[definition["key"]] = {
            "name": definition["name"], "description": definition["description"],
            "metrics": evaluate(y[test_indices], test_logits),
            "validation_metrics": best["validation_metrics"],
            "training": {"epochs": best["training"]["epochs_run"], "best_epoch": best["training"]["best_epoch"],
                         "seconds": round(total_fit_seconds, 3), "fit_samples": int(len(fit_indices)),
                         "validation_samples": int(len(validation_indices)), "test_samples": int(len(test_indices)),
                         "range_bound": best["bound"], "range_penalty": best["penalty"]},
            "predictions": test_logits.tolist(), "labels": y[test_indices].tolist(),
        }
        print(f"{definition['name']}: complete ({total_fit_seconds:.1f}s)", flush=True)

    result = {
        "run_id": f"model-lab-{int(time.time())}", "completed_at": datetime.now(timezone.utc).isoformat(),
        "task": args.task, "dataset": "scikit-learn Digits", "class_mapping": args.task.removeprefix("digits_").replace("_vs_", " vs "),
        "split": {"method": "stratified train/test split with validation-only model selection", "seed": task_seed,
                  "fit_samples": int(len(fit_indices)), "validation_samples": int(len(validation_indices)), "test_samples": int(len(test_indices))},
        "models": output_models,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Both models trained and evaluated on the held-out test split.", flush=True)


if __name__ == "__main__":
    main()
