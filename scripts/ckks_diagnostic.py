from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import sklearn
import tenseal as ts

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.dataset import load_binary_digits
from he.ckks_inference import (
    EncryptedPathError,
    PARAMETER_SETS,
    encrypted_features,
    error_summary,
    make_context,
    plaintext_stages,
)
from metrics.compute import compute_metrics
from models.polynomial_graph import PolynomialGraph


def load_artifacts():
    model_doc = json.loads((ROOT / "artifacts/model_parameters.json").read_text(encoding="utf-8"))
    split_doc = json.loads((ROOT / "artifacts/splits.json").read_text(encoding="utf-8"))
    tiny = model_doc["tinycnn"]
    return tiny["parameters"], np.asarray(tiny["scaler_mean"]), np.asarray(tiny["scaler_scale"]), split_doc


def run_sample(index: int, label: int, raw: np.ndarray, scaler_mean: np.ndarray, scaler_scale: np.ndarray,
               parameters: dict, context: ts.Context) -> dict:
    standardized = (raw - scaler_mean) / scaler_scale
    stages = plaintext_stages(standardized, parameters)
    graph = PolynomialGraph(parameters)
    graph_logit = float(graph.logits(standardized.reshape(1, 64))[0])
    head_weights = np.asarray(parameters["head_weights"], dtype=np.float64)
    plaintext_head_logit = float(stages["pooled"] @ head_weights + float(parameters["head_bias"]))
    if not np.isclose(graph_logit, plaintext_head_logit, rtol=0.0, atol=1e-10):
        raise AssertionError("PolynomialGraph and explicit plaintext stages disagree")

    start = time.perf_counter()
    result = encrypted_features(standardized, parameters, context)
    encrypted_feature_seconds = time.perf_counter() - start
    encrypted_logit = float(np.asarray(result.pooled) @ head_weights + float(parameters["head_bias"]))
    encrypted_prediction = int(encrypted_logit >= 0.0)
    return {
        "filtered_dataset_index": int(index),
        "label": int(label),
        "raw_input": raw.tolist(),
        "standardized_input": standardized.tolist(),
        "input_decrypted_error": error_summary(result.input_decrypted, standardized),
        "plaintext": {
            "convolution": stages["convolution"].tolist(),
            "square": stages["squared"].tolist(),
            "pooled": stages["pooled"].tolist(),
            "classification_logit": graph_logit,
            "prediction": int(graph_logit >= 0.0),
        },
        "encrypted_decrypted": {
            "convolution": result.convolution,
            "square": result.squared,
            "pooled": result.pooled,
            "classification_logit_plaintext_head": encrypted_logit,
            "prediction": encrypted_prediction,
        },
        "errors": {
            stage: error_summary(result_values, stages[stage])
            for stage, result_values in (
                ("convolution", result.convolution),
                ("squared", result.squared),
                ("pooled", result.pooled),
            )
        } | {"classification_logit": error_summary([encrypted_logit], [graph_logit])},
        "ciphertext_states": result.states,
        "timings_seconds": result.timings_seconds | {"total_feature_extraction": encrypted_feature_seconds},
    }


def exception_record(exc: BaseException) -> dict:
    return {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}


def main() -> None:
    parser = argparse.ArgumentParser(description="CipherBench CKKS feature-extraction diagnostics")
    parser.add_argument("--mode", choices=["smoke", "validation", "final-test"], default="smoke")
    parser.add_argument("--config", choices=list(PARAMETER_SETS))
    parser.add_argument("--count", type=int)
    args = parser.parse_args()

    parameters, scaler_mean, scaler_scale, split_doc = load_artifacts()
    data = load_binary_digits()
    if split_doc["split_id"] != json.loads((ROOT / "artifacts/model_parameters.json").read_text())["split_id"]:
        raise RuntimeError("Saved split and model parameters have different split identifiers")
    if args.mode == "smoke":
        indices = [int(split_doc["indices"]["fit"][0])]
        configs = [args.config] if args.config else ["paper_literal", "diagnostic_40bit", "depth3_40bit_exploratory"]
    elif args.mode == "validation":
        indices = [int(i) for i in split_doc["indices"]["validation"]]
        if args.count is not None:
            indices = indices[: args.count]
        configs = [args.config or "depth3_40bit_exploratory"]
    else:
        if args.config not in (None, "depth3_40bit_exploratory"):
            raise ValueError("Final test permits only the frozen depth3_40bit_exploratory configuration")
        indices = [int(i) for i in split_doc["indices"]["test"]]
        configs = ["depth3_40bit_exploratory"]

    results = {
        "run_id": "ckks-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "mode": args.mode,
        "split_id": split_doc["split_id"],
        "seed": split_doc["seed"],
        "positive_class": 1,
        "label_order": [0, 1],
        "preprocessing": "Saved outer-train StandardScaler from Phase 1 TinyCNN; no refit or retraining.",
        "head_execution": "Plaintext after decrypting pooled ciphertext features; not end-to-end encrypted.",
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "tenseal": ts.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "parameter_results": {},
    }
    for config_name in configs:
        config_result = {
            "parameters": PARAMETER_SETS[config_name],
            "samples": [],
            "exceptions": [],
        }
        try:
            context = make_context(config_name)
            config_result["context_created"] = True
            config_result["context_flags"] = {
                "auto_rescale": context.auto_rescale,
                "auto_mod_switch": context.auto_mod_switch,
                "auto_relin": context.auto_relin,
            }
            for index in indices:
                try:
                    config_result["samples"].append(
                        run_sample(index, int(data.labels[index]), data.features[index], scaler_mean, scaler_scale,
                                   parameters, context)
                    )
                except Exception as exc:
                    record = exception_record(exc) | {"filtered_dataset_index": index}
                    if isinstance(exc, EncryptedPathError):
                        record["failed_stage"] = exc.stage
                        record["partial_results"] = exc.partial
                        expected = plaintext_stages(
                            (data.features[index] - scaler_mean) / scaler_scale, parameters
                        )
                        partial = exc.partial
                        partial_errors = {}
                        if "convolution_decrypted" in partial:
                            partial_errors["convolution"] = error_summary(
                                partial["convolution_decrypted"], expected["convolution"]
                            )
                        if "square_decrypted" in partial:
                            partial_errors["square"] = error_summary(partial["square_decrypted"], expected["squared"])
                        record["partial_numerical_errors"] = partial_errors
                    config_result["exceptions"].append(record)
                    # This failure is preserved in the output; stop this parameter set at the first failure.
                    break
        except Exception as exc:
            config_result["context_created"] = False
            config_result["exceptions"].append(exception_record(exc))

        if args.mode != "smoke" and config_result["samples"]:
            labels = [sample["label"] for sample in config_result["samples"]]
            predictions = [sample["encrypted_decrypted"]["prediction"] for sample in config_result["samples"]]
            plain_predictions = [sample["plaintext"]["prediction"] for sample in config_result["samples"]]
            config_result["metrics"] = compute_metrics(np.asarray(labels), np.asarray(predictions))
            config_result["prediction_agreement"] = {
                "agreement_count": int(np.sum(np.asarray(predictions) == plain_predictions)),
                "sample_count": len(predictions),
                "flipped_predictions": int(np.sum(np.asarray(predictions) != plain_predictions)),
            }
            config_result["predictions"] = [
                {"filtered_dataset_index": sample["filtered_dataset_index"], "label": sample["label"],
                 "plaintext_prediction": sample["plaintext"]["prediction"], "encrypted_prediction": sample["encrypted_decrypted"]["prediction"],
                 "plaintext_logit": sample["plaintext"]["classification_logit"],
                 "encrypted_logit": sample["encrypted_decrypted"]["classification_logit_plaintext_head"]}
                for sample in config_result["samples"]
            ]
        results["parameter_results"][config_name] = config_result

    out_name = {"smoke": "ckks_smoke.json", "validation": "ckks_validation.json", "final-test": "ckks_final_test.json"}[args.mode]
    out_path = ROOT / "artifacts" / out_name
    out_path.write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({name: {"samples": len(value["samples"]), "exceptions": len(value["exceptions"]),
                              "metrics": value.get("metrics"), "prediction_agreement": value.get("prediction_agreement")}
                      for name, value in results["parameter_results"].items()}, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
