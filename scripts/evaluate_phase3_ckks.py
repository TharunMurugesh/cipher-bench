"""Recompute Phase 3 validation-only CKKS diagnostics from frozen artifacts."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.dataset import load_binary_digits
from he.ckks_inference import make_context
from scripts.run_phase3_ablation import _ckks_validation


ARTIFACTS = ROOT / "artifacts"


def main() -> None:
    ablation = json.loads((ARTIFACTS / "phase3_ablation.json").read_text(encoding="utf-8"))
    models = json.loads((ARTIFACTS / "phase3_models.json").read_text(encoding="utf-8"))
    splits = json.loads((ARTIFACTS / "phase3_splits.json").read_text(encoding="utf-8"))
    task = load_binary_digits()
    context_start = time.perf_counter()
    context = make_context("depth3_40bit_exploratory")
    keygen_seconds = time.perf_counter() - context_start
    output = {
        "parameter_set": "depth3_40bit_exploratory",
        "one_time_context_and_evaluation_key_generation_seconds": keygen_seconds,
        "thread_count": 1, "batch_size": 1, "warmup_count": 0,
        "primary_task_validation_only": True,
        "recomputed_from_frozen_phase3_artifacts": True,
        "models": {},
    }
    split_doc = splits["digits_0_vs_1"]
    for name in ("A0_standard_scaler_control", "A3_pixel_scale_plus_range_penalty_RC_TinyCNN"):
        info = models["digits_0_vs_1"][name]
        config = ablation["benchmarks"]["digits_0_vs_1"]["models"][name]["configuration"]
        scaler = None
        if config["preprocessing"] == "standard":
            pre = info["preprocessing"]
            scaler = StandardScaler()
            scaler.mean_ = np.asarray(pre["mean"], dtype=np.float64)
            scaler.scale_ = np.asarray(pre["scale"], dtype=np.float64)
            scaler.var_ = np.square(scaler.scale_)
            scaler.n_features_in_ = 64
            scaler.n_samples_seen_ = int(pre["fit_rows"])
        for method in ("mask_dot", "im2col_partial"):
            key = f"{name}:{method}"
            output["models"][key] = _ckks_validation(
                task, split_doc, info, name, config["preprocessing"], scaler, context, method
            )
    target = ARTIFACTS / "phase3_ckks_validation.json"
    target.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "output": str(target),
                      "models": list(output["models"])}, indent=2))


if __name__ == "__main__":
    main()
