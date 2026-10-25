from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class SplitIndices:
    seed: int
    test_fraction: float
    validation_fraction_of_train: float
    train_indices: np.ndarray
    test_indices: np.ndarray
    fit_indices: np.ndarray
    validation_indices: np.ndarray
    split_id: str

    def to_dict(self, source_indices: np.ndarray | None = None) -> dict:
        payload = {
            "seed": self.seed,
            "test_fraction": self.test_fraction,
            "validation_fraction_of_train": self.validation_fraction_of_train,
            "split_id": self.split_id,
            "index_semantics": "indices into the filtered 0-vs-1 dataset",
            "indices": {
                "train": self.train_indices.tolist(),
                "test": self.test_indices.tolist(),
                "fit": self.fit_indices.tolist(),
                "validation": self.validation_indices.tolist(),
            },
        }
        if source_indices is not None:
            payload["source_indices"] = {
                key: source_indices[idx].tolist()
                for key, idx in {
                    "train": self.train_indices,
                    "test": self.test_indices,
                    "fit": self.fit_indices,
                    "validation": self.validation_indices,
                }.items()
            }
            payload["source_index_semantics"] = "indices into sklearn.datasets.load_digits() before filtering"
        return payload


def build_split_indices(
    labels: np.ndarray,
    *,
    seed: int = 2026,
    test_fraction: float = 0.30,
    validation_fraction_of_train: float = 0.15,
) -> SplitIndices:
    labels = np.asarray(labels)
    if labels.ndim != 1 or set(np.unique(labels)) != {0, 1}:
        raise ValueError("labels must be a one-dimensional array containing classes 0 and 1")
    if not 0 < test_fraction < 1 or not 0 < validation_fraction_of_train < 1:
        raise ValueError("split fractions must be strictly between zero and one")

    all_indices = np.arange(labels.size, dtype=np.int64)
    train_idx, test_idx = train_test_split(
        all_indices,
        test_size=test_fraction,
        random_state=seed,
        stratify=labels,
    )
    fit_idx, validation_idx = train_test_split(
        train_idx,
        test_size=validation_fraction_of_train,
        random_state=seed + 1,
        stratify=labels[train_idx],
    )

    parts = {"train": train_idx, "test": test_idx, "fit": fit_idx, "validation": validation_idx}
    if np.intersect1d(train_idx, test_idx).size or np.intersect1d(fit_idx, validation_idx).size:
        raise RuntimeError("Split overlap detected")
    if set(np.concatenate([fit_idx, validation_idx])) != set(train_idx):
        raise RuntimeError("Inner fit and validation sets do not partition the outer training set")
    if set(np.concatenate([train_idx, test_idx])) != set(all_indices):
        raise RuntimeError("Outer training and test sets do not partition the dataset")

    digest_payload = {
        "seed": seed,
        "test_fraction": test_fraction,
        "validation_fraction_of_train": validation_fraction_of_train,
        "indices": {key: value.tolist() for key, value in parts.items()},
    }
    digest = hashlib.sha256(json.dumps(digest_payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return SplitIndices(
        seed=seed,
        test_fraction=test_fraction,
        validation_fraction_of_train=validation_fraction_of_train,
        train_indices=train_idx,
        test_indices=test_idx,
        fit_indices=fit_idx,
        validation_indices=validation_idx,
        split_id=f"digits01-seed{seed}-{digest}",
    )
