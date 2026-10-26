from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class MajorityClassifier:
    def fit(self, labels: np.ndarray) -> "MajorityClassifier":
        labels = np.asarray(labels, dtype=np.int64)
        classes, counts = np.unique(labels, return_counts=True)
        self.label_ = int(classes[np.flatnonzero(counts == counts.max())[0]])
        self.classes_ = classes
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if not hasattr(self, "label_"):
            raise RuntimeError("MajorityClassifier must be fitted first")
        return np.full(len(features), self.label_, dtype=np.int64)


def fit_logistic_regression(train_x: np.ndarray, train_y: np.ndarray, *, seed: int) -> LogisticRegression:
    model = LogisticRegression(
        solver="lbfgs",
        max_iter=2000,
        random_state=seed,
        C=1.0,
    )
    return model.fit(train_x, train_y)
