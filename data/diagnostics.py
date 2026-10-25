from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler


def fit_standardizer(train_features: np.ndarray) -> StandardScaler:
    """Fit StandardScaler only on the rows supplied by the training partition."""
    x = np.asarray(train_features, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Training features must be a finite 2D matrix")
    return StandardScaler().fit(x)


def _range_summary(values: np.ndarray) -> dict:
    x = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(x)
    if not finite.any():
        return {"count": int(x.size), "finite_count": 0, "nan_count": int(np.isnan(x).sum()),
                "positive_infinity_count": int(np.isposinf(x).sum()),
                "negative_infinity_count": int(np.isneginf(x).sum())}
    clean = x[finite]
    return {
        "count": int(x.size),
        "finite_count": int(finite.sum()),
        "nan_count": int(np.isnan(x).sum()),
        "positive_infinity_count": int(np.isposinf(x).sum()),
        "negative_infinity_count": int(np.isneginf(x).sum()),
        "min": float(clean.min()),
        "max": float(clean.max()),
        "mean": float(clean.mean()),
        "std_population": float(clean.std()),
        "max_abs": float(np.abs(clean).max()),
        "quantiles_0_01_0_50_0_99": [float(v) for v in np.quantile(clean, [0.01, 0.5, 0.99])],
        "over_1e6_count": int((np.abs(clean) > 1e6).sum()),
    }


def pixel_feature_diagnostics(features: np.ndarray, images: np.ndarray, train_indices: np.ndarray) -> dict:
    x = np.asarray(features, dtype=np.float64)
    train_x = x[np.asarray(train_indices, dtype=np.int64)]
    std = train_x.std(axis=0)
    near_constant = np.flatnonzero(std <= 1e-8)
    border = np.array(
        [r * 8 + c for r in range(8) for c in range(8) if r in (0, 7) or c in (0, 7)],
        dtype=np.int64,
    )
    border_near_constant = np.intersect1d(border, near_constant)
    return {
        "raw_pixel_value_range_expected": [0.0, 16.0],
        "all_filtered_images_shape": list(np.asarray(images).shape),
        "feature_count": int(x.shape[1]),
        "training_partition_count": int(len(train_indices)),
        "train_raw_feature_summary": _range_summary(train_x),
        "train_feature_mean_range": [float(train_x.mean(axis=0).min()), float(train_x.mean(axis=0).max())],
        "train_feature_std_range": [float(std.min()), float(std.max())],
        "near_constant_threshold_std_le": 1e-8,
        "near_constant_feature_indices": near_constant.tolist(),
        "near_constant_feature_count": int(near_constant.size),
        "border_feature_indices": border.tolist(),
        "border_near_constant_feature_indices": border_near_constant.tolist(),
        "border_near_constant_feature_count": int(border_near_constant.size),
    }


def preprocessing_diagnostics(scaler: StandardScaler, train_features: np.ndarray, test_features: np.ndarray) -> dict:
    train_scaled = scaler.transform(train_features)
    test_scaled = scaler.transform(test_features)
    return {
        "method": "sklearn.preprocessing.StandardScaler, per feature, fit on outer training split only",
        "fit_sample_count": int(scaler.n_samples_seen_),
        "constant_feature_scale_behavior": "StandardScaler uses scale 1 for zero-variance features",
        "fitted_mean_min_max": [float(scaler.mean_.min()), float(scaler.mean_.max())],
        "fitted_scale_min_max": [float(scaler.scale_.min()), float(scaler.scale_.max())],
        "training_transformed": _range_summary(train_scaled),
        "test_transformed_unlabelled": _range_summary(test_scaled),
        "test_used_for_fit": False,
    }


def activation_diagnostics(model, standardized_features: np.ndarray, labels: np.ndarray) -> dict:
    details = model.intermediates(standardized_features)
    return {
        "diagnostic_sample_count": int(len(labels)),
        "pre_square_convolution": _range_summary(details["pre_square"]),
        "square_activation": _range_summary(details["squared"]),
        "2x2_stride2_average_pool": _range_summary(details["pooled"]),
        "head_logits": _range_summary(details["logits"]),
        "nonfinite_any": bool(any(not np.isfinite(value).all() for value in details.values())),
    }
