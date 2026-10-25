from __future__ import annotations

import copy

import numpy as np


def _windows(images: np.ndarray) -> np.ndarray:
    x = np.asarray(images, dtype=np.float64)
    if x.ndim == 2:
        if x.shape[1] != 64:
            raise ValueError("Expected 64 flattened image features")
        x = x.reshape(-1, 8, 8)
    if x.ndim != 3 or x.shape[1:] != (8, 8):
        raise ValueError("Expected image input shaped (n, 8, 8) or (n, 64)")
    return np.lib.stride_tricks.sliding_window_view(x, (3, 3), axis=(1, 2))


class TinyCNN:
    """Trainable 3x3 valid conv -> square -> 2x2/stride-2 average pool -> logistic head."""

    def __init__(self, kernels: np.ndarray, conv_bias: np.ndarray, head_weights: np.ndarray, head_bias: float):
        self.kernels = np.asarray(kernels, dtype=np.float64)
        self.conv_bias = np.asarray(conv_bias, dtype=np.float64)
        self.head_weights = np.asarray(head_weights, dtype=np.float64)
        self.head_bias = float(head_bias)
        if self.kernels.ndim != 3 or self.kernels.shape[1:] != (3, 3):
            raise ValueError("Convolution kernels must have shape (kernel_count, 3, 3)")
        if self.conv_bias.shape != (self.kernels.shape[0],):
            raise ValueError("Convolution bias must have one value per kernel")
        if self.head_weights.shape != (self.kernels.shape[0] * 9,):
            raise ValueError("Logistic head must have 9 pooled values per kernel")

    @classmethod
    def initialize(cls, *, kernel_count: int = 1, seed: int = 2026) -> "TinyCNN":
        rng = np.random.default_rng(seed)
        kernels = rng.normal(0.0, 0.20, size=(kernel_count, 3, 3))
        return cls(kernels, np.zeros(kernel_count), rng.normal(0.0, 0.02, size=kernel_count * 9), 0.0)

    def intermediates(self, features: np.ndarray) -> dict[str, np.ndarray]:
        windows = _windows(features)
        pre_square = np.einsum("nhwij,kij->nkhw", windows, self.kernels, optimize=True)
        pre_square += self.conv_bias[None, :, None, None]
        squared = np.square(pre_square)
        n, kernel_count, _, _ = squared.shape
        pooled = squared.reshape(n, kernel_count, 3, 2, 3, 2).mean(axis=(3, 5))
        flattened = pooled.reshape(n, -1)
        logits = flattened @ self.head_weights + self.head_bias
        return {"pre_square": pre_square, "squared": squared, "pooled": pooled, "logits": logits}

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        return self.intermediates(features)["logits"]

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        logits = np.clip(self.decision_function(features), -700.0, 700.0)
        positive = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack([1.0 - positive, positive])

    def predict(self, features: np.ndarray) -> np.ndarray:
        return (self.predict_proba(features)[:, 1] >= 0.5).astype(np.int64)

    def parameters(self) -> dict:
        return {
            "kernels": self.kernels.tolist(),
            "conv_bias": self.conv_bias.tolist(),
            "head_weights": self.head_weights.tolist(),
            "head_bias": self.head_bias,
        }

    @classmethod
    def from_parameters(cls, parameters: dict) -> "TinyCNN":
        return cls(parameters["kernels"], parameters["conv_bias"], parameters["head_weights"], parameters["head_bias"])


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.clip(logits, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-logits))


def range_penalty_value_and_gradient(
    preactivation: np.ndarray, *, bound: float, coefficient: float
) -> tuple[float, np.ndarray]:
    """Squared boundary penalty and its exact elementwise derivative."""
    z = np.asarray(preactivation, dtype=np.float64)
    if bound <= 0 or coefficient < 0:
        raise ValueError("bound must be positive and coefficient cannot be negative")
    excess = np.maximum(np.abs(z) - bound, 0.0)
    value = float(coefficient * np.square(excess).mean())
    gradient = (2.0 * coefficient / z.size) * excess * np.sign(z)
    return value, gradient


def _loss(
    model: TinyCNN,
    x: np.ndarray,
    y: np.ndarray,
    l2: float,
    range_bound: float | None = None,
    range_penalty: float = 0.0,
) -> float:
    logits = model.decision_function(x)
    cross_entropy = np.logaddexp(0.0, logits) - y * logits
    value = cross_entropy.mean() + 0.5 * l2 * np.square(model.head_weights).sum()
    if range_penalty and range_bound is not None:
        preactivation = model.intermediates(x)["pre_square"]
        value += range_penalty_value_and_gradient(
            preactivation, bound=range_bound, coefficient=range_penalty
        )[0]
    return float(value)


def train_tinycnn(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    *,
    seed: int = 2026,
    kernel_count: int = 1,
    max_epochs: int = 250,
    patience: int = 25,
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    l2: float = 1e-4,
    range_bound: float | None = None,
    range_penalty: float = 0.0,
    restore_best: bool = True,
) -> tuple[TinyCNN, dict]:
    """Train the square-activation graph end-to-end, selecting the epoch on validation only."""
    x = np.asarray(train_x, dtype=np.float64)
    y = np.asarray(train_y, dtype=np.float64)
    vx = np.asarray(validation_x, dtype=np.float64)
    vy = np.asarray(validation_y, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 64 or vx.ndim != 2 or vx.shape[1] != 64:
        raise ValueError("TinyCNN expects flattened 8x8 input")
    if range_penalty < 0 or (range_penalty > 0 and (range_bound is None or range_bound <= 0)):
        raise ValueError("A positive range penalty requires a positive bound; penalty cannot be negative")
    if len(x) != len(y) or len(vx) != len(vy) or set(np.unique(y)).difference({0.0, 1.0}):
        raise ValueError("Training inputs and binary labels are inconsistent")

    model = TinyCNN.initialize(kernel_count=kernel_count, seed=seed)
    params = [model.kernels, model.conv_bias, model.head_weights, np.asarray([model.head_bias])]
    first_moments = [np.zeros_like(p) for p in params]
    second_moments = [np.zeros_like(p) for p in params]
    rng = np.random.default_rng(seed + 17)
    best_state = None
    best_loss = float("inf")
    best_epoch = 0
    no_improvement = 0
    history = []
    step = 0
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8

    for epoch in range(1, max_epochs + 1):
        permutation = rng.permutation(len(x))
        for start in range(0, len(x), batch_size):
            batch_idx = permutation[start : start + batch_size]
            xb, yb = x[batch_idx], y[batch_idx]
            windows = _windows(xb)
            pre_square = np.einsum("nhwij,kij->nkhw", windows, model.kernels, optimize=True)
            pre_square += model.conv_bias[None, :, None, None]
            squared = np.square(pre_square)
            pooled = squared.reshape(len(xb), kernel_count, 3, 2, 3, 2).mean(axis=(3, 5))
            flattened = pooled.reshape(len(xb), -1)
            logits = flattened @ model.head_weights + model.head_bias
            error = (_sigmoid(logits) - yb) / len(xb)

            grad_head_weights = flattened.T @ error + l2 * model.head_weights
            grad_head_bias = np.asarray([error.sum()])
            grad_pooled = (error[:, None] * model.head_weights[None, :]).reshape(len(xb), kernel_count, 3, 3)
            grad_squared = np.repeat(np.repeat(grad_pooled / 4.0, 2, axis=2), 2, axis=3)
            grad_pre = 2.0 * pre_square * grad_squared
            if range_penalty and range_bound is not None:
                grad_pre += range_penalty_value_and_gradient(
                    pre_square, bound=range_bound, coefficient=range_penalty
                )[1]
            grad_kernels = np.einsum("nkhw,nhwij->kij", grad_pre, windows, optimize=True)
            grad_conv_bias = grad_pre.sum(axis=(0, 2, 3))
            grads = [grad_kernels, grad_conv_bias, grad_head_weights, grad_head_bias]

            norm = np.sqrt(sum(float(np.square(g).sum()) for g in grads))
            if norm > 5.0:
                grads = [g * (5.0 / norm) for g in grads]
            step += 1
            for i, (param, grad) in enumerate(zip(params, grads)):
                first_moments[i] = beta1 * first_moments[i] + (1.0 - beta1) * grad
                second_moments[i] = beta2 * second_moments[i] + (1.0 - beta2) * np.square(grad)
                m_hat = first_moments[i] / (1.0 - beta1**step)
                v_hat = second_moments[i] / (1.0 - beta2**step)
                param -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
            model.head_bias = float(params[3][0])

        train_loss = _loss(model, x, y, l2, range_bound, range_penalty)
        validation_loss = _loss(model, vx, vy, l2, range_bound, range_penalty)
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss})
        if not np.isfinite(train_loss) or not np.isfinite(validation_loss):
            raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")
        if validation_loss < best_loss - 1e-9:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.parameters())
            no_improvement = 0
        else:
            no_improvement += 1
        if no_improvement >= patience:
            break

    if best_state is None:
        raise RuntimeError("TinyCNN did not produce a finite validation checkpoint")
    returned_model = TinyCNN.from_parameters(best_state) if restore_best else model
    return returned_model, {
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "best_validation_loss": best_loss,
        "validation_history": history,
        "early_stopping_patience": patience,
        "gradient_global_norm_clip": 5.0,
        "l2_head_penalty": l2,
        "range_penalty": float(range_penalty),
        "range_bound": None if range_bound is None else float(range_bound),
        "range_penalty_definition": "lambda * mean(max(abs(pre_square)-B, 0)^2)",
        "batch_size": batch_size,
        "learning_rate": learning_rate,
    }


def _fit_fixed_epochs(
    train_x, train_y, *, epochs: int, seed: int, kernel_count: int,
    range_bound: float | None = None, range_penalty: float = 0.0,
) -> TinyCNN:
    if epochs < 1:
        raise ValueError("epochs must be positive")
    model, _ = train_tinycnn(
        train_x,
        train_y,
        train_x,
        train_y,
        seed=seed,
        kernel_count=kernel_count,
        max_epochs=epochs,
        patience=epochs + 1,
        range_bound=range_bound,
        range_penalty=range_penalty,
        restore_best=False,
    )
    return model
