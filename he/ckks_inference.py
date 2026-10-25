from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import tenseal as ts


PARAMETER_SETS = {
    "paper_literal": {
        "poly_modulus_degree": 8192,
        "coeff_mod_bit_sizes": [40, 21, 21, 40],
        "global_scale": 2**40,
    },
    "diagnostic_40bit": {
        "poly_modulus_degree": 8192,
        "coeff_mod_bit_sizes": [60, 40, 40, 60],
        "global_scale": 2**40,
    },
    "depth3_40bit_exploratory": {
        "poly_modulus_degree": 8192,
        "coeff_mod_bit_sizes": [49, 40, 40, 40, 49],
        "global_scale": 2**40,
    },
}


class EncryptedPathError(RuntimeError):
    def __init__(self, stage: str, message: str, partial: dict[str, Any], cause: BaseException):
        super().__init__(message)
        self.stage = stage
        self.partial = partial
        self.__cause__ = cause


def ciphertext_state(value: ts.CKKSVector) -> dict[str, Any]:
    ciphertext = value.ciphertext()[0]
    return {
        "scale": float(ciphertext.scale),
        "coeff_modulus_size": int(ciphertext.coeff_modulus_size()),
        "parms_id": list(ciphertext.parms_id()),
    }


def _mask_for_window(kernel: np.ndarray, top: int, left: int) -> np.ndarray:
    mask = np.zeros((8, 8), dtype=np.float64)
    mask[top : top + 3, left : left + 3] = kernel
    return mask.reshape(-1)


def make_context(parameter_set: str) -> ts.Context:
    config = PARAMETER_SETS[parameter_set]
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=config["poly_modulus_degree"],
        coeff_mod_bit_sizes=config["coeff_mod_bit_sizes"],
    )
    context.global_scale = config["global_scale"]
    context.generate_relin_keys()
    context.generate_galois_keys()
    return context


@dataclass
class EncryptedFeatures:
    input_decrypted: list[float]
    convolution: list[float]
    squared: list[float]
    pooled: list[float]
    states: dict[str, Any]
    timings_seconds: dict[str, float]
    serialized_ciphertext_bytes: dict[str, int]


def encrypted_features(
    standardized_features: np.ndarray,
    parameters: dict,
    context: ts.Context,
) -> EncryptedFeatures:
    """Mask-dot valid convolution, square, and encrypted 2x2 average pooling.

    Each convolution output is a one-slot encrypted dot product. The method
    deliberately preserves each operation boundary so level exhaustion and
    numerical errors are observable.
    """
    x = np.asarray(standardized_features, dtype=np.float64)
    if x.shape != (64,) or not np.isfinite(x).all():
        raise ValueError("Expected 64 finite standardized input features")
    kernels = np.asarray(parameters["kernels"], dtype=np.float64)
    biases = np.asarray(parameters["conv_bias"], dtype=np.float64)
    if kernels.ndim != 3 or kernels.shape[1:] != (3, 3):
        raise ValueError("Expected kernels shaped (kernel_count, 3, 3)")

    started = time.perf_counter()
    encrypted_input = ts.ckks_vector(context, x.tolist())
    input_seconds = time.perf_counter() - started
    states: dict[str, Any] = {"input": ciphertext_state(encrypted_input)}
    started = time.perf_counter()
    input_decrypted = encrypted_input.decrypt()
    decrypt_seconds = time.perf_counter() - started
    conv_ciphertexts: list[ts.CKKSVector] = []

    started = time.perf_counter()
    for k, kernel in enumerate(kernels):
        for row in range(6):
            for col in range(6):
                mask = _mask_for_window(kernel, row, col)
                value = encrypted_input.dot(mask.tolist())
                value = value + float(biases[k])
                conv_ciphertexts.append(value)
    conv_seconds = time.perf_counter() - started
    states["convolution"] = [ciphertext_state(value) for value in conv_ciphertexts]
    started = time.perf_counter()
    conv_values = [float(value.decrypt()[0]) for value in conv_ciphertexts]
    stage_decrypt_seconds = time.perf_counter() - started
    decrypt_seconds += stage_decrypt_seconds

    started = time.perf_counter()
    try:
        square_ciphertexts = [value.square() for value in conv_ciphertexts]
    except Exception as exc:
        raise EncryptedPathError(
            "square",
            f"square activation failed after encrypted convolution: {type(exc).__name__}: {exc}",
            {"states": states, "convolution_decrypted": conv_values,
             "timings_seconds": {"encrypt": input_seconds, "encrypted_convolution": conv_seconds,
                                 "decrypt_before_failure": decrypt_seconds}},
            exc,
        ) from exc
    square_seconds = time.perf_counter() - started
    started = time.perf_counter()
    squared_values = [float(value.decrypt()[0]) for value in square_ciphertexts]
    stage_decrypt_seconds = time.perf_counter() - started
    decrypt_seconds += stage_decrypt_seconds
    states["square"] = [ciphertext_state(value) for value in square_ciphertexts]

    # The reconstructed model uses non-overlapping 2x2 pooling at stride 2.
    # Sum encrypted values, then multiply by 1/4 while still encrypted.
    started = time.perf_counter()
    pooled_ciphertexts: list[ts.CKKSVector] = []
    conv_h, conv_w = 6, 6
    for k in range(len(kernels)):
        for pool_row in range(3):
            for pool_col in range(3):
                flat_positions = [
                    (pool_row * 2 + dy) * conv_w + (pool_col * 2 + dx)
                    for dy in range(2)
                    for dx in range(2)
                ]
                offset = k * conv_h * conv_w
                pooled_value = square_ciphertexts[offset + flat_positions[0]]
                for position in flat_positions[1:]:
                    pooled_value = pooled_value + square_ciphertexts[offset + position]
                try:
                    pooled_value = pooled_value * 0.25
                except Exception as exc:
                    raise EncryptedPathError(
                        "pooling",
                        f"encrypted average pooling scale failed for kernel={k}, row={pool_row}, col={pool_col}: "
                        f"{type(exc).__name__}: {exc}",
                        {"states": states, "convolution_decrypted": conv_values, "square_decrypted": squared_values,
                         "pool_sum_state_before_scale": ciphertext_state(pooled_value),
                         "timings_seconds": {"encrypt": input_seconds, "encrypted_convolution": conv_seconds,
                                             "encrypted_square": square_seconds,
                                             "decrypt_before_failure": decrypt_seconds}},
                        exc,
                    ) from exc
                pooled_ciphertexts.append(pooled_value)
    pool_seconds = time.perf_counter() - started
    started = time.perf_counter()
    pooled_values = [float(value.decrypt()[0]) for value in pooled_ciphertexts]
    stage_decrypt_seconds = time.perf_counter() - started
    decrypt_seconds += stage_decrypt_seconds
    states["pooled"] = [ciphertext_state(value) for value in pooled_ciphertexts]

    return EncryptedFeatures(
        input_decrypted=input_decrypted,
        convolution=conv_values,
        squared=squared_values,
        pooled=pooled_values,
        states=states,
        timings_seconds={
            "encrypt": input_seconds,
            "encrypted_convolution": conv_seconds,
            "encrypted_square": square_seconds,
            "encrypted_pooling": pool_seconds,
            "decrypt_intermediates": decrypt_seconds,
        },
        serialized_ciphertext_bytes={
            "input_ciphertext_bytes": len(encrypted_input.serialize()),
            "convolution_ciphertexts_bytes_sum": sum(len(value.serialize()) for value in conv_ciphertexts),
            "square_ciphertexts_bytes_sum": sum(len(value.serialize()) for value in square_ciphertexts),
            "pooled_ciphertexts_bytes_sum": sum(len(value.serialize()) for value in pooled_ciphertexts),
            "convolution_ciphertext_count": len(conv_ciphertexts),
            "square_ciphertext_count": len(square_ciphertexts),
            "pooled_ciphertext_count": len(pooled_ciphertexts),
        },
    )


def encrypted_im2col_partial_inference(
    standardized_features: np.ndarray,
    parameters: dict,
    head_weights: np.ndarray,
    head_bias: float,
    context: ts.Context,
) -> dict[str, Any]:
    """TenSEAL's packed/im2col convolution plus square, with early map decryption.

    TenSEAL 0.3.18 exposes conv2d_im2col but no CKKSVector rotation API for the
    grouped 2x2 pooling layout. This diagnostic therefore decrypts the squared
    6x6 map before pooling, and explicitly reports that changed privacy boundary.
    """
    x = np.asarray(standardized_features, dtype=np.float64)
    if x.shape != (64,) or not np.isfinite(x).all():
        raise ValueError("Expected 64 finite standardized input features")
    kernels = np.asarray(parameters["kernels"], dtype=np.float64)
    if kernels.shape != (1, 3, 3):
        raise ValueError("The current im2col diagnostic supports the one-kernel Phase 1 model")
    start = time.perf_counter()
    encrypted_im2col, windows = ts.im2col_encoding(context, x.reshape(8, 8).tolist(), 3, 3, 1)
    encrypt_seconds = time.perf_counter() - start
    state_input = ciphertext_state(encrypted_im2col)

    start = time.perf_counter()
    encrypted_conv = encrypted_im2col.conv2d_im2col(kernels[0].tolist(), windows)
    encrypted_conv = encrypted_conv + float(parameters["conv_bias"][0])
    conv_seconds = time.perf_counter() - start
    conv_state = ciphertext_state(encrypted_conv)

    start = time.perf_counter()
    encrypted_square = encrypted_conv.square()
    square_seconds = time.perf_counter() - start
    square_state = ciphertext_state(encrypted_square)

    start = time.perf_counter()
    conv_values = np.asarray(encrypted_conv.decrypt(), dtype=np.float64)
    square_values = np.asarray(encrypted_square.decrypt(), dtype=np.float64)
    decrypt_seconds = time.perf_counter() - start
    # Early decryption is required because the installed vector API does not
    # expose rotations needed to gather each non-overlapping 2x2 window.
    start = time.perf_counter()
    pooled = square_values.reshape(3, 2, 3, 2).mean(axis=(1, 3)).reshape(-1)
    logit = float(pooled @ np.asarray(head_weights, dtype=np.float64) + head_bias)
    plaintext_pool_head_seconds = time.perf_counter() - start
    return {
        "convolution": conv_values.tolist(),
        "square": square_values.tolist(),
        "pooled_plaintext_after_early_decryption": pooled.tolist(),
        "classification_logit_plaintext_head": logit,
        "prediction": int(logit >= 0.0),
        "states": {"input": state_input, "convolution": conv_state, "square": square_state},
        "timings_seconds": {
            "im2col_encrypt_and_layout": encrypt_seconds,
            "encrypted_convolution": conv_seconds,
            "encrypted_square": square_seconds,
            "decrypt_convolution_and_square": decrypt_seconds,
            "plaintext_pooling_and_head": plaintext_pool_head_seconds,
            "total_until_plaintext_prediction": encrypt_seconds + conv_seconds + square_seconds + decrypt_seconds + plaintext_pool_head_seconds,
        },
        "serialized_ciphertext_bytes": {
            "input": len(encrypted_im2col.serialize()),
            "convolution": len(encrypted_conv.serialize()),
            "square": len(encrypted_square.serialize()),
        },
        "privacy_boundary": "The squared 6x6 convolution map is decrypted before pooling; pooling and head are plaintext.",
    }


def plaintext_stages(features: np.ndarray, parameters: dict) -> dict[str, np.ndarray]:
    x = np.asarray(features, dtype=np.float64).reshape(8, 8)
    kernels = np.asarray(parameters["kernels"], dtype=np.float64)
    biases = np.asarray(parameters["conv_bias"], dtype=np.float64)
    conv = np.empty((len(kernels), 6, 6), dtype=np.float64)
    for k, kernel in enumerate(kernels):
        for row in range(6):
            for col in range(6):
                conv[k, row, col] = np.sum(x[row : row + 3, col : col + 3] * kernel) + biases[k]
    squared = conv * conv
    pooled = squared.reshape(len(kernels), 3, 2, 3, 2).mean(axis=(2, 4))
    return {"convolution": conv.reshape(-1), "squared": squared.reshape(-1), "pooled": pooled.reshape(-1)}


def error_summary(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    delta = np.abs(np.asarray(actual, dtype=np.float64) - np.asarray(expected, dtype=np.float64))
    return {"mean_absolute_error": float(delta.mean()), "max_absolute_error": float(delta.max())}
