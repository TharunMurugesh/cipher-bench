import json
import math
import platform
import sys
import traceback

import numpy as np
import tenseal as ts

CASES = {
    "paper_literal": {
        "poly_modulus_degree": 8192,
        "coeff_mod_bit_sizes": [40, 21, 21, 40],
        "global_scale": 2**40,
    },
    "seal_documented_example": {
        "poly_modulus_degree": 8192,
        "coeff_mod_bit_sizes": [60, 40, 40, 60],
        "global_scale": 2**40,
    },
}
A = [1.25, -2.0]
B = [0.5, 3.0]
ADD_EXPECTED = [1.75, 1.0]
MUL_EXPECTED = [0.625, -6.0]


def emit(name, value):
    print(f"{name}={json.dumps(value, default=str)}", flush=True)


def errors(actual, expected):
    absolute = [abs(a - e) for a, e in zip(actual, expected)]
    return {"max_abs": max(absolute), "mean_abs": sum(absolute) / len(absolute)}


emit("runtime", {
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "tenseal": ts.__version__,
    "numpy": np.__version__,
})

for case_name, params in CASES.items():
    emit(f"{case_name}.parameters", params)
    try:
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=params["poly_modulus_degree"],
            coeff_mod_bit_sizes=params["coeff_mod_bit_sizes"],
        )
        context.global_scale = params["global_scale"]
        context.generate_relin_keys()
        context.generate_galois_keys()
        emit(f"{case_name}.context_created", True)
        emit(f"{case_name}.auto_flags", {
            "auto_rescale": context.auto_rescale,
            "auto_mod_switch": context.auto_mod_switch,
            "auto_relin": context.auto_relin,
        })

        left = ts.ckks_vector(context, A)
        right = ts.ckks_vector(context, B)
        emit(f"{case_name}.input_ciphertext", {
            "scale": left.ciphertext()[0].scale,
            "coeff_modulus_size": left.ciphertext()[0].coeff_modulus_size(),
            "parms_id": left.ciphertext()[0].parms_id(),
        })

        summed = left + right
        sum_values = summed.decrypt()
        emit(f"{case_name}.addition", {
            "decrypted": sum_values,
            "expected": ADD_EXPECTED,
            "error": errors(sum_values, ADD_EXPECTED),
            "ciphertext": {
                "scale": summed.ciphertext()[0].scale,
                "coeff_modulus_size": summed.ciphertext()[0].coeff_modulus_size(),
                "parms_id": summed.ciphertext()[0].parms_id(),
            },
        })

        product = left * right
        product_values = product.decrypt()
        emit(f"{case_name}.ciphertext_multiplication", {
            "decrypted": product_values,
            "expected": MUL_EXPECTED,
            "error": errors(product_values, MUL_EXPECTED),
            "ciphertext": {
                "scale": product.ciphertext()[0].scale,
                "coeff_modulus_size": product.ciphertext()[0].coeff_modulus_size(),
                "parms_id": product.ciphertext()[0].parms_id(),
            },
        })
    except Exception as exc:
        emit(f"{case_name}.exception", {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        })

