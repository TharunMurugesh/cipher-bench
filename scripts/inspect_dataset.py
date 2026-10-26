from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.benchmark import load_digit_pair
from data.dataset import load_binary_digits


PAIRS = {"digits_0_vs_1": (0, 1), "digits_2_vs_7": (2, 7), "digits_3_vs_8": (3, 8), "digits_4_vs_9": (4, 9)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=tuple(PAIRS), required=True)
    args = parser.parse_args()
    low, high = PAIRS[args.task]
    data = load_binary_digits() if args.task == "digits_0_vs_1" else load_digit_pair(low, high)
    headers = ["sample_id", "class_label", "digit"] + [f"pixel_{i}" for i in range(64)]
    rows = []
    for index, (features, label, source_index) in enumerate(zip(data.features, data.labels, data.source_indices)):
        digit = high if int(label) == 1 else low
        rows.append([int(source_index), int(label), digit, *[int(pixel) for pixel in features]])
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(headers)
    writer.writerows(rows)
    response = {
        "dataset": "scikit-learn Digits",
        "task": args.task,
        "class_mapping": {str(low): 0, str(high): 1},
        "source_url": "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html",
        "description": "8×8 grayscale handwritten digit images. Pixel values are integers from 0 to 16. class_label is the binary label used by the benchmark; digit is the original digit.",
        "sample_count": len(rows),
        "feature_count": 64,
        "class_counts": {"0": int(np.sum(data.labels == 0)), "1": int(np.sum(data.labels == 1))},
        "columns": headers,
        "preview": rows[:10],
        "csv": buffer.getvalue(),
    }
    print(json.dumps(response, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
