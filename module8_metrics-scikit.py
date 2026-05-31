#!/usr/bin/env python3
"""Read N (x, y) binary-label points and print precision and recall.

- x is the ground-truth label (0 or 1)
- y is the predicted label (0 or 1)
- NumPy is used for data storage and insertion
- scikit-learn is used for metric computation
"""

from __future__ import annotations

import sys

import numpy as np
from sklearn.metrics import precision_score, recall_score


def read_positive_int(prompt: str) -> int:
    """Read a positive integer from stdin."""
    while True:
        try:
            value = int(input(prompt).strip())
            if value <= 0:
                print("Please enter a positive integer.")
                continue
            return value
        except ValueError:
            print("Invalid input. Please enter a positive integer.")


def read_binary_label(prompt: str) -> int:
    """Read a binary label (0 or 1) from stdin."""
    while True:
        try:
            value = int(input(prompt).strip())
            if value not in (0, 1):
                print("Please enter either 0 or 1.")
                continue
            return value
        except ValueError:
            print("Invalid input. Please enter either 0 or 1.")


def main() -> int:
    n = read_positive_int("Enter N (positive integer): ")

    # Data initialization with NumPy
    points = np.zeros((n, 2), dtype=int)

    # Data insertion with NumPy
    for i in range(n):
        print(f"Point {i + 1}:")
        x = read_binary_label("  Enter x (ground truth, 0 or 1): ")
        y = read_binary_label("  Enter y (prediction, 0 or 1): ")
        points[i, 0] = x
        points[i, 1] = y

    y_true = points[:, 0]
    y_pred = points[:, 1]

    # ML / metric computation using scikit-learn
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
