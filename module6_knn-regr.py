#!/usr/bin/env python3
"""
module6_knn-regr.py

Interactive k-NN regression program.

Behavior:
1) Ask for N (positive integer)
2) Ask for k (positive integer)
3) Read N training points one by one: x then y (real numbers)
4) Ask for test input X
5) Print the k-NN regression prediction if k <= N, otherwise print an error message

Notes:
- Training data are stored in a NumPy array of shape (N, 2).
- Distances are computed with NumPy.
- For 1D inputs, Euclidean and Manhattan distance lead to the same neighbor ordering.
"""

import sys
from typing import Tuple

import numpy as np


def read_positive_int(prompt: str) -> int:
    """Read a positive integer from stdin."""
    while True:
        try:
            value = int(input(prompt).strip())
            if value <= 0:
                print("Error: please enter a positive integer.")
                continue
            return value
        except ValueError:
            print("Error: please enter a valid integer.")


def read_real(prompt: str) -> float:
    """Read a real number from stdin."""
    while True:
        try:
            return float(input(prompt).strip())
        except ValueError:
            print("Error: please enter a valid real number.")


def knn_regression_predict(train_points: np.ndarray, x_query: float, k: int) -> float:
    """
    Return the k-NN regression prediction for a 1D input feature.

    Parameters
    ----------
    train_points : np.ndarray
        Array of shape (N, 2), where column 0 is x and column 1 is y.
    x_query : float
        Query input x.
    k : int
        Number of nearest neighbors.

    Returns
    -------
    float
        Mean y-value among the k nearest neighbors.
    """
    # 1D Euclidean distance; in 1D this equals abs difference.
    distances = np.abs(train_points[:, 0] - x_query)

    # Indices of k smallest distances.
    nearest_idx = np.argsort(distances)[:k]

    # Average the corresponding labels.
    return float(np.mean(train_points[nearest_idx, 1]))


def main() -> None:
    n = read_positive_int("Enter N (positive integer): ")
    k = read_positive_int("Enter k (positive integer): ")

    if k > n:
        print("Error: k must be less than or equal to N.")
        return

    # Preallocate training data with NumPy.
    train_points = np.empty((n, 2), dtype=float)

    for i in range(n):
        x_i = read_real(f"Enter x for point {i + 1}: ")
        y_i = read_real(f"Enter y for point {i + 1}: ")
        train_points[i] = (x_i, y_i)

    x_query = read_real("Enter X for prediction: ")

    prediction = knn_regression_predict(train_points, x_query, k)
    print(f"Predicted Y: {prediction}")


if __name__ == "__main__":
    main()
