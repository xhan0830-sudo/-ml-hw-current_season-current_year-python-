import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def read_positive_int(prompt: str) -> int:
    while True:
        try:
            value = int(input(prompt).strip())
            if value > 0:
                return value
            print("Please enter a positive integer.")
        except ValueError:
            print("Invalid input. Please enter a positive integer.")


def read_nonnegative_int(prompt: str) -> int:
    while True:
        try:
            value = int(input(prompt).strip())
            if value >= 0:
                return value
            print("Please enter a non-negative integer.")
        except ValueError:
            print("Invalid input. Please enter a non-negative integer.")


def read_float(prompt: str) -> float:
    while True:
        try:
            return float(input(prompt).strip())
        except ValueError:
            print("Invalid input. Please enter a real number.")


def read_dataset(n: int, name: str):
    xs = np.empty(n, dtype=float)
    ys = np.empty(n, dtype=int)

    for i in range(n):
        print(f"{name} pair {i + 1}:")
        x = read_float("  x = ")
        y = read_nonnegative_int("  y = ")
        xs[i] = x
        ys[i] = y

    return xs.reshape(-1, 1), ys


def main():
    print("kNN Classifier with hyperparameter search (k = 1..10)\n")

    n = read_positive_int("Enter N (training size): ")
    X_train, y_train = read_dataset(n, "Training")

    m = read_positive_int("\nEnter M (test size): ")
    X_test, y_test = read_dataset(m, "Test")

    max_k = min(10, n)
    best_k = 1
    best_acc = -1.0

    for k in range(1, max_k + 1):
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        if acc > best_acc:
            best_acc = acc
            best_k = k

    print(f"\nBest k: {best_k}")
    print(f"Test accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
