import numpy as np
from sklearn.neighbors import KNeighborsRegressor


def read_positive_int(prompt: str) -> int:
    while True:
        try:
            value = int(input(prompt).strip())
            if value > 0:
                return value
            print("Error: please enter a positive integer.")
        except ValueError:
            print("Error: please enter a valid integer.")


def read_real(prompt: str) -> float:
    while True:
        try:
            return float(input(prompt).strip())
        except ValueError:
            print("Error: please enter a valid real number.")


def main():
    N = read_positive_int("Enter N (positive integer): ")
    k = read_positive_int("Enter k (positive integer): ")

    if k > N:
        print("Error: k must be less than or equal to N.")
        return

    points = np.empty((N, 2), dtype=float)

    for i in range(N):
        x = read_real(f"Enter x for point {i + 1}: ")
        y = read_real(f"Enter y for point {i + 1}: ")
        points[i, 0] = x
        points[i, 1] = y

    X_train = points[:, 0].reshape(-1, 1)
    y_train = points[:, 1]

    label_variance = np.var(y_train)
    print(f"Variance of labels: {label_variance}")

    X_query = read_real("Enter X for prediction: ")

    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X_train, y_train)

    y_pred = model.predict(np.array([[X_query]]))[0]
    print(f"Predicted Y: {y_pred}")


if __name__ == "__main__":
    main()
