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
    # Read N and k
    N = read_positive_int("Enter N (positive integer): ")
    k = read_positive_int("Enter k (positive integer): ")

    if k > N:
        print("Error: k must be less than or equal to N.")
        return

    # Initialize training data using NumPy
    points = np.empty((N, 2), dtype=float)

    # Read N points: (x, y)
    for i in range(N):
        x = read_real(f"Enter x for point {i + 1}: ")
        y = read_real(f"Enter y for point {i + 1}: ")
        points[i, 0] = x
        points[i, 1] = y

    # Split into features and labels
    X_train = points[:, 0].reshape(-1, 1)
    y_train = points[:, 1]

    # Variance of labels
    label_variance = np.var(y_train)
    print(f"Variance of labels: {label_variance}")

    # Read query X
    X_query = read_real("Enter X for prediction: ")

    # k-NN Regression using scikit-learn
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X_train, y_train)

    y_pred = model.predict(np.array([[X_query]]))[0]
    print(f"Predicted Y: {y_pred}")


if __name__ == "__main__":
    main()
