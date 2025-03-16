import numpy as np

from n_02.nn import FullyConnectedNN

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def xor(x, y):
    return int(x != y)

def create_dataset(num_samples):
    X = np.random.randint(0, 2, (num_samples, 2))
    y = np.array([xor(x, y) for x, y in X])
    return X, y

def main():
    num_samples = 100
    X, y = create_dataset(num_samples)

    model = FullyConnectedNN(2, [2],
                             [tanh], tanh,
                             [tanh_derivative], tanh_derivative)
    model.train(X, y, 100, 0.1, verbose=True, early_stop=True, early_stop_loss=1e-3)

    X_test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_test = model.predict(X_test)
    print(f"raw: {y_test}")
    y_test = model.predict(X_test, classification_threshold=0.5)
    print(f"classified: {y_test}")


if __name__ == "__main__":
    main()