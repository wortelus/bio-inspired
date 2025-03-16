import numpy as np


def he_normal(input_num):
    return np.random.randn(input_num) * np.sqrt(2.0 / input_num)


class Perceptron:
    def __init__(self, input_num, activator, init_weights: str = "he_normal", classification: bool = True):
        self.activator = activator
        if init_weights == "he_normal":
            self.weights = he_normal(input_num)
        elif init_weights == "zero":
            self.weights = np.zeros(input_num)
        else:
            raise ValueError("init_weights must be set")

        self.bias = 0
        self.classification = classification

    def __str__(self):
        return f"weights:\t{self.weights}\nbias\t:{self.bias}"

    def predict(self, input_vec):
        predict = self.activator(np.dot(input_vec, self.weights) + self.bias)

        if self.classification:
            # vstup může být batch, proto np.where
            return np.where(predict > 0, 1, 0)
        else:
            return predict

    def train(self, X, y,
              epochs: int,
              learning_rate: float,
              batch_size: int,
              early_stop=True,
              verbose: bool = False):
        for _ in range(epochs):
            batch_indices = np.random.choice(len(X), batch_size, replace=False)

            # indexace pres numpy [np.array][np.array] -> np.array
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]

            self._update_weights(X_batch, y_batch, learning_rate)

            if verbose or early_stop:
                err = np.mean(y_batch - self.predict(X_batch))
                if verbose:
                    print(f"avg delta error: {err}")
                if early_stop and err == 0:
                    print("Early stopping")
                    break

    def plot(self, X, y, x_min, x_max):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        # Pointy na plotu - červené pro třídu 1, modré pro třídu 0
        for i in range(len(X)):
            if y[i] == 1:
                plt.plot(X[i][0], X[i][1], 'ro')
            else:
                plt.plot(X[i][0], X[i][1], 'bo')

        # y = 3x + 2
        x_line = np.linspace(x_min, x_max, 2)
        y_line = 3 * x_line + 2
        ax.plot(x_line, y_line, 'k--', label="y = 3x + 2")

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()
        ax.set_title("Vizualizace klasifikace perceptronu")

        plt.xlim(x_min, x_max)
        plt.ylim(x_min, x_max)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True)

        plt.tight_layout()
        plt.show()

    def _update_weights(self, X, y, learning_rate):
        output = self.predict(X)

        # y_true − y_pred
        delta = y - output

        # w ← w + η(y_true − y_pred) * x
        self.weights += learning_rate * np.dot(delta, X)

        # b ← b + η(y_true − y_pred)
        self.bias += learning_rate * delta.sum()
