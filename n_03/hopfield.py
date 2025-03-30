import numpy as np


def _to_bipolar(pattern):
    # [0, 1] -> [-1, 1]
    return pattern * 2 - 1


class Hopfield:
    def __init__(self, pattern):
        self.pattern = pattern
        valid = np.all(np.isin(pattern, [0, 1]))
        if not valid:
            raise ValueError("Pattern must be binary")

        # Velikost řádku/sloupce patternu
        self.size = pattern.shape[1]
        if pattern.shape[0] != pattern.shape[1]:
            raise ValueError("Pattern must be square matrix")

        self.weights = self._calculate_weights()

    def _calculate_weights(self):
        # Výpočet vah

        # Column vector
        flattened = self.pattern.reshape(-1, 1)
        flattened = _to_bipolar(flattened)
        # Weights matrix
        weights = flattened @ flattened.T
        # Nastavení diagonály na nuly
        np.fill_diagonal(weights, 0)
        return weights

    def asynchronous_recovery(self, input_pattern):
        x = input_pattern.copy()
        x = _to_bipolar(x)

        # Zde projdeme iterativně všechny sloupce matice vah
        for i in range(self.size ** 2):
            # násobení sloupce ve weights matici s x
            h_i = np.dot(self.weights[i], x)

            # x[i] = 1 if h_i >= 0 else -1
            x[i] = np.sign(h_i)
        return x

    def synchronous_recovery(self, input_pattern, steps=5):
        x = input_pattern.copy()
        x = _to_bipolar(x)

        # Celý weights matrix je vynásobený x
        for _ in range(steps):
            x = np.sign(self.weights @ x)

        return x

    def plot(self, pattern, title=None):
        import matplotlib.pyplot as plt
        plt.imshow(pattern, cmap='gray')
        if title:
            plt.title(title)

        size = pattern.shape[0]
        plt.xticks(np.arange(size))
        plt.yticks(np.arange(size))
        plt.colorbar()
        plt.show()
