import numpy as np

from perceptron import Perceptron


def task_func(x, y):
    if y > 3 * x + 2:
        return 1
    else:
        return 0


def generate_data(max_xy, num_samples):
    # X - vstupní data
    # y - 1/0 dle task_func()
    X = (np.random.rand(num_samples, 2) - 0.5) * max_xy
    y = np.array([task_func(x, y) for x, y in X])
    return X, y


def main():
    # generování dat
    xy_range = 10
    x, y = generate_data(xy_range, 100)

    # lineární aktivační funkce + klasifikace
    lin = lambda l: l

    # pro 2 features, lin aktivaci a inicializaci na nuly
    model = Perceptron(2, lin, init_weights="zero", classification=True)

    # trénování modelu
    # můžeme nastavit batch_size na 1 abychom viděli postupnou konvergenci mezi epochami
    model.train(x, y, 100, 0.1, 100, verbose=True)

    # plotík
    model.plot(x, y, -xy_range / 1.5, xy_range / 1.5)


if __name__ == "__main__":
    main()
