import numpy as np
import matplotlib.pyplot as plt


def logistic_map(x, a):
    return a * x * (1 - x)


def generate_bifurcation_data(a_values, n_iterations, n_transients, x0=0.5):
    a_plot_values = []
    x_plot_values = []

    for a in a_values:
        x = x0
        
        # zahoďme prvních n_transient hodnot
        for _ in range(n_transients):
            x = logistic_map(x, a)

        # uložme hodnoty 'a' a 'x', které rekurentně generujem
        for _ in range(n_iterations - n_transients):
            x = logistic_map(x, a)
            a_plot_values.append(a)
            x_plot_values.append(x)

    return a_plot_values, x_plot_values


def plot_bifurcation(a_plot_values, x_plot_values, title="Bifurcation Diagram"):
    plt.figure(figsize=(12, 7))

    # černé tečky jako a,x body
    plt.plot(a_plot_values, x_plot_values, ',k', alpha=0.25)  # Use small black dots

    plt.xlabel("Parametr 'a'")
    plt.ylabel("Parametr 'x'")
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.plot()
    plt.show()
