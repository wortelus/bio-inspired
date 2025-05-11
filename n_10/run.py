import numpy as np
from bifurcation import generate_bifurcation_data, plot_bifurcation


def main():
    # počet rekurentních iterací
    xn_iterations = 1000
    # kolik zahodíme x_n hodnot před záznamem
    # potřebujeme hodnoty zahodit, než se rekurentní hodnoty <~3.4 ustálí
    n_transients = 0

    # konstantní rozsah hodnoty 'a'
    a_values_count = 1000
    a_values = np.linspace(0.0, 4.0, a_values_count)

    a_values, x_values = generate_bifurcation_data(a_values, xn_iterations, n_transients)
    plot_bifurcation(a_values, x_values, title="Logistic Map Bifurcation Diagram (Actual)")


if __name__ == "__main__":
    main()