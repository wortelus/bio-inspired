import numpy as np

from n_04.ql import QLearning

environment = np.array([
    [0, 0, 2, 0, 0, 1, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 1, 1, 0, 1],
    [1, 1, 1, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, 0, 1, 3, 0],
])


def main():
    ql = QLearning(environment)

    # pokud neexistuje cesta, nejsme schopni ji vždy zachytit
    # jsme schopni poznat z Q-hodnot, že se nemáme jak posunout
    # avšak .solve() je prakticky úkaz halting problemu

    ql.solve(iterations=1000)
    optimal_path = ql.get_optimal_path()

    ql.plot_result(optimal_path, "Myška :3")
    print("Optimální cesta:", [f"{int(x), int(y)}" for y, x in optimal_path])


if __name__ == "__main__":
    main()
