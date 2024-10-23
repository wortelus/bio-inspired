from collections.abc import Callable
from functools import partial

import numpy as np

f_eval = Callable[[np.ndarray], float]

f_names = ["sphere", "Schwefel", "Rosenbrock", "Rastrigin", "Griewank", "Levy", "Michalewicz", "Zakharov", "Ackley"]


class Function:
    def __init__(self, name, func):
        self.name = name
        self.func = partial(func, self)

    def sphere(self, params):
        params = np.array(params)
        suma = np.sum(params ** 2)
        return suma

    def Schwefel(self, params):
        params = np.array(params)
        d = len(params)
        suma = 418.9829 * d - np.sum(params * np.sin(np.sqrt(np.abs(params))))
        return suma

    def Rosenbrock(self, params):
        params = np.array(params)
        # interesting way to index [x+1] by using [1:] together with [:-1]
        suma = np.sum(100 * (params[1:] - params[:-1] ** 2) ** 2 + (params[:-1] - 1) ** 2)
        return suma

    def Rastrigin(self, params):
        params = np.array(params)
        d = len(params)
        suma = 10 * d * np.sum(params ** 2 - 10 * np.cos(2 * np.pi * params))
        return suma

    def Griewank(self, params):
        params = np.array(params)
        suma1 = np.sum(params ** 2)
        suma2 = np.prod(np.cos(params / np.sqrt(np.arange(1, len(params) + 1))))
        return 1 + suma1 / 4000 - suma2

    def Levy(self, params):
        params = np.array(params)
        w = 1 + (params - 1) / 4

        term1 = (w[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1) ** 2)
        term2 = (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)

        return np.sum(term1) + term2

    def Michalewicz(self, params):
        params = np.array(params)
        m = 10
        suma = -np.sum(
            np.sin(params) * np.pow(np.sin(np.divide([np.arange(1, len(params) + 1)] * params ** 2, np.pi)), 2 * m))
        return suma

    def Zakharov(self, params):
        params = np.array(params)
        d = len(params)
        suma = (
                np.sum(params ** 2) +
                np.sum(0.5 * np.arange(1, d + 1) * params) ** 2 +
                np.sum(0.5 * np.arange(1, d + 1) * params) ** 4
        )
        return suma

    def Ackley(self, params):
        params = np.array(params)
        a = 20
        b = 0.2
        c = 2 * np.pi
        d = len(params)
        suma = (
                -a *
                np.exp(-b * np.sqrt(np.sum(params ** 2) / d)) -
                np.exp(np.sum(np.cos(c * params)) / d) +
                a + np.exp(1)
        )
        return suma

    def get_func(name: str) -> (object, (int, int)):
        if name == "sphere":
            return Function(name, Function.sphere), (-4, 4)
        elif name == "Schwefel":
            return Function(name, Function.Schwefel), (-500, 500)
        elif name == "Rosenbrock":
            return Function(name, Function.Rosenbrock), (-6, 6)
        elif name == "Rastrigin":
            return Function(name, Function.Rastrigin), (-5, 5)
        elif name == "Griewank":
            return Function(name, Function.Griewank), (-10, 10)
        elif name == "Levy":
            return Function(name, Function.Levy), (-10, 10)
        elif name == "Michalewicz":
            return Function(name, Function.Michalewicz), (-4, 4)
        elif name == "Zakharov":
            return Function(name, Function.Zakharov), (-10, 10)
        elif name == "Ackley":
            return Function(name, Function.Ackley), (-40, 40)
