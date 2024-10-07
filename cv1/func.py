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
        suma = 0
        for p in params:
            suma += p ** 2
        return suma

    def Schwefel(self, params):
        suma = 0
        for p in params:
            suma += -p * np.sin(np.sqrt(np.abs(p)))
        return suma

    def Rosenbrock(self, params):
        suma = 0
        for i in range(len(params) - 1):
            suma += 100 * (params[i + 1] - params[i] ** 2) ** 2 + (params[i] - 1) ** 2
        return suma

    def Rastrigin(self, params):
        suma = 0
        for p in params:
            suma += p ** 2 - 10 * np.cos(2 * np.pi * p) + 10
        return suma

    def Griewank(self, params):
        suma1 = 0
        suma2 = 1
        for i in range(len(params)):
            suma1 += params[i] ** 2
            suma2 *= np.cos(params[i] / np.sqrt(i + 1))
        return 1 + suma1 / 4000 - suma2

    def Levy(self, params):
        if type(params) is not np.ndarray:
            params = np.array(params)

        suma = 0
        w = 1 + (params - 1) / 4
        for i in range(len(params) - 1):
            suma += (w[i] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[i] + 1) ** 2) + (w[-1] - 1) ** 2 * (
                        1 + np.sin(2 * np.pi * w[-1]) ** 2)
        return suma

    def Michalewicz(self, params):
        suma = 0
        for i in range(len(params)):
            suma += -np.sin(params[i]) * np.sin((i + 1) * params[i] ** 2 / np.pi) ** 20
        return suma

    def Zakharov(self, params):
        suma1 = 0
        suma2 = 0
        for i in range(len(params)):
            suma1 += params[i] ** 2
            suma2 += 0.5 * (i + 1) * params[i]
        return suma1 + suma2 ** 2 + suma2 ** 4

    def Ackley(self, params):
        suma1 = 0
        suma2 = 0
        for i in range(len(params)):
            suma1 += params[i] ** 2
            suma2 += np.cos(2 * np.pi * params[i])
        return -20 * np.exp(-0.2 * np.sqrt(suma1 / len(params))) - np.exp(suma2 / len(params)) + 20 + np.e

    def get_func(name: str):
        if name == "sphere":
            return Function(name, Function.sphere)
        elif name == "Schwefel":
            return Function(name, Function.Schwefel)
        elif name == "Rosenbrock":
            return Function(name, Function.Rosenbrock)
        elif name == "Rastrigin":
            return Function(name, Function.Rastrigin)
        elif name == "Griewank":
            return Function(name, Function.Griewank)
        elif name == "Levy":
            return Function(name, Function.Levy)
        elif name == "Michalewicz":
            return Function(name, Function.Michalewicz)
        elif name == "Zakharov":
            return Function(name, Function.Zakharov)
        elif name == "Ackley":
            return Function(name, Function.Ackley)