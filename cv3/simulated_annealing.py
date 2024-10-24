from core.solution import Solution
from core.func import Function, f_names

def main():
    for name in f_names:
        f, (lb, ub), temp_mult = Function.get_func(name)
        func = Solution(2, lb, ub, f)
        func.simulated_annealing(ub / 100, 1, 
                                 100 * temp_mult, 0.95, 0.05 * temp_mult, 
                                 500)
        func.animate()


if __name__ == '__main__':
    main()