from b_core.solution import Solution
from b_core.func import Function, f_names

def main():
    for name in f_names:
        f, (lb, ub) = Function.get_func(name)
        func = Solution(2, lb, ub, f)
        func.hill_climb(0.05, 100, 1000)
        func.animate()


if __name__ == '__main__':
    main()