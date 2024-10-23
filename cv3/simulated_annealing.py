from cv1_2.main import Solution
from cv1_2.func import Function, f_names

def main():
    for name in f_names:
        f, (lb, ub) = Function.get_func(name)
        func = Solution(2, lb, ub, f)
        func.simulated_annealing(ub / 100, 1, 
                                 100, 0.95, 0.5, 
                                 200)
        func.animate()


if __name__ == '__main__':
    main()