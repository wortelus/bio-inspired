from cv1_2.main import Solution
from cv1_2.func import Function, f_names

def main():
    for name in f_names:
        func = Solution(2, -4, 4, Function.get_func(name))
        func.hill_climb(0.05, 100, 1000)
        func.animate()


if __name__ == '__main__':
    main()