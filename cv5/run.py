from cv5.differential_evolution import DifferentialEvolution
from core.func import Function, f_names

def main():
    for name in f_names:
        f, bounds, _ = Function.get_func(name)
        differential_evolution = DifferentialEvolution(f, 10, 300, bounds)
        differential_evolution.run()
        differential_evolution.animate()

if __name__ == '__main__':
    main()