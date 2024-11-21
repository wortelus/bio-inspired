from os.path import join

from cv7.soma_ato import SomaAtO
from core.func import Function, f_names

def main():
    for name in f_names:
        f, bounds, _ = Function.get_func(name)
        soma = SomaAtO(f, 100, 20, bounds, 0.11, 0.4, 3.0)
        soma.run()
        soma.animate(join("plots", name + ".mp4"))

if __name__ == '__main__':
    main()