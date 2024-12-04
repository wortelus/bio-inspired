from os.path import join

from cv9.firefly import Fireflies
from core.func import Function, f_names

def main():
    for name in f_names:
        f, bounds, _ = Function.get_func(name)
        swarm = Fireflies(f, bounds, 20, 0.5)
        swarm.optimize(100)
        swarm.animate(join("plots", name + ".mp4"))

if __name__ == '__main__':
    main()