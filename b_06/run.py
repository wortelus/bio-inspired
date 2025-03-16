from os.path import join

from b_06.particle_swarm import Swarm
from b_core.func import Function, f_names

def main():
    for name in f_names:
        f, bounds, _ = Function.get_func(name)
        swarm = Swarm(f, 15, bounds, 0.5, 0.5, 50, 0.9, 0.4)
        swarm.optimize(100)
        swarm.animate(join("plots", name + ".mp4"))

if __name__ == '__main__':
    main()