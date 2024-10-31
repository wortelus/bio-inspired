from random import random

from core.tsp import Tsp, TspList
from core.func import Function, f_names


def get_random_positions(cities: TspList, min_x, max_x, min_y, max_y):
    for i, city in enumerate(cities):
        cities[i] = (city[0],
                     min_x + random() * (max_x - min_x),
                     min_y + random() * (max_y - min_y)
                     )


def main():
    cities = [
        ('Prague', 50.08, 14.43),
        ('Brno', 49.20, 16.37),
        ('Ostrava', 49.50, 18.15),
        ('Plzen', 49.45, 13.22),
        ('Liberec', 50.45, 15.03),
        ('Olomouc', 49.35, 17.15),
        ('Ceske Budejovice', 48.58, 14.28),
        ('Hradec Kralove', 50.12, 15.50),
        ('Usti nad Labem', 50.40, 14.02),
        ('Pardubice', 50.02, 15.46),
        ('Zlin', 49.13, 17.40),
        ('Frýdek-Místek', 49.41, 18.21),
        ('Karviná', 49.51, 18.32),
        ('Jihlava', 49.24, 15.35),
        ('Mladá Boleslav', 50.24, 14.54),
        ('Opava', 49.56, 17.54),
        ('Teplice', 50.38, 13.50),
        ('Děčín', 50.46, 14.12),
        ('Karlovy Vary', 50.13, 12.52),
        ('Chomutov', 50.27, 13.24),
        ('Jablonec nad Nisou', 50.43, 15.10),
        ('Třebíč', 49.13, 15.52),
        ('Prostějov', 49.28, 17.07),
        ('Přerov', 49.27, 17.27),
        ('Cheb', 50.04, 12.23),
        ('Kolín', 50.01, 15.12),
        ('Trutnov', 50.34, 15.55),
        ('Kroměříž', 49.18, 17.23),
        ('Šumperk', 49.58, 16.58),
        ('Vsetín', 49.20, 17.59),
        ('Valašské Meziříčí', 49.28, 17.58),
        ('Litvínov', 50.36, 13.38),
        ('Nový Jičín', 49.35, 18.00),
        ('Uherské Hradiště', 49.04, 17.27),
        ('Havířov', 49.46, 18.25),
        ('Žďár nad Sázavou', 49.34, 15.56),
        ('Česká Lípa', 50.41, 14.32),
        ('Klatovy', 49.24, 13.17),
        ('Bohumín', 49.54, 18.21),
        ('Hodonín', 48.51, 17.08),
    ]
    get_random_positions(cities, 48, 51, 12, 19)
    tsp = Tsp(cities, 1000, 20, 0.5)
    paths, costs = tsp.run()
    tsp.plot_animation(paths, costs)
    print("done")


if __name__ == '__main__':
    main()
