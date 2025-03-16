from copy import deepcopy

import numpy as np
import matplotlib.animation as animation
from datetime import datetime
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap

from b_core.tsp import TspList
from b_08.ant import Ant


# Kalkulace po dokončení průchodu všemi mravenci
def vaporization_func(pheromones: np.ndarray, vaporization_rate: float):
    pheromones *= (1 - vaporization_rate)


class ACO:
    def __init__(self,
                 cities: TspList,
                 generations: int,
                 ant_count: int,
                 evaporating_rate: float,
                 pheromone_constant: float,
                 alpha: float,
                 beta: float,
                 ):
        self.cities = cities
        self.num_cities = len(cities)

        self.generations = generations
        self.ant_count = ant_count
        self.evaporating_rate = evaporating_rate
        self.pheromone_constant = pheromone_constant
        self.alpha = alpha
        self.beta = beta

        self.city_matrix = self.init_city_matrix()
        self.pheromone_matrix = self.init_pheromone_matrix(self.pheromone_constant)
        self.ants = self.init_ants()

        self.best_paths = []
        self.best_costs = []

        self.all_paths = []
        self.all_costs = []

        self.pheromone_matrices = []

        if self.num_cities == self.ant_count:
            print("IDEÁL - Každý mravenec bude začínat v jiném městě")
        else:
            print("POZOR - Předpokládáme náhodné rozdělení startovních měst")

    def init_city_matrix(self) -> np.ndarray:
        # 0 - id, 1 - x, 2 - y
        return np.array(
            [[np.sqrt(
                (city1[1] - city2[1]) ** 2 +
                (city1[2] - city2[2]) ** 2
            )
                for city2 in self.cities]
                for city1 in self.cities]
        )

    def init_pheromone_matrix(self, initiation_value: float = 1.0) -> np.ndarray:
        return np.ones((len(self.cities), len(self.cities))) * initiation_value

    def init_ants(self) -> list:
        if self.num_cities == self.ant_count:
            # Každý mravenec začíná v jiném městě
            # print("Každý mravenec bude začínat v jiném městě")
            return [Ant(i, self.alpha, self.beta, self.city_matrix, self.pheromone_matrix, i) for i in
                    range(self.num_cities)]
        else:
            # Náhodné rozdělení startovních měst
            # print("Předpokládáme náhodné rozdělení startovních měst")
            return [Ant(i, self.alpha, self.beta, self.city_matrix, self.pheromone_matrix) for i in
                    range(self.num_cities)]

    def evaporate_pheromone(self):
        vaporization_func(self.pheromone_matrix, self.evaporating_rate)

    def run(self):
        for g in range(self.generations):
            print(f"{g + 1}/{self.generations}\tGeneration")

            # Reset ants and evaporate pheromone
            self.ants = self.init_ants()
            self.evaporate_pheromone()

            for ant in self.ants:
                while len(ant.visited) < self.num_cities:
                    ant.next()
                ant.update_all_pheromones()

            best_ant = min(self.ants, key=lambda x: x.calculate_path())

            # Collect best ant from this generation
            self.all_costs.append(best_ant.calculate_path())
            self.all_paths.append(best_ant.visited)
            self.pheromone_matrices.append(deepcopy(self.pheromone_matrix))

            # Collect best ant from this generation (IF BETTER)
            if len(self.best_costs) == 0 or best_ant.calculate_path() < self.best_costs[-1]:
                self.best_costs.append(best_ant.calculate_path())
                self.best_paths.append(best_ant.visited)
                print(f"Better:\t{g}, Z={self.best_costs[-1]}")
            else:
                self.best_paths.append(self.best_paths[-1])
                self.best_costs.append(self.best_costs[-1])
                print(f"Worse:\t{g}")

        return self.best_paths, self.best_costs

    # převzato a upraveno z b_core/tsp.py
    def animate(self):
        paths = self.best_paths
        costs = self.best_costs

        # Počet snímků
        frames = len(paths)
        assert len(paths) == len(costs)

        fig, ax = plt.subplots()
        x = [city[1] for city in self.cities]
        y = [city[2] for city in self.cities]
        ax.scatter(x, y)

        for city_name, city_x, city_y in self.cities:
            ax.annotate(city_name,
                        (city_x, city_y),
                        textcoords="offset points", xytext=(5, 5),
                        ha='center', fontsize=8)

        line, = ax.plot([], [], lw=2)

        def init():
            line.set_data([], [])
            return line,

        def animate(i):
            # Počítáme se všemi cestami (i s těmi zhoršujícími se)            
            cur_path = paths[i]
            cur_cost = costs[i]
            ax.set_title(f"Generace {i + 1}, Nejlepší mravenec: {cur_cost}")

            _x = [self.cities[i][1] for i in cur_path]
            _y = [self.cities[i][2] for i in cur_path]

            # Barvičky
            val_min, val_max = 0., 1.
            normalized_value = (cur_cost - val_min) / (val_max - val_min)
            normalized_value = np.clip(normalized_value, val_min, val_max)
            cmap = get_cmap('viridis')  # Replace 'viridis' with your preferred colormap
            color = cmap(normalized_value)
            line.set_color(color)

            # Hotfix: zjistil jsem že TSP počítá aj zpáteční cestu
            # _x.append(_x[0])
            # _y.append(_y[0])
            line.set_data(_x, _y)
            return line,

        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=frames, interval=200, blit=True)
        anim.save(f'plots/{datetime.now().isoformat()}.mp4', writer='ffmpeg', fps=20, dpi=300)

    def animate_pheromones(self):
        frames = len(self.pheromone_matrices)
        fig, ax = plt.subplots()

        img = ax.imshow(self.pheromone_matrices[0], vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
        cbar = fig.colorbar(img, ax=ax)
        ax.set_title("Generace 1")

        def animate(i):
            img.set_data(self.pheromone_matrices[i])  # Update the image data
            ax.set_title(f"Generace {i + 1}")
            return [img]

        anim = animation.FuncAnimation(fig, animate, frames=frames, repeat=False)
        anim.save(f'plots/{datetime.now().isoformat()}_pheromones.mp4', writer='ffmpeg', fps=20, dpi=300)

    def animate_all(self):
        paths = self.best_paths
        costs = self.best_costs
        pheromone_matrices = self.pheromone_matrices

        # Počet snímků
        frames = len(paths)
        assert len(paths) == len(costs) == len(pheromone_matrices)

        # Dva subploty - 1. TSP Cesty, 2. Pheromone Matrix
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        plt.tight_layout()

        # Subplot 1: TSP Cesty
        x = [city[1] for city in self.cities]
        y = [city[2] for city in self.cities]
        
        ax1.scatter(x, y)
        for city_name, city_x, city_y in self.cities:
            ax1.annotate(city_name,
                         (city_x, city_y),
                         textcoords="offset points", xytext=(5, 5),
                         ha='center', fontsize=8)

        line, = ax1.plot([], [], lw=2)
        ax1.set_aspect('equal', adjustable='box')

        # Subplot 2: Pheromone Matrix
        img = ax2.imshow(pheromone_matrices[0], vmin=0.0, vmax=1.0, cmap="viridis", aspect="equal")
        cbar = fig.colorbar(img, ax=ax2)
        ax2.set_title("Generace 1")        
        ax2.set_aspect('equal', adjustable='box')

        def init():
            # Inicializace TSP Cest a Pheromone Matrix
            line.set_data([], [])
            img.set_data(np.zeros_like(pheromone_matrices[0]))
            return line, img

        def animate(i):
            # Aktualizace TSP cest
            cur_path = paths[i]
            cur_cost = costs[i]
            ax1.set_title(f"Generace {i + 1}, Nejlepší mravenec: {cur_cost}")

            _x = [self.cities[j][1] for j in cur_path]
            _y = [self.cities[j][2] for j in cur_path]

            # Barvičky
            val_min, val_max = 0.0, 1.0
            normalized_value = (cur_cost - val_min) / (val_max - val_min)
            normalized_value = np.clip(normalized_value, val_min, val_max)
            cmap = get_cmap('viridis')
            color = cmap(normalized_value)
            line.set_color(color)

            # TSP čára
            line.set_data(_x, _y)

            # Aktualizace Pheromone Matrix
            img.set_data(pheromone_matrices[i])
            ax2.set_title(f"Generace {i + 1}")

            return line, img

        # Animace
        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=frames, interval=200, blit=True)

        # Uložit animaci
        anim.save(f'plots/{datetime.now().isoformat()}_combined.mp4', writer='ffmpeg', fps=10, dpi=300)
