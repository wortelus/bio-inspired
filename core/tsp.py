from random import sample, random
from typing import Callable

TspList = list[(str, float, float)]
TspDict = dict[str, (float, float)]

CrossoverFunc = Callable[[TspList, TspList], TspList]
EvalFunc = Callable[[TspDict, list[str]], float]
MutationFunc = Callable[[TspList], TspList]


# Helper function cus I'm lazy
def tsp_list_to_dict(tsp_list: TspList) -> TspDict:
    tsp_dict = {city[0]: (city[1], city[2]) for city in tsp_list}
    return tsp_dict


# Another helper function cus I'm lazy
def tsp_list_to_list_of_cities(tsp_list: TspList) -> list[str]:
    return [city[0] for city in tsp_list]


def tsp_eval_func(cities: TspDict, path: list[str]) -> float:
    cost = 0
    for i in range(len(path) - 1):
        city1 = path[i]
        city2 = path[i + 1]
        cost += ((
                         (cities[city1][0] - cities[city2][0]) ** 2 +
                         (cities[city1][1] - cities[city2][1]) ** 2
                 )** 0.5)
    return cost


def tsp_mutation_func(path: TspList) -> TspList:
    # Copy the path
    new_path = path[:]
    # Swap two random cities
    i, j = sorted(sample(range(len(new_path)), 2))
    new_path[i], new_path[j] = new_path[j], new_path[i]
    return new_path


def tsp_crossover_func(path1: TspList, path2: TspList) -> TspList:
    # Vybereme dva náhodné indexy
    i, j = sorted(sample(range(len(path1)), 2))
    # Nová cesta dle P1
    child = [None] * len(path1)
    # "prostředek" bude z P1
    child[i:j] = path1[i:j]
    # Pozice, kde začneme vkládat z P2 (po konci prostředku, ale 'j' se vrátí zase na začátek)
    position = j
    for city in path2:
        if city not in child:
            if position >= len(child):
                position = 0
            child[position] = city
            position += 1
    return child


def default_population(cities: TspList, size: int) -> list[TspList]:
    return [sample(cities, len(cities)) for _ in range(size)]


class Tsp:
    def __init__(self,
                 cities: list[(str, float, float)],
                 generations: int,
                 population_size: int = 100,
                 crossover: CrossoverFunc = tsp_crossover_func,
                 eval_func: EvalFunc = tsp_eval_func,
                 mutation_func: MutationFunc = tsp_mutation_func,
                 ):
        self.cities_all = cities
        self.cities_list = tsp_list_to_list_of_cities(cities)
        self.cities_dict = tsp_list_to_dict(cities)
        
        self.crossover = crossover
        self.population_size = population_size
        self.generations = generations

        self.mutation_rate = 0.5

        self.eval_func = eval_func
        self.mutation_func = mutation_func

        self.populations = [default_population(self.cities_list, population_size)]

    def run(self):
        # Stvořme Adama a Evu
        population = self.populations[0]
        # Ohodnoťme je
        costs = [self.eval_func(self.cities_dict, path) for path in population]
        
        # Listy pro nejlepší cesty a ceny z dané populace v dané generaci
        best_cost = [min(costs)]
        best_path = [population[costs.index(best_cost[0])]]

        for gen in range(self.generations):
            print(f"{gen + 1}/{self.generations}\tGeneration")
            # Zkopírujme si populaci a ceny
            new_population = population[:]
            new_costs = costs[:]

            for pop in range(self.population_size):
                maminka, tatinek = sample(population, 2)
                # Zkřížíme maminku a tatínka (pusinky)
                babetko = self.crossover(maminka, tatinek)
                # Zmutujme babetko
                if random() < self.mutation_rate:
                    babetko = self.mutation_func(babetko)
                # Ohodnoťme babetko
                child_cost = self.eval_func(self.cities_dict, babetko)

                # Pokud je babetko lepší než maminka, tak ji nahraďme a její cenu taky za babetko
                if child_cost < self.eval_func(self.cities_dict, maminka):
                    new_population[pop] = babetko
                    new_costs[pop] = child_cost

            # 
            population = new_population
            costs = new_costs
            
            best_cost.append(min(costs))
            best_path.append(population[costs.index(best_cost[-1])])
            
            print(f"{gen + 1}/{self.generations}\tBest cost: {best_cost[-1]}")
            
        return best_path, best_cost

    def plot_animation(self, path: list[TspList], costs: list[float]):
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        fig, ax = plt.subplots()
        x = [city[0] for city in self.cities_dict.values()]
        y = [city[1] for city in self.cities_dict.values()]
        ax.scatter(x, y)
        line, = ax.plot([], [], lw=2)

        def init():
            line.set_data([], [])
            return line,

        def animate(i):
            cur_path = path[i]
            cur_cost = costs[i]
            ax.set_title(f"Generation {i + 1}, Cost: {cur_cost}")
            
            _x = [self.cities_dict[city][0] for city in cur_path]
            _y = [self.cities_dict[city][1] for city in cur_path]
            line.set_data(_x, _y)
            return line,

        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(path), interval=200, blit=True)
        # broken pipe for large gifs ???
        # anim.save(f'plots/{(random() * 1000) % 1000}.gif', writer='imagemagick', fps=10)
        anim.save(f'plots/{int(random() * 1000 % 1000)}.mp4', writer='ffmpeg', fps=10, dpi=200)
        # plt.show()
