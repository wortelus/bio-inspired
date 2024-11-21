from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt, animation

from core.func import Function

class SomaAtO:
    def __init__(self,
                 function: Function,
                 max_migrations: int,
                 population_size: int,
                 bounds: tuple,
                 step: float,
                 prt: float,
                 path_length: float):
        self.function = function
        self.max_migrations = max_migrations
        self.population_size = population_size
        self.bounds = bounds
        self.dimensions = len(bounds)
        self.step = step
        self.prt = prt
        self.path_length = path_length
        
        # Nejlepší
        self.best_costs = []
        self.best_parameters = []
        
        # Pro animaci
        self.migration_positions = []
        self.migration_costs = []
        self.leader_idx = []
        
    def run(self):
        population = [np.random.uniform(self.bounds[0], self.bounds[1], size=2) 
                      for _ in range(self.population_size)]
        costs = [self.function.func(p) for p in population]
        
        # Nejlepší z populace
        best_idx = np.argmin(costs)        
        self.best_costs.append(self.function.func(population[best_idx]))
        self.best_parameters.append(population[best_idx])
        
        # Pro animaci
        self.migration_positions.append(population[:])
        self.migration_costs.append(costs[best_idx])
        self.leader_idx.append(best_idx)
        
        
        for migration in range(self.max_migrations):
            # Vůdce je ten, který má nejlepší cost
            costs = [self.function.func(p) for p in population]
            leader_idx = np.argmin(costs)
            leader = population[leader_idx]
            
            # Všichni z populace až na leadera
            for i in range(self.population_size):
                individual_moving = population[i]
                offsprings = np.empty([self.dimensions, 0])
                if i == best_idx:
                    continue
                    
                # po STEP od 0 až po PATH_LENGTH dělej
                # prt vektor (1/0)
                # k je index
                # GENERACE POTOMKŮ
                for k in np.arange(0, self.path_length, self.step):
                    prt_vector = (np.random.rand(self.dimensions,) < self.prt) * 1
                    offspring = individual_moving + (leader - individual_moving) * k * prt_vector
                    offspring = offspring.reshape(self.dimensions, 1)
                    offsprings = np.append(offsprings, offspring, axis=1)
                    
                
                # Pokud je off-bounds, tak nahraď náhodným číslem z intervalu bounds
                for column in range(offsprings.shape[1]):
                    for row in range(self.dimensions):
                        if offsprings[row, column] < self.bounds[0] or offsprings[row, column] > self.bounds[1]:
                            offsprings[row, column] = np.random.uniform(self.bounds[0], self.bounds[1])
                            
                # Evaluace
                offspring_costs = [self.function.func(offsprings[:, i]) for i in range(offsprings.shape[1])]
                
                # Výběr nejlepšího potomka
                best_idx = np.argmin(offspring_costs)
                
                # Pokud je jeden z potomků lepší než current individuál z populace
                if offspring_costs[best_idx] < costs[i]:
                    self.best_costs.append(offspring_costs[best_idx])
                    self.best_parameters.append(offsprings[:, best_idx])
                    
                    # Nahraď individuála nejlepším potomkem
                    population[i] = offsprings[:, best_idx]
                    
                    print(f"Migration: {migration}\tBest cost: {self.best_costs[-1]}")
                else:
                    print(f"Migration: {migration}\tNo update: {offspring_costs[best_idx]}")
                        
            self.migration_positions.append(population[:])
            self.migration_costs.append(self.best_costs[-1])
            self.leader_idx.append(leader_idx)


    # Taken from cv6/particle_swarm.py
    def animate(self, filename: str, resolution=100):
        # Generate a grid for the heatmap
        x = np.linspace(self.bounds[0], self.bounds[1], resolution)
        y = np.linspace(self.bounds[0], self.bounds[1], resolution)
        X, Y = np.meshgrid(x, y)

        # Výsledky na Z ose pro heatmapu
        Z = np.array([[self.function.func(np.array([xi, yi])) for xi, yi in zip(row_x, row_y)] for row_x, row_y in zip(X, Y)])

        # Matplotlib figura s heatmap & scatterplot
        fig, ax = plt.subplots(figsize=(8, 8))
        heatmap = ax.imshow(Z, extent=(self.bounds[0], self.bounds[1], self.bounds[0], self.bounds[1]),
                            origin="lower", cmap="viridis", alpha=0.75)
        scatter = ax.scatter([], [], color="red", label="Individuály")
        scatter_leader = ax.scatter([], [], color="blue", label="Vůdce")
        ax.set_xlim(self.bounds[0], self.bounds[1])
        ax.set_ylim(self.bounds[0], self.bounds[1])
        ax.set_aspect('equal')
        ax.legend()
    
        # Frame funkce
        def update(frame):
            positions = self.migration_positions[frame]
            leader_idx = self.leader_idx[frame]
    
            # Leader z 'positions' pryč
            positions_no_leader = np.delete(positions, leader_idx, axis=0)
    
            # Scatter
            scatter.set_offsets(positions_no_leader)
            scatter_leader.set_offsets(positions[leader_idx].reshape(1, -1))
            
            ax.set_title(f"Migrace {frame}, Nejmenší cost: {self.migration_costs[frame]}")
            return scatter, scatter_leader
    
        # Animace
        plt.tight_layout()
        ani = animation.FuncAnimation(fig, update, frames=len(self.migration_positions), blit=True)
        ani.save(filename, fps=2, writer="ffmpeg", dpi=300)
        plt.close(fig)
                    