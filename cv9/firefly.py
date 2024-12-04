import numpy as np
from matplotlib import pyplot as plt, animation

from core.func import Function


class Firefly:
    def __init__(self, func: Function, bounds: tuple, beta_0: float = 1, gamma: float = None):
        self.func = func
        self.position = np.random.uniform(bounds[0], bounds[1], size=2)
        self.orig_light_intensity = self.func.func(self.position)
        
        self.gamma = gamma
        self.attract = self.atractiveness
        if gamma is None:
            self.attract = self.atractiveness_no_tuning
            self.gamma = 1.0
        
        self.bounds = bounds
        self.beta_0 = beta_0
        
    def euclidean_distance(self, other_firefly):
        return np.linalg.norm(self.position - other_firefly.position)
        
    def intensity(self, other_firefly):
        return self.func.func(self.position) * np.exp(-self.gamma * self.euclidean_distance(other_firefly))
    
    def atractiveness(self, other_firefly):
        return self.beta_0 * np.exp(-self.gamma * self.euclidean_distance(other_firefly))
    
    def atractiveness_no_tuning(self, other_firefly):
        return self.beta_0 / (1 + self.gamma * self.euclidean_distance(other_firefly))


class Fireflies:
    def __init__(self,
                 function: Function,
                 bounds: tuple,
                 firefly_count: int,
                 alpha: float = 0.3,
                 beta_0: float = 1, 
                 gamma: float = None):
        self.function = function
        self.bounds = bounds
        self.alpha = alpha
        self.beta_0 = beta_0
        self.gamma = gamma
        
        self.fireflies = [Firefly(function, bounds, beta_0, gamma) for _ in range(firefly_count)]

        # Animation data
        self.generations = []
        self.positions = []
        self.costs = []

    def optimize(self, n_iterations):
        for i in range(n_iterations):
            print(f"{i + 1}/{n_iterations}\tGeneration")
            for firefly in self.fireflies:
                for other in self.fireflies:
                    a_intensity = firefly.intensity(other)
                    b_intensity = other.intensity(firefly)
                    if a_intensity > b_intensity:
                        firefly.position = (firefly.position + firefly.attract(other) *
                                            (other.position - firefly.position) +
                                            self.alpha * np.random.normal(size=2))
                        
                        firefly.position = np.clip(firefly.position, self.bounds[0], self.bounds[1])
                        
            self.generations.append(i)
            self.positions.append([firefly.position for firefly in self.fireflies])
            self.costs.append(min([firefly.func.func(firefly.position) for firefly in self.fireflies]))
            
            print("Best firefly:", self.costs[-1])


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
        scatter = ax.scatter([], [], color="red")
        ax.set_xlim(self.bounds[0], self.bounds[1])
        ax.set_ylim(self.bounds[0], self.bounds[1])
        ax.set_aspect('equal')

        # Frame funkce
        def update(frame):
            positions = self.positions[frame]
            scatter.set_offsets(positions)
            ax.set_title(f"Iterace {frame}, nalezené minimum cost funkce: {self.costs[frame]}")
            return scatter,

        # Animace
        plt.tight_layout()
        ani = animation.FuncAnimation(fig, update, frames=len(self.positions), blit=True)
        ani.save(filename, fps=10, writer="ffmpeg", dpi=300)
        plt.close(fig)