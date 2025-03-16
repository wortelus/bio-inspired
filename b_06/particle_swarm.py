import numpy as np
from matplotlib import pyplot as plt, animation


class Particle:
    def __init__(self,
                 bounds: tuple,
                 c1: float,
                 c2: float,
                 w: float):
        self.position = np.random.uniform(bounds[0], bounds[1], size=2)
        self.velocity = np.random.uniform(-1, 1, size=2)
        self.bounds = bounds
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.best_position = self.position
        self.best_value = np.inf

    def update(self,
               global_best_position: float,
               w: float):
        r1 = np.random.uniform(size=2)
        r2 = np.random.uniform(size=2)

        # Nový vektor pohybu
        self.velocity = (w * self.velocity +
                         self.c1 * r1 * (self.best_position - self.position) +
                         self.c2 * r2 * (global_best_position - self.position))
        
        # Aktualizace pozice
        self.position = self.position + self.velocity
        
        # Cliping s bounds
        self.position = np.clip(self.position, self.bounds[0], self.bounds[1])
        
        return self.position

    def __str__(self):
        return f"Particle: at {self.position} of best value {self.best_value}"


class Swarm:
    def __init__(self,
                 function: callable,
                 particle_count: int,
                 bounds: tuple,
                 c1: float,
                 c2: float,
                 m_max: int,
                 w_start: float,
                 w_end: float):
        self.w_start = w_start
        self.w_end = w_end
        self.m_max = m_max
        self.function = function
        self.bounds = bounds
        self.particles = [Particle(bounds, c1, c2, w_start) for _ in range(particle_count)]
        self.global_best_position = np.inf
        self.global_best_value = np.inf

        # Animation data
        self.generations = []
        self.positions = []
        self.vectors = []
        self.costs = []

    def optimize(self, n_iterations):
        for i in range(n_iterations):
            print(f"{i + 1}/{n_iterations}\tGeneration")
            
            # Každou iteraci nový inertia_weight
            w = self.w_start - ((self.w_start - self.w_end) * i / self.m_max)

            generation_positions = []
            generation_vectors = []

            # Aktualizace částic
            for particle in self.particles:
                # Uloz pozice a vektory pro animaci
                generation_positions.append(particle.position)
                generation_vectors.append(particle.velocity)

                # Cost funkce evaluace
                value = self.function.func(particle.position)

                # Pokud je hodnota lepší než nejlepší hodnota částice, ulož ji
                if value < particle.best_value:
                    particle.best_value = value
                    particle.best_position = particle.position

                # Pokud je hodnota lepší než nejlepší hodnota globální, ulož ji
                if value < self.global_best_value:
                    self.global_best_value = value
                    self.global_best_position = particle.position

            # Aktualizace pozic a vektorů
            for particle in self.particles:
                particle.update(self.global_best_position, w)

            # Uložení animačních dat po dokončení generace
            self.positions.append(generation_positions[:])
            self.vectors.append(generation_vectors[:])
            self.generations.append(i)
            self.costs.append(self.global_best_value)

        return self.global_best_position

    def __str__(self):
        return f"Swarm: at {self.global_best_position} of best value {self.global_best_value}"


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