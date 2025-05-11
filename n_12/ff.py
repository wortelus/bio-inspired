import random
import numpy as np
import tkinter as tk

from n_12.const import TREE, EMPTY, FIRE, cmap, BURNED
from n_12.utils import rgba_to_hex


class ForestFire:
    def __init__(self,
                 grid_size: int,
                 initial_density: float,
                 p_growth: float = 0.01,
                 p_fire_spontaneous: float = 0.0001,
                 neighborhood_type: str = 'von_neumann',
                 tk_master: tk.Tk = None,
                 tk_cell_size: int = 10):
        self.grid_size = grid_size
        self.initial_density = initial_density
        self.grid = np.zeros((grid_size, grid_size), dtype=int)
        self.p_growth = p_growth
        self.p_fire_spontaneous = p_fire_spontaneous

        self.cell_size = tk_cell_size
        if tk_master:
            self.master = tk_master
            total_size = tk_cell_size * grid_size
            self.canvas = tk.Canvas(tk_master,
                                    width=total_size,
                                    height=total_size,
                                    bg="black")
            self.canvas.pack()

        if neighborhood_type not in ['von_neumann', 'moore']:
            raise ValueError("nesprávný typ okolí políčka")

        self.neighborhood_type = neighborhood_type

    def populate(self):
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if random.random() < self.initial_density:
                    self.grid[r, c] = TREE

    def get_neighbors(self, grid, r, c, neighborhood_type='von_neumann'):
        neighbors = []
        rows, cols = grid.shape

        if neighborhood_type == 'von_neumann':
            # pouze 4 pravoúhlé sousedy
            # N, S, E, W
            indices = [
                ((r - 1) % rows, c),
                ((r + 1) % rows, c),
                (r, (c - 1) % cols),
                (r, (c + 1) % cols),
            ]
        elif neighborhood_type == 'moore':
            # všichni sousedi (včetně diagonálních)
            # N, S, E, W, NE, NW, SE, SW
            indices = [
                ((r - 1) % rows, c),  # N
                ((r + 1) % rows, c),  # S
                (r, (c - 1) % cols),  # W
                (r, (c + 1) % cols),  # E
                ((r - 1) % rows, (c - 1) % cols),  # NW
                ((r - 1) % rows, (c + 1) % cols),  # NE
                ((r + 1) % rows, (c - 1) % cols),  # SW
                ((r + 1) % rows, (c + 1) % cols),  # SE
            ]
        else:
            raise ValueError("nesprávný typ okolí políčka")

        for nr, nc in indices:
            neighbors.append(grid[nr, nc])

        return neighbors

    def update_grid(self):
        current_grid = self.grid
        new_grid = self.grid.copy()
        rows, cols = new_grid.shape

        for r in range(rows):
            for c in range(cols):
                current_state = current_grid[r, c]
                neighbors = self.get_neighbors(current_grid, r, c, self.neighborhood_type)

                is_neighbor_burning = any(n == FIRE for n in neighbors)

                if current_state == EMPTY or current_state == BURNED:
                    # prázdný nebo spálený má šanci p na vzkvétnutí nového stromu
                    if random.random() < self.p_growth:
                        new_grid[r, c] = TREE

                elif current_state == TREE:
                    # pokud kterýkoli ze sousedů hoří -> hoříme taky
                    if is_neighbor_burning:
                        new_grid[r, c] = FIRE
                    else:
                        # pokud je strom, pak náhodně zapálíme
                        if random.random() < self.p_fire_spontaneous:
                            new_grid[r, c] = FIRE

                elif current_state == FIRE:
                    # byli jsme předtím zapálení -> teď shoříme
                    new_grid[r, c] = BURNED

        self.grid = new_grid

    def draw_grid(self):
        if not self.canvas:
            raise ValueError("tkinter canvas není inicializován")

        self.canvas.delete("all")

        gs = self.grid_size
        cs = self.cell_size

        for r in range(gs):
            for c in range(gs):
                x1, y1 = c * cs, r * cs
                x2, y2 = x1 + cs, y1 + cs
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=rgba_to_hex(cmap(self.grid[r, c])), outline="black")
        # vynutí překreslení
        self.master.update_idletasks()

    def animate(self, ms=50):
        self.update_grid()
        self.draw_grid()
        self.master.after(ms, self.animate)
