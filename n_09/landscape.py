import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import random


def generate_fractal_landscape(
        seed: 10,
        iterations: int,
        initial_points: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
        initial_perturbation: float = 1.0,
        smoothness: float = 2.0) -> np.ndarray:
    # seed
    np.random.seed(seed)
    random.seed(seed)

    # finalni velikost gridu
    # 4 startovni body -> iterativne do 3x3, 5x5, 9x9... atd.
    grid_size = 2 ** iterations + 1
    # inicializace gridu s "budouci" velikosti s default=0
    heightmap = np.zeros((grid_size, grid_size), dtype=float)
    # nastaveni 4 bodu do jejich respective rohu
    tl, tr, bl, br = initial_points
    heightmap[0, 0] = tl
    heightmap[0, grid_size - 1] = tr
    heightmap[grid_size - 1, 0] = bl
    heightmap[grid_size - 1, grid_size - 1] = br

    current_perturbation = initial_perturbation
    # v diamond smycce je step_size // 2 offset pro ziskani stredu 4 bodu
    # kazdou iteraci se zmensuje // 2
    step_size = grid_size - 1

    for i in range(iterations):
        half_step = step_size // 2
        # pokud half_step je moc mala, zastav
        if half_step == 0:
            break

        # diamond-square algoritmus
        # 2 kroky iterativne ->
        # diamond: nastavi centry kazdeho ctverce, co uz ma 4 body present a nenastaveny stred
        # square: nastavi centry kazdeho diamantu, co uz ma 4 body present a nenastaveny stred
        # diamant je prakticky definovan stredovymi body ctvercu

        # 1/2 DIAMOND STEP
        # začátek od half_step s step_size kroky
        for r in range(half_step, grid_size, step_size):
            for c in range(half_step, grid_size, step_size):
                # vezmeme 4 rohove body dle half_step
                # hledáme sousedy or (r, c)
                c1 = heightmap[r - half_step, c - half_step]
                c2 = heightmap[r - half_step, c + half_step]
                c3 = heightmap[r + half_step, c - half_step]
                c4 = heightmap[r + half_step, c + half_step]

                # prumer vysky 4 rohu v dane iteraci
                avg = (c1 + c2 + c3 + c4) / 4.0
                # nahodny offset k nove vysce
                offset = np.random.uniform(-current_perturbation / 2.0, current_perturbation / 2.0)
                # nastaveni prumer 4 bodu + nahodny offset
                # jelikoz iterujeme s pomoci step size: half_step
                # mame garantovano, ze nenarazime na jiz nastaveny prumer
                heightmap[r, c] = avg + offset

        # 2/2 SQUARE STEP
        # print("iteration:", i)
        # naopak oproti diamond step:
        # začátek 0 s half_step kroky
        # (oproti half_step se step_size kroky v diamond stepu)
        for r in range(0, grid_size, half_step):
            # avšak zde musíme alternovat
            # "na oko" je to jen sudé/liché shift
            # dobrá vizualizace je zde:
            # https://en.wikipedia.org/wiki/Diamond-square_algorithm#Visualization
            for c in range((r + half_step) % step_size, grid_size, step_size):
                # print(f"r:{r}\tc:{c}")

                # opět, hledáme sousedy or (r, c)
                neighbors = []
                if r - half_step >= 0:
                    neighbors.append(heightmap[r - half_step, c])
                if r + half_step < grid_size:
                    neighbors.append(heightmap[r + half_step, c])
                if c - half_step >= 0:
                    neighbors.append(heightmap[r, c - half_step])
                if c + half_step < grid_size:
                    neighbors.append(heightmap[r, c + half_step])

                # opet -> avg + nahodny offset dle soucasne perturbace
                if neighbors:
                    avg = sum(neighbors) / len(neighbors)
                    offset = np.random.uniform(
                        -current_perturbation / 2.0,
                        current_perturbation / 2.0)
                    heightmap[r, c] = avg + offset

        # snizeni perturbace a step_size (half_step se samozrejme snizi // 2 na zacatku iterace)
        current_perturbation /= smoothness
        step_size //= 2

    return heightmap
