import numpy as np
from landscape import generate_fractal_landscape
from plot import plot_landscape_3d

def main():
    ITERATIONS = 8
    SEED = 10

    # placatý čtverec
    INITIAL_POINTS = (0, 0, 0, 0)
    # vyšší hodnota -> drsnější výkyvy a vyšší rozsah
    INITIAL_PERTURBATION = 10.0
    # konstanta, kterou dělíme každou iteraci perturbation
    # ovlivňuje "drastičnost" nových bodů v pozdějších iteracích
    SMOOTHNESS = 2.0

    print(f"Generování mřížky ({2 ** ITERATIONS + 1}x{2 ** ITERATIONS + 1})...")
    heightmap = generate_fractal_landscape(
        seed=SEED,
        iterations=ITERATIONS,
        initial_points=INITIAL_POINTS,
        initial_perturbation=INITIAL_PERTURBATION,
        smoothness=SMOOTHNESS,
    )
    min_h, max_h = np.min(heightmap), np.max(heightmap)
    print(f"Hotovo, Z range: {min_h:.2f} to {max_h:.2f}")

    # nastavení levelu vody
    SEA_LEVEL = 0.0
    heightmap = np.maximum(SEA_LEVEL, heightmap)
    plot_landscape_3d(heightmap, SEA_LEVEL)

if __name__ == "__main__":
    main()