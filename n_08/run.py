import matplotlib.pyplot as plt

from n_08.mandelbrot import Mandelbrot


def main():
    # Parameters
    width, height = 800, 600
    max_iter = 50
    zoom = 1.0
    center_x, center_y = -0.5, 0.0

    # Mendel brot
    mendelbrot = Mandelbrot(max_iter, (center_x, center_y), zoom, width, height)
    mendelbrot.plot_simple(zoom)

    # zoomnem na zajímavý bod
    # vizuálně řečeno "mezi ty 2 největší kuličky"
    center = (-0.7435, 0.1314)
    max_iter = 200
    mendelbrot = Mandelbrot(max_iter, center, zoom, width, height)
    mendelbrot.animate_zoom(10.0, 2000.0, 120)

if __name__ == "__main__":
    main()
