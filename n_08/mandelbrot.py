import colorsys

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import animation


class Mandelbrot:
    def __init__(self,
                 max_iter: int = 1000,
                 center: (float, float) = (0., 0.),
                 zoom: float = 1.0,
                 width: int = 800,
                 height: int = 800,
                 m: int = 2):
        self.max_iter = max_iter
        self.center = center
        self.zoom = zoom
        self.width = width
        self.height = height
        self.m = m

        # rozsah komplexní roviny dle zoomu
        # jelikož pro X -> <-2; 1> a pro Y -> <-1; 1>
        # pak je rozsah 3 a 2
        w = 3.0 / zoom / 2.
        h = 2.0 / zoom / 2.

        # rozsah okolo centru
        cx, cy = center
        self.x_min = cx - w
        self.x_max = cx + w
        self.y_min = cy - h
        self.y_max = cy + h

        # vytvoření mřížky X/Y hodnot
        # X je reálná složka
        # Y je imaginární složka
        x = np.linspace(self.x_min, self.x_max, width)
        y = np.linspace(self.y_min, self.y_max, height)
        X, Y = np.meshgrid(x, y)

        # z real. složky X a imaginární složky Y komplexní plane
        C = X + 1j * Y

        # nastavení z_0 (nultá iterace) na samé 0
        Z = np.zeros_like(C)
        self.iterations_count = np.zeros(C.shape, dtype=int)
        escaped = np.zeros(C.shape, dtype=bool)

        # iterace Mendelbrot
        # začneme od 1... iterations_count[x,y] = 0 značí, že bod (x,y) je uvnitř Mandelbrot setu
        # 1 na druhou stranu značí, že bod ihned překročil threshold (2)
        for i in range(1, self.max_iter):
            # boolean maska tam, kde to ještě nepřekročilo hodnotu m
            mask = np.abs(Z) <= 2
            # tam, kde to nepřekročilo hodnotu m (mask je pole o velikosti like Z a C)
            # Z[mask] a C[max] vrátí konkrétní flat seznam hodnot z Z/C, kde mask=True

            # jádro výpočetu pro Mandelbrot set
            Z[mask] = Z[mask] ** self.m + C[mask]

            # tato maska nám zkontroluje jestli hodnota
            # překročila thresholf 2, a zda již nebyla nastavena
            # (abychom nepřepsali)
            newly_escaped = (np.abs(Z) > 2) & (~escaped)
            self.iterations_count[newly_escaped] = i
            escaped |= newly_escaped

    def np_rgb(self):
        rgb_image = np.zeros((self.height, self.width, 3), dtype=np.float32)

        for i in range(self.height):
            for j in range(self.width):
                # za kolik iterací kandidát na Mandelbrot unikl ?
                # pokud neutekl nikdy, iterace nikdy nenašla iteaci
                # kde by výsledná hodnota Z byla větší než threshold (2)
                escape_factor = self.iterations_count[i, j] / self.max_iter
                # neutekla ? nastavíme černou (vnitřek mandelbrot setu)
                # utekla ? využijeme HSV colorspace, abychom
                # měli pěkný r-g-b přechod dle toho, kolik iterací bylo třeba
                # aby kandidát na unikl
                r, g, b = (0, 0, 0) \
                    if escape_factor == 0 \
                    else colorsys.hsv_to_rgb(
                    escape_factor,
                    1.0,
                    1.0
                )
                rgb_image[i, j] = [r, g, b]

        return rgb_image

    def plot_simple(self, zoom: float):
        img = self.np_rgb()
        plt.figure(figsize=(10, 7))
        plt.imshow(img, extent=[self.x_min, self.x_max, self.y_min, self.y_max])
        plt.title(f"Mandelbrot Set (zoom={zoom})")
        # pro vizualizaci vypneme
        # avšak pro kontext, Y je imaginární složka a X je reálná složka
        # plt.axis("off")
        plt.show()

    def animate_zoom(self, start_zoom: float, end_zoom: float, steps: int, interval: int = 100, filename="mandelbrot_animation.gif"):
        fig, ax = plt.subplots(figsize=(8, 8))

        # prázný np rgb array s default rozsahem
        img_display = ax.imshow(np.zeros((self.height, self.width, 3)),
                                extent=[self.x_min, self.x_max, self.y_min, self.y_max])
        ax.axis("off")

        # připravíme zoom kroky
        # lineární
        # zooms = np.linspace(start_zoom, end_zoom, steps)
        # exponencionální ale vypadá líp
        zooms = start_zoom * (end_zoom / start_zoom) ** (np.linspace(0, 1, steps))

        # animační krok, kde zoomujem
        def update(frame):
            zoom = zooms[frame]
            self.__init__(self.max_iter, self.center, zoom, self.width, self.height, self.m)

            img = self.np_rgb()
            img_display.set_data(img)
            img_display.set_extent([self.x_min, self.x_max, self.y_min, self.y_max])
            ax.set_title(f"Zoom: {zoom:.2f}")

            return [img_display]

        # blit=True pro rychlejší překreslování
        ani = FuncAnimation(fig, update, frames=steps, interval=interval, blit=True,
                            repeat=False)
        print(f"Ukládání do {filename}...")
        # použijem PillowWriter pro GIF
        ani.save(filename, writer=animation.PillowWriter(fps=10))
        print("Uloženo")
        plt.close(fig)
