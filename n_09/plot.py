import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def plot_landscape_3d(heightmap: np.ndarray,
                      sea_level: float | None = None,
                      title: str = "Fraktální krajina (3D pohled)"):
    # 3D plane pro matplotlib
    # 2x2 plot
    fig, axs = plt.subplots(2, 2, figsize=(12, 12), subplot_kw={'projection': '3d'})

    # X/Y souřadnicový meshgrid
    grid_size = heightmap.shape[0]
    x = np.arange(0, grid_size, 1)
    y = np.arange(0, grid_size, 1)
    X, Y = np.meshgrid(x, y)

    # více úhlů pohledu
    angles = [
        (45, 45),
        (30, 135),
        (45, 215),
        # top view
        (90, 0),
    ]

    # zip ax[0..3] a angles[0..3]
    for ax, angle in zip(axs.ravel(), angles):
        surf = ax.plot_surface(
            X, Y,
            heightmap,
            cmap=cm.terrain,
            linewidth=0,
            antialiased=True,
            rstride=2,
            cstride=2
        )

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z výška')
        ax.set_title(f"{title} - Pohled {angle}")

        # nastavení angle
        # výška + azimut (směr)
        ax.view_init(elev=angle[0], azim=angle[1])

    plt.tight_layout()
    plt.show()
