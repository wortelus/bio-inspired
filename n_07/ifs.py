import random
import matplotlib.pyplot as plt

from affine import Affine3DTransform


class Ifs:
    def __init__(self,
                 transformations: list[Affine3DTransform],
                 iterations: int,
                 seed: int = None):
        if iterations < 0:
            raise ValueError("Iterations must be greater or equal to 0")

        self.transformations = transformations
        self.iterations = iterations
        self.points = []
        if seed is not None:
            random.seed(seed)

    def generate(self):
        # počáteční bod
        x, y, z = 0., 0., 0.

        # opakujme iterace pro x, y, z
        for _ in range(self.iterations):
            # náhodný výběr afinní transformace
            t = random.choice(self.transformations)

            # aplikace transformace
            x, y, z = self._transform(x, y, z, t)

            # přidání bodu do seznamu
            self.points.append((x, y, z))
        return self.points

    def _transform(self,
                   x: float,
                   y: float,
                   z: float,
                   t: Affine3DTransform):
        a, b, c, d, e, f, g, h, i, j, k, l = t

        # transformace bodu (x, y, z) pomocí dané matice t (a posunu j, k, l)
        x_new = a * x + b * y + c * z + j
        y_new = d * x + e * y + f * z + k
        z_new = g * x + h * y + i * z + l

        return x_new, y_new, z_new

    def plot(self, title: str = None):
        fig = plt.figure()

        # 1 subplot na 1x1 grid
        ax = fig.add_subplot(111, projection='3d')

        # rozložení bodů do prostoru
        x, y, z = zip(*self.points)
        ax.scatter(x, y, z, s=1)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        if title:
            plt.title(title)

        plt.show()
