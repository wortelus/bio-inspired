from matplotlib.colors import ListedColormap
from matplotlib import pyplot as plt

# typ pole
EMPTY = 0
TREE = 1
FIRE = 2
BURNED = 3

# barvy pro jednotlivé stavy lesa
# černá = prázdné pole
# tmavě zelená = strom
# červená = oheň
# hnědá = spálený strom
cmap = ListedColormap(['black', 'darkgreen', 'red', 'brown'])
bounds = [0, 1, 2, 3]
norm = plt.Normalize(vmin=0, vmax=3)