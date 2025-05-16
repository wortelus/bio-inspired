import numpy as np

PI = np.pi

# gravitační zrychlení
G = 9.81

# první rameno (délka a hmotnost)
L1 = 1.0
M1 = 1.0

# druhé rameno (délka a hmotnost)
L2 = 1.0
M2 = 1.0

#
# časové parametry
#

# maximální čas simulace
T_MAX = 30.0
FPS = 60
# časové body
T_POINTS = np.arange(0, T_MAX, 1 / FPS)

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 700
CENTER_X = SCREEN_WIDTH // 2
CENTER_Y = SCREEN_HEIGHT // 2 - 100

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED_COL = (255, 0, 0)
BLUE_COL = (0, 0, 255)
GREEN_COL = (0, 255, 0)
GRAY_COL = (150, 150, 150)
LIGHT_BLUE_COL = (173, 216, 230)