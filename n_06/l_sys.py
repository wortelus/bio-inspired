from math import cos, sin, radians

from matplotlib import pyplot as plt


class LSystem:
    def __init__(self, axiom: str,
                 rules: dict[str, str],
                 angle: float,
                 nesting: int):
        if nesting < 0:
            raise ValueError("Iterations must be greater or equal to 0")

        self.axiom = axiom
        self.rules = rules
        self.current_string = axiom
        self.iterations = nesting
        self.angle = angle

        self._generate()

    def _generate(self):
        for _ in range(self.iterations):
            next_string = ""
            # zpracování tokenů pro rekurzivní překlad do finální formy
            for char in self.current_string:
                # buď použijeme pravidlo z self.rules, nebo ponecháme znak beze změny
                next_string += self.rules.get(char, char)
            self.current_string = next_string

    def plot(self, length: int = 2, title: str = None):
        stack = []
        # současná souřadnice
        x, y = 0, 0
        # směr pohybu
        direction = (length, 0)  # (dx, dy)
        for char in self.current_string:
            if char == "F":
                # dopředu dle aktuálního směru

                plt.plot([x, x + direction[0]], [y, y + direction[1]], 'b-')
                # update souřadnic na novou pozici
                x += direction[0]
                y += direction[1]
            elif char == "+":
                # doprava
                direction = self._turn(direction, self.angle)
            elif char == "-":
                # doleva
                direction = self._turn(direction, -self.angle)
            elif char == "[":
                # uložení aktuálního směru
                stack.append((x, y, direction))
            elif char == "]":
                # návrat do uloženého směru
                x, y, direction = stack.pop()
        # vykreslení
        # otočení osy y aby střed souřadného systému byl v TL
        plt.gca().invert_yaxis()
        # 1:1 poměr
        plt.axis('equal')
        plt.axis('off')
        if title:
            plt.title(title)
        plt.show()

    def _turn(self, direction: tuple[float, float], angle: float) -> tuple[float, float]:
        angle = radians(angle)
        # otočení směru dle známého vzorce (ZPG)
        return (direction[0] * cos(angle) - direction[1] * sin(angle),
                direction[0] * sin(angle) + direction[1] * cos(angle))
