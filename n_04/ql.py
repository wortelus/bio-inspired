import numpy as np
from matplotlib import pyplot as plt


class QLearning:
    def __init__(self, environment: np.ndarray):
        self.environment = environment
        self.h, self.w = environment.shape

        # (y, x) - nahoru, doprava, dolů, doleva
        self.moves = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        self.q = np.zeros((self.h, self.w, 4))

    def solve(self,
              iterations=500,
              gamma=0.1,
              # epsilon -> měnící se poměr mezi exploration a exploitation
              epsilon=1.0,
              epsilon_decay=0.995,
              min_epsilon=0.01):
        """
        Dle zadaných parametrů provede počet epizod tréninku a naučí se
        optimální cestu průchodu myškou s pomocí rewards/trestů.
            episodes        - počet epizod tréninku
            gamma           - learning rate
            epsilon         - přidán pro poměr náhodného průzkumu (exploration) a exploitation
            epsilon_decay   - každé epizode se epsilon sníží o tento multiplier
                            -> více exploitation v pozdějších fázích
            min_epsilon     - minimální hodnota epsilon
        """
        # 1 == zeď (nepovolené pole)
        #
        # 2 == start myšky
        start_pos = np.argwhere(self.environment == 2)[0]
        # 3 == sýr
        end_pos = np.argwhere(self.environment == 3)[0]

        for episode in range(iterations):
            current_pos = start_pos.copy()

            # done - krokuje dokud nenalezne sýr nebo narazí do zdi
            done = False
            while not done:
                y, x = current_pos
                if np.random.rand() < epsilon:
                    # exploration: náhodná akce
                    action = np.random.choice(4)
                else:
                    # exploitation: nejlepší akce podle Q-hodnot
                    action = np.argmax(self.q[y, x])

                # výpočet nové pozice (new_pos) podle zvolené akce
                move = self.moves[action]
                new_y = np.clip(y + move[0], 0, self.h - 1)
                new_x = np.clip(x + move[1], 0, self.w - 1)
                new_pos = np.array([new_y, new_x])

                # odměna nebo trest za 'new_pos'
                if np.array_equal(new_pos, end_pos):
                    # sýr nalezen
                    reward = 10
                    done = True
                elif self.environment[new_y, new_x] == 1:
                    # velký trest, bum do zdi
                    reward = -10
                    done = True
                else:
                    # malý trest, každý krok něco stojí
                    reward = -1

                # Q hodnota aktualizace
                # Q(s, a) = Q(s, a) + (reward + gamma * max_a' Q(s', a') - Q(s, a))
                best_next = np.max(self.q[new_y, new_x])
                self.q[y, x, action] += reward + gamma * best_next - self.q[y, x, action]

                current_pos = new_pos

            # po konci epizody snížíme epsilon
            epsilon = max(min_epsilon, epsilon * epsilon_decay)

    def get_optimal_path(self):
        """
        Vrátí optimální cestu z startu do cíle dle naučených Q-hodnot.
        :return: seznam (x, y) tuple, které reprezentují pozice na cestě
        """
        # myška
        start_pos = np.argwhere(self.environment == 2)[0]
        # sýr
        end_pos = np.argwhere(self.environment == 3)[0]

        # aktuální pozice myšky
        current_pos = start_pos.copy()
        # 'path' obsahuje pozice, které myška navštívila
        path = [tuple(current_pos)]

        # opakuj, dokud myška nenalezne sýr
        while not np.array_equal(current_pos, end_pos):
            y, x = current_pos

            # vybereme nejlepší akci podle Q-hodnot (exploitace)
            action = np.argmax(self.q[y, x])
            # posun na 'new_pos' podle zvolené akce
            move = self.moves[action]
            new_y = np.clip(y + move[0], 0, self.h - 1)
            new_x = np.clip(x + move[1], 0, self.w - 1)
            new_pos = np.array([new_y, new_x])

            # pokud jsmě se nepohnuli, tak jsme se zasekli
            # skonči bez nalezení sýra
            if np.array_equal(new_pos, current_pos):
                print("Myška is stuck.")
                break

            path.append((new_y, new_x))
            current_pos = new_pos

        return path

    def plot_result(self, path, title=None):
        # vytvoření vlastní colormap pro environment:
        # 0 - volné pole: bílá
        # 1 - díra: černá
        # 2 - start: zelená
        # 3 - sýr: žlutá
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(['white', 'black', 'green', 'yellow'])

        fig, ax = plt.subplots(figsize=(6, 6))

        # zobrazení prostředí
        ax.imshow(self.environment, cmap=cmap, origin='upper')

        # převod cesty na numpy pole pro vykreslení
        path = np.array(path)
        # vykreslení optimální cesty červenou čarou s kolečky
        ax.plot(path[:, 1], path[:, 0], marker='o', color='red', linewidth=2, markersize=6)

        # plot mřížka
        ax.set_xticks(np.arange(-0.5, self.w, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.h, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=1)

        if title:
            ax.set_title(title)
        plt.show()
