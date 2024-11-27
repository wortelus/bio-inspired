import numpy as np


class Ant:
    def __init__(self,
                 k: int,
                 alpha: float,
                 beta: float,
                 city_matrix: np.ndarray,
                 pheromone_matrix: np.ndarray,
                 start_city: int = None):
        self.k = k
        
        # Pokud None, předpokládáme náhodný start
        if start_city is None:
            self.start_city = np.random.randint(0, len(city_matrix))
        else:
            self.start_city = start_city
        
        self.alpha = alpha
        self.beta = beta
        self.city_matrix = city_matrix
        
        # Start od k-města
        self.visited = [self.start_city]

        self.pheromone_matrix = pheromone_matrix

    def next(self):
        # Přidáváme města postupně
        assert len(self.visited) < len(self.city_matrix)
        
        # Výběr následujícího města
        next_k = self.select_next_city(self.visited)
        self.visited.append(next_k)

    def select_next_city(self, visited: list) -> int:
        # Pravděpodobnosti
        probabilities = self.calculate_probabilities(visited)

        # Výběr města
        return np.random.choice(len(self.city_matrix), p=probabilities)

    def calculate_probabilities(self, visited: list) -> np.ndarray:
        # Výpočet vektoru pravděpodobností navštívení
        p = np.zeros(len(self.city_matrix))
        p[visited] = 0

        for i in range(len(self.city_matrix)):
            if i not in visited:
                p[i] = (
                        self.pheromone_matrix[visited[-1], i] ** self.alpha *  # Feromon
                        (1 / self.city_matrix[visited[-1], i]) ** self.beta  # Vzdálenost (inverzní)
                )

        return p / np.sum(p)
    
    def calculate_path(self):
        # Výpočet pouze na dokončené cestě
        assert len(self.visited) == len(self.city_matrix)
        
        path = 0
        for i, city in enumerate(self.visited[:-1]):
            next_city = self.visited[i + 1]
            path += self.city_matrix[city, next_city]
            
        return path
    
    def update_all_pheromones(self):
        # Výpočet pouze na dokončené cestě
        assert len(self.visited) == len(self.city_matrix)
        
        for i, city in enumerate(self.visited[:-1]):
            next_city = self.visited[i + 1]
            self.update_pheromone(city, next_city)

    def update_pheromone(self, start_from: int, end_to: int):
        # Výpočet pouze na dokončené cestě
        assert len(self.visited) == len(self.city_matrix)
        
        self.pheromone_matrix[start_from, end_to] += 1 / len(self.visited)
