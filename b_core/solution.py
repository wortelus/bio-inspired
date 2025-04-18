from matplotlib.animation import FuncAnimation

from b_core.func import Function, f_names
import matplotlib.pyplot as plt

import numpy as np


class Solution:
    def __init__(self, dimension: int,
                 lower_bound: float,
                 upper_bound: float,
                 func: Function):
        self.dimension = dimension
        self.lB = lower_bound
        self.uB = upper_bound
        self.func = func

        self.best_cost = np.empty(dtype=np.float64, shape=(0,))
        self.best_params = np.empty(shape=(0, self.dimension)).reshape(-1, self.dimension)

        self.fig, self.ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(7, 7), dpi=100)

    def blind(self, iterations, population):
        # for every i iteration
        for i in range(iterations):
            iter_cost = np.inf
            iter_params = np.zeros(self.dimension)
            # we create j population size, of which we choose one best parameter set with the lowest cost
            # TODO: should be NumPy-ed instead of this terrible for loop
            # for j in range(population):
            #     params = np.random.uniform(self.lB, self.uB, self.dimension)
            #     fit = self.func.func(params)
            #     if fit < iter_fit:
            #         iter_fit = fit
            #         iter_params = params

            # vectorized approach instead of previous terrible for loops
            population_params = np.random.uniform(self.lB, self.uB, (population, self.dimension))

            # entire population computation
            cost_values = np.array([self.func.func(params) for params in population_params])

            best_index = np.argmin(cost_values)
            iter_cost = cost_values[best_index]
            iter_params = population_params[best_index]

            # after running j populations, we append the best iteration data to best_fit and best_params NumPy arrays
            if (len(self.best_cost) == 0 or
                    iter_cost < self.best_cost[-1]):
                self.best_cost = np.append(self.best_cost, iter_cost)
                self.best_params = np.vstack([self.best_params, iter_params])

                # stdout if we did better or worse than the previous best iteration
                print(f'Better:\t{i}, Z={iter_cost}')
            else:
                print(f'Worse:\t{i}')

        return

    def hill_climb(self, step_size: float, neighbour_count: int, max_iterations=1000):
        previous_params = np.random.uniform(self.lB, self.uB, self.dimension)
        for i in range(1, max_iterations):
            neighbour_params = previous_params[:, np.newaxis] + np.random.uniform(-step_size, step_size,
                                                                                  (self.dimension, neighbour_count))
            # flip the axis, TODO: this is not invariant to more axis selection
            neighbour_params = np.array(list(zip(neighbour_params[0], neighbour_params[1])))
            cost_values = np.array(list(map(self.func.func, neighbour_params)))

            # Find the index of the minimum cost value
            best_index = np.argmin(cost_values)
            iter_fit = cost_values[best_index]
            iter_params = neighbour_params[best_index]

            if (len(self.best_cost) == 0 or
                    iter_fit < self.best_cost[-1]):
                self.best_cost = np.append(self.best_cost, iter_fit)
                self.best_params = np.vstack([self.best_params, iter_params])
                previous_params = iter_params

                # stdout if we did better or worse than the previous best iteration
                print(f'Better:\t{i}, Z={iter_fit}')
            else:
                # in hill climb, stop if current iteration made no improvement
                print(f'Worse:\t{i}')
                print("Stopping")
                break

            print(f"{i}/{max_iterations} iteration of population {neighbour_count}")

    def simulated_annealing(self, step_size: float, neighbour_count: int,
                            initial_temp: float, cooling_rate: float, stop_temp: float,
                            max_iterations=1000):
        # Initialize parameters randomly
        current_params = np.random.uniform(self.lB, self.uB, self.dimension)
        current_cost = self.func.func(current_params)

        # Initial temperature
        temperature = initial_temp

        # Best solutions so far
        self.best_cost = np.append(self.best_cost, current_cost)
        self.best_params = np.vstack([self.best_params, current_params])

        for i in range(1, max_iterations):
            # Generate neighbours around current parameters
            neighbour_params = (current_params[:, np.newaxis] +
                                np.random.uniform(-step_size, step_size,
                                                  (self.dimension, neighbour_count)))
            # TODO: only works for 2D, should be generalized
            neighbour_params = np.array(list(zip(neighbour_params[0], neighbour_params[1])))

            # Compute the cost of all generated neighbours in current iteration
            cost_values = np.array(list(map(self.func.func, neighbour_params)))

            # Find the index of the minimum cost value (best neighbour)
            best_index = np.argmin(cost_values)
            best_neighbour_cost = cost_values[best_index]
            best_neighbour_params = neighbour_params[best_index]

            # If the new solution is better, accept it
            if best_neighbour_cost < self.best_cost[-1]:
                current_params = best_neighbour_params
                current_cost = best_neighbour_cost
                print(f'Better:\t{i}, Z={current_cost}')
            else:
                # If the new solution is worse, accept it based on following probability
                delta_cost = best_neighbour_cost - current_cost
                acceptance_probability = np.exp(-delta_cost / temperature)

                if np.random.rand() < acceptance_probability:
                    current_params = best_neighbour_params
                    current_cost = best_neighbour_cost
                    print(f'Accepted worse solution at iteration {i} with cost {current_cost}')
                else:
                    print(f"Rejected worse solution at iteration {i} with cost {current_cost}")
                    current_cost = self.best_cost[-1]
                    current_params = self.best_params[-1]

            # Update the Solution
            self.best_cost = np.append(self.best_cost, current_cost)
            self.best_params = np.vstack([self.best_params, current_params])

            print(f"Iteration {i}/{max_iterations} | Current cost: {current_cost} | Temperature: {temperature}")

            # Cooldown
            temperature = temperature * cooling_rate

            # Stop if the temperature is low enough
            if temperature < stop_temp:
                print("Temperature has cooled down, stopping.")
                break

        return self.best_params, self.best_cost

    def plot(self, i: int):
        self.ax.clear()

        #
        # surface
        #
        x = np.linspace(self.lB, self.uB, 100)
        y = np.linspace(self.lB, self.uB, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for j in range(X.shape[0]):
            for k in range(X.shape[1]):
                params = [X[j, k], Y[j, k]]
                Z[j, k] = np.array(self.func.func(params))

        #
        # best points
        #
        x = self.best_params[i, 0]
        y = self.best_params[i, 1]
        z = self.best_cost[i]

        # surface
        surf = self.ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.3)

        # best points
        scatter = self.ax.scatter(x, y, z, c='r', marker='o')

        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        plt.title(f'Iter: {i}, Z={self.best_cost[i]}')

        return surf, scatter

    def animate(self):
        # Creating the animation
        anim = FuncAnimation(self.fig, self.plot, frames=self.best_cost.shape[0], repeat=False)
        anim.save(f'plots/{self.func.name}.gif', writer='imagemagick', fps=10)

        # Save or show the animation
        # plt.show()

    def animate_best(self):
        def rotate(i):
            self.ax.view_init(elev=30., azim=i)
            return self.plot(len(self.best_cost) - 1)  # Plot the last frame (best solution)

        anim = FuncAnimation(self.fig, rotate, frames=np.arange(0, 360, 2), repeat=False)
        anim.save(f'plots/{self.func.name}_rotate.gif', writer='imagemagick', fps=30)