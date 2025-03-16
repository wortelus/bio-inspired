import numpy as np
from matplotlib import pyplot as plt

from n_01.perceptron import Perceptron, he_normal


class FullyConnectedNN:
    def __init__(self,
                 input_size: int,
                 layer_sizes: [int],
                 layer_activators: [],
                 output_activation,
                 layer_activators_derivatives: [],
                 output_activation_derivative):
        self.input_size = input_size
        self.layer_sizes = layer_sizes
        self.layer_activators = layer_activators
        self.output_activation = output_activation
        self.layer_activators_derivatives = layer_activators_derivatives
        self.output_activation_derivative = output_activation_derivative

        self.layers = []

        # [[Perceptron]], kde vnitřní list je jeden hidden layer
        previous_size = input_size
        for size, activator, activator_derivative in zip(layer_sizes, layer_activators, layer_activators_derivatives):
            hidden = []
            for _ in range(size):
                hidden.append(Perceptron(input_num=previous_size, activator=activator, init_weights="he_normal",
                                         classification=False, activator_derivative=activator_derivative))

            previous_size = size
            self.layers.append(hidden)

        # poslední vrstva je JEDEN perceptron, ten se taky stara i o to, zda je vystup klasifikace/regrese
        self.layers.append(Perceptron(input_num=previous_size, activator=output_activation, init_weights="he_normal",
                                      classification=False, activator_derivative=output_activation_derivative))

    def predict(self, X, classification_threshold=None):
        y_pred = X

        # první vrstva + hidden layers
        for layer in self.layers[:-1]:
            y_pred = np.array([neuron.predict(y_pred) for neuron in layer])
            y_pred = y_pred.T  # transpozice pro další vrstvu (nic konkrétního, pouze aby shapes byly kompatibilní)

        # posledni vrstva
        y_pred = self.layers[-1].predict(y_pred)
        if classification_threshold is not None:
            return np.where(y_pred > classification_threshold, 1, 0)
        else:
            return y_pred

    def train(self, X, y,
              epochs: int,
              learning_rate: float,
              early_stop=True,
              verbose: bool = False,
              loss_function=lambda y_true, y_pred: np.mean((y_true - y_pred) ** 2),  # mse
              early_stop_loss=1e-6):
        for epoch in range(epochs):

            epoch_loss = 0

            # vzorky po jednom (SGD) bez mini-batchů
            for i in range(0, len(X)):
                # seznam aktivací
                # [[float]], v posledním listu bude vždy jeden prvek (skalár)
                pre_activations = []
                activations = []

                X_current = X[i]
                y_current = y[i]

                # první vrstva
                last_preactivation = np.array(
                    [np.dot(X_current, neuron.weights) + neuron.bias for neuron in self.layers[0]])
                pre_activations.append(last_preactivation)
                last_activations = np.array([neuron.predict(X_current) for neuron in self.layers[0]])
                activations.append(last_activations)

                # skryté vrstvy ([1:-1] - bez vstupní a výstupní)
                for layer in self.layers[1:-1]:
                    last_preactivation = np.array(
                        [np.dot(last_activations, neuron.weights) + neuron.bias for neuron in layer])
                    pre_activations.append(last_preactivation)
                    last_activations = np.array([neuron.predict(last_activations) for neuron in layer])
                    activations.append(last_activations)

                # výstupní vrstva - JEDEN perceptron, čili v listu bude jako poslední hodnota skalár
                output_neuron = self.layers[-1]
                last_preactivation = np.array(
                    [np.dot(last_activations, output_neuron.weights) + output_neuron.bias])
                pre_activations.append(last_preactivation)
                last_activations = np.array(self.layers[-1].predict(last_activations)).reshape(1, )
                activations.append(last_activations)

                # přidání loss hodnoty jednoho samplu do celku
                epoch_loss += loss_function(y_current, last_activations[0])

                #
                # zpětná propagace
                #

                # pro poslední neuron spočítáme jeho gradient pomocí = rozdíl * derivace aktivační funkce
                error = y_current - last_activations[0]
                delta = error * output_neuron.activator_derivative(pre_activations[-1][0])

                # aktualizace vah a biasu v posledním neuronu
                output_neuron.weights += learning_rate * delta * activations[-2]
                output_neuron.bias += learning_rate * delta

                # Uložíme delta z výstupní vrstvy pro propagaci zpět
                next_delta = np.array([delta])

                # od předposlední vrstvy - len(self.layers) - 2
                # po první vrstvu - 0
                for l in range(len(self.layers) - 2, -1, -1):
                    layer_current = self.layers[l]
                    pre_activation_current = pre_activations[l]
                    new_delta = []

                    # pokud l == 0, čili první vrstva, vstupem do vrstvy je X_current
                    # jinak předchozí aktivace
                    if l == 0:
                        a_prev = X_current
                    else:
                        a_prev = activations[l - 1]


                    if l == len(self.layers) - 2:
                        # předposlední vrstva
                        # váhy výstupního neuronu
                        next_weights = output_neuron.weights

                        # pro každý neuron v předposlední vrstvě
                        for j, neuron in enumerate(layer_current):
                            # vytáhneme váhu a deltu z posledního perceptronu v síti
                            next_w_times_delta = next_weights[j] * delta

                            # vynásobíme next_w_times_delta s derivací aktivační funkce neuronu v této vrstvě
                            delta_j = neuron.activator_derivative(pre_activation_current[j]) * next_w_times_delta
                            # aktualizace vah a biasů
                            neuron.weights += learning_rate * delta_j * a_prev
                            neuron.bias += learning_rate * delta_j

                            # nová delta pro příští vrstvu
                            new_delta.append(delta_j)
                    else:
                        # není předposlední vrstva
                        next_layer = self.layers[l + 1]

                        # pro každý neuron v hidden layer
                        for j, neuron in enumerate(layer_current):
                            # SUMA váh a delt z té (ne-výstpní) vrstvy vpravo o jednu
                            next_w_times_delta = 0
                            for k, next_neuron in enumerate(next_layer):
                                next_w_times_delta += next_neuron.weights[j] * next_delta[k]

                            # vynásobíme next_w_times_delta s derivací aktivační funkce neuronu v této vrstvě
                            delta_j = neuron.activator_derivative(pre_activation_current[j]) * next_w_times_delta
                            neuron.weights += learning_rate * delta_j * a_prev
                            neuron.bias += learning_rate * delta_j

                            # nová delta pro příští vrstvu
                            new_delta.append(delta_j)

                    # seznam nových delta pro další vrstvu
                    next_delta = np.array(new_delta)

            avg_loss = epoch_loss / len(X)
            if verbose:
                print(f"Epocha {epoch + 1}/{epochs}\tPrůměrná ztráta: {avg_loss}")

            # jednoduchý early stopping
            if early_stop and avg_loss < early_stop_loss:
                if verbose:
                    print(f"Early stopping: loss klesla pod hodnotu {early_stop_loss}.")
                break
