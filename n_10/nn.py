import os.path

import keras
import numpy as np
from keras.src.layers import LSTM, BatchNormalization, Dropout

import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Activation

from n_10.bifurcation import generate_bifurcation_data


def main():
    # počet rekurentních iterací
    xn_iterations = 200
    # kolik zahodíme x_n hodnot před záznamem
    # pro zajímavost dáme neuronce hodnoty i z prvních iterací rekurence
    n_transients = 0

    # konstantní rozsah hodnoty 'a'
    a_values_count = 1000
    a_values = np.linspace(0.0, 4.0, a_values_count)

    a_values, x_values = generate_bifurcation_data(a_values, xn_iterations, n_transients)
    X = np.array(a_values).reshape(-1, 1)
    y = np.array(x_values).reshape(-1, 1)

    print(f"Generated data shapes before shuffling: X={X.shape}, y={y.shape}")
    if X.shape[0] == 0:
        print("Error: No data generated!")
        return

    # hodnoty 'a' budou tvořit náš vstup neuronce, ona pak bude předpovídat hodnoty 'x' (kterých generujeme n=<0; xn_iterations>
    # vytvoříme si indexový rozsah, který následně zamícháme
    indices = np.arange(X.shape[0])
    # aby síť dostala data ze vše rozsahů hodnot 'a' data, pak je musíme zamíchat
    np.random.seed(10)
    np.random.shuffle(indices)

    # rozdělíme data na trénovací, validační a testovací sady
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    train_len = int(0.6 * len(X_shuffled))
    val_len = int(0.2 * len(X_shuffled))

    X_train, y_train = X_shuffled[:train_len], y_shuffled[:train_len]
    X_val, y_val = X_shuffled[train_len: train_len + val_len], y_shuffled[train_len: train_len + val_len]
    X_test, y_test = X_shuffled[train_len + val_len:], y_shuffled[train_len + val_len:]

    checkpoint_filepath = "bifurcation_model.keras"
    if not os.path.exists(checkpoint_filepath):
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_filepath,
            save_best_only=True,
            monitor="val_loss",
            mode="min")

        input_layer = Input(shape=(1,))
        x = Dense(64)(input_layer)
        # x = BatchNormalization()(x)
        x = Activation('relu')(x)
        # x = Dropout(0.2)(x)
        x = Dense(32)(x)
        # x = BatchNormalization()(x)
        x = Activation('relu')(x)
        # x = Dropout(0.2)(x)
        output_layer = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer='adam', loss='mae', metrics=['mse'])
        model.summary()

        model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_val, y_val),
                  callbacks=[checkpoint], verbose=1)
        model.load_weights(checkpoint_filepath)
    else:
        model = keras.saving.load_model(checkpoint_filepath)

    y_pred = model.predict(X_test)
    y_pred = y_pred.reshape(-1)

    # Plot the predicted values
    plt.plot(X_test, y_test, ',k', markersize=1, label='Test data')
    plt.plot(X_test, y_pred, 'o', alpha=0.2, markersize=1)

    plt.xlabel("Parametr 'a'")
    plt.ylabel("Parametr 'x'")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.plot()
    plt.show()



if __name__ == "__main__":
    main()