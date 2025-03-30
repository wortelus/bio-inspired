import numpy as np

from hopfield import Hopfield

# 2x2 priklad dle navy-task3.pdf
pattern = np.array([[1, 0],
                    [0, 1]])

# 3x3 ctverec
pattern_a = np.array([[1, 1, 1],
                      [1, 0, 1],
                      [1, 1, 1]])

# 4x4 osmicka
pattern_b = np.array([[0, 1, 1, 1, 0],
                      [0, 1, 0, 1, 0],
                      [0, 1, 1, 1, 0],
                      [0, 1, 0, 1, 0],
                      [0, 1, 1, 1, 0]])

# 5x5 hvezda
pattern_c = np.array([[1, 0, 1, 0, 1],
                      [0, 1, 1, 1, 0],
                      [1, 1, 0, 1, 1],
                      [0, 1, 1, 1, 0],
                      [1, 0, 1, 0, 1]])


def main():
    # Kontrolni zkouska dle navy-task3.pdf
    simple = Hopfield(pattern)
    # Bitflip dle matice 'V' navy-task3.pdf
    simple_in = np.array([[0, 1], [0, 1]]).flatten()
    simple_out = simple.asynchronous_recovery(simple_in)
    simple.plot(pattern, title="Simple (original) pattern")
    simple.plot(simple_in.reshape(2, 2), title="Simple (broken) input pattern")
    simple.plot(simple.weights, title="Simple weights")
    simple.plot(simple_out.reshape(2, 2), title="Simple recovered pattern")

    # Synchronní update, zde jdou vidět halucinace
    steps = 5
    simple_out_sync = simple.synchronous_recovery(simple_in, steps=steps)
    simple.plot(simple_out_sync.reshape(2, 2), title=f"Simple recovered pattern (synchronous, {steps} steps)")
    steps = 20
    simple_out_sync = simple.synchronous_recovery(simple_in, steps=steps)
    simple.plot(simple_out_sync.reshape(2, 2), title=f"Simple recovered pattern (synchronous, {steps} steps)")

    a = Hopfield(pattern_a)
    a_in = pattern_a.flatten()
    a.plot(pattern_a, title="Clean pattern A")
    # Bitflip v pixelu na pozici 4
    a_in[4] = 0 if a_in[4] == 1 else 1
    a_out = a.asynchronous_recovery(a_in)
    a.plot(a_in.reshape(3, 3), title="Broken pattern A")
    a.plot(a.weights, title="Weights A")
    a.plot(a_out.reshape(3, 3), title="Recovered broken pattern A")
    # Synchrónní update, zde to funguje
    steps = 1
    a_out_sync = a.synchronous_recovery(a_in, steps=steps)
    a.plot(a_out_sync.reshape(3, 3), title=f"Recovered Pattern A (synchronous, {steps} steps)")
    steps = 5
    a_out_sync = a.synchronous_recovery(a_in, steps=steps)
    a.plot(a_out_sync.reshape(3, 3), title=f"Recovered Pattern A (synchronous, {steps} steps)")

    b = Hopfield(pattern_b)
    b_in = pattern_b.flatten()
    # Náhodná změna v pixelu z 8 na 6
    b_in[8] = 0 if b_in[8] == 1 else 1

    b_out = b.asynchronous_recovery(b_in)
    b.plot(b_in.reshape(5, 5), title="B Broken osmička")
    b.plot(b.weights, title="Weights B")
    b.plot(b_out.reshape(5, 5), title="Recovered Pattern B")
    # Synchrónní update, zde to funguje
    steps = 1
    b_out_sync = b.synchronous_recovery(b_in, steps=steps)
    b.plot(b_out_sync.reshape(5, 5), title=f"Recovered Pattern B (synchronous, {steps} steps)")
    steps = 5
    b_out_sync = b.synchronous_recovery(b_in, steps=steps)
    b.plot(b_out_sync.reshape(5, 5), title=f"Recovered Pattern B (synchronous, {steps} steps)")

    c = Hopfield(pattern_c)
    c_in = pattern_c.flatten()
    c_out = c.asynchronous_recovery(c_in)
    c.plot(c_in.reshape(5, 5), title="Pattern C")
    c.plot(c.weights, title="Weights C")
    c.plot(c_out.reshape(5, 5), title="Recovered Pattern C")
    # Synchrónní update, zde to funguje
    steps = 1
    c_out_sync = c.synchronous_recovery(c_in, steps=steps)
    c.plot(c_out_sync.reshape(5, 5), title=f"Recovered Pattern C (synchronous, {steps} steps)")
    steps = 5
    c_out_sync = c.synchronous_recovery(c_in, steps=steps)
    c.plot(c_out_sync.reshape(5, 5), title=f"Recovered Pattern C (synchronous, {steps} steps)")


if __name__ == "__main__":
    main()
