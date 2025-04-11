from n_07.affine import Affine3DTransform
from n_07.ifs import Ifs


def main():
    iterations = 100000
    seed = 10

    # první příklad
    at = [
        Affine3DTransform.from_list([0.00, 0.00, 0.01, 0.00, 0.26, 0.00, 0.00, 0.00, 0.05, 0.00, 0.00, 0.00]),
        Affine3DTransform.from_list([0.20, -0.26, -0.01, 0.23, 0.22, -0.07, 0.07, 0.00, 0.24, 0.00, 0.80, 0.00]),
        Affine3DTransform.from_list([-0.25, 0.28, 0.01, 0.26, 0.24, -0.07, 0.07, 0.00, 0.24, 0.00, 0.22, 0.00]),
        Affine3DTransform.from_list([0.85, 0.04, -0.01, -0.04, 0.85, 0.09, 0.00, 0.08, 0.84, 0.00, 0.80, 0.00]),
    ]
    a = Ifs(at, iterations, seed)
    a.generate()
    a.plot(title="První příklad")

    # druhý příklad
    bt = [
        Affine3DTransform.from_list([0.05, 0.00, 0.00, 0.00, 0.60, 0.00, 0.00, 0.00, 0.05, 0.00, 0.00, 0.00]),
        Affine3DTransform.from_list([0.45, -0.22, 0.22, 0.22, 0.45, 0.22, -0.22, 0.22, -0.45, 0.00, 1.00, 0.00]),
        Affine3DTransform.from_list([- 0.45, 0.22, - 0.22, 0.22, 0.45, 0.22, 0.22, - 0.22, 0.45, 0.00, 1.25, 0.00]),
        Affine3DTransform.from_list([0.49, - 0.08, 0.08, 0.08, 0.49, 0.08, 0.08, - 0.08, 0.49, 0.00, 2.00, 0.00]),
    ]
    b = Ifs(bt, iterations, seed)
    b.generate()
    b.plot(title="Druhý příklad")

if __name__ == '__main__':
    main()
