from n_06.l_sys import LSystem


def main():
    axiom = "F+F"
    rules = {
        "F": "F+F-F"
    }
    test = LSystem(axiom, rules, 90, 1)
    test.plot(length=5, title="Test")
    print(f"První example: {test.current_string}")

    axiom = "F+F+F+F"
    rules = {
        "F": "F+F-F-FF+F+F-F"
    }
    a = LSystem(axiom, rules, 90, 3)
    a.plot(length=5, title="Example 1")

    axiom = "F++F++F"
    rules = {
        "F": "F+F--F+F"
    }
    b = LSystem(axiom, rules, 60, 3)
    b.plot(length=5, title="Example 2")

    axiom = "F"
    rules = {
        "F": "F[+F]F[-F]F"
    }
    b = LSystem(axiom, rules, 180 / 7, 3)
    b.plot(length=5, title="Example 3")

    axiom = "F"
    rules = {
        "F": "FF+[+F-F-F]-[-F+F+F]"
    }
    b = LSystem(axiom, rules, 180 / 8, 3)
    b.plot(length=5, title="Example 4")


if __name__ == "__main__":
    main()
