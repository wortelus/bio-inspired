from julia import Julia

def main():
    # Některé zajímavé hodnoty c pro Julia set (m=2):
    # c = -0.8 + 0.156j
    # c = 0.285 + 0.01j
    # c = -0.4 + 0.6j
    # c = -0.70176 - 0.3842j (připomíná mořského koníka)
    # c = -0.7 + 0.27015j

    c_value = -0.7 + 0.27015j
    julia_set = Julia(c_julia=c_value, max_iter=150, width=600, height=600)

    julia_set.plot_simple(zoom=1.0)
    julia_set.animate_zoom(start_zoom=1.0, end_zoom=100.0, steps=50)

if __name__ == "__main__":
    main()