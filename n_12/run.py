import tkinter as tk

from n_12.ff import ForestFire


def main():
    root = tk.Tk()
    ff = ForestFire(
        # velikost strany pole
        grid_size=100,
        # hustota lesa (pravděpodobnost stromu)
        initial_density=0.5,
        # šance na vzkvétnutí nového stromu na prázdné/spálené půdě
        p_growth=0.05,
        # šance na samovznícení stromu
        p_fire_spontaneous=0.001,
        # 4/8 sousedů
        neighborhood_type='von_neumann',
        # Tkinter rodič
        tk_master=root
    )
    ff.populate()
    ff.animate(ms=10)
    root.mainloop()


if __name__ == "__main__":
    main()
