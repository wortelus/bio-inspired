from n_12.ff import ForestFire


def main():
    ff = ForestFire(
        # velikost strany pole
        grid_size=2,
        # hustota lesa (pravděpodobnost stromu)
        initial_density=0.5,
        # šance na vzkvétnutí nového stromu na prázdné/spálené půdě
        p_growth=0.05,
        # šance na samovznícení stromu
        p_fire_spontaneous=0.001,
        # 4/8 sousedů
        neighborhood_type='von_neumann',
    )
    ff.populate()
    ff.animate()


if __name__ == "__main__":
    main()
