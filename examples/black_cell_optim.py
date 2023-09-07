import numpy as np
from ecopv.main_optimization import cell_optimization

import pygmo as pg
from solcore.light_source import LightSource
from time import time


interval = 0.01
wl_cell = np.arange(300, 4000, interval)

light_source = LightSource(
    source_type="standard",
    version="AM1.5g",
    x=wl_cell,
    output_units="photon_flux_per_nm",
)

photon_flux = np.array(light_source.spectrum(wl_cell))

close_guesses = [
    [1.337],
                 [0.96159811, 1.63271755],
                 [0.93351, 1.36649659, 1.89924796],
                 [0.71523332, 1.11437635, 1.49345409, 2.00180808],
                 [0.70973005, 1.0107807, 1.33084337, 1.66754078, 2.14036662],
                 [0.75716868, 1.02538943, 1.31171125, 1.5692605, 1.87432925, 2.3132805],
                 ]

n_junctions = [1, 2, 3, 4, 5, 6]

for i1, n_juncs in enumerate(n_junctions):
    start = time()
    print(n_juncs, "Junctions")

    p_init = cell_optimization(
        n_juncs, photon_flux, power_in=light_source.power_density, eta_ext=1
    )

    def get_bounds():
        Eg_bounds_lower = [0.9*Eg for Eg in close_guesses[i1]]
        Eg_bounds_upper = [1.1*Eg for Eg in close_guesses[i1]]

        return [Eg_bounds_lower, Eg_bounds_upper]

    p_init.get_bounds = get_bounds

    prob = pg.problem(p_init)
    algo = pg.algorithm(
        pg.de(
            # gen=1000*n_juncs,
            gen=500,
            F=1,
            CR=1,
            xtol=0,
            ftol=0,
        )
    )

    pop = pg.population(prob, 50 * n_juncs)
    pop = algo.evolve(pop)

    print("Optimized efficiency: ", -pop.champion_f[0] * 100)
    print("Optimized bandgaps: ", np.sort(pop.champion_x))
    print("Time taken (s): ", time() - start)

