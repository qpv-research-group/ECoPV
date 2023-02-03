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

for n_juncs in [1, 2, 3]:
    start = time()
    print(n_juncs, "Junctions")

    p_init = cell_optimization(
        n_juncs, photon_flux, power_in=light_source.power_density, eta_ext=1
    )

    prob = pg.problem(p_init)
    algo = pg.algorithm(
        pg.de(
            gen=1000,
            F=1,
            CR=1,
        )
    )

    pop = pg.population(prob, 20 * n_juncs)
    pop = algo.evolve(pop)
    print("Optimized efficiency: ", -pop.champion_f[0] * 100)
    print("Optimized bandgaps: ", np.sort(pop.champion_x))
    print("Time taken (s): ", time() - start)

# import matplotlib.pyplot as plt
# Egs = np.linspace(0.9, 1.4)
# Pmax = np.empty_like(Egs)
# for i1, Eg in enumerate(Egs):
#     Pmax[i1] = getPmax([Eg], photon_flux[1], photon_flux[0], interval, 1)
# plt.figure()
# plt.plot(Egs, Pmax)
# plt.show()
