import numpy as np
from colour_optimisation import getPmax
import pygmo as pg
from solcore.light_source import LightSource
from time import time

class black_cell_optim:
    def __init__(self, n_juncs, photon_flux, power_in=1000, eta_ext=1):

        self.n_juncs = n_juncs

        self.cell_wl = photon_flux[0]
        self.solar_flux = photon_flux[1]
        self.solar_spec = self.solar_flux[np.all([self.cell_wl >= 380, self.cell_wl <= 780], axis=0)]
        self.incident_power = power_in
        self.interval = np.round(np.diff(self.cell_wl)[0], 6)
        self.dim = n_juncs
        self.eta_ext = eta_ext


    def fitness(self, x):
        Egs = -np.sort(-x)  # [0]

        eta = getPmax(Egs, self.solar_flux, self.cell_wl, self.interval, self.eta_ext) / self.incident_power

        return [-eta]

    def get_bounds(self):

        j1 = [0.7, 1.4]
        j2 = [0.9, 1.8]
        j3 = [1.1, 2]
        j4 = [1.3, 2.2]
        j5 = [1.5, 2.4]

        lims = [j1, j2, j3, j4, j5]

        lower_bounds = []
        upper_bounds = []

        for k1 in range(self.dim):
            if k1 < 5:
                lower_bounds.append(lims[k1][0])
                upper_bounds.append(lims[k1][1])

            else:
                lower_bounds.append(1.5)
                upper_bounds.append(3)

        return (lower_bounds, upper_bounds)

    def get_name(self):
        return "Two-dip colour generation function"

    def get_extra_info(self):
        return "\tDimensions: " + str(self.dim)

    def get_nobj(self):
        return 1


interval = 0.01
wl_cell = np.arange(300, 4000, interval)

photon_flux = np.array(LightSource(
    source_type="standard", version="AM1.5g", x=wl_cell, output_units="photon_flux_per_nm"
).spectrum(wl_cell))

for n_juncs in [1,2,3,4,5,6]:
    start = time()
    print(n_juncs, "Junctions")

    p_init = black_cell_optim(n_juncs, photon_flux, eta_ext=0.1)

    prob = pg.problem(p_init)
    algo = pg.algorithm(pg.de(gen=1000, F=1, CR=1, ))

    pop = pg.population(prob,20*n_juncs)
    pop = algo.evolve(pop)
    print(-pop.champion_f*100)
    print(np.sort(pop.champion_x))
    print(time() - start)

# import matplotlib.pyplot as plt
# Egs = np.linspace(0.9, 1.4)
# Pmax = np.empty_like(Egs)
# for i1, Eg in enumerate(Egs):
#     Pmax[i1] = getPmax([Eg], photon_flux[1], photon_flux[0], interval, 1)
# plt.figure()
# plt.plot(Egs, Pmax)
# plt.show()