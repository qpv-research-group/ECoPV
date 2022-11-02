from colour_optimisation import *

class single_colour_archi:

    def __init__(self, plot_pareto=None, fix_height=True):
        self.plot_pareto = plot_pareto
        self.fix_height = fix_height
        pass

    def run(self, target, photon_flux, n_peaks=2, n_gaps=1, popsize=80, gen=1000, n_trials=10, archi=None):

        p_init = n_dip_colour_function_mobj(n_peaks, n_gaps, target, photon_flux, 1000, self.fix_height, 1)
        udp = pg.problem(p_init)
        algo = pg.algorithm(pg.moead(gen=gen, CR=1, F=1,
                                     preserve_diversity=True,#))
                                     decomposition="bi"))#, preserve_diversity=True, decomposition="bi"))

        if archi is None: archi = pg.archipelago(n=n_trials, algo=algo, prob=udp, pop_size=popsize)

        archi.evolve()

        # print(archi)

        archi.wait()

        # all_fs = np.stack([archi[j1].get_population().get_f() for j1 in range(n_trials)])
        # all_xs = np.stack([archi[j1].get_population().get_x() for j1 in range(n_trials)])

        if self.plot_pareto is not None:
            for j1 in range(n_trials):
                pg.plot_non_dominated_fronts(archi[j1].get_population().get_f(), axes=self.plot_pareto)

        return archi # all_xs, all_fs


from Spectrum_Function import delta_E_CIE2000, convert_xyY_to_XYZ, convert_xyY_to_Lab, convert_XYZ_to_Lab
import numpy as np
from solcore.light_source import LightSource
import matplotlib.pyplot as plt
import pandas as pd
from time import time
from colour_optimisation import *
import seaborn as sns

n_peaks = 2
n_junctions = 1
col_thresh = 0.004 # for a wavelength interval of 0.1, minimum achievable error will be ~ 0.001
pop_size = 60
n_iters = 800

max_iters = 15*n_iters

acceptable_sd = 0.005

n_trials = 1

interval = 0.1  # interval between each two wavelength points, 0.02 needed for low dE values

wl_cell = np.arange(300, 4000, interval)

photon_flux_cell = np.array(LightSource(
    source_type="standard", version="AM1.5g", x=wl_cell, output_units="photon_flux_per_nm"
).spectrum(wl_cell))

photon_flux_colour = photon_flux_cell[:, np.all((photon_flux_cell[0] >= 380, photon_flux_cell[0] <= 780), axis=0)]

color_names = np.array([
    "DarkSkin", "LightSkin", "BlueSky", "Foliage", "BlueFlower", "BluishGreen",
    "Orange", "PurplishBlue", "ModerateRed", "Purple", "YellowGreen", "OrangeYellow",
    "Blue", "Green", "Red", "Yellow", "Magenta", "Cyan", "White-9-5", "Neutral-8",
    "Neutral-6-5", "Neutral-5", "Neutral-3-5", "Black-2"
])

purple_only = color_names == "Purple"

single_J_result = pd.read_csv("paper_colours.csv")

color_names = color_names[purple_only]
color_xyY = np.array(single_J_result[['x', 'y', 'Y']])[purple_only]

color_XYZ = np.array([convert_xyY_to_XYZ(x) for x in color_xyY])

color_Lab = np.array([convert_xyY_to_Lab(x) for x in color_xyY])

fig, ax = plt.subplots()
internal_run = single_colour_archi()

for j1 in range(10):
    archi = internal_run.run(color_XYZ[0], photon_flux_cell,
                             n_peaks, n_junctions, pop_size,
                             n_iters, n_trials=n_trials, archi=None)

    obj_vals = archi[0].get_population().get_f()
    acc_vals = obj_vals[obj_vals[:,0] < col_thresh, :]

    n_fronts = len(pg.fast_non_dominated_sorting(obj_vals))
    max_eff = -np.min(acc_vals[:,1])

    ax.plot(*obj_vals.T, 'o', mfc='none', label=str(n_fronts) + '   ' + str(np.round(max_eff, 6)))

    ax.axvline(x=col_thresh)

# plt.xlim(0, 3*col_thresh)
# plt.ylim(-0.329, -0.32)
plt.legend()
plt.show()
