from ecopv.main_optimization import (
    load_colorchecker,
    multiple_color_cells,
    cell_optimization,
)
import numpy as np
from solcore.light_source import LightSource
import matplotlib.pyplot as plt
import pygmo as pg
from os import path
import pandas as pd


col_thresh = 0.004  # for a wavelength interval of 0.1, minimum achievable color error will be (very rough estimate!) ~ 0.001.
# This is the maximum allowed fractional error in X, Y, or Z colour coordinates.

acceptable_eff_change = 1e-4  # how much can the efficiency (in %) change between iteration sets? Stop when have reached
# col_thresh and efficiency change is less than this.

n_trials = 10  # number of islands which will run concurrently
interval = 0.1  # wavelength interval (in nm)
wl_cell = np.arange(
    300, 4000, interval
)  # wavelengths used for cell calculations (range of wavelengths in AM1.5G solar
# spectrum. For calculations relating to colour perception, only the visible range (380-780 nm) will be used.

single_J_result = pd.read_csv("../ecopv/data/paper_colors.csv")

initial_iters = 100  # number of initial evolutions for the archipelago
add_iters = 100  # additional evolutions added each time if color threshold/convergence condition not met
# every color will run a minimum of initial_iters + add_iters evolutions before ending!

max_trials_col = 3 * add_iters
# how many population evolutions happen before giving up if there are no populations
# which meet the color threshold

type = "sharp"  # "sharp" for rectangular dips or "gauss" for gaussians
fixed_height = True  # fixed height peaks (will be at the value of max_height) if True, or peak height is an optimization
# variable if False
light_source_name = "AM1.5g"
j01_methods = ["no_R", "perfect_R"]

max_height = 1
# maximum height of reflection peaks; fixed at this value of if fixed_height = True

base = 0
# baseline fixed reflection (fixed at this value for both fixed_height = True and False).

n_junc_loop = [1, 2, 3, 4, 5, 6]  # loop through these numbers of junctions
n_peak_loop = [2]  # loop through these numbers of reflection peaks

color_names, color_XYZ = load_colorchecker()
# load the names and XYZ coordinates of the 24 default Babel colors
start_ind = 0
end_ind = len(color_names)
color_names = color_names[start_ind:end_ind]
color_XYZ = color_XYZ[start_ind:end_ind]

# Use AM1.5G spectrum:
light_source = LightSource(
    source_type="standard",
    version=light_source_name,
    x=wl_cell,
    output_units="photon_flux_per_nm",
)

photon_flux_cell = np.array(light_source.spectrum(wl_cell))

photon_flux_color = photon_flux_cell[
    :, np.all((photon_flux_cell[0] >= 380, photon_flux_cell[0] <= 780), axis=0)
]

shapes = ["+", "o", "^", ".", "*", "v", "s", "x"]

loop_n = 0

# precalculate optimal bandgaps for junctions:
save_path = path.join(path.dirname(path.abspath(__file__)), "results")

# Need this __main__ construction because otherwise the parallel running of the different islands (n_trials) may throw an error

for n_peaks in n_peak_loop:
    for n_junctions in n_junc_loop:
        plt.figure()
        champion_effs = np.zeros((len(j01_methods), len(color_names)))
        for l1, j01_method in enumerate(j01_methods):

            save_loc = (
                "results/champion_eff_"
                + type
                + str(n_peaks)
                + "_"
                + str(n_junctions)
                + "_"
                + str(fixed_height)
                + str(max_height)
                + "_"
                + str(base)
                + "_"
                + j01_method + ".txt"
            )


            champion_effs[l1] = np.loadtxt(
                "results/champion_eff_"
                + type
                + str(n_peaks)
                + "_"
                + str(n_junctions)
                + "_"
                + str(fixed_height)
                + str(max_height)
                + "_"
                + str(base)
                + "_"  + j01_method + ".txt",
            )

            champion_pops = np.loadtxt(
                "results/champion_pop_"
                + type
                + str(n_peaks)
                + "_"
                + str(n_junctions)
                + "_"
                + str(fixed_height)
                + str(max_height)
                + "_"
                + str(base)
                + "_" + j01_method + ".txt",
            )
            champion_bandgaps = champion_pops[:, -n_junctions:]

        plt.plot(
            color_names,
            100*(champion_effs[1]-champion_effs[0])/champion_effs[1],
            marker=shapes[l1],
            mfc="none",
            linestyle="none",
        )

        plt.xticks(rotation=45)
        plt.legend()
        plt.ylabel("Efficiency (%)")
        plt.title("Peaks:" + str(n_peaks) + "Junctions:" + str(n_junctions))
        plt.tight_layout()

        plt.show()

