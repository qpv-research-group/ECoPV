# colours of interest:

# Blue, Red, Magenta, YellowGreen, Neutral 6-5
inds = [12, 14, 16, 10, 20]

from ecopv.main_optimization import (
    load_colorchecker,
    multiple_color_cells,
    cell_optimization,
)
from ecopv.spectrum_functions import gen_spectrum_ndip
import numpy as np
from solcore.light_source import LightSource
import matplotlib.pyplot as plt
import pygmo as pg

from os import path
import pandas as pd
import seaborn as sns
from matplotlib import rc

rc("font", **{"family": "sans-serif",
              "sans-serif": ["Helvetica"]})

# sns.set_style("whitegrid")
cols = sns.color_palette('husl', 2)

force_rerun = False

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

R_type = "sharp"  # "sharp" for rectangular dips or "gauss" for gaussians
fixed_height = True  # fixed height peaks (will be at the value of max_height) if True, or peak height is an optimization
# variable if False
light_source_name = "AM1.5g"
j01_method = "perfect_R"

max_height = 1
# maximum height of reflection peaks; fixed at this value of if fixed_height = True

base = 0
# baseline fixed reflection (fixed at this value for both fixed_height = True and False).

n_junc_loop = [1, 2, 3, 4, 5, 6]  # loop through these numbers of junctions
n_peak_loop = [2]  # loop through these numbers of reflection peaks

linestyles = ['-', '--', '-.', ':']
color_names, color_XYZ = load_colorchecker(illuminant="AM1.5g", output_coords="XYZ")
# load the names and XYZ coordinates of the 24 default Babel colors

color_names = color_names[inds]
color_XYZ = color_XYZ[inds]

# Use AM1.5G spectrum for cell calculations:
light_source = LightSource(
    source_type="standard",
    version=light_source_name,
    x=wl_cell,
    output_units="photon_flux_per_nm",
)

photon_flux_cell = np.array(light_source.spectrum(wl_cell))

loop_n = 0

n_tests = 10

# precalculate optimal bandgaps for junctions:
save_path = path.join(path.dirname(path.abspath(__file__)), "results")

if __name__ == "__main__":

    eff_array = np.zeros((len(n_junc_loop), len(n_peak_loop), n_tests, len(color_names)))

    for j1, n_junctions in enumerate(n_junc_loop):
        # pop_array = np.zeros((len(n_peak_loop), n_tests, len(color_names), 2 * n_peaks + n_junctions))
        Eg_array = np.zeros((len(n_peak_loop), n_tests, len(color_names), n_junctions))

        for i1, n_peaks in enumerate(n_peak_loop):

            for k1 in range(n_tests):
                # print(f"Trial {k1}")

                Eg_guess = np.loadtxt(
                    save_path
                    + "/champion_pop_{}juncs_{}spec.txt".format(
                        n_junctions, light_source_name
                    ),
                    ndmin=1,
                )

                save_loc = (
                    "results/champion_eff_"
                    + R_type
                    + str(n_peaks)
                    + "_"  + str(j01_method)
                    + str(n_junctions) + f"conv_{k1}_ESI4.txt"
                )

                if not path.exists(save_loc) or force_rerun:

                    print(
                        n_peaks, "peaks,", n_junctions, "junction(s),"
                    )

                    result = multiple_color_cells(
                        color_XYZ,
                        color_names,
                        photon_flux_cell,
                        n_peaks=n_peaks,
                        n_junctions=n_junctions,
                        R_type=R_type,
                        fixed_height=fixed_height,
                        n_trials=n_trials,
                        initial_iters=initial_iters,
                        add_iters=add_iters,
                        col_thresh=col_thresh,
                        acceptable_eff_change=acceptable_eff_change,
                        max_trials_col=max_trials_col,
                        base=base,
                        max_height=max_height,
                        Eg_black=Eg_guess,
                        plot=False,
                        power_in=light_source.power_density,
                        return_archipelagos=True,
                        j01_method=j01_method,
                        illuminant=light_source_name,
                    )

                    champion_effs = result["champion_eff"]
                    champion_pops = result["champion_pop"]

                    # final_populations = result["archipelagos"]

                    np.savetxt(
                        "results/champion_eff_"
                        + R_type
                        + str(n_peaks)
                        + "_"  + str(j01_method)
                        + str(n_junctions) + f"conv_{k1}_ESI4.txt",
                        champion_effs,
                    )
                    np.savetxt(
                        "results/champion_pop_"
                        + R_type
                        + str(n_peaks)
                        + "_"  + str(j01_method)
                        + str(n_junctions) + f"conv_{k1}_ESI4.txt",
                        champion_pops,
                    )

                else:

                    champion_effs = np.loadtxt(
                        "results/champion_eff_"
                        + R_type
                        + str(n_peaks)
                        + "_"  + str(j01_method)
                        + str(n_junctions) + f"conv_{k1}_ESI4.txt",
                    )
                    champion_pops = np.loadtxt(
                        "results/champion_pop_"
                        + R_type
                        + str(n_peaks)
                        + "_"  + str(j01_method)
                        + str(n_junctions) + f"conv_{k1}_ESI4.txt",
                    )

                    champion_bandgaps = champion_pops[:, -n_junctions:]

                Eg_array[i1, k1] = champion_pops[:, -n_junctions:]

                eff_array[j1, i1, k1, :] = champion_effs
                # pop_array[i1, k1, :, :] = champion_pops

        Eg_min = np.min(Eg_array, axis=1)
        Eg_max = np.max(Eg_array, axis=1)

        Eg_range = Eg_max - Eg_min
        Eg_range_max = np.max(np.max(Eg_range, axis=1), axis=1)
        print(f"{n_junctions} junctions, max range: {Eg_range_max}")

    fig, axs = plt.subplots(len(n_junc_loop), len(color_names), figsize=(10, 1.5*len(n_junc_loop)))

    max_per_col_junc = np.max(eff_array, axis=2)
    min_per_col_junc = np.min(eff_array, axis=2)
    range_per_col_junc = max_per_col_junc - min_per_col_junc
    std_per_col_junc = np.std(eff_array, axis=2)
    mean_per_col_junc = np.mean(eff_array, axis=2)

    # range_per_junc = np.max(np.max(range_per_col_junc, axis=1), axis=1)
    range_per_junc = np.max(range_per_col_junc[:,0,:], axis=1)

    for j1, n_junctions in enumerate(n_junc_loop):
            # pop_array = np.zeros((len(n_peak_loop), n_tests, len(color_names), 2 * n_peaks + n_junctions))

        for l1 in range(len(color_names)):

            # bins = np.arange(np.min(eff_array[:, :, l1]), np.max(eff_array[:, :, l1]), 2*acceptable_eff_change)
            bins = np.linspace(np.max(max_per_col_junc[j1, 0, l1]) - 1.1*range_per_junc[j1],
                               np.max(max_per_col_junc[j1, 0, l1]) + 0.1* range_per_junc[j1],
                               15)

            print("range:", np.max(eff_array[j1, :, :, l1]) - np.min(eff_array[j1, :, :, l1]))
            axs[j1, l1].hist(eff_array[j1, 0, :, l1], alpha=0.3, color='k', linewidth=2,
                             bins=bins, label='2 peaks')
            # axs[j1, l1].hist(eff_array[j1, 1, :, l1], alpha=0.5, color=cols[1], histtype='step', linewidth=2,
            #                     bins=bins, label='3 peaks')
            if j1 == 0:
                axs[j1, l1].set_title(color_names[l1])

            if l1 == 0:
                axs[j1, l1].set_ylabel(f'{n_junctions} junction(s)' + '\n \n Counts')


            max_count = np.max([np.max(np.histogram(eff_array[j1, 0, :, l1], bins=bins)[0])])# for m1 in range(len(n_peak_loop))])
                # axs[j1, l1].text(np.max(max_per_col_junc[j1, 0, l1]) - 1.05*range_per_junc[j1],
                #                  0.85*max_count, f"{n_junctions} junction(s)", weight='bold')

            # if j1 == 0 and l1 == len(color_names) - 1:
            #     axs[j1, l1].legend()

            axs[j1, l1].text(np.max(max_per_col_junc[j1, 0, l1]) - 1.05*range_per_junc[j1], 0.85*max_count,
                             f"{mean_per_col_junc[j1, 0, l1]:.3f} ({std_per_col_junc[j1, 0, l1]:.4f})", weight='bold')

            if j1 == len(n_junc_loop) - 1:
                axs[j1, l1].set_xlabel('Efficiency (%)')

            axs[j1, l1].ticklabel_format(useOffset=False)

    plt.tight_layout()
    plt.show()