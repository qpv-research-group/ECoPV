# colours of interest:

# Green, BlueFlower, YellowGreen, Neutral 3-5, Neutral 5, Neutral 6-5
inds = [13, 4, 10, 22, 21, 20]

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

cols = sns.color_palette('husl', 7)

force_rerun = False

col_thresh = 0.004  # for a wavelength interval of 0.1, minimum achievable color error will be (very rough estimate!) ~ 0.001.
# This is the maximum allowed fractional error in X, Y, or Z colour coordinates.

acceptable_eff_change = 1e-4  # how much can the efficiency (in %) change between iteration sets? Stop when have reached
# col_thresh and efficiency change is less than this.

n_trials = 8  # number of islands which will run concurrently
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
j01_method = "numerical_R"

max_height = 1
# maximum height of reflection peaks; fixed at this value of if fixed_height = True

base = 0
# baseline fixed reflection (fixed at this value for both fixed_height = True and False).

n_junc_loop = [5, 6]  # loop through these numbers of junctions
n_peak_loop = [2, 3]  # loop through these numbers of reflection peaks

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

shapes = ["+", "o", "^", ".", "*", "v", "s", "x"]

loop_n = 0

n_tests = 10

# precalculate optimal bandgaps for junctions:
save_path = path.join(path.dirname(path.abspath(__file__)), "results")

if __name__ == "__main__":

    fig, ax2 = plt.subplots(1, 1)

    for j1, n_junctions in enumerate(n_junc_loop):

        fig, axs = plt.subplots(3, 2, figsize=(10, 10))
        axs = axs.flatten()

        for i1, n_peaks in enumerate(n_peak_loop):

            for i2, j01_method in enumerate(['', 'numerical_R']):

                eff_array = np.zeros((n_tests, len(color_names)))
                pop_array = np.zeros((n_tests, len(color_names), 2*n_peaks + n_junctions))

                for k1 in range(n_tests):
                    # print(f"Trial {k1}")
                    champion_bandgaps = np.zeros((len(color_names), n_junctions))

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
                        + str(n_junctions) + f"conv_{k1}j0fix.txt"
                    )

                    if not path.exists(save_loc) or force_rerun:

                        print(
                            n_peaks, "peaks,", n_junctions, "junctions,"
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
                        champion_bandgaps = champion_pops[:, -n_junctions:]

                        final_populations = result["archipelagos"]

                        np.savetxt(
                            "results/champion_eff_"
                            + R_type
                            + str(n_peaks)
                            + "_"  + str(j01_method)
                            + str(n_junctions) + f"conv_{k1}j0fix.txt",
                            champion_effs,
                        )
                        np.savetxt(
                            "results/champion_pop_"
                            + R_type
                            + str(n_peaks)
                            + "_"  + str(j01_method)
                            + str(n_junctions) + f"conv_{k1}j0fix.txt",
                            champion_pops,
                        )

                    else:

                        champion_effs = np.loadtxt(
                            "results/champion_eff_"
                            + R_type
                            + str(n_peaks)
                            + "_"  + str(j01_method)
                            + str(n_junctions) + f"conv_{k1}j0fix.txt",
                        )
                        champion_pops = np.loadtxt(
                            "results/champion_pop_"
                            + R_type
                            + str(n_peaks)
                            + "_"  + str(j01_method)
                            + str(n_junctions) + f"conv_{k1}j0fix.txt",
                        )

                        champion_bandgaps = champion_pops[:, -n_junctions:]

                    eff_array[k1, :] = champion_effs
                    pop_array[k1, :, :] = champion_pops

                # for l1 in range(len(color_names)):
                #     axs[l1].hist(eff_array[:, l1], alpha=0.5, color=cols[i1], histtype='step',
                #                  label='{} peaks, {}'.format(n_peaks, j01_method),
                #                  linestyle=linestyles[i2], linewidth=2)
                #     axs[l1].set_title(color_names[l1])

                for l1 in range(len(color_names)):
                    max_eff_ind = np.argmax(eff_array[:, l1])
                    R_spec = gen_spectrum_ndip(pop_array[max_eff_ind, l1, :], n_peaks,
                                               wl_cell)
                    axs[l1].plot(wl_cell, R_spec, color=cols[i1],
                                 label='{} peaks, {}'.format(n_peaks, j01_method),
                                 linestyle=linestyles[i2], linewidth=1)
                    axs[l1].set_title(color_names[l1])
                    axs[l1].set_xlim([420, 600])

        plt.legend()

        plt.title(f"{n_junctions} junctions")

                    # if n_peaks == 2:
                    #
                    #     two_peak_ref = champion_effs
                    #     two_peak_pop = champion_pops
                    #
                    # else:
                    #     print(n_junctions)
                    #     eff_diff = (
                    #             100
                    #             * (champion_effs - two_peak_ref)
                    #             / two_peak_ref
                    #     )  # this is NEGATIVE if two_peak if higher, POSITIVE if two_peak is lower
                    #
                    #     ax2.plot(
                    #         color_names,
                    #         eff_diff,
                    #         mfc="none",
                    #         linestyle="none",
                    #         color=cols[k1],
                    #         marker=shapes[j1],
                    #         markersize=4,
                    #         label="{} peaks, {} junctions".format(n_peaks, n_junctions),
                    #     )

                    # ax2.plot(color_names, champion_effs, mfc="none", linestyle="none", color=cols[j1], marker=shapes[i1])

                        # for i2 in range(len(eff_diff)):
                        #     if eff_diff[i2] > 0.4:
                        #         two_peak_spec = gen_spectrum_ndip(two_peak_pop[i2], 2, wl_cell,
                        #                                           )
                        #         n_peak_spec = gen_spectrum_ndip(champion_pops[i2], n_peaks, wl_cell)
                        #
                        #         plt.figure()
                        #         plt.plot(wl_cell, two_peak_spec, '--k')
                        #         plt.plot(wl_cell, n_peak_spec, '-r')
                        #         plt.plot(wl_cell, photon_flux_cell[1] / np.max(photon_flux_cell), 'y', alpha=0.5)
                        #
                        #         for i3 in range(n_junctions):
                        #             plt.axvline(1240 / two_peak_pop[i2][-i3], color='k', alpha=0.6, linestyle='--')
                        #             plt.axvline(1240 / champion_pops[i2][-i3], color='r', alpha=0.6)
                        #
                        #         plt.title(str(color_names[i2]) + str(n_junctions) + str(n_peaks))
                        #
                        #         plt.xlim(300, 900)
                        #         plt.show()

        # plt.legend()
        plt.show()