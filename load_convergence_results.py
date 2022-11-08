from Spectrum_Function import delta_E_CIE2000, convert_xyY_to_XYZ, convert_xyY_to_Lab, convert_XYZ_to_Lab
import numpy as np
from solcore.light_source import LightSource
import matplotlib.pyplot as plt
import pandas as pd
from colour_optimisation import *
import seaborn as sns

n_junctions = [1, 2]
n_i = [100, 200, 300, 400, 500]
pop_sizes = [50, 75, 100, 150]

pal = sns.color_palette("husl", n_colors=len(n_i))

n_trials = 10
n_peaks = 2
col_thresh = 0.002  # for a wavelength interval of 0.1, minimum achievable error will be ~ 0.001
pop_size = 80
n_iters = 300

interval = 0.1  # interval between each two wavelength points, 0.02 needed for low dE values

wl_cell = np.arange(300, 4000, interval)

photon_flux_cell = np.array(LightSource(
    source_type="standard", version="AM1.5g", x=wl_cell, output_units="photon_flux_per_nm"
).spectrum(wl_cell))

photon_flux_colour = photon_flux_cell[:, np.all((photon_flux_cell[0] >= 380, photon_flux_cell[0] <= 780), axis=0)]

color_names = [
    "DarkSkin", "LightSkin", "BlueSky", "Foliage", "BlueFlower", "BluishGreen",
    "Orange", "PurplishBlue", "ModerateRed", "Purple", "YellowGreen", "OrangeYellow",
    "Blue", "Green", "Red", "Yellow", "Magenta", "Cyan", "White-9-5", "Neutral-8",
    "Neutral-6-5", "Neutral-5", "Neutral-3-5", "Black-2"
]

single_J_result = pd.read_csv("paper_colors.csv")
black_result = single_J_result.loc[single_J_result['Colour'] == 'black 2']
black_result.loc[23, 'Colour'] = 'black'

single_J_result = pd.concat([black_result, single_J_result])

color_xyY = np.array(single_J_result[['x', 'y', 'Y']])

color_XYZ = np.array([convert_xyY_to_XYZ(x) for x in color_xyY])
color_YXZ = np.insert(color_XYZ, 0, [0, 0, 0], axis=0)
color_names.insert(0, 'Black')

color_Lab = np.array([convert_xyY_to_Lab(x) for x in color_xyY])



for n_gaps in n_junctions:

    fig, axs = plt.subplots(6, 4, figsize=(10, 20))
    axs = axs.flatten()

    for k1, pop_size in enumerate(pop_sizes):
        for l1, n_iters in enumerate(n_i):

            # XYZ_final = np.empty((len(color_names), n_trials, 3))
            # Lab_final = np.empty((len(color_names), n_trials, 3))
            # dE = np.empty((len(color_names), n_trials))
            #

            save_name = 'results/compare_results_' + str(n_gaps) + '_' + str(n_trials) + '_' + str(n_peaks) + '_' + \
                        str(n_gaps) + '_' + str(pop_size) + '_' + str(n_iters) + '_' + str(col_thresh * 100) + '.npy'

            save_name_nevals = 'results/compare_results_' + str(n_gaps) + '_' + str(n_trials) + '_' + str(
                n_peaks) + '_' + \
                               str(n_gaps) + '_' + str(pop_size) + '_' + str(n_iters) + '_' + str(
                col_thresh * 100) + 'nevals.npy'

            compare_results = np.load(save_name)
            n_evals = np.load(save_name_nevals)

            means = np.mean(compare_results, axis=1)
            stds = np.std(compare_results, axis=1)
            lolims = np.min(compare_results, axis=1)
            uplims = np.max(compare_results, axis=1)
            print(n_iters, pop_size, np.unique(n_evals))

            for col_index in np.arange(1, 25, 1):

                if k1 == 1:
                    axs[col_index-1].plot(pop_size*n_iters, means[col_index, 1], 'o', color=pal[l1],
                             label=str(n_iters), markersize=5, mfc='none')
                    axs[col_index-1].errorbar(pop_size * n_iters, means[col_index, 1],
                                 stds[col_index,1], color=pal[l1], capsize=10)
                else:
                    axs[col_index-1].plot(pop_size*n_iters, means[col_index, 1], 'o', color=pal[l1],
                                 markersize=5, mfc='none')
                    axs[col_index-1].errorbar(pop_size * n_iters, means[col_index, 1],
                                 stds[col_index,1], color=pal[l1], capsize=10)


                # plt.xlabel("Total evaluations")
                axs[col_index-1].set_title(color_names[col_index])
                # axs[col_index-1].set_ylim(means[col_index,0] + 0.02, means[col_index,0] - 0.02)
    axs[col_index - 1].legend(title="Iterations")
    plt.tight_layout()
    plt.show()