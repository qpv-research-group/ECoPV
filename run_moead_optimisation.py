from Spectrum_Function import delta_E_CIE2000, convert_xyY_to_XYZ, convert_xyY_to_Lab, convert_XYZ_to_Lab
import numpy as np
from solcore.light_source import LightSource
import matplotlib.pyplot as plt
import pandas as pd
from joblib import Parallel, delayed
from time import time
from colour_optimisation import *

n_junctions = [1, 2, 3]
n_i = [100, 200, 300, 400, 500]
pop_sizes = [50, 75, 100, 150]

n_trials = 10
n_peaks = 2
col_thresh = 0.002 # for a wavelength interval of 0.1, minimum achievable error will be ~ 0.001
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

single_J_result = pd.read_csv("paper_colours.csv")
black_result = single_J_result.loc[single_J_result['Colour'] == 'black 2']
black_result.loc[23, 'Colour'] = 'black'

single_J_result = pd.concat([black_result, single_J_result])

color_xyY = np.array(single_J_result[['x', 'y', 'Y']])

color_XYZ = np.array([convert_xyY_to_XYZ(x) for x in color_xyY])
color_YXZ = np.insert(color_XYZ, 0, [0, 0, 0], axis=0)
color_names.insert(0, 'Black')

color_Lab = np.array([convert_xyY_to_Lab(x) for x in color_xyY])

for n_gaps in n_junctions:
    for pop_size in pop_sizes:
        for n_iters in n_i:

            compare_results = np.zeros((len(color_XYZ), n_trials, n_peaks*2 + n_gaps + 1))
            n_evals = np.zeros((len(color_XYZ), n_trials))

            start = time()

            XYZ_final = np.empty((len(color_names), n_trials, 3))
            Lab_final = np.empty((len(color_names), n_trials, 3))
            dE = np.empty((len(color_names), n_trials))

            for j1 in range(n_trials):

                print(n_gaps, pop_size, n_iters)

                internal_run = single_colour()

                eff_result_par = Parallel(n_jobs=-1)(delayed(internal_run.run)
                                                     (color_XYZ[i1], color_names[i1],
                                                      col_thresh, photon_flux_cell,
                                                      n_peaks, n_gaps,
                                                      80, 500) for i1 in range(len(color_XYZ)))


                loop_result = np.stack([item[0] for item in eff_result_par])
                loop_n_evals = np.stack([item[1] for item in eff_result_par])



                compare_results[:, j1] = loop_result
                n_evals[:, j1] = loop_n_evals
                best_pops = loop_result[:, 1:]

                XYZ_final[:, j1, :] = np.stack([XYZ_from_pop_dips(x, n_peaks, photon_flux_colour, interval) for x in best_pops])
                Lab_final[:, j1, :] = np.stack([convert_XYZ_to_Lab(x) for x in XYZ_final[:, j1, :]])

                dE[:, j1] = np.stack([delta_E_CIE2000(x, color_Lab[i1]) for i1, x in enumerate(Lab_final[:, j1, :])])

            time_taken = time()-start

            save_name = 'results/compare_results_' + str(n_gaps) + '_' + str(n_trials) + '_' + str(n_peaks) + '_' + \
                        str(n_gaps) + '_' + str(pop_size) + '_' + str(n_iters) + '_' + str(col_thresh*100) + '.npy'

            save_name_nevals = 'results/compare_results_' + str(n_gaps) + '_' + str(n_trials) + '_' + str(n_peaks) + '_' + \
                        str(n_gaps) + '_' + str(pop_size) + '_' + str(n_iters) + '_' + str(col_thresh*100) + 'nevals.npy'

            np.save(save_name, compare_results)

            np.save(save_name_nevals, n_evals)


            # dXYZ = (XYZ_final - color_XYZ[:, None, :])/color_XYZ[:, None, :]
            #
            # # Maximum dLab observed per colour:
            # dE_max = np.max(dE, 1)

            # plt.figure()
            # plt.plot(np.sum(np.abs(dXYZ), axis=2), dLab, 'o', mfc='none')
            # plt.show()
            #
            # plt.figure()
            # plt.plot(dXYZ[:,:,2], dLab, 'o', mfc='none')
            # plt.show()

            # plt.figure(figsize=(7.5,3))
            # plt.plot(color_names, compare_results[:,:,0]*100, 'o', mfc='none')
            # plt.plot(color_names, single_J_result['eta'], 'or', mfc='none', label= '1J eff')
            # # plt.ylim(750,)
            # plt.xticks(rotation=45)
            # plt.legend()
            # plt.ylabel("Efficiency (%)")
            # plt.title("Pop:" + str(pop_size) + "Iters:" + str(n_iters) + "Time:" + str(time_taken))
            # plt.tight_layout()
            # plt.show()

            # plt.figure(figsize=(8,3))
            # plt.plot(color_names, compare_results[:,:,1], 'o', mfc='none')
            # # plt.ylim(750,)
            # plt.xticks(rotation=45)
            # plt.tight_layout()
            # plt.show()


            # max_eff_index = np.argmax(compare_results[:,:, 0], 1)
            # Eg_max_eta = [compare_results[i1,max_eff_index[i1],2*n_peaks+1] for i1 in range(len(color_XYZ))]
            #
            # plt.figure(figsize=(7.5,3))
            # plt.plot(color_names, compare_results[:,:, 2*n_peaks+1], 'o', mfc='none')
            # # plt.plot(color_names, compare_results[:,:, 2*n_peaks+2], 'o', mfc='none')
            # # plt.plot(color_names, compare_results[:,:, 2*n_peaks+3], 'o', mfc='none')
            # # plt.plot(color_names, Eg_max_eta, 'ko', mfc='none')
            # # plt.ylim(750,)
            # plt.legend()
            # plt.xticks(rotation=45)
            # plt.ylabel("Bandgaps (eV)")
            # plt.tight_layout()
            # plt.show()
