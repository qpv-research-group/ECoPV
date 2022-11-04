from colour_optimisation import *
from Spectrum_Function import delta_E_CIE2000, convert_xyY_to_XYZ, convert_xyY_to_Lab, \
    convert_XYZ_to_Lab, gen_spectrum_twogauss, gen_spectrum_ndip
import numpy as np
from solcore.light_source import LightSource
import matplotlib.pyplot as plt
import pandas as pd
from time import time
from colour_optimisation import *
import seaborn as sns

n_peaks = 3
n_junctions = 1
col_thresh = 0.004 # for a wavelength interval of 0.1, minimum achievable error will be ~ 0.001
pop_size = 50
initial_iters = 100
add_iters = 100

acceptable_eff_change = 1e-4

n_trials = 10

n_params = 2*n_peaks + n_junctions

interval = 0.1  # interval between each two wavelength points, 0.02 needed for low dE values

junc_loop = [3,4]

n_peak_loop = [2]

class single_colour_archi:

    def __init__(self, plot_pareto=False, fix_height=True):
        self.plot_pareto = plot_pareto
        self.fix_height = fix_height
        pass

    def run(self, target, photon_flux, n_peaks=2, n_gaps=1, popsize=80, gen=1000, n_trials=10, w_bounds=None, archi=None):

        p_init = n_gauss_colour_function_mobj(n_peaks, n_gaps, target, photon_flux, 1000, self.fix_height, 1, w_bounds)

        udp = pg.problem(p_init)
        algo = pg.algorithm(pg.moead(gen=gen, CR=1, F=1,
                                     preserve_diversity=True))
                                     # decomposition="bi"))#, preserve_diversity=True, decomposition="bi"))

        if archi is None: archi = pg.archipelago(n=n_trials, algo=algo, prob=udp, pop_size=popsize)

        archi.evolve()

        archi.wait()

        # all_fs = np.stack([archi[j1].get_population().get_f() for j1 in range(n_trials)])
        # all_xs = np.stack([archi[j1].get_population().get_x() for j1 in range(n_trials)])

        return archi # all_xs, all_fs


wl_cell = np.arange(300, 4000, interval)

photon_flux_cell = np.array(LightSource(
    source_type="standard", version="AM1.5g", x=wl_cell, output_units="photon_flux_per_nm"
).spectrum(wl_cell))

photon_flux_colour = photon_flux_cell[:, np.all((photon_flux_cell[0] >= 380, photon_flux_cell[0] <= 780), axis=0)]

# plt.figure()
# plt.plot(photon_flux_colour[0], gen_spectrum_twogauss(np.array([500, 600]), np.array([15, 15]), photon_flux_colour[0]))
# plt.plot(photon_flux_colour[0], gen_spectrum_ndip(np.array([500, 600]), np.array([20, 20]), photon_flux_colour[0]))
# plt.show()

color_names = np.array([
    "DarkSkin", "LightSkin", "BlueSky", "Foliage", "BlueFlower", "BluishGreen",
    "Orange", "PurplishBlue", "ModerateRed", "Purple", "YellowGreen", "OrangeYellow",
    "Blue", "Green", "Red", "Yellow", "Magenta", "Cyan", "White-9-5", "Neutral-8",
    "Neutral-6-5", "Neutral-5", "Neutral-3-5", "Black-2"
])[:18]

# white_only = color_names == "White-9-5"
#
# color_names = color_names[white_only]

single_J_result = pd.read_csv("paper_colours.csv")

color_xyY = np.array(single_J_result[['x', 'y', 'Y']])[:18]

single_J_result = np.array(single_J_result['eta'])[:18]

color_XYZ = np.array([convert_xyY_to_XYZ(x) for x in color_xyY])

color_Lab = np.array([convert_xyY_to_Lab(x) for x in color_xyY])

internal_run = single_colour_archi(plot_pareto=False)

colors = sns.color_palette("rocket", n_colors=n_trials)



if __name__ == "__main__":

    for n_peaks in n_peak_loop:
        for n_junctions in junc_loop:
            n_params = 2 * n_peaks + n_junctions
            pop_size = n_params*10
            print(n_peaks, 'peaks,', n_junctions, 'junctions', 'Population size:', pop_size)

            for ntest in [1,2,3,4,5]:

                # width_bounds = [None]*len(color_XYZ)

                mean_sd_effs = np.empty((len(color_XYZ), 4))

                all_converged = False

                conv = np.array([False]*len(color_XYZ))

                color_indices = np.arange(len(color_XYZ))

                champion_eff = np.zeros(len(color_XYZ))
                champion_pop = np.empty((len(color_XYZ), n_params))

                archipelagos = [None]*len(color_XYZ)

                iters_needed = np.zeros(len(color_XYZ))

                n_fronts = np.zeros((len(color_XYZ), n_trials))

                current_iters = initial_iters

                start_time = time()

                while not all_converged:

                    start = time()
                    print("Add iters:", current_iters)

                    for k1 in color_indices:

                        iters_needed[k1] += current_iters

                        archi = internal_run.run(color_XYZ[k1], photon_flux_cell,
                                                          n_peaks, n_junctions, pop_size,
                                                          current_iters, n_trials=n_trials, archi=archipelagos[k1])#,
                                                 # w_bounds=width_bounds[k1])

                        archipelagos[k1] = archi

                        all_fs = np.stack([archi[j1].get_population().get_f() for j1 in range(n_trials)])
                        all_xs = np.stack([archi[j1].get_population().get_x() for j1 in range(n_trials)])

                        sln = all_fs[:, :, 0] < col_thresh

                        acc_fs = all_fs[sln]

                        best_acc_ind = np.array([np.argmin(x[sln[i1],1]) if len(x[sln[i1]]) > 0 else 0 for i1, x in enumerate(all_fs)])
                        best_acc_eff = np.array([-np.min(x[sln[i1],1]) if len(x[sln[i1]]) > 0 else 0 for i1, x in enumerate(all_fs)])
                        best_acc_pop = np.array([x[sln[i1]][best_acc_ind[i1]] if len(x[sln[i1]]) > 0 else [0]*n_params for i1, x in enumerate(all_xs)])
                        # ragged array warning

                        # print(best_acc_pop)
                        all_acc_eff = -all_fs[sln, 1]*100

                        # plt.scatter([color_names[k1]]*n_trials, best_acc_eff * 100, color=colors, facecolors='none')

                        max_eff_acc = best_acc_eff[best_acc_eff > 0]*100
                        best_acc_pop = best_acc_pop[best_acc_eff > 0]

                        if len(max_eff_acc) > 0:

                            print(color_names[k1], np.round(np.max(max_eff_acc), 3),
                                  np.round(np.mean(max_eff_acc),3), np.round(np.std(max_eff_acc), 6))

                            ch_eff = np.max(max_eff_acc)
                            ch_eff_ind = np.argmax(max_eff_acc)

                            ch_pop = best_acc_pop[ch_eff_ind]
                            # if not hasattr(ch_pop, "shape"):
                            #     print("weird population", ch_eff, ch_eff_ind, max_eff_acc, best_acc_pop)
                            mean_sd_effs[k1] = [np.min(max_eff_acc), ch_eff, np.mean(max_eff_acc), np.std(max_eff_acc)]

                            delta_eta = ch_eff - champion_eff[k1]

                            if delta_eta >= acceptable_eff_change:
                                champion_eff[k1] = ch_eff
                                champion_pop[k1] = ch_pop
                                print(np.round(delta_eta, 5), "delta eff - New champion efficiency")

                            else:
                                print("No change/worse champion efficiency")
                                conv[k1] = True


                        else:
                            print(color_names[k1], "no acceptable populations")

                    time_taken = time() - start

                    color_indices = np.where(~conv)[0]
                    print(len(color_indices), "colour(s) are still above acceptable std. dev. threshold. Took", time_taken, "s")

                    if len(color_indices) == 0:
                        print("All colours are converged")
                        all_converged = True

                    else:
                         # n_iters = n_iters + 200
                        print("Running for another", add_iters, "iterations")
                        current_iters = add_iters

                print("TOTAL TIME:", time()-start_time)

                plt.figure(figsize=(7.5, 3))
                # plt.plot(color_names, mean_sd_effs[:,0], 'ok', mfc='none', label='Converged mean')
                plt.plot(color_names, champion_eff, 'ob', mfc='none', label='Champion efficiency')
                plt.plot(color_names, single_J_result, 'or', mfc='none', label='1J eff')

                plt.xticks(rotation=45)
                plt.legend()
                plt.ylabel("Efficiency (%)")
                # plt.title("Pop:" + str(pop_size) + "Iters:" + str(n_iters) + "Time:" + str(time_taken))
                plt.tight_layout()
                plt.show()

                champion_pop = np.array([reorder_peaks(x, n_peaks) for x in champion_pop])
                np.save("results/champion_eff_tcheb_adaptpopsize_gauss" + str(n_peaks) + '_' + str(n_junctions) + '_' + str(ntest), champion_eff)
                np.save("results/champion_pop_tcheb_adaptpopsize_gauss" + str(n_peaks) + '_' + str(n_junctions) + '_' + str(ntest), champion_pop)
                np.save("results/niters_tcheb_adaptpopsize_gauss" + str(n_peaks) + '_' + str(n_junctions) + '_' + str(ntest), iters_needed)



            # # plt.figure()
            #
            # for k1 in range(len(color_XYZ)):
            #
            #     archi = archipelagos[k1]
            #
            #     all_fs = np.stack([archi[j1].get_population().get_f() for j1 in range(n_trials)])
            #     all_xs = np.stack([archi[j1].get_population().get_x() for j1 in range(n_trials)])
            #
            #     n_fronts[k1,:] = [len(pg.fast_non_dominated_sorting(archi[j1].get_population().get_f())[0]) for j1 in range(n_trials)]
            #
            #     sln = all_fs[:, :, 0] < col_thresh
            #
            #     acc_fs = all_fs[sln]
            #
            #     best_acc_ind = np.array([np.argmin(x[sln[i1], 1]) if len(x[sln[i1]]) > 0 else 0 for i1, x in enumerate(all_fs)])
            #
            #     acc_pops = [x[sln[i1]] for i1, x in enumerate(all_xs)]
            #
            #     best_pop = np.array([x[best_acc_ind[i1]] for i1, x in enumerate(acc_pops)])
            #     best_pop = np.array([reorder_peaks(x, n_peaks) for x in best_pop])
            #
            #     # print(np.max(best_pop[:, n_peaks:2*n_peaks], 0))
            #
            #     # plt.plot([color_XYZ[k1,1]]*2, np.max(best_pop[:, n_peaks:2*n_peaks], 0), 'o')
            #     # plt.plot(color_XYZ[k1,1], 200*color_XYZ[k1,1], 'ok')
            #     # plt.figure()
            #     # for pop in best_pop:
            #     #     plt.plot(photon_flux_colour[0], gen_spectrum_ndip(pop[:n_peaks], pop[n_peaks:2*n_peaks], photon_flux_colour[0]))
            #     #
            #     # plt.title(color_names[k1])
            #     # plt.show()
            #
            #     plt.figure()
            #
            #     for j1, fs in enumerate(all_fs):
            #
            #         plt.plot(*fs.T, 'o', mfc='none', label=str(n_fronts[k1, j1]))
            #
            #     plt.legend()
            #     plt.title(color_names[k1])
            #     plt.show()

            # plt.figure(figsize=(7.5, 3))
            # champion_effs = np.zeros((len(color_XYZ), 5))
            #
            # for ntest in [1,2,3, 4,5]:
            #
            #
            #     champion_eff = np.load("results/champion_eff_tcheb" + str(ntest) + ".npy")
            #     champion_pop = np.load("results/champion_pop_tcheb" + str(ntest) + ".npy")
            #
            #     champion_effs[:, ntest-1] = champion_eff
            #
            #     plt.plot(color_names, champion_pop[:,2], 'ob', mfc='none')
            #     # plt.plot(color_names, champion_pop[:,-1], 'ob', mfc='none', label='Champion efficiency')
            #     # plt.plot(color_XYZ[:,1], champion_pop[:, n_peaks], 'ob', mfc='none')
            #     # plt.plot(color_XYZ[:,1], champion_pop[:, n_peaks+1], 'ob', mfc='none')
            #
            #
            # # plt.plot(color_XYZ[:,1], 160*color_XYZ[:,1], 'or', mfc='none')
            # plt.xticks(rotation=45)
            # plt.legend()
            # plt.ylabel("Efficiency (%)")
            # plt.tight_layout()
            # plt.show()

