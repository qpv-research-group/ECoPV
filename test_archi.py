from colour_optimisation import *
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
n_iters = 400

max_iters = 5*n_iters

acceptable_sd = 0.05

n_trials = 4

interval = 0.1  # interval between each two wavelength points, 0.02 needed for low dE values

class single_colour_archi:

    def __init__(self, plot_pareto=False, fix_height=True):
        self.plot_pareto = plot_pareto
        self.fix_height = fix_height
        pass

    def run(self, target, photon_flux, n_peaks=2, n_gaps=1, popsize=80, gen=1000, n_trials=10, archi=None):

        p_init = n_dip_colour_function_mobj(n_peaks, n_gaps, target, photon_flux, 1000, self.fix_height, 1)
        udp = pg.problem(p_init)
        algo = pg.algorithm(pg.moead(gen=gen, CR=1, F=1,
                                     preserve_diversity=True,
                                     decomposition="bi"))#, preserve_diversity=True, decomposition="bi"))

        if archi is None: archi = pg.archipelago(n=n_trials, algo=algo, prob=udp, pop_size=popsize)

        archi.evolve()

        # print(archi)

        archi.wait()

        # all_fs = np.stack([archi[j1].get_population().get_f() for j1 in range(n_trials)])
        # all_xs = np.stack([archi[j1].get_population().get_x() for j1 in range(n_trials)])

        return archi # all_xs, all_fs


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

color_xyY = np.array(single_J_result[['x', 'y', 'Y']])

color_XYZ = np.array([convert_xyY_to_XYZ(x) for x in color_xyY])

color_Lab = np.array([convert_xyY_to_Lab(x) for x in color_xyY])

internal_run = single_colour_archi(plot_pareto=False)

colors = sns.color_palette("rocket", n_colors=n_trials)

mean_sd_effs = np.empty((len(color_XYZ), 4))

all_converged = False

color_indices = np.arange(len(color_XYZ))

champion_eff = np.zeros(len(color_XYZ))
champion_pop = np.empty((len(color_XYZ), 2*n_peaks+n_junctions))

archipelagos = [None]*len(color_XYZ)

iters_needed = np.zeros(len(color_XYZ))

n_fronts = np.zeros((len(color_XYZ), n_trials))

if __name__ == "__main__":

    while not all_converged:

        start = time()

        for k1 in color_indices:

            iters_needed[k1] += n_iters

            archi = internal_run.run(color_XYZ[k1], photon_flux_cell,
                                              n_peaks, n_junctions, pop_size,
                                              n_iters, n_trials=n_trials, archi=archipelagos[k1])

            archipelagos[k1] = archi

            all_fs = np.stack([archi[j1].get_population().get_f() for j1 in range(n_trials)])
            all_xs = np.stack([archi[j1].get_population().get_x() for j1 in range(n_trials)])

            sln = all_fs[:, :, 0] < col_thresh

            acc_fs = all_fs[sln]

            best_acc_ind = np.array([np.argmin(x[sln[i1],1]) if len(x[sln[i1]]) > 0 else 0 for i1, x in enumerate(all_fs)])
            best_acc_eff = np.array([-np.min(x[sln[i1],1]) if len(x[sln[i1]]) > 0 else 0 for i1, x in enumerate(all_fs)])
            best_acc_pop = np.array([x[sln[i1]][best_acc_ind[i1]] if len(x[sln[i1]]) > 0 else 0 for i1, x in enumerate(all_xs)])
            # ragged array warning

            # print(best_acc_pop)
            all_acc_eff = -all_fs[sln, 1]*100

            # plt.scatter([color_names[k1]]*n_trials, best_acc_eff * 100, color=colors, facecolors='none')

            max_eff_acc = best_acc_eff[best_acc_eff > 0]*100

            print(color_names[k1], np.max(max_eff_acc), np.mean(max_eff_acc), np.std(max_eff_acc))

            ch_eff = np.max(max_eff_acc)
            ch_eff_ind = np.argmax(max_eff_acc)

            ch_pop = best_acc_pop[ch_eff_ind]

            mean_sd_effs[k1] = [np.min(max_eff_acc), ch_eff, np.mean(max_eff_acc), np.std(max_eff_acc)]

            if np.max(max_eff_acc) >= champion_eff[k1]:
                champion_eff[k1] = np.max(max_eff_acc)
                champion_pop[k1] = ch_pop
                print("New (or same) champion efficiency")

            else:
                for i1 in range(n_trials):
                    current_pop = archipelagos[i1][0].get_population()
                    # print(current_pop_0)
                    current_pop.set_x(0, champion_pop[k1])
                    # print(current_pop_0)
                    archipelagos[i1][0].set_population(current_pop)
                    # print(archipelagos[k1][0].get_population())
                mean_sd_effs[k1,-1] = 2*acceptable_sd
                print("Putting champion x back in population of all islands")

            if iters_needed[k1] > max_iters:
                # print(color_names[k1], "- restarting with new populations")
                # archipelagos[k1] = None # restart with new populations
                print("Stopping because maximum iterations reached")
                all_converged = True

        time_taken = time() - start

        color_indices = np.where(mean_sd_effs[:,-1] > acceptable_sd)[0]
        print(len(color_indices), "colour(s) are still above acceptable std. dev. threshold. Took ", time_taken, "s")

        if len(color_indices) == 0:
            print("All colours are converged")
            all_converged = True

        else:
             # n_iters = n_iters + 200
            print("Running for another", n_iters, "iterations")


    plt.figure(figsize=(7.5, 3))
    plt.plot(color_names, mean_sd_effs[:,0], 'ok', mfc='none', label='Converged mean (sd <' + str(acceptable_sd) +')')
    plt.plot(color_names, champion_eff, 'ob', mfc='none', label='Champion efficiency')
    plt.plot(color_names, single_J_result['eta'], 'or', mfc='none', label='1J eff')

    plt.xticks(rotation=45)
    plt.legend()
    plt.ylabel("Efficiency (%)")
    plt.title("Pop:" + str(pop_size) + "Iters:" + str(n_iters) + "Time:" + str(time_taken))
    plt.tight_layout()
    plt.show()


    plt.figure()

    for k1 in range(len(color_XYZ)):

        archi = archipelagos[k1]

        all_fs = np.stack([archi[j1].get_population().get_f() for j1 in range(n_trials)])
        all_xs = np.stack([archi[j1].get_population().get_x() for j1 in range(n_trials)])

        n_fronts[k1,:] = [len(pg.fast_non_dominated_sorting(archi[j1].get_population().get_f())[0]) for j1 in range(n_trials)]

        sln = all_fs[:, :, 0] < col_thresh

        acc_fs = all_fs[sln]

        best_acc_ind = np.array([np.argmin(x[sln[i1], 1]) if len(x[sln[i1]]) > 0 else 0 for i1, x in enumerate(all_fs)])

        acc_pops = [x[sln[i1]] for i1, x in enumerate(all_xs)]

        best_pop = np.array([x[best_acc_ind[i1]] for i1, x in enumerate(acc_pops)])
        best_pop = np.array([reorder_peaks(x, n_peaks) for x in best_pop])

        # print(np.max(best_pop[:, n_peaks:2*n_peaks], 0))

        plt.plot([color_XYZ[k1,1]]*2, np.max(best_pop[:, n_peaks:2*n_peaks], 0), 'o')
        plt.plot(color_XYZ[k1,1], 200*color_XYZ[k1,1], 'ok')
        # plt.figure()
        # for pop in best_pop:
        #     plt.plot(photon_flux_colour[0], gen_spectrum_ndip(pop[:n_peaks], pop[n_peaks:2*n_peaks], photon_flux_colour[0]))
        #
        # plt.title(color_names[k1])
        # plt.show()

        # plt.figure()
        #
        # for j1, fs in enumerate(all_fs):
        #
        #     plt.plot(*fs.T, 'o', mfc='none', label=str(n_fronts[k1, j1]))
        #
        # plt.legend()
        # plt.title(color_names[k1])
        # plt.show()

    plt.show()


    # two_peaks = np.loadtxt("2_peak_1_junction.txt")
    # three_peaks = np.loadtxt("3_peak_1_junction.txt")
    #
    # plt.figure()
    # plt.plot(two_peaks[:,0], 'o', label="Two peaks")
    # plt.plot(three_peaks[:,0], 'o')
    # plt.legend()
    # plt.show()
    #
    # plt.figure()
    # plt.plot(two_peaks[:,0]-three_peaks[:,0], 'o', label="2 peaks - 3 peaks")
    # plt.legend()
    # plt.show()



