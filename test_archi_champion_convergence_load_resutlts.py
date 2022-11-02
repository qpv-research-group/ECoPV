from colour_optimisation import *
from Spectrum_Function import delta_E_CIE2000, convert_xyY_to_XYZ, convert_xyY_to_Lab, convert_XYZ_to_Lab
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

junc_loop = [1,2,3,4]
n_peak_loop = [2,3,4]


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
])[:18]

# white_only = color_names == "White-9-5"
#
# color_names = color_names[white_only]

single_J_result = pd.read_csv("paper_colours.csv")

color_xyY = np.array(single_J_result[['x', 'y', 'Y']])[:18]

single_J_result = np.array(single_J_result['eta'])[:18]

color_XYZ = np.array([convert_xyY_to_XYZ(x) for x in color_xyY])

color_Lab = np.array([convert_xyY_to_Lab(x) for x in color_xyY])

colors = sns.color_palette("husl", n_colors=len(n_peak_loop))

ntest_load = [1,2,3,4,5]

for n_junctions in junc_loop:

    plt.figure()
    for i1, n_peaks in enumerate(n_peak_loop):

        champion_effs = np.empty((5, len(color_XYZ)))
        champion_pops = np.empty((5, len(color_XYZ), 2*n_peaks + n_junctions))
        iters_needed = np.empty((5, len(color_XYZ)))

        for ntest in ntest_load:
            champion_effs[ntest-1] = np.load("results/champion_eff_tcheb_" + str(n_peaks) + '_' + str(n_junctions) + '_' + str(ntest) + '.npy')
            champion_pops[ntest-1] = np.load("results/champion_pop_tcheb_" + str(n_peaks) + '_' + str(n_junctions) + '_' + str(ntest) + '.npy')
            iters_needed[ntest-1] = np.load("results/niters_tcheb_" + str(n_peaks) + '_' + str(n_junctions) + '_' + str(ntest) + '.npy')

        mean_std_eff = np.array([np.mean(champion_effs, 0), np.std(champion_effs, 0)]).T
        print(n_junctions, n_peaks, np.max(mean_std_eff[:,1]))
        plt.plot(color_names, mean_std_eff[:,0], 'o', color=colors[i1], mfc='none', label=str(n_peaks))
        plt.errorbar(color_names, mean_std_eff[:,0], yerr=mean_std_eff[:,1], marker='none', linestyle='none', ecolor=colors[i1])
        plt.xticks(rotation=45)
        plt.title(str(n_junctions) + ' junctions')

    plt.legend()
    plt.show()

for n_junctions in junc_loop:

    plt.figure()
    for i1, n_peaks in enumerate(n_peak_loop):

        champion_effs = np.empty((len(ntest_load), len(color_XYZ)))
        champion_pops = np.empty((len(ntest_load), len(color_XYZ), 2 * n_peaks + n_junctions))
        iters_needed = np.empty((len(ntest_load), len(color_XYZ)))

        for ntest in ntest_load:
            champion_effs[ntest - 1] = np.load(
                "results/champion_eff_tcheb_" + str(n_peaks) + '_' + str(n_junctions) + '_' + str(ntest) + '.npy')
            champion_pops[ntest - 1] = np.load(
                "results/champion_pop_tcheb_" + str(n_peaks) + '_' + str(n_junctions) + '_' + str(ntest) + '.npy')
            iters_needed[ntest - 1] = np.load(
                "results/niters_tcheb_" + str(n_peaks) + '_' + str(n_junctions) + '_' + str(ntest) + '.npy')

        for k1 in np.arange(1, n_junctions+1):
            plt.plot(color_names, champion_pops[:,:,-k1].T, 'o', color=colors[i1], mfc='none')

    plt.xticks(rotation=45)
    plt.title(str(n_junctions) + ' junctions')
    plt.legend()
    plt.show()


for n_junctions in junc_loop:

    plt.figure()
    for i1, n_peaks in enumerate(n_peak_loop):

        champion_effs = np.empty((len(ntest_load), len(color_XYZ)))
        champion_pops = np.empty((len(ntest_load), len(color_XYZ), 2 * n_peaks + n_junctions))
        iters_needed = np.empty((len(ntest_load), len(color_XYZ)))

        for ntest in ntest_load:
            champion_effs[ntest - 1] = np.load(
                "results/champion_eff_tcheb_" + str(n_peaks) + '_' + str(n_junctions) + '_' + str(ntest) + '.npy')
            champion_pops[ntest - 1] = np.load(
                "results/champion_pop_tcheb_" + str(n_peaks) + '_' + str(n_junctions) + '_' + str(ntest) + '.npy')
            iters_needed[ntest - 1] = np.load(
                "results/niters_tcheb_" + str(n_peaks) + '_' + str(n_junctions) + '_' + str(ntest) + '.npy')

        plt.plot(color_names, np.mean(iters_needed,0), 'o', color=colors[i1], mfc='none', label=str(n_peaks))

    plt.xticks(rotation=45)
    plt.title(str(n_junctions) + ' junctions')
    plt.legend()
    plt.show()


time_taken = np.loadtxt("time_taken.csv", delimiter=",", skiprows=1)

n_peaks = 2
n_junctions = 3

# compare with changing population size:
champion_effs = np.empty((len(ntest_load), len(color_XYZ)))
champion_pops = np.empty((len(ntest_load), len(color_XYZ), 2 * n_peaks + n_junctions))
iters_needed = np.empty((len(ntest_load), len(color_XYZ)))

for ntest in ntest_load:
    champion_effs[ntest - 1] = np.load(
        "results/champion_eff_tcheb_" + str(n_peaks) + '_' + str(n_junctions) + '_' + str(ntest) + '.npy')
    champion_pops[ntest - 1] = np.load(
        "results/champion_pop_tcheb_" + str(n_peaks) + '_' + str(n_junctions) + '_' + str(ntest) + '.npy')
    iters_needed[ntest - 1] = np.load(
        "results/niters_tcheb_" + str(n_peaks) + '_' + str(n_junctions) + '_' + str(ntest) + '.npy')


champion_effs_adapt = np.empty((len(ntest_load), len(color_XYZ)))
champion_pops_adapt = np.empty((len(ntest_load), len(color_XYZ), 2 * n_peaks + n_junctions))
iters_needed_adapt = np.empty((len(ntest_load), len(color_XYZ)))

for ntest in ntest_load:
    champion_effs_adapt[ntest - 1] = np.load(
        "results/champion_eff_tcheb_adaptpopsize" + str(n_peaks) + '_' + str(n_junctions) + '_' + str(ntest) + '.npy')
    champion_pops_adapt[ntest - 1] = np.load(
        "results/champion_pop_tcheb_adaptpopsize" + str(n_peaks) + '_' + str(n_junctions) + '_' + str(ntest) + '.npy')
    iters_needed_adapt[ntest - 1] = np.load(
        "results/niters_tcheb_adaptpopsize" + str(n_peaks) + '_' + str(n_junctions) + '_' + str(ntest) + '.npy')

plt.figure()
plt.plot(color_names, np.mean(champion_effs, 0), 'ok')
plt.plot(color_names, np.mean(champion_effs_adapt, 0), 'or')
plt.show()

plt.figure()
plt.plot(color_names, np.std(champion_effs, 0), 'ok')
plt.plot(color_names, np.std(champion_effs_adapt, 0), 'or')
plt.show()