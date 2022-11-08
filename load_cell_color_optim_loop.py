from color_cell_optimization import load_babel, multiple_color_cells
import numpy as np
from solcore.light_source import LightSource
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

col_thresh = 0.004 # for a wavelength interval of 0.1, minimum achievable color error will be ~ 0.001. Fractional error in X, Y, Z.
acceptable_eff_change = 1e-4 # how much can the efficiency (in %) change between iteration sets?
n_trials = 8 # number of islands which will run concurrently
interval = 0.1 # wavelength interval (in nm)
wl_cell = np.arange(300, 4000, interval) # wavelengths

single_J_result = pd.read_csv("data/paper_colors.csv")

initial_iters = 100 # number of initial evolutions for the archipelago
add_iters = 100 # additional evolutions added each time if color threshold/convergence condition not met
# every color will run a minimum of initial_iters + add_iters evolutions before ending!

max_trials_col = 3*add_iters # how many population evolutions happen before giving up if there are no populations
                            # which meet the color threshold

type = "sharp" # "sharp" for rectangular dips or "gauss" for gaussians
fixed_height = False # fixed height peaks (will be at the value of max_height) or not

max_height = 1 # maximum height of reflection peaks
base = 0 # baseline fixed reflection

n_junc_loop = [1, 2, 3]

n_peak_loop = [2]
# also run for 1 junc/1 peak but no more junctions.

color_names, color_XYZ = load_babel() # 24 default Babel colors
# color_names = color_names[:5]
# color_XYZ = color_XYZ[:5]

photon_flux_cell = np.array(LightSource(
    source_type="standard", version="AM1.5g", x=wl_cell, output_units="photon_flux_per_nm"
).spectrum(wl_cell))

photon_flux_color = photon_flux_cell[:, np.all((photon_flux_cell[0] >= 380, photon_flux_cell[0] <= 780), axis=0)]

shapes = ['+', 'o', '^', '.', '*', "v", "s", "x"]

loop_n = 0

cols = sns.color_palette("husl", n_colors=len(n_junc_loop))

fig, (ax1, ax2) = plt.subplots(2, figsize=(8,7))

ax1.plot(color_names, single_J_result['eta'], 'k.', mfc='none')

ax2.plot(color_names, single_J_result['Eg'], 'k.', mfc='none')
ax2.axhline(y=1240/780, alpha=0.5, linestyle='--', color='k')
ax2.axhline(y=1240/600, alpha=0.5, linestyle='--', color='k')

for i1, n_peaks in enumerate(n_peak_loop):
    for j1, n_junctions in enumerate(n_junc_loop):
        for k1, fixed_height in enumerate([True, False]):

            champion_effs = np.loadtxt("results/champion_eff_tcheb_" + type + str(n_peaks) + '_' + str(
                n_junctions) + '_' + str(fixed_height) + str(max_height) + '_' + str(base)+'.txt')
            champion_pops = np.loadtxt("results/champion_pop_tcheb_" + type + str(n_peaks) + '_' + str(
                n_junctions) + '_' + str(fixed_height) + str(max_height) + '_' + str(base) +'.txt')
            # final_populations = np.load("results/final_pop_tcheb_" + type + str(n_peaks) + '_' + str(
            #     n_junctions) + '_' + str(fixed_height) + str(max_height) + '_' + str(base)+'.npy', allow_pickle=True)

            if j1 == 0:
                ax1.plot(color_names, champion_effs, mfc='none', linestyle='none',
                         color=cols[j1], marker=shapes[(k1+1)*(i1+1)-1],
                         label=fixed_height)

            else:
                ax1.plot(color_names, champion_effs, mfc='none', linestyle='none',
                         color=cols[j1], marker=shapes[(k1+1)*(i1+1)-1])

            ax1.set_xticklabels(color_names, rotation=45, ha='right', rotation_mode='anchor')
            # plt.legend(title="Fixed h:")
            ax1.set_ylabel("Efficiency (%)")

            if j1 == 0:
                ax2.plot(color_names, champion_pops[:, -n_junctions:], mfc='none', linestyle='none',
                         color=cols[j1], marker=shapes[(k1+1)*(i1+1)-1],
                         label=fixed_height)

            else:
                ax2.plot(color_names, champion_pops[:, -n_junctions:], mfc='none', linestyle='none',
                         color=cols[j1], marker=shapes[(k1+1)*(i1+1)-1])


            ax2.set_xticklabels(color_names, rotation=45, ha='right', rotation_mode='anchor')
            # plt.legend(title="Fixed h:")
            ax2.set_ylabel("Bandgap (eV)")


plt.tight_layout()
plt.show()


