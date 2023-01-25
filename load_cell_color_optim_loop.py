from color_cell_optimization import load_babel, make_spectrum_ndip, load_cmf, spec_to_XYZ
from colormath.color_objects import sRGBColor, XYZColor
from colormath.color_conversions import convert_color

import numpy as np
from solcore.light_source import LightSource
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.patches import Rectangle
from cycler import cycler


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

n_junc_loop = [6]

n_peak_loop = [2,3 ,4]
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

cols = sns.color_palette("Set2", n_colors=len(n_junc_loop))

black_cell_eff = np.array([33.8, 45.9, 51.8, 55.5, 57.8, 59.7])
black_cell_Eg = [[1.34], [0.96, 1.63], [0.93, 1.37, 1.90],
                 [0.72, 1.11, 1.49, 2.00], [0.70, 1.01, 1.33, 1.67, 2.14],
                 [0.70, 0.96, 1.20, 1.47, 1.79, 2.24]]

fig, (ax1, ax2) = plt.subplots(2, figsize=(8,7))

# ax1.plot(color_names, single_J_result['eta'], 'k.', mfc='none')

ax2.plot(color_names, single_J_result['Eg'], 'k.', mfc='none')
ax2.axhline(y=1240/780, alpha=0.5, linestyle='--', color='k') # limit of visible light
ax2.axhline(y=1240/600, alpha=0.5, linestyle='--', color='k') # peak of longest-wavelength cmf

alphas = [1, 0.5]

fixed_height_loop = [True]

for j1, n_junctions in enumerate(n_junc_loop):
    for i1, n_peaks in enumerate(n_peak_loop):
        for k1, fixed_height in enumerate(fixed_height_loop):

            champion_effs = np.loadtxt("results/champion_eff_tcheb_" + type + str(n_peaks) + '_' + str(
                n_junctions) + '_' + str(fixed_height) + str(max_height) + '_' + str(base)+'.txt')
            champion_pops = np.loadtxt("results/champion_pop_tcheb_" + type + str(n_peaks) + '_' + str(
                n_junctions) + '_' + str(fixed_height) + str(max_height) + '_' + str(base) +'.txt')
            # final_populations = np.load("results/final_pop_tcheb_" + type + str(n_peaks) + '_' + str(
            #     n_junctions) + '_' + str(fixed_height) + str(max_height) + '_' + str(base)+'.npy', allow_pickle=True)

            if j1 == 0:
                ax1.plot(color_names, champion_effs, mfc='none', linestyle='none',
                         color=cols[j1], marker=shapes[i1],
                         # label=fixed_height,
                         label=n_junctions,
                         alpha=alphas[k1])

            else:
                ax1.plot(color_names, champion_effs, mfc='none', linestyle='none',
                         color=cols[j1], marker=shapes[i1], alpha=alphas[k1],
                         label=n_junctions,
                         )

            ax1.plot('Black', black_cell_eff[n_junctions-1], 'x', mfc='none', linestyle='none',
                     color=cols[j1])

            ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
            # plt.legend(title="Fixed h:")
            ax1.set_ylabel("Efficiency (%)")

            if j1 == 0:
                ax2.plot(color_names, champion_pops[:, -n_junctions:], mfc='none', linestyle='none',
                         color=cols[j1], marker=shapes[i1],
                         # label=fixed_height,
                         alpha=alphas[k1])

            else:
                ax2.plot(color_names, champion_pops[:, -n_junctions:], mfc='none', linestyle='none',
                         color=cols[j1], marker=shapes[i1], alpha=alphas[k1])


            ax2.plot(['Black']*len(black_cell_Eg[n_junctions-1]), black_cell_Eg[n_junctions-1], 'x', mfc='none', linestyle='none',
                     color=cols[j1])
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
            # plt.legend(title="Fixed h:")
            ax2.set_ylabel("Bandgap (eV)")

# ax1.set_ylim(50, 55)
ax1.legend(bbox_to_anchor=(1.15, 0.8))
plt.tight_layout()
plt.show()




fixed_height = True
fig, ax = plt.subplots(1, figsize=(8,3.2))

for j1, n_junctions in enumerate(n_junc_loop):
    champion_effs = np.empty((len(n_peak_loop), len(color_XYZ)))

    for i1, n_peaks in enumerate(n_peak_loop):
        champion_effs[i1] = np.loadtxt("results/champion_eff_tcheb_" + type + str(n_peaks) + '_' + str(
            n_junctions) + '_' + str(fixed_height) + str(max_height) + '_' + str(base)+'.txt')

    for l1 in range(0, len(n_peak_loop)-1):
        print(l1)
        diff = champion_effs[l1+1, :] - champion_effs[0, :] # negative if more peaks don't help
        print(diff)
        ax.plot(color_names, diff, mfc='none', linestyle='none', marker=shapes[i1], color=cols[j1], label=str(n_junctions))

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
plt.legend()
plt.tight_layout()
plt.show()


# cmf = load_cmf(photon_flux_cell[0])
# interval = np.round(np.diff(photon_flux_cell[0])[0], 6)
#
#
# type = "sharp"
# max_height = 1
# base = 0
#
# n_junc_loop = [6]
#
# n_peak_loop = [2, 3]
# # also run for 1 junc/1 peak but no more junctions.
#
# alphas = [1, 0.5]
#
# for j1, n_junctions in enumerate(n_junc_loop):
#     for l1, target in enumerate(color_XYZ):
#
#         fig, ax = plt.subplots()
#
#
#         for i1, n_peaks in enumerate(n_peak_loop):
#             for k1, fixed_height in enumerate([True]):
#                 placeholder_obj = make_spectrum_ndip(n_peaks=n_peaks, type=type, fixed_height=fixed_height)
#
#                 champion_effs = np.loadtxt("results/champion_eff_tcheb_" + type + str(n_peaks) + '_' + str(
#                     n_junctions) + '_' + str(fixed_height) + str(max_height) + '_' + str(base) + '.txt')
#                 champion_pops = np.loadtxt("results/champion_pop_tcheb_" + type + str(n_peaks) + '_' + str(
#                     n_junctions) + '_' + str(fixed_height) + str(max_height) + '_' + str(base) + '.txt')
#                 # final_populations = np.load("results/final_pop_tcheb_" + type + str(n_peaks) + '_' + str(
#                 #     n_junctions) + '_' + str(fixed_height) + str(max_height) + '_' + str(base)+'.npy', allow_pickle=True)
#
#                 spec = placeholder_obj.spectrum_function(champion_pops[l1], n_peaks,
#                                                          photon_flux_cell[0], max_height, base)
#
#                 found_xyz = spec_to_XYZ(spec, photon_flux_cell[1], cmf, interval)
#                 color_xyz_f = XYZColor(*found_xyz)
#                 color_xyz_t = XYZColor(*target)
#                 color_srgb_f = convert_color(color_xyz_f, sRGBColor)
#                 color_srgb_t = convert_color(color_xyz_t, sRGBColor)
#
#                 ax.set_prop_cycle(color=['red', 'green', 'blue'])
#                 ax.fill_between(photon_flux_cell[0], 1, 1 - spec, color=cols[i1], alpha=0.3)
#                 ax.plot(photon_flux_cell[0], cmf / np.max(cmf))
#                 ax.plot(photon_flux_cell[0], photon_flux_cell[1] / np.max(photon_flux_cell[1]), '-k',
#                         alpha=0.5)
#
#                 # plt.xlim(300, 1000)
#                 for Eg in champion_pops[l1][-n_junctions:]:
#                     ax.axvline(x=1240 / Eg, color=cols[i1], linestyle='--')
#
#         plt.xlim(300, 1000)
#         ax.set_title(color_names[l1])
#         plt.tight_layout()
#         plt.show()
#
#


cmf = load_cmf(photon_flux_cell[0])
interval = np.round(np.diff(photon_flux_cell[0])[0], 6)

colors = ['k', 'b', 'r']

pal = ['r', 'g', 'b']
cols = cycler('color', pal)
params = {"axes.prop_cycle": cols}
plt.rcParams.update(params)

type = "sharp"
fixed_height_loop = [True, False]
max_height = 1
base = 0

patch_width = 0.75
n_junc_loop = [2]

n_peak_loop = [2]

data_width = 0.6

offset = np.linspace(0, data_width, 3)
# also run for 1 junc/1 peak but no more junctions.

alphas = [1, 0.5]
fig, axes = plt.subplots(2,2, gridspec_kw={'height_ratios':[1,2],
                                           'width_ratios':[3,1],
                                           'hspace': 0.1,
                                           'wspace': 0.05},
                         figsize=(8, 5))

offset_ind = 0

for j1, n_junctions in enumerate(n_junc_loop):


    for i1, n_peaks in enumerate(n_peak_loop):
        for k1, fixed_height in enumerate(fixed_height_loop):
            for l1, target in enumerate(color_XYZ):

                placeholder_obj = make_spectrum_ndip(n_peaks=n_peaks, type=type, fixed_height=fixed_height)

                champion_effs = np.loadtxt("results/champion_eff_" + type + str(n_peaks) + '_' + str(
                    n_junctions) + '_' + str(fixed_height) + str(max_height) + '_' + str(base) + '.txt')
                champion_pops = np.loadtxt("results/champion_pop_" + type + str(n_peaks) + '_' + str(
                    n_junctions) + '_' + str(fixed_height) + str(max_height) + '_' + str(base) + '.txt')
                # final_populations = np.load("results/final_pop_tcheb_" + type + str(n_peaks) + '_' + str(
                #     n_junctions) + '_' + str(fixed_height) + str(max_height) + '_' + str(base)+'.npy', allow_pickle=True)

                spec = placeholder_obj.spectrum_function(champion_pops[l1], n_peaks,
                                                         photon_flux_cell[0], max_height, base)

                centres = champion_pops[l1][:n_peaks]
                widths = champion_pops[l1][n_peaks:2 * n_peaks]

                found_xyz = spec_to_XYZ(spec, photon_flux_cell[1], cmf, interval)
                color_xyz_f = XYZColor(*found_xyz)
                color_xyz_t = XYZColor(*target)
                color_srgb_f = convert_color(color_xyz_f, sRGBColor)
                color_srgb_t = convert_color(color_xyz_t, sRGBColor).get_value_tuple()

                axes[0,0].plot(l1+offset[offset_ind]-data_width/2, champion_effs[l1], '.', color=colors[offset_ind],
                             markersize=4)
                axes[1,0].errorbar([l1+offset[offset_ind]-data_width/2]*len(centres), centres, yerr=widths/2,fmt='none',
                            ecolor=colors[offset_ind])

                axes[1,0].add_patch(
                    Rectangle(xy=(l1-patch_width/2, 370), width=patch_width,
                              height=20,
                              facecolor=color_srgb_t)
                )

            offset_ind += 1


axes[0,0].set_xticks(np.arange(0, len(color_XYZ)))
axes[0,0].set_xticklabels([])
axes[0,0].grid(axis='y', color='0.95')
axes[0,0].set_xlim(-0.6, len(color_XYZ)-0.4)
axes[1,0].set_xlim(-0.6, len(color_XYZ)-0.4)
axes[0,0].set_ylabel("Efficiency (%)")
axes[1,0].set_xticks(np.arange(0, len(color_XYZ)))
axes[1,0].set_xticklabels(color_names, rotation=90, ha='right', rotation_mode='anchor')

axes[1,1].plot(photon_flux_cell[1]/np.max(photon_flux_cell), wl_cell, 'k', alpha=0.5)
axes[1,1].plot(cmf, wl_cell)
axes[1,1].set_yticklabels([])
axes[1,0].set_ylim(360, 670)
axes[1,1].set_ylim(360, 670)
axes[1,0].grid(axis='y', color='0.95')
axes[1,1].grid(axis='y', color='0.95')

for l1 in range(len(color_XYZ)+1):
    axes[0, 0].axvline(x=l1 - 0.5, color='0.95', linewidth=0.5)
    axes[1, 0].axvline(x=l1-0.5, color='0.95', linewidth=0.5)

axes[0,1].axis('off')
axes[1,0].set_ylabel("Wavelength (nm)")
axes[1,1].set_xlabel(r"Spectral sensitivity / "
           "\n" 
           r"Normalised photon flux")
axes[0,1].plot(0,0, color=colors[0], label='2 peaks')
# axes[0,1].plot(0,0, color=colors[1], label='3 peaks')
# axes[0,1].plot(0,0, color=colors[2], label='4 peaks')
axes[0,1].legend(frameon=False, loc='center')
plt.subplots_adjust(bottom=0.25, top=0.95, left=0.1, right=0.97)
plt.tight_layout()
plt.show()