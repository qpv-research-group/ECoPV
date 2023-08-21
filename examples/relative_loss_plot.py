from ecopv.optimization_functions import getIVmax, getPmax, db_cell_calculation_perfectR
from ecopv.spectrum_functions import make_spectrum_ndip, gen_spectrum_ndip, load_cmf

import numpy as np
from solcore.light_source import LightSource
import seaborn as sns
import pandas as pd
from cycler import cycler
from solcore.constants import kb, q

k = kb / q
T = 298
kbT = k * T

from ecopv.plot_utilities import *

def make_sorted_xr(arr, color_names, append_black=None, ascending=False):
    if arr.ndim == 1:
        dims = ["color"]

    else:
        dims = ["color", "n"]


    eff_xr_col = xr.DataArray(data=arr[:18], dims=dims, coords={"color": Y_cols})

    eff_xr_col = eff_xr_col.sortby("color", ascending=ascending)
    eff_xr_col = eff_xr_col.assign_coords(color=col_names.data)

    if append_black is not None:
        eff_xr_bw = xr.DataArray(
            data=np.append(arr[18:], [append_black], axis=0),
            dims=dims,
            coords={"color": np.append(color_names[18:], "Black")},
        )

    else:
        eff_xr_bw = xr.DataArray(
            data=arr[18:], dims=dims, coords={"color": color_names[18:]}
        )

    if ascending:
        eff_xr_bw.data = eff_xr_bw.data[::-1]
        eff_xr_bw.coords["color"] = eff_xr_bw.coords["color"][::-1]

    eff_xr = xr.concat([eff_xr_col, eff_xr_bw], dim="color")

    return eff_xr


interval = 0.1  # wavelength interval (in nm)
wl_cell = np.arange(300, 4000, interval)  # wavelengths

single_J_result = pd.read_csv("../ecopv/data/paper_colors.csv")

initial_iters = 100  # number of initial evolutions for the archipelago
add_iters = 100  # additional evolutions added each time if color threshold/convergence condition not met
# every color will run a minimum of initial_iters + add_iters evolutions before ending!

max_trials_col = (
    3 * add_iters
)  # how many population evolutions happen before giving up if there are no populations
# which meet the color threshold

R_type = "sharp"  # "sharp" for rectangular dips or "gauss" for gaussians
fixed_height = True  # fixed height peaks (will be at the value of max_height) or not
j01_method = "perfect_R"

max_height = 1  # maximum height of reflection peaks
base = 0  # baseline fixed reflection

patch_width = 0.75

n_junc_loop = [1, 2, 3, 4, 5, 6]

# also run for 1 junc/1 peak but no more junctions.

color_names, color_XYZ = load_colorchecker()  # 24 default Babel colors

color_XYZ_xr = xr.DataArray(
    color_XYZ[:18],
    dims=["color", "XYZ"],
    coords={"color": color_XYZ[:18, 1], "XYZ": ["X", "Y", "Z"]},
)

color_XYZ_xr = color_XYZ_xr.sortby("color", ascending=True)
color_XYZ_bw = xr.DataArray(
    color_XYZ[18:][::-1],
    dims=["color", "XYZ"],
    coords={"color": color_XYZ[18:, 1][::-1], "XYZ": ["X", "Y", "Z"]},
)
color_XYZ_xr = xr.concat([color_XYZ_xr, color_XYZ_bw], dim="color")

# color_names = color_names[:5]
# color_XYZ = color_XYZ[:5]

light_source_name = "AM1.5g"

photon_flux_cell = np.array(
    LightSource(
        source_type="standard",
        version="AM1.5g",
        x=wl_cell,
        output_units="photon_flux_per_nm",
    ).spectrum(wl_cell)
)

photon_flux_color = photon_flux_cell[
    :, np.all((photon_flux_cell[0] >= 380, photon_flux_cell[0] <= 780), axis=0)
]

shapes = ["o", "^", "+", "v", "*", "x", "s"]

### Efficiency and relative efficiency loss for each color, 1-6 junctions, 2-4 peaks ###
n_peak_loop = [2,3,4]

loop_n = 0

cols = sns.color_palette("Set2", n_colors=len(n_junc_loop))
cols = ["k", "r", "g"]

black_cell_eff = np.array([33.79, 45.85, 51.76, 55.49, 57.82, 59.71])
black_cell_Eg = [
    [1.34],
    [0.96, 1.63],
    [0.93, 1.37, 1.90],
    [0.72, 1.11, 1.49, 2.00],
    [0.70, 1.01, 1.33, 1.67, 2.14],
    [0.69, 0.96, 1.20, 1.47, 1.79, 2.24],
]

Y = np.hstack((color_XYZ[:, 1], [0]))
Y_cols = Y[:18]
col_names = xr.DataArray(data=color_names[:18],
                         dims=["Y"], coords={"Y": Y_cols})
col_names = col_names.sortby("Y", ascending=True)

col_names_all_desc = xr.DataArray(data=color_names, dims=["Y"],
                                  coords={"Y": color_XYZ[:,1]})

col_names_all_desc = col_names_all_desc.sortby("Y", ascending=True)

alphas = [1, 0.5]

fixed_height_loop = [True]

fig, (ax1, ax_l, ax2) = plt.subplots(3, figsize=(5.5, 7),
                                     gridspec_kw={"height_ratios": [1, 0.2, 1]})


for j1, n_junctions in enumerate(n_junc_loop):
    for i1, n_peaks in enumerate(n_peak_loop):

        champion_effs = np.loadtxt(
            "results/champion_eff_"
            + R_type
            + str(n_peaks)
            + "_"
            + str(n_junctions)
            + "_"
            + "True"
            + str(max_height)
            + "_"
            + str(base) + "_"  + j01_method + light_source_name
            + ".txt"
        )
        champion_pops = np.loadtxt(
            "results/champion_pop_"
            + R_type
            + str(n_peaks)
            + "_"
            + str(n_junctions)
            + "_"
            + "True"
            + str(max_height)
            + "_"
            + str(base) + "_"  + j01_method + light_source_name
            + ".txt"
        )

        append_black = black_cell_eff[n_junctions - 1]
        eff_xr = make_sorted_xr(champion_effs, color_names, append_black,
                                ascending=True)

        eff_loss = (
                100
                * (eff_xr - black_cell_eff[n_junctions - 1])
                / black_cell_eff[n_junctions - 1]
        )

        if n_peaks == 2:
            ax1.plot(
                eff_loss.color.data,
                eff_loss.data,
                mfc="none",
                linestyle="none",
                color=cols[i1],
                marker=shapes[j1],
                markersize=4,
            )

            two_peak_ref = eff_xr.data
            two_peak_pop = champion_pops

        else:
            eff_diff = (
                100
                * (eff_xr - two_peak_ref)
                / two_peak_ref
            ) # this is NEGATIVE if two_peak if higher, POSITIVE if two_peak is lower

            ax2.plot(
                eff_diff.color.data,
                eff_diff.data,
                mfc="none",
                linestyle="none",
                color=cols[i1],
                marker=shapes[j1],
                markersize=4,
            )

        # plt.legend(title="Fixed h:")

apply_formatting(ax1, n_colors=len(eff_diff.color.data))
apply_formatting(ax2, eff_diff.color.data)

for i1, n_junctions in enumerate(n_junc_loop):
    ax_l.plot(0, 0, marker=shapes[i1], color='k', linestyle='none', label=n_junctions, mfc='none')

ax_l.set_xlim(10, 20)
ax_l.legend(title="Junctions:", ncol=3, loc="center left")

ax_l2 = ax_l.twinx()

for i1, n_peaks in enumerate(n_peak_loop):
    ax_l2.plot(0, 0, marker='o', color=cols[i1], linestyle='none', label=n_peaks, mfc='none')

ax_l2.legend(title="Reflectance peaks:", ncol=3, loc="center right")
ax_l2.set_xlim(10, 20)
ax1.set_ylim(-38, 1)

# plt.legend(title="Fixed h:")
ax1.set_ylabel("Relative efficiency change (%)")

ax2.set_ylabel("Relative efficiency change (%)")

# ax1.set_ylim(50, 55)
# leg = ax1.legend(bbox_to_anchor=(0.2, 0), loc="upper center",
#                  title="Junctions:", ncol=6,
#            )
# # leg.get_title().set_fontsize('9')
# leg = ax2.legend(bbox_to_anchor=(0.95, 1.2), loc="upper right",
#                  title="Peaks:", ncol=3,
#            )
# leg.get_title().set_fontsize('9')

ax_l.axis("off")
ax_l2.axis("off")
plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
add_colour_patches(ax2, patch_width, eff_diff.color.data, color_XYZ_xr.data)
add_colour_patches(ax1, patch_width, eff_diff.color.data, color_XYZ_xr.data)
plt.show()


# Pick some specific spectra where 3 or 4 peaks does better.
# YellowGreen, 6 junctions

champion_pops_2peaks = np.loadtxt(
    "results/champion_pop_"+ R_type+ str(2) +"_"+ str(6)+ "_"+ "True"+ str(max_height)+ "_"+ str(base) + "_"
    + j01_method + light_source_name + ".txt")[10]

champion_pops_4peaks = np.loadtxt(
    "results/champion_pop_"+ R_type+ str(4) +"_"+ str(6)+ "_"+ "True"+ str(max_height)+ "_"+ str(base) + "_"
    + j01_method + light_source_name + ".txt")[10]

R_2 = gen_spectrum_ndip(champion_pops_2peaks, 2, wl_cell)
R_4 = gen_spectrum_ndip(champion_pops_4peaks, 4, wl_cell)

plt.figure()
plt.plot(wl_cell, R_2)
plt.plot(wl_cell, R_4, '--')
plt.xlim(300, 800)
plt.show()