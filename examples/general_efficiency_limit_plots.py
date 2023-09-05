from ecopv.optimization_functions import getIVmax, getPmax
from ecopv.spectrum_functions import make_spectrum_ndip, gen_spectrum_ndip, load_cmf

import numpy as np
from solcore.light_source import LightSource
import seaborn as sns
import pandas as pd
from cycler import cycler


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
fixed_height = True

max_height = 1  # maximum height of reflection peaks
base = 0  # baseline fixed reflection

patch_width = 0.75

n_junc_loop = [1, 2, 3, 4, 5, 6]

n_peak_loop = [2]
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

shapes = ["o", "+", "^", ".", "*", "v", "s", "x"]

### Efficiency and relative efficiency loss for each color, 1-6 junctions, 2-4 peaks ###
n_peak_loop = [2,3]

loop_n = 0

cols = sns.color_palette("Set2", n_colors=len(n_junc_loop))
cols = ["r", "g", "k"]

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

# optimal bandgaps - new plot

color_list = sRGB_color_list(order="sorted", include_black=True)
color_list_patches = sRGB_color_list(order="sorted", include_black=False)
# color_list = np.insert(color_list, 0, [0,0,0], axis=0)

n_peak_loop = [2]
n_peaks = 2

fig, axs = plt.subplots(3, 2, figsize=(11.5, 7))

axs = axs.flatten()

for j1, n_junctions in enumerate(n_junc_loop):

    champion_pops = np.loadtxt(
        "results/champion_pop_"
        + R_type
        + str(n_peaks)
        + "_"
        + str(n_junctions)
        + "_"
        + str(fixed_height)
        + str(max_height)
        + "_"
        + str(base) + "_"  + j01_method + light_source_name
        + ".txt"
    )

    # Eg_xr = np.vstack(
    #     (champion_pops[:, -n_junctions:], black_cell_Eg[n_junctions - 1]),
    # )

    # ordered = np.append(color_XYZ[:,1], 0).argsort()
    Eg_values = make_sorted_xr(champion_pops[:, -n_junctions:], color_names,
                       append_black=black_cell_Eg[n_junctions - 1],
                             ascending=True)

    # Y_values = np.append(color_XYZ[:,1], 0)[ordered]
    # Eg_values = Eg_xr[ordered]

    # champion_pops = champion_pops[ordered[1:]]

    champion_pops = make_sorted_xr(champion_pops[:, :-n_junctions], color_names,
                                   append_black=np.zeros(champion_pops[:,
                                                         :-n_junctions].shape[1]),
                                   ascending=True)

    if j1 == 0:
        lowest_edge_1 = champion_pops[:,0] - champion_pops[:,2]/2
        highest_edge_1 = champion_pops[:,0] + champion_pops[:,2]/2

        lowest_edge = champion_pops[:,1] - champion_pops[:,3]/2
        highest_edge = champion_pops[:,1] + champion_pops[:,3]/2


    axs[j1].plot(Eg_values, '--k', alpha=0.5)

    for k1 in range(len(Eg_values)):
        axs[j1].plot([k1]*len(Eg_values[k1]), Eg_values[k1], 'o',
                color=color_list[k1],
                markeredgecolor='k', )

        if highest_edge_1[k1].data != 0:
            axs[j1].fill_between([k1-0.5, k1+0.5],
                                 y1=1240 / highest_edge_1[k1],
                                 y2=1240 / lowest_edge_1[k1],
                                 color='k',
                                 alpha=0.1)

            axs[j1].fill_between([k1-0.5, k1+0.5],
                                 y1=1240 / highest_edge[k1],
                                 y2=1240 / lowest_edge[k1],
                                 color='k',
                                 alpha=0.1)

    # axs[j1].fill_between(Y_values, y1=1240 / highest_edge,
    #                      y2=1240 / lowest_edge,
    #                      alpha=0.1, color='k')


    # axs[j1].set_xlim(-0.02, 0.9326)

    range_Eg = np.max(Eg_values) - np.min(Eg_values)

    axs[j1].set_ylim(np.min(Eg_values)-0.13*range_Eg,
                     np.max(Eg_values)+0.1*range_Eg)

    # if j1 < 4:
    #     apply_formatting(axs[j1], grid="x", n_colors=len(Eg_xr.color.data))
    #
    # else:
    #     apply_formatting(axs[j1], Eg_xr.color.data, grid="x")

    if n_junctions == 1:
        axs[j1].text(20.5, np.max(Eg_values) - 0.02*(np.max(Eg_values)-np.min(
            Eg_values)), "1 junction", weight="bold")

    else:
        axs[j1].text(20.5, np.max(Eg_values) - 0.02*(np.max(Eg_values)-np.min(
            Eg_values)), str(n_junctions) + " junctions", weight="bold")

    if j1 == 0 or j1 == 2 or j1 == 4:
        axs[j1].set_ylabel("Bandgap (eV)")

    # if j1 % 2 == 1:
    #     axs[j1].set_yticklabels([])
    #     axs[j1].tick_params(direction="in", which="both", axis="y", right=False)
        # f = lambda x: 1240 / x
        # ax2 = axs[j1].secondary_yaxis("right", functions=(f, f))
        # ax2.set_yticks([1, 1.5, 2, 3])
        # ax2.yaxis.set_minor_locator(tck.AutoMinorLocator())
        # ax2.set_ylabel("Bandgap (eV)")

    if j1 > 3:
        # axs[j1].set_xlabel(r"$Y$")
        add_colour_patches(axs[j1], patch_width, labels=champion_pops.color.data,
                           color_XYZ=color_list_patches, color_coords='sRGB')

    apply_formatting(axs[j1], color_labels=champion_pops.color.data, n_colors=25)
    axs[j1].grid(axis='x')



plt.tight_layout()
plt.subplots_adjust(hspace=0.1, wspace=0.15)
# add_colour_patches(axs[4], patch_width, Eg_xr.color.data, color_XYZ_xr)
# add_colour_patches(axs[5], patch_width, Eg_xr.color.data, color_XYZ_xr)
plt.show()


light_source_name = "AM1.5g"

####

cmf = load_cmf(photon_flux_cell[0])
interval = np.round(np.diff(photon_flux_cell[0])[0], 6)

RGBA = wavelength_to_rgb(photon_flux_color[0])

colors = ["k", "b", "r", "y", "m", "g"]

pal = ["r", "g", "b"]
cols = cycler("color", pal)
params = {"axes.prop_cycle": cols}
plt.rcParams.update(params)

R_type = "sharp"
fixed_height = True
max_height = 1
base = 0

patch_width = 0.9

n_junc_loop = [1, 2, 3, 4, 5, 6]

n_peak_loop = [2]

data_width = 0.75

offset = np.linspace(0, data_width, len(n_junc_loop))
# also run for 1 junc/1 peak but no more junctions.

alphas = [1, 0.5]

from matplotlib import rc
rc("font", **{"family": "sans-serif",
              "sans-serif": ["Helvetica"],
              })

fig, axes = plt.subplots(
    2,
    2,
    gridspec_kw={
        "height_ratios": [1, 2],
        "width_ratios": [5, 1],
        "hspace": 0.15,
        "wspace": 0.05,
    },
    figsize=(10.5, 5),
)

eff_data = np.zeros((len(n_junc_loop), 24))
Eg_data = np.zeros((len(n_junc_loop), 24))

offset_ind = 0

for j1, n_junctions in enumerate(n_junc_loop):
    for i1, n_peaks in enumerate(n_peak_loop):
        placeholder_obj = make_spectrum_ndip(
            n_peaks=n_peaks, R_type=R_type, fixed_height=fixed_height
        )

        champion_effs = np.loadtxt(
            "results/champion_eff_"
            + R_type
            + str(n_peaks)
            + "_"
            + str(n_junctions)
            + "_"
            + str(fixed_height)
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
            + str(fixed_height)
            + str(max_height)
            + "_"
            + str(base) + "_"  + j01_method + light_source_name
            + ".txt"
        )
        eff_xr = make_sorted_xr(champion_effs, color_names, ascending=True)
        c_xr = make_sorted_xr(champion_pops[:, :n_peaks], color_names,
                              ascending=True)
        w_xr = make_sorted_xr(champion_pops[:, n_peaks : 2 * n_peaks],
                              color_names, ascending=True)
        Eg_xr = make_sorted_xr(champion_pops[:, -n_junctions:], color_names,
                               ascending=True)

        eff_data[j1, :] = eff_xr.data
        Eg_data[j1, :] = Eg_xr.data[:,-1]

        for l1, target in enumerate(color_XYZ_xr):
            centres = c_xr[l1]
            widths = w_xr[l1]

            axes[0, 0].plot(
                l1 + offset[offset_ind] - data_width / 2,
                eff_xr.data[l1],
                ".",
                color=colors[offset_ind],
                markersize=4,
            )

            axes[1, 0].errorbar(
                [l1 + offset[offset_ind] - data_width / 2] * len(centres),
                centres,
                yerr=widths / 2,
                fmt="none",
                ecolor=colors[offset_ind],
            )

            # axes[1, 0].plot(
            #     l1 + offset[offset_ind] - data_width / 2,
            #     1240 / Eg_xr.data[l1, -1],
            #     "o",
            #     mfc="none",
            #     markersize=3,
            #     color=colors[offset_ind],
            # )

        offset_ind += 1

for i1 in range(len(RGBA)):
    axes[1, 1].add_patch(
        Rectangle(
            xy=(0, photon_flux_color[0][i1]),
            width=photon_flux_color[1][i1] / np.max(photon_flux_color),
            height=interval,
            facecolor=RGBA[i1, :3],
            alpha=0.6 * RGBA[i1, 3],
        )
    )

axes[1, 1].plot(photon_flux_cell[1] / np.max(photon_flux_cell), wl_cell, "k", alpha=0.5)
axes[1, 1].plot(cmf, wl_cell)

axes[0, 0].set_ylabel("Efficiency (%)")
axes[1, 1].set_yticklabels([])
axes[1, 0].set_ylim(370, 670)
axes[1, 1].set_ylim(370, 670)
axes[1, 1].grid(axis="both", color="0.9")

axes[0, 1].axis("off")
axes[1, 0].set_ylabel("Wavelength (nm)")
axes[1, 1].set_xlabel(r"Spectral sensitivity / " "\n" r"Normalised photon flux")
axes[0, 1].plot(0, 0, color=colors[0], label="1 junction")

for i1 in range(1, max(n_junc_loop)):
    axes[0, 1].plot(0, 0, color=colors[i1], label=f"{i1 + 1} junctions")

axes[0, 1].legend(frameon=False, loc="center")
axes[1, 1].set_xlim(0, 1.8)

for i1, subs in enumerate(eff_data.T):

    axes[0, 0].plot(i1 + offset - data_width / 2, subs, "-k", alpha=0.3)
#     axes[1, 0].plot(i1 + offset - data_width / 2, 1240/Eg_data.T[i1], "-k", alpha=0.3)

plt.subplots_adjust(bottom=0.25, top=0.95, left=0.1, right=0.97)
apply_formatting(axes[0, 0], n_colors=24)
apply_formatting(axes[1, 0], eff_xr.color.data)
add_colour_patches(axes[1, 0], patch_width, eff_xr.color.data, color_XYZ_xr)

axes[0,0].set_title('(a)', loc='left')
axes[1,0].set_title('(b)', loc='left')
axes[1,1].set_title('(c)', loc='left')

plt.tight_layout()
fig.savefig("fig4.pdf", bbox_inches="tight")
plt.show()



### Efficiency and relative efficiency loss for each color, 1-6 junctions, 2-4 peaks ###
n_peak_loop = [2, 3]

loop_n = 0

cols = ["k", "r"]

table_for_paper = np.zeros((25, 9))

_, xyY_coords = load_colorchecker('xyY')

xyY_sorted = make_sorted_xr(xyY_coords, color_names, append_black=[0,0,0], ascending=True)

table_for_paper[:, :3] = xyY_sorted


fig, (ax1, ax_l, ax2) = plt.subplots(3, figsize=(5.5, 7),
                                     gridspec_kw={"height_ratios": [1, 0.2, 1]})

ax2.axhline(0, color="k", linestyle="--", alpha=0.5)

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

        if n_peaks == 2:
            table_for_paper[:, j1 + 3] = eff_xr.data

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
ax1.set_title('(a)', loc='left')

ax2.set_ylabel("Relative efficiency change (%)")
ax2.set_title('(b)', loc='left')
# ax2.set_ylim(-0.3, 2)

ax_l.axis("off")
ax_l2.axis("off")
plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
add_colour_patches(ax2, patch_width, eff_diff.color.data, color_XYZ_xr.data)
add_colour_patches(ax1, patch_width, eff_diff.color.data, color_XYZ_xr.data)
plt.show()

np.savetxt("results/efficiency_table.csv", table_for_paper, delimiter=",",)