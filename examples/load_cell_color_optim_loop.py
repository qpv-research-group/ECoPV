from os.path import join
from ecopv.optimization_functions import getPmax, getIVmax
from ecopv.spectrum_functions import make_spectrum_ndip, gen_spectrum_ndip

from solcore.light_source import LightSource
import seaborn as sns
from cycler import cycler

from ecopv.plot_utilities import *


def make_sorted_xr(arr, color_names, append_black=None):
    if arr.ndim == 1:
        dims = ["color"]

    else:
        dims = ["color", "n"]

    eff_xr_col = xr.DataArray(data=arr[:18], dims=dims, coords={"color": Y_cols})

    eff_xr_col = eff_xr_col.sortby("color", ascending=False)
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

    eff_xr = xr.concat([eff_xr_col, eff_xr_bw], dim="color")

    return eff_xr


interval = 0.1  # wavelength interval (in nm)
wl_cell = np.arange(300, 4000, interval)  # wavelengths

initial_iters = 100  # number of initial evolutions for the archipelago
add_iters = 100  # additional evolutions added each time if color threshold/convergence condition not met
# every color will run a minimum of initial_iters + add_iters evolutions before ending!

max_trials_col = (
    3 * add_iters
)  # how many population evolutions happen before giving up if there are no populations
# which meet the color threshold

type = "sharp"  # "sharp" for rectangular dips or "gauss" for gaussians
fixed_height = True  # fixed height peaks (will be at the value of max_height) or not

max_height = 1  # maximum height of reflection peaks
base = 0  # baseline fixed reflection

patch_width = 0.75

n_junc_loop = [1, 2, 3, 4, 5, 6]

n_peak_loop = [4, 3, 2]
# also run for 1 junc/1 peak but no more junctions.

color_names, color_XYZ = load_colorchecker()  # 24 default Babel colors

color_XYZ_xr = xr.DataArray(
    color_XYZ[:18],
    dims=["color", "XYZ"],
    coords={"color": color_XYZ[:18, 1], "XYZ": ["X", "Y", "Z"]},
)

color_XYZ_xr = color_XYZ_xr.sortby("color", ascending=False)
color_XYZ_bw = xr.DataArray(
    color_XYZ[18:],
    dims=["color", "XYZ"],
    coords={"color": color_XYZ[18:, 1], "XYZ": ["X", "Y", "Z"]},
)
color_XYZ_xr = xr.concat([color_XYZ_xr, color_XYZ_bw], dim="color")

# color_names = color_names[:5]
# color_XYZ = color_XYZ[:5]

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


### Compare 1J limiting efficiencies with those from paper ###
single_J_result = np.loadtxt(
    join("../ecopv/data", "paper_colors.csv"),
    skiprows=1,
    usecols=[2, 3, 4, 5, 6, 7, 8, 9],
    delimiter=",",
)
n_peaks = 2
n_junctions = 1

champion_effs = np.loadtxt(
    "results/champion_eff_"
    + type
    + str(n_peaks)
    + "_"
    + str(n_junctions)
    + "_"
    + str(fixed_height)
    + str(max_height)
    + "_"
    + str(base)
    + "_spd.txt"
)
champion_pops = np.loadtxt(
    "results/champion_pop_"
    + type
    + str(n_peaks)
    + "_"
    + str(n_junctions)
    + "_"
    + str(fixed_height)
    + str(max_height)
    + "_"
    + str(base)
    + "_spd.txt"
)

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax.plot(color_names, single_J_result[:, 3], "ok", label="Published results", mfc="none")
ax.plot(color_names, champion_effs, "or", label="MO-DE results", mfc="none")
ax.legend()
ax2.plot(color_names, champion_pops[:, -1], "ok", mfc="none")
ax2.plot(color_names, single_J_result[:, 7], "or", mfc="none")
apply_formatting(ax, color_labels=color_names)
apply_formatting(ax2, color_labels=color_names)

plt.tight_layout()
plt.show()


### Efficiency and relative efficiency loss for each color, 1-6 junctions, 2-4 peaks ###
loop_n = 0

cols = sns.color_palette("Set2", n_colors=len(n_junc_loop))
cols = ["r", "g", "k"]

black_cell_eff = np.array([33.8, 45.9, 51.8, 55.5, 57.8, 59.7])
black_cell_Eg = [
    [1.34],
    [0.96, 1.63],
    [0.93, 1.37, 1.90],
    [0.72, 1.11, 1.49, 2.00],
    [0.70, 1.01, 1.33, 1.67, 2.14],
    [0.70, 0.96, 1.20, 1.47, 1.79, 2.24],
]

Y = np.hstack((color_XYZ[:, 1], [0]))
Y_cols = Y[:18]
col_names = xr.DataArray(data=color_names[:18], dims=["Y"], coords={"Y": Y_cols})
col_names = col_names.sortby("Y", ascending=False)

alphas = [1, 0.5]

fixed_height_loop = [True]

fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 7))

for j1, n_junctions in enumerate(n_junc_loop):
    for i1, n_peaks in enumerate(n_peak_loop):
        for k1, fixed_height in enumerate(fixed_height_loop):

            champion_effs = np.loadtxt(
                "results/champion_eff_"
                + type
                + str(n_peaks)
                + "_"
                + str(n_junctions)
                + "_"
                + str(fixed_height)
                + str(max_height)
                + "_"
                + str(base)
                + "_spd.txt"
            )
            champion_pops = np.loadtxt(
                "results/champion_pop_"
                + type
                + str(n_peaks)
                + "_"
                + str(n_junctions)
                + "_"
                + str(fixed_height)
                + str(max_height)
                + "_"
                + str(base)
                + "_spd.txt"
            )

            append_black = black_cell_eff[n_junctions - 1]
            eff_xr = make_sorted_xr(champion_effs, color_names, append_black)

            if i1 == len(n_peak_loop) - 1:
                ax1.plot(
                    eff_xr.color.data,
                    eff_xr.data,
                    mfc="none",
                    linestyle="none",
                    color=cols[i1],
                    marker=shapes[j1],
                    # label=fixed_height,
                    label=n_junctions,
                    alpha=alphas[k1],
                )

            else:
                ax1.plot(
                    eff_xr.color.data,
                    eff_xr.data,
                    mfc="none",
                    linestyle="none",
                    color=cols[i1],
                    marker=shapes[j1],
                    alpha=alphas[k1],
                )

            eff_loss = (
                100
                * (eff_xr - black_cell_eff[n_junctions - 1])
                / black_cell_eff[n_junctions - 1]
            )

            if j1 == len(n_junc_loop) - 1:
                ax2.plot(
                    eff_loss.color.data,
                    eff_loss.data,
                    mfc="none",
                    linestyle="none",
                    color=cols[i1],
                    marker=shapes[j1],
                    label=n_peaks,
                    alpha=alphas[k1],
                )

            else:
                ax2.plot(
                    eff_loss.color.data,
                    eff_loss.data,
                    mfc="none",
                    linestyle="none",
                    color=cols[i1],
                    marker=shapes[j1],
                    alpha=alphas[k1],
                )

            # plt.legend(title="Fixed h:")

apply_formatting(ax1, n_colors=len(eff_loss.color.data))
apply_formatting(ax2, eff_loss.color.data)

ax2.set_ylim(-40, 0.6)

# plt.legend(title="Fixed h:")
ax1.set_ylabel("Efficiency (%)")
ax2.set_ylabel("Relative efficiency loss (%)")

# ax1.set_ylim(50, 55)
ax1.legend(bbox_to_anchor=(1.15, 0.8), title="Junctions:")
ax2.legend(bbox_to_anchor=(1.15, 0.8), title="Peaks:")
plt.tight_layout()
add_colour_patches(ax2, patch_width, eff_loss.color.data, color_XYZ_xr)
plt.show()


### Efficiency and relative efficiency loss for each color, 1-6 junctions, 2-4 peaks - BLACK BODY ###

black_cell_eff_bb = []
black_cell_Eg_bb = []
light_source = LightSource(
    source_type="black body",
    x=wl_cell,
    output_units="photon_flux_per_nm",
    entendue="Sun",
    T=5778,
)

for n_junctions in n_junc_loop:
    Egs = np.loadtxt("results/champion_pop_{}juncs_bbspec.txt".format(n_junctions))
    if Egs.size > 1:
        Egs = Egs[::-1]

    else:
        Egs = [Egs.tolist()]
    black_cell_Eg_bb.append(Egs)

    black_cell_eff_bb.append(
        100
        * getPmax(
            Egs,
            light_source.spectrum(wl_cell)[1],
            light_source.spectrum(wl_cell)[0],
            interval,
        )
        / light_source.power_density
    )

alphas = [1, 0.5]

n_junc_loop = [1, 2, 3, 4, 5, 6]
n_peak_loop = [2]
fixed_height_loop = [True]

fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 7))

for j1, n_junctions in enumerate(n_junc_loop):
    for i1, n_peaks in enumerate(n_peak_loop):
        for k1, fixed_height in enumerate(fixed_height_loop):

            champion_effs = np.loadtxt(
                "results/champion_eff_"
                + type
                + str(n_peaks)
                + "_"
                + str(n_junctions)
                + "_"
                + str(fixed_height)
                + str(max_height)
                + "_"
                + str(base)
                + "bb_spd.txt"
            )
            champion_pops = np.loadtxt(
                "results/champion_pop_"
                + type
                + str(n_peaks)
                + "_"
                + str(n_junctions)
                + "_"
                + str(fixed_height)
                + str(max_height)
                + "_"
                + str(base)
                + "bb_spd.txt"
            )

            append_black = black_cell_eff_bb[n_junctions - 1]
            eff_xr = make_sorted_xr(champion_effs, color_names, append_black)

            if i1 == len(n_peak_loop) - 1:
                ax1.plot(
                    eff_xr.color.data,
                    eff_xr.data,
                    mfc="none",
                    linestyle="none",
                    color=cols[i1],
                    marker=shapes[j1],
                    # label=fixed_height,
                    label=n_junctions,
                    alpha=alphas[k1],
                )

            else:
                ax1.plot(
                    eff_xr.color.data,
                    eff_xr.data,
                    mfc="none",
                    linestyle="none",
                    color=cols[i1],
                    marker=shapes[j1],
                    alpha=alphas[k1],
                )

            eff_loss = (
                100
                * (eff_xr - black_cell_eff_bb[n_junctions - 1])
                / black_cell_eff_bb[n_junctions - 1]
            )

            if j1 == len(n_junc_loop) - 1:
                ax2.plot(
                    eff_loss.color.data,
                    eff_loss.data,
                    mfc="none",
                    linestyle="none",
                    color=cols[i1],
                    marker=shapes[j1],
                    label=n_peaks,
                    alpha=alphas[k1],
                )

            else:
                ax2.plot(
                    eff_loss.color.data,
                    eff_loss.data,
                    mfc="none",
                    linestyle="none",
                    color=cols[i1],
                    marker=shapes[j1],
                    alpha=alphas[k1],
                )

            # plt.legend(title="Fixed h:")

apply_formatting(ax1, n_colors=len(eff_loss.color.data))
apply_formatting(ax2, eff_loss.color.data)

# ax2.set_ylim(-40, 0.6)

# plt.legend(title="Fixed h:")
ax1.set_ylabel("Efficiency (%)")
ax2.set_ylabel("Relative efficiency loss (%)")

# ax1.set_ylim(50, 55)
ax1.legend(bbox_to_anchor=(1.15, 0.8), title="Junctions:")
ax2.legend(bbox_to_anchor=(1.15, 0.8), title="Peaks:")
plt.tight_layout()
add_colour_patches(ax2, patch_width, eff_loss.color.data, color_XYZ_xr)
plt.show()


### Optimal bandgaps (2 peaks) ###

n_junc_loop = [1, 2, 3, 4, 5, 6]
n_peak_loop = [2]
fixed_height_loop = [True]

pal = sns.color_palette("husl", max(n_junc_loop))
cols = cycler("color", pal)
params = {"axes.prop_cycle": cols}
plt.rcParams.update(params)

fig, axs = plt.subplots(3, 2, figsize=(10, 7))

axs = axs.flatten()

for j1, n_junctions in enumerate(n_junc_loop):
    for i1, n_peaks in enumerate(n_peak_loop):
        for k1, fixed_height in enumerate(fixed_height_loop):
            champion_effs = np.loadtxt(
                "results/champion_eff_"
                + type
                + str(n_peaks)
                + "_"
                + str(n_junctions)
                + "_"
                + str(fixed_height)
                + str(max_height)
                + "_"
                + str(base)
                + "bb_spd.txt"
            )
            champion_pops = np.loadtxt(
                "results/champion_pop_"
                + type
                + str(n_peaks)
                + "_"
                + str(n_junctions)
                + "_"
                + str(fixed_height)
                + str(max_height)
                + "_"
                + str(base)
                + "bb_spd.txt"
            )

            Eg_xr = make_sorted_xr(
                champion_pops[:, -n_junctions:],
                color_names,
                append_black=black_cell_Eg[n_junctions - 1],
            )

            axs[j1].plot(
                Eg_xr.color.data,
                1240 / Eg_xr.data,
                mfc="none",
                linestyle="--",
                # color=cols[i1], marker=shapes[j1],
                marker="o",
                markersize=4,
                # label=n_junctions,
                alpha=alphas[k1],
            )

    axs[j1].set_ylim(350, 2400)

    if j1 < 4:
        apply_formatting(axs[j1], grid="x", n_colors=len(Eg_xr.color.data))

    else:
        apply_formatting(axs[j1], Eg_xr.color.data, grid="x")

    if n_junctions == 1:
        axs[j1].text(1, 2050, "1 junction")

    else:
        axs[j1].text(1, 2050, str(n_junctions) + " junctions")

    if j1 == 0 or j1 == 2 or j1 == 4:
        axs[j1].set_ylabel("Bandgap (nm)")

    if j1 % 2 == 1:
        axs[j1].set_yticklabels([])
        axs[j1].tick_params(direction="in", which="both", axis="y", right=False)
        f = lambda x: 1240 / x
        ax2 = axs[j1].secondary_yaxis("right", functions=(f, f))
        ax2.set_yticks([1, 1.5, 2, 3])
        ax2.yaxis.set_minor_locator(tck.AutoMinorLocator())
        ax2.set_ylabel("Bandgap (eV)")

plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)
add_colour_patches(axs[4], patch_width, Eg_xr.color.data, color_XYZ_xr)
add_colour_patches(axs[5], patch_width, Eg_xr.color.data, color_XYZ_xr)
plt.show()

# Compare fixed and variable peak height results:
n_junc_loop = [1, 2, 3, 4, 5, 6]
x
n_peak_loop = [2]

alphas = [1, 0.5]

fixed_height_loop = [True, False]

for j1, n_junctions in enumerate(n_junc_loop):
    for i1, n_peaks in enumerate(n_peak_loop):
        champion_eff_comp = np.empty((2, 24))

        for k1, fixed_height in enumerate(fixed_height_loop):
            champion_effs = np.loadtxt(
                "results/champion_eff_"
                + type
                + str(n_peaks)
                + "_"
                + str(n_junctions)
                + "_"
                + str(fixed_height)
                + str(max_height)
                + "_"
                + str(base)
                + ".txt"
            )
            champion_pops = np.loadtxt(
                "results/champion_pop_"
                + type
                + str(n_peaks)
                + "_"
                + str(n_junctions)
                + "_"
                + str(fixed_height)
                + str(max_height)
                + "_"
                + str(base)
                + ".txt"
            )
            # final_populations = np.load("results/final_pop_tcheb_" + type + str(n_peaks) + '_' + str(
            #     n_junctions) + '_' + str(fixed_height) + str(max_height) + '_' + str(base)+'.npy', allow_pickle=True)
            champion_eff_comp[k1] = champion_effs

    print(
        n_junctions, np.max(champion_eff_comp[1] - champion_eff_comp[0])
    )  # if positive, variable height is better;
    # should be < acceptable_eff_change which is 1e-4

    # plt.legend(title="Fixed h:")
#
# fixed_height = True
# fig, ax = plt.subplots(1, figsize=(8, 3.2))
#
# for j1, n_junctions in enumerate(n_junc_loop):
#     champion_effs = np.empty((len(n_peak_loop), len(color_XYZ)))
#
#     for i1, n_peaks in enumerate(n_peak_loop):
#         champion_effs[i1] = np.loadtxt("results/champion_eff_tcheb_" + type + str(n_peaks) + '_' + str(
#             n_junctions) + '_' + str(fixed_height) + str(max_height) + '_' + str(base) + '.txt')
#
#     for l1 in range(0, len(n_peak_loop) - 1):
#         print(l1)
#         diff = champion_effs[l1 + 1, :] - champion_effs[0, :]  # negative if more peaks don't help
#         print(diff)
#         ax.plot(color_names, diff, mfc='none', linestyle='none', marker=shapes[i1], color=cols[j1],
#                 label=str(n_junctions))
#
# ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
# plt.legend()
# plt.tight_layout()
# plt.show()

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

### Efficiency/peak placements with number of R peaks ###
cmf = load_cmf(photon_flux_cell[0])
interval = np.round(np.diff(photon_flux_cell[0])[0], 6)

RGBA = wavelength_to_rgb(photon_flux_color[0])

colors = ["k", "b", "r"]

pal = ["r", "g", "b"]
cols = cycler("color", pal)
params = {"axes.prop_cycle": cols}
plt.rcParams.update(params)

type = "sharp"
fixed_height_loop = [True]
max_height = 1
base = 0

n_junc_loop = [5]

n_peak_loop = [2, 3, 4]

data_width = 0.6

offset = np.linspace(0, data_width, 3)
# also run for 1 junc/1 peak but no more junctions.

alphas = [1, 0.5]

fig, axes = plt.subplots(
    2,
    2,
    gridspec_kw={
        "height_ratios": [1, 2],
        "width_ratios": [3, 1],
        "hspace": 0.1,
        "wspace": 0.05,
    },
    figsize=(8, 5),
)

offset_ind = 0

for j1, n_junctions in enumerate(n_junc_loop):
    for i1, n_peaks in enumerate(n_peak_loop):
        for k1, fixed_height in enumerate(fixed_height_loop):
            placeholder_obj = make_spectrum_ndip(
                n_peaks=n_peaks, type=type, fixed_height=fixed_height
            )

            champion_effs = np.loadtxt(
                "results/champion_eff_"
                + type
                + str(n_peaks)
                + "_"
                + str(n_junctions)
                + "_"
                + str(fixed_height)
                + str(max_height)
                + "_"
                + str(base)
                + "_spd.txt"
            )
            champion_pops = np.loadtxt(
                "results/champion_pop_"
                + type
                + str(n_peaks)
                + "_"
                + str(n_junctions)
                + "_"
                + str(fixed_height)
                + str(max_height)
                + "_"
                + str(base)
                + "_spd.txt"
            )

            eff_xr = make_sorted_xr(champion_effs, color_names)
            c_xr = make_sorted_xr(champion_pops[:, :n_peaks], color_names)
            w_xr = make_sorted_xr(champion_pops[:, n_peaks : 2 * n_peaks], color_names)

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
axes[0, 1].plot(0, 0, color=colors[0], label="2 peaks")
axes[0, 1].plot(0, 0, color=colors[1], label="3 peaks")
axes[0, 1].plot(0, 0, color=colors[2], label="4 peaks")
axes[0, 1].legend(frameon=False, loc="center")
axes[1, 1].set_xlim(0, 1.8)

plt.subplots_adjust(bottom=0.25, top=0.95, left=0.1, right=0.97)
apply_formatting(axes[0, 0], n_colors=24)
apply_formatting(axes[1, 0], eff_xr.color.data)
add_colour_patches(axes[1, 0], patch_width, eff_xr.color.data, color_XYZ_xr)
plt.tight_layout()
plt.show()

### Efficiency/peak placements with number of junctions ###
n_junc_loop = [4, 5, 6]

n_peak_loop = [2]

data_width = 0.6

offset = np.linspace(0, data_width, 3)
# also run for 1 junc/1 peak but no more junctions.

alphas = [1, 0.5]

fig, axes = plt.subplots(
    2,
    2,
    gridspec_kw={
        "height_ratios": [1, 2],
        "width_ratios": [3, 1],
        "hspace": 0.1,
        "wspace": 0.05,
    },
    figsize=(8, 5),
)

offset_ind = 0

for j1, n_junctions in enumerate(n_junc_loop):
    for i1, n_peaks in enumerate(n_peak_loop):
        for k1, fixed_height in enumerate(fixed_height_loop):
            placeholder_obj = make_spectrum_ndip(
                n_peaks=n_peaks, type=type, fixed_height=fixed_height
            )

            champion_effs = np.loadtxt(
                "results/champion_eff_"
                + type
                + str(n_peaks)
                + "_"
                + str(n_junctions)
                + "_"
                + str(fixed_height)
                + str(max_height)
                + "_"
                + str(base)
                + "_spd.txt"
            )
            champion_pops = np.loadtxt(
                "results/champion_pop_"
                + type
                + str(n_peaks)
                + "_"
                + str(n_junctions)
                + "_"
                + str(fixed_height)
                + str(max_height)
                + "_"
                + str(base)
                + "_spd.txt"
            )
            # final_populations = np.load("results/final_pop_tcheb_" + type + str(n_peaks) + '_' + str(
            #     n_junctions) + '_' + str(fixed_height) + str(max_height) + '_' + str(base)+'.npy', allow_pickle=True)
            eff_xr = make_sorted_xr(champion_effs, color_names)
            c_xr = make_sorted_xr(champion_pops[:, :n_peaks], color_names)
            w_xr = make_sorted_xr(champion_pops[:, n_peaks : 2 * n_peaks], color_names)
            Eg_xr = make_sorted_xr(champion_pops[:, -n_junctions:], color_names)

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

                axes[1, 0].plot(
                    l1 + offset[offset_ind] - data_width / 2,
                    1240 / Eg_xr.data[l1, -1],
                    "o",
                    mfc="none",
                    markersize=3,
                    color=colors[offset_ind],
                )

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

axes[0, 1].axis("off")
axes[1, 0].set_ylabel("Wavelength (nm)")
axes[1, 1].set_xlabel(r"Spectral sensitivity / " "\n" r"Normalised photon flux")
axes[0, 1].plot(0, 0, color=colors[0], label="4 junctions")
axes[0, 1].plot(0, 0, color=colors[1], label="5 junctions")
axes[0, 1].plot(0, 0, color=colors[2], label="6 junctions")
axes[0, 1].legend(frameon=False, loc="center")
plt.subplots_adjust(bottom=0.25, top=0.95, left=0.1, right=0.97)
apply_formatting(axes[0, 0], n_colors=24)
apply_formatting(axes[1, 0], Eg_xr.color.data)
add_colour_patches(axes[1, 0], patch_width, Eg_xr.color.data, color_XYZ_xr)
plt.tight_layout()
plt.show()

### Peak placement, top bandgap and efficiency - subset of colours ###

n_junc_loop = [4, 5, 6]

n_peaks = 2

data_width = 0.6

offset = np.linspace(0, data_width, 3)
# also run for 1 junc/1 peak but no more junctions.

alphas = [1, 0.5]

subset_inds = [0, 3, 7, 13, 17, 18, 19, 20, 21, 22, 23]

color_XYZ_subs = color_XYZ_xr[subset_inds]

fig, axes = plt.subplots(2, 1, gridspec_kw={"height_ratios": [1, 2]}, figsize=(4, 5.5))
eff_array = np.zeros((len(color_XYZ_subs), len(n_junc_loop)))
Eg_array = np.zeros((len(color_XYZ_subs), len(n_junc_loop)))

offset_ind = 0

for j1, n_junctions in enumerate(n_junc_loop):

    placeholder_obj = make_spectrum_ndip(
        n_peaks=n_peaks, type=type, fixed_height=fixed_height
    )

    champion_effs = np.loadtxt(
        "results/champion_eff_"
        + type
        + str(n_peaks)
        + "_"
        + str(n_junctions)
        + "_"
        + str(fixed_height)
        + str(max_height)
        + "_"
        + str(base)
        + "_spd.txt"
    )
    champion_pops = np.loadtxt(
        "results/champion_pop_"
        + type
        + str(n_peaks)
        + "_"
        + str(n_junctions)
        + "_"
        + str(fixed_height)
        + str(max_height)
        + "_"
        + str(base)
        + "_spd.txt"
    )
    # final_populations = np.load("results/final_pop_tcheb_" + type + str(n_peaks) + '_' + str(
    #     n_junctions) + '_' + str(fixed_height) + str(max_height) + '_' + str(base)+'.npy', allow_pickle=True)
    eff_xr = make_sorted_xr(champion_effs, color_names)[subset_inds]
    c_xr = make_sorted_xr(champion_pops[:, :n_peaks], color_names)[subset_inds]
    w_xr = make_sorted_xr(champion_pops[:, n_peaks : 2 * n_peaks], color_names)[
        subset_inds
    ]
    Eg_xr = make_sorted_xr(champion_pops[:, -n_junctions:], color_names)[subset_inds]

    for l1, target in enumerate(color_XYZ_subs):
        centres = c_xr[l1]
        widths = w_xr[l1]

        eff_array[l1, j1] = eff_xr.data[l1]
        Eg_array[l1, j1] = 1240 / Eg_xr.data[l1, -1]

        axes[0].plot(
            l1 + offset[offset_ind] - data_width / 2,
            eff_xr.data[l1],
            ".",
            color=colors[offset_ind],
            markersize=4,
        )

        axes[1].errorbar(
            [l1 + offset[offset_ind] - data_width / 2] * len(centres),
            centres,
            yerr=widths / 2,
            fmt="none",
            ecolor=colors[offset_ind],
        )

        axes[1].plot(
            l1 + offset[offset_ind] - data_width / 2,
            1240 / Eg_xr.data[l1, -1],
            "o",
            mfc="none",
            markersize=3,
            color=colors[offset_ind],
        )

    offset_ind += 1

for i1, effs in enumerate(eff_array):
    axes[0].plot(offset + i1 - data_width / 2, effs, "--k", alpha=0.4)
    axes[1].plot(offset + i1 - data_width / 2, Eg_array[i1], "--k", alpha=0.4)

axes[0].set_ylabel("Efficiency (%)")

axes[1].set_ylim(370, 800)
axes[1].set_ylabel("Wavelength (nm)")

apply_formatting(axes[0], n_colors=len(subset_inds))
apply_formatting(axes[1], Eg_xr.color.data)
add_colour_patches(axes[1], patch_width, Eg_xr.color.data, color_XYZ_subs)
plt.tight_layout()
plt.show()


photon_flux_eV = LightSource(
    source_type="standard",
    version="AM1.5g",
    x=1240 / wl_cell,
    output_units="photon_flux_per_ev",
)

champion_pops = np.loadtxt(
    "results/champion_pop_"
    + type
    + str(2)
    + "_"
    + str(1)
    + "_"
    + "True"
    + str(1)
    + "_"
    + str(0)
    + ".txt"
)

white_ch_pop = champion_pops[-6]
spec_white = gen_spectrum_ndip(
    white_ch_pop, 2, photon_flux_cell[0], max_height=1, base=0
)

photon_flux_white = (1 - spec_white) * photon_flux_eV.spectrum(1240 / wl_cell)[1]

Imax = np.zeros_like(wl_cell)
Vmax = np.zeros_like(wl_cell)

Imax_w = np.zeros_like(wl_cell)
Vmax_w = np.zeros_like(wl_cell)

for i1, Eg in enumerate(1240 / wl_cell):
    Vmax[i1], Imax[i1] = getIVmax(
        [Eg], photon_flux_cell[1], photon_flux_cell[0], interval
    )
    Vmax_w[i1], Imax_w[i1] = getIVmax(
        [Eg], (1 - spec_white) * photon_flux_cell[1], photon_flux_cell[0], interval
    )

pal = sns.color_palette("husl", 3)

fig, ax = plt.subplots()
ax2 = ax.twinx()

ax2.fill_between(
    1240 / wl_cell,
    photon_flux_eV.spectrum(1240 / wl_cell)[1] / 1e21,
    alpha=0.2,
    color="k",
)

ax.plot(1240 / wl_cell, Imax / max(Imax), color=pal[0])
ax.plot(1240 / wl_cell, Vmax / max(Vmax), color=pal[1])
ax.plot(1240 / wl_cell, (Imax * Vmax) / np.max(Imax * Vmax), color=pal[2])

ax.plot(1240 / wl_cell, Imax_w / max(Imax_w), "--", color=pal[0])
ax.plot(1240 / wl_cell, Vmax_w / max(Vmax_w), "--", color=pal[1])
ax.plot(1240 / wl_cell, (Imax_w * Vmax_w) / np.max(Imax_w * Vmax_w), "--", color=pal[2])

ax.plot([0, 0.6], [-1, -1], "k", label="Ideal black cell")
ax.plot([0, 0.6], [-1, -1], "--k", label="Ideal white cell")
ax2.axvline(1.12, color="k", linestyle="--", alpha=0.5)
ax2.axvline(1.34, color="k", linestyle="--", alpha=0.5)
ax2.set_xlim(0.65, 2.1)
ax2.set_ylim(0, 4)
ax.set_ylim(0, 1.02)
ax.set_xlabel("Bandgap (eV)")
ax2.set_ylabel(r"Photon flux ($\times 10^{21}$ eV$^{-1}$ m$^{-2}$ s$^{-1}$)")
ax.set_ylabel("Normalized $J_{max}$, $V_{max}$, $\eta_{max}$")
ax.text(1.2, 0.56, r"$J_{max}$", color=pal[0], size=12, weight="bold")
ax.text(1.15, 0.2, r"$V_{max}$", color=pal[1], size=12, weight="bold")
ax.text(1.3, 0.95, r"$\eta$", color=pal[2], size=12, weight="bold")

ax2.spines["right"].set_color("gray")
ax2.yaxis.label.set_color("gray")
ax2.tick_params(axis="y", colors="gray")

ax.legend(loc="upper right")
plt.show()

# from colour.plotting import plot_RGB_chromaticities_in_chromaticity_diagram_CIE1931
# RGB = np.random.random((128, 128, 3))
# plot_RGB_chromaticities_in_chromaticity_diagram_CIE1931(
#     RGB, "ITU-R BT.709"
# )
# plt.show()
