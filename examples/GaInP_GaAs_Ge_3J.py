from ecopv.main_optimization import (
    load_colorchecker,
    multiple_color_cells,
    cell_optimization,
)
from ecopv.optimization_functions import getIVmax, getPmax
from ecopv.spectrum_functions import gen_spectrum_ndip
from ecopv.plot_utilities import *
import numpy as np
from solcore.light_source import LightSource
import matplotlib.pyplot as plt
import pygmo as pg
from os import path
import xarray as xr
import seaborn as sns

from matplotlib import rc
rc("font", **{"family": "sans-serif",
              "sans-serif": ["Helvetica"]})

force_rerun = False

col_thresh = 0.004  # for a wavelength interval of 0.1, minimum achievable color error will be (very rough estimate!) ~ 0.001.
# This is the maximum allowed fractional error in X, Y, or Z colour coordinates.

acceptable_eff_change = 1e-4  # how much can the efficiency (in %) change between iteration sets? Stop when have reached
# col_thresh and efficiency change is less than this.

n_trials = 10  # number of islands which will run concurrently
interval = 0.1  # wavelength interval (in nm)
wl_cell = np.arange(
    300, 4000, interval
)  # wavelengths used for cell calculations (range of wavelengths in AM1.5G solar
# spectrum. For calculations relating to colour perception, only the visible range (380-780 nm) will be used.

initial_iters = 100  # number of initial evolutions for the archipelago
add_iters = 100  # additional evolutions added each time if color threshold/convergence condition not met
# every color will run a minimum of initial_iters + add_iters evolutions before ending!

max_trials_col = (
    3 * add_iters
)  # how many population evolutions happen before giving up if there are no populations
# which meet the color threshold

R_type = "sharp"  # "sharp" for rectangular dips or "gauss" for gaussians
fixed_height = True  # fixed height peaks (will be at the value of max_height) if True, or peak height is an optimization
# variable if False
light_source_name = "AM1.5g"

max_height = (
    1  # maximum height of reflection peaks; fixed at this value of fixed_height = True
)
base = 0  # baseline fixed reflection (fixed at this value for both fixed_height = True and False).

n_junc_loop = [0]  # loop through these numbers of junctions

n_peak_loop = [2]  # loop through these numbers of reflection peaks

(
    color_names,
    color_XYZ,
) = (
    load_colorchecker()
)  # load the names and XYZ coordinates of the 24 default Babel colors

Y = np.hstack((color_XYZ[:, 1], [0]))
Y_cols = Y[:18]
col_names = xr.DataArray(data=color_names[:18], dims=["Y"], coords={"Y": Y_cols})
col_names = col_names.sortby("Y", ascending=False)
col_names_all = col_names.data.tolist() + color_names[18:].tolist()

# Use AM1.5G spectrum:
light_source = LightSource(
    source_type="standard",
    version=light_source_name,
    x=wl_cell,
    output_units="photon_flux_per_nm",
)

photon_flux_cell = np.array(light_source.spectrum(wl_cell))

photon_flux_color = photon_flux_cell[
    :, np.all((photon_flux_cell[0] >= 380, photon_flux_cell[0] <= 780), axis=0)
]

shapes = ["+", "o", "^", ".", "*", "v", "s", "x"]

loop_n = 0

# precalculate optimal bandgaps for junctions:
save_path = path.join(path.dirname(path.abspath(__file__)), "results")

fixed_height_loop = [True]

fixed_bandgaps = [1.90, 1.44, 0.67]


if __name__ == "__main__":
    # Need this __main__ construction because otherwise the parallel running of the different islands (n_trials) may throw an error

    max_effs = np.zeros((len(n_junc_loop), len(color_XYZ)))
    champion_pop_array = np.zeros((len(n_junc_loop), len(color_XYZ)), dtype=object)
    best_Eg = np.empty((len(n_junc_loop), len(color_XYZ)), dtype=object)

    for n_peaks in n_peak_loop:
        for i1, n_junctions in enumerate(n_junc_loop):

            Eg_guess = None

            for fixed_height in fixed_height_loop:

                save_name = (
                    "results/champion_eff_"
                    + R_type
                    + str(n_peaks)
                    + "_"
                    + str(n_junctions)
                    + "_"
                    + str(fixed_height)
                    + str(max_height)
                    + "_"
                    + str(base)
                    + "_3J.txt"
                )

                if not path.exists(save_name) or force_rerun:
                    print(
                        n_peaks,
                        "peaks,",
                        n_junctions,
                        "junctions,",
                        "fixed height:",
                        fixed_height,
                    )
                    result = multiple_color_cells(
                        color_XYZ,
                        color_names,
                        photon_flux_cell,
                        n_peaks=n_peaks,
                        n_junctions=n_junctions,
                        R_type=R_type,
                        fixed_height=fixed_height,
                        n_trials=n_trials,
                        initial_iters=initial_iters,
                        add_iters=add_iters,
                        col_thresh=col_thresh,
                        acceptable_eff_change=acceptable_eff_change,
                        max_trials_col=max_trials_col,
                        base=base,
                        max_height=max_height,
                        Eg_black=Eg_guess,
                        fixed_bandgaps=fixed_bandgaps,
                        plot=False,
                        return_archipelagos=False,
                        n_reset=2,
                    )

                    champion_effs = result["champion_eff"]
                    champion_pops = result["champion_pop"]

                    np.savetxt(
                        "results/champion_eff_"
                        + R_type
                        + str(n_peaks)
                        + "_"
                        + str(n_junctions)
                        + "_"
                        + str(fixed_height)
                        + str(max_height)
                        + "_"
                        + str(base)
                        + "_3J.txt",
                        champion_effs,
                    )
                    np.savetxt(
                        "results/champion_pop_"
                        + R_type
                        + str(n_peaks)
                        + "_"
                        + str(n_junctions)
                        + "_"
                        + str(fixed_height)
                        + str(max_height)
                        + "_"
                        + str(base)
                        + "_3J.txt",
                        champion_pops,
                    )

                    max_effs[i1] = champion_effs
                    # best_Eg[i1] = champion_pops[-n_junctions:] if n_junctions > 0 else fixed_bandgaps
                    for l1 in range(len(color_XYZ)):
                        # best_Eg[i1, l1] = champion_pops[l1, -n_junctions:] if n_junctions > 0 else fixed_bandgaps
                        champion_pop_array[i1, l1] = champion_pops[l1]

                else:
                    print("Existing saved result found")

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
                        + str(base)
                        + "_3J.txt"
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
                        + str(base)
                        + "_3J.txt"
                    )

                    max_effs[i1] = champion_effs

                    for l1 in range(len(color_XYZ)):
                        # best_Eg[i1, l1] = champion_pops[l1, -n_junctions:] if n_junctions > 0 else fixed_bandgaps
                        champion_pop_array[i1, l1] = champion_pops[l1]


    black_cell_eff = (
        getPmax(fixed_bandgaps, photon_flux_cell[1], wl_cell, interval,
                upperE=1240/min(wl_cell), method="no_R") / 10
    )


    eff_xr = make_sorted_xr(max_effs[0], color_names, append_black=black_cell_eff,
                            ascending=True)
    pop_xr = make_sorted_xr(champion_pops, color_names,
                            append_black=[0, 0, 0, 0], ascending=True)

    pal = sns.color_palette("husl", 3)

    fig, (ax, ax2) = plt.subplots(2, 1,
                           figsize=(5.5, 4.5),
                           gridspec_kw={'height_ratios': [1, 1.65]},
                           )

    # ax2 = ax.twinx()

    J1_c = np.zeros(len(color_XYZ) + 1)
    J2_c = np.zeros(len(color_XYZ) + 1)
    J3_c = np.zeros(len(color_XYZ) + 1)

    for k1 in range(len(pop_xr)):
        spec = gen_spectrum_ndip(pop_xr[k1].data, n_peaks=2, wl=wl_cell)

        if pop_xr[k1].data[0] != 0:
            _, Is = getIVmax(
                fixed_bandgaps, (1 - spec) * photon_flux_cell[1], wl_cell, interval,
                upperE=1240/min(wl_cell), method="perfect_R", n_peaks=2,
                x=pop_xr[k1].data, rad_eff=[1]*3
            )
            print(pop_xr[k1].data)

        else:
            _, Is = getIVmax(
                fixed_bandgaps, photon_flux_cell[1], wl_cell, interval,
                upperE=1240/min(wl_cell), method="no_R",
                rad_eff=[1]*3
            )

        J1_c[k1] = Is[0]
        J2_c[k1] = Is[1]
        J3_c[k1] = Is[2]

    _, Is = getIVmax(fixed_bandgaps, photon_flux_cell[1], wl_cell, interval,
                     upperE=1240 / min(wl_cell), method="perfect_R", n_peaks=2,
                     x=pop_xr[k1].data,
                     rad_eff=[1]*3,
                     )

    ax2.plot(
        eff_xr.color,
        J2_c / 10,
        label="GaAs",
        color=pal[1],
        # marker=shapes[2],
        alpha=0.5,
        linestyle="--",
    )
    ax2.plot(
        eff_xr.color,
        J3_c / 10,
        label="Ge",
        color=pal[2],
        # marker="s",
        alpha=0.5,
        linestyle="--",
    )
    ax2.plot(
        eff_xr.color,
        J1_c / 10,
        label="GaInP",
        color=pal[0],
        marker=shapes[1],
        alpha=0.5,
        linestyle="-",
    )
    ax.plot(
        eff_xr.color.data,
        eff_xr,
        marker='o',
        linestyle="none",
        markersize=6,
        color="k",
    )
    ax.xaxis.set_ticks(ax.get_xticks())
    ax2.set_xticklabels(
        eff_xr.color.data, rotation=45, ha="right", rotation_mode="anchor"
    )
    ax.set_xticklabels([])
    ax.set_ylabel("Efficiency (%)")
    # ax.arrow(
    #     4.5,
    #     41,
    #     -4.1,
    #     0,
    #     head_width=0.7,
    #     head_length=0.5,
    #     overhang=0.2,
    #     color="k",
    #     alpha=0.6,
    # )
    # ax2.arrow(
    #     22,
    #     15,
    #     1.75,
    #     0,
    #     head_width=0.7,
    #     head_length=0.5,
    #     overhang=0.2,
    #     color=pal[0],
    #     alpha=0.6,
    # )
    ax2.set_ylabel(r"$J_{max}$ per junction (mA/cm$^2$)")
    plt.tight_layout()
    ax2.legend(loc=(0.05, 0.58))
    ax.set_ylim(12, 45)
    ax2.set_ylim(0, 30)

    apply_formatting(
        ax2, color_labels=eff_xr.color.data, n_colors=len(eff_xr.color.data)
    )
    apply_formatting(
        ax, n_colors=len(eff_xr.color.data)
    )
    # ax2.yaxis.label.set_color(pal[0])  # setting up Y-axis label color to blue
    ax2.yaxis.set_minor_locator(tck.AutoMinorLocator())
    # ax2.tick_params(direction="in", which="both", top=True, right=True, colors=pal[0])
    # ax2.spines["right"].set_color(pal[0])  # setting up Y-axis tick color to red
    rgb_colors = sRGB_color_list(order="sorted", include_black=True)
    add_colour_patches(ax2, 0.75, eff_xr.color.data, rgb_colors,
                       color_coords='rgb')
    ax.set_title('(a)', loc='left')
    ax2.set_title('(b)', loc='left')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.17)
    plt.show()
