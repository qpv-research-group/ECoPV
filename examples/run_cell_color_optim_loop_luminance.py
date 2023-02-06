from ecopv.main_optimization import (
    load_colorchecker,
    multiple_color_cells,
    cell_optimization,
)
from ecopv.optimization_functions import getIVtandem
import numpy as np
from colormath.color_conversions import convert_color
from colormath.color_objects import xyYColor, XYZColor
from solcore.light_source import LightSource
import matplotlib.pyplot as plt
import pygmo as pg
from os import path
from cycler import cycler
from ecopv.plot_utilities import *

force_rerun = True
calc = True

col_thresh = 0.004  # for a wavelength interval of 0.1, minimum achievable color error will be (very rough estimate!) ~ 0.001.
# This is the maximum allowed fractional error in X, Y, or Z colour coordinates.

acceptable_eff_change = 1e-4  # how much can the efficiency (in %) change between iteration sets? Stop when have reached
# col_thresh and efficiency change is less than this.

n_trials = 10  # number of islands which will run concurrently in parallel
interval = 0.1  # wavelength interval (in nm)
wl_cell = np.arange(
    300, 4000, interval
)  # wavelengths used for cell calculations (range of wavelengths in AM1.5G solar
# spectrum. For calculations relating to colour perception, only the visible range (380-780 nm) will be used.

n_peaks = 2

initial_iters = 100  # number of initial evolutions for the archipelago
add_iters = 100  # additional evolutions added each time if color threshold/convergence condition not met
# every color will run a minimum of initial_iters + add_iters evolutions before ending!

max_trials_col = (
    5 * add_iters
)  # how many population evolutions happen before giving up if there are no populations
# which meet the color threshold

type = "sharp"  # "sharp" for rectangular dips or "gauss" for gaussians
fixed_height = True  # fixed height peaks (will be at the value of max_height) if True, or peak height is an optimization
# variable if False
light_source_name = "AM1.5g"

max_height = (
    1  # maximum height of reflection peaks; fixed at this value of fixed_height = True
)
base = 0  # baseline fixed reflection (fixed at this value for both fixed_height = True and False).

n_junc_loop = [1, 2, 3, 4]  # loop through these numbers of junctions

color_names, color_xyY = load_colorchecker(
    output_coords="xyY"
)  # load the names and XYZ coordinates of the 24 default Babel colors
color_names_pre = color_names[[12, 13, 14, 18]]
color_xyY = color_xyY[[12, 13, 14, 18]]

Y_values = np.linspace(0.02, 0.91, 20)  # loop through these Y values (luminance)
color_suff = [str(np.round(x, 2)) for x in Y_values]

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


for n_junctions in n_junc_loop:

    save_loc = save_path + "/champion_pop_{}juncs_{}spec.txt".format(
        n_junctions, light_source_name
    )

    if not path.exists(save_loc) or force_rerun:

        p_init = cell_optimization(
            n_junctions,
            photon_flux_cell,
            power_in=light_source.power_density,
            eta_ext=1,
        )

        prob = pg.problem(p_init)
        algo = pg.algorithm(
            pg.de(
                gen=1000,
                F=1,
                CR=1,
            )
        )

        pop = pg.population(prob, 20 * n_junctions)
        pop = algo.evolve(pop)

        champion_pop = np.sort(pop.champion_x)

        np.savetxt(
            save_path
            + "/champion_pop_{}juncs_{}spec.txt".format(n_junctions, light_source_name),
            champion_pop,
        )


champion_effs_array = np.zeros((len(n_junc_loop), len(Y_values), len(color_xyY)))
champion_pops_array = np.empty(
    (len(n_junc_loop), len(Y_values), len(color_xyY)), dtype=object
)


if __name__ == "__main__":
    # Need this __main__ construction because otherwise the parallel running of the different islands (n_trials) may throw an error
    if calc:

        for i1, n_junctions in enumerate(n_junc_loop):

            poss_colors = np.arange(0, len(color_xyY))

            for j1, Y in enumerate(Y_values):

                save_loc = (
                    "results/champion_eff_Y_"
                    + color_suff[j1]
                    + "_"
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

                # if not path.exists(save_loc):

                xyY_coords = color_xyY[poss_colors]
                color_xyY[:, 2] = Y

                color_XYZ = np.array(
                    [
                        convert_color(xyYColor(*x), XYZColor).get_value_tuple()
                        for x in color_xyY
                    ]
                )
                # color_XYZ[color_XYZ > 1] = 1
                color_XYZ = color_XYZ[poss_colors]
                # print(color_XYZ)
                Eg_guess = np.loadtxt(
                    save_path
                    + "/champion_pop_{}juncs_{}spec.txt".format(
                        n_junctions, light_source_name
                    ),
                    ndmin=1,
                )

                color_names = [
                    x + "_" + color_suff[j1] for x in color_names_pre[poss_colors]
                ]
                print("still possible: ", color_names)

                print(n_junctions, "junctions,", "fixed height:", fixed_height)
                result = multiple_color_cells(
                    color_XYZ,
                    color_names,
                    photon_flux_cell,
                    n_peaks=n_peaks,
                    n_junctions=n_junctions,
                    type=type,
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
                    plot=False,
                    return_archipelagos=True
                )

                champion_effs = result["champion_eff"]
                champion_pops = result["champion_pop"]

                final_populations = result["archipelagos"]

                champion_effs_array[i1, j1, poss_colors] = champion_effs

                for ind, l1 in enumerate(poss_colors):
                    champion_pops_array[i1, j1, l1] = champion_pops[ind]

                poss_colors = np.delete(poss_colors, np.where(champion_effs == 0))

                np.savetxt(
                    "results/champion_eff_Y_"
                    + color_suff[j1]
                    + "_"
                    + type
                    + str(n_peaks)
                    + "_"
                    + str(n_junctions)
                    + "_"
                    + str(fixed_height)
                    + str(max_height)
                    + "_"
                    + str(base)
                    + "_spd.txt",
                    champion_effs,
                )
                np.savetxt(
                    "results/champion_pop_Y_"
                    + color_suff[j1]
                    + "_"
                    + type
                    + str(n_peaks)
                    + "_"
                    + str(n_junctions)
                    + "_"
                    + str(fixed_height)
                    + str(max_height)
                    + "_"
                    + str(base)
                    + "_spd.txt",
                    champion_pops,
                )
                np.save(
                    "results/final_pop_Y_"
                    + color_suff[j1]
                    + "_"
                    + type
                    + str(n_peaks)
                    + "_"
                    + str(n_junctions)
                    + "_"
                    + str(fixed_height)
                    + str(max_height)
                    + "_"
                    + str(base)
                    + "_spd.npy",
                    final_populations,
                )

                # else:
                #     print("Result already exists")

        np.save("results/champion_effs_array_spd.npy", champion_effs_array)
        np.save("results/champion_pops_array_spd.npy", champion_pops_array)

    else:
        champion_effs_array = np.load(
            "results/champion_effs_array_spd.npy", allow_pickle=True
        )
        champion_pops_array = np.load(
            "results/champion_pops_array_spd.npy", allow_pickle=True
        )

    zs = champion_effs_array == 0
    champion_effs_array[zs] = np.nan

    black_cell_eff = np.array([33.8, 45.9, 51.8, 55.5, 57.8, 59.7])

    Y_plot = np.insert(Y_values, 0, 0)

    pal = ["blue", "green", "red", "gray"]
    cols = cycler("color", pal)
    params = {"axes.prop_cycle": cols}
    plt.rcParams.update(params)

    color_labels = ["Blue", "Green", "Red", "Grey"]

    shapes = ["o", "+", "^", "*", "v", "."]
    linestyle = ["-", "--", "-.", ":", "-", "--"]
    fig, ax = plt.subplots()
    for i1, n_junc in enumerate([1, 2, 3, 4]):
        plot_data = np.insert(champion_effs_array[i1], 0, black_cell_eff[i1], axis=0)

        ax.plot(
            Y_plot, plot_data, linestyle=linestyle[i1], marker=shapes[i1], mfc="none"
        )

        ax.plot(
            0,
            1,
            "k",
            linestyle=linestyle[i1],
            marker=shapes[i1],
            mfc="none",
            label=str(n_junc),
        )

    ax1 = ax.twinx()
    for j1 in range(len(color_xyY)):
        ax1.plot(-100, -100, "-o", mfc="none", color=pal[j1], label=color_labels[j1])

    ax1.set_ylim(1, 1.1)
    ax1.get_yaxis().set_visible(False)
    ax.grid(axis="both", color="0.8")
    ax.set_xlabel("Luminance (Y)")
    ax.set_ylabel("Efficiency (%)")
    ax.set_xlim(-0.03, 1)
    ax.set_ylim(23, 56)
    ax.legend(title="Junctions:", bbox_to_anchor=(1.22, 0.8), frameon=False)
    plt.tight_layout()
    ax1.legend(title="Colour:", bbox_to_anchor=(1.25, 0.4), frameon=False)
    plt.tight_layout()
    fig.savefig("fig5.pdf", bbox_inches="tight")
    plt.show()

    # Red, green, blue and greys: single junction has less power loss than higher number of junctions. Why?

    fig, ax = plt.subplots()
    for i1, n_junc in enumerate([1, 2, 3, 4]):
        plot_data = np.insert(
            champion_effs_array[n_junc - 1], 0, black_cell_eff[n_junc - 1], axis=0
        )
        ax.plot(
            Y_plot,
            plot_data / np.nanmax(plot_data),
            linestyle=linestyle[i1],
            marker=shapes[i1],
            mfc="none",
        )

        ax.plot(
            0,
            2,
            "k",
            linestyle=linestyle[i1],
            marker=shapes[i1],
            mfc="none",
            label=str(n_junc),
        )

    ax1 = ax.twinx()
    for j1 in range(len(color_xyY)):
        ax1.plot(-100, -100, "-o", mfc="none", color=pal[j1], label=color_labels[j1])

    ax1.set_ylim(1, 1.1)
    ax1.get_yaxis().set_visible(False)
    ax.grid(axis="both", color="0.8")
    ax.set_xlabel("Luminance (Y)")
    ax.set_ylabel("Normalized efficiency")
    ax.set_xlim(-0.03, 1)
    ax.set_ylim(0.64, 1.02)

    ax.legend(title="Junctions:", bbox_to_anchor=(1.22, 0.8), frameon=False)
    plt.tight_layout()
    ax1.legend(title="Colour:", bbox_to_anchor=(1.25, 0.4), frameon=False)
    plt.tight_layout()

    plt.show()

    fig, axes = plt.subplots(2)
    for i1, n_junc in enumerate([1, 2, 3, 4]):
        pops = champion_pops_array[i1].flatten()
        IVs = np.zeros((len(pops), 2))
        n_params = len(pops[0])
        for l1 in range(len(pops)):
            if pops[l1] is None:
                pops[l1] = np.full((n_params), np.nan)
                IVs[l1] = np.nan

            else:
                IVs[l1] = getIVtandem(
                    pops[l1][-n_junc:][::-1],
                    photon_flux_cell[1],
                    photon_flux_cell[0],
                    interval,
                )

        pops = np.vstack(pops)
        pops = np.reshape(pops, (20, 4, n_params))
        IVs = np.reshape(IVs, (20, 4, 2))
        axes[0].plot(
            Y_values,
            IVs[:, :, 1],
            linestyle=linestyle[i1],
            marker=shapes[i1],
            mfc="none",
        )

        axes[0].plot(
            0,
            1,
            "k",
            linestyle=linestyle[i1],
            marker=shapes[i1],
            mfc="none",
            label=str(n_junc),
        )

        axes[1].plot(
            Y_values,
            IVs[:, :, 0],
            linestyle=linestyle[i1],
            marker=shapes[i1],
            mfc="none",
        )

    for j1 in range(len(color_xyY)):
        axes[1].plot(
            -100, -100, "-o", mfc="none", color=pal[j1], label=color_labels[j1]
        )

    axes[0].grid(axis="both", color="0.8")
    axes[1].grid(axis="both", color="0.8")
    axes[0].set_xlabel("Jsc")
    axes[1].set_ylabel("Voc")
    axes[0].set_xlim(-0.03, 1)
    axes[1].set_xlim(-0.03, 1)
    axes[0].set_ylim(50, 450)
    axes[1].set_ylim(0.5, 4)

    # ax.set_ylim(0.5, 1)
    axes[0].legend(title="Junctions:", bbox_to_anchor=(1.22, 0.8), frameon=False)
    plt.tight_layout()
    axes[1].legend(title="Colour:", bbox_to_anchor=(1.25, 0.4), frameon=False)
    plt.tight_layout()

    plt.show()
