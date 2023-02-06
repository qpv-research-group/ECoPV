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

type = "sharp"  # "sharp" for rectangular dips or "gauss" for gaussians
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


for n_junctions in n_junc_loop:

    if n_junctions > 0:

        save_loc = save_path + "/champion_pop_{}juncs_{}spec_3J.txt".format(
            n_junctions, light_source_name
        )

        if not path.exists(save_loc) or force_rerun:

            p_init = cell_optimization(
                n_junctions,
                photon_flux_cell,
                power_in=light_source.power_density,
                eta_ext=1,
                fixed_bandgaps=fixed_bandgaps,
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

            np.savetxt(save_loc, champion_pop)


if __name__ == "__main__":
    # Need this __main__ construction because otherwise the parallel running of the different islands (n_trials) may throw an error

    max_effs = np.zeros((len(n_junc_loop), len(color_XYZ)))
    champion_pop_array = np.zeros((len(n_junc_loop), len(color_XYZ)), dtype=object)
    best_Eg = np.empty((len(n_junc_loop), len(color_XYZ)), dtype=object)

    for n_peaks in n_peak_loop:
        for i1, n_junctions in enumerate(n_junc_loop):

            if n_junctions > 0:
                Eg_guess = np.loadtxt(
                    save_path
                    + "/champion_pop_{}juncs_{}spec_3J.txt".format(
                        n_junctions, light_source_name
                    ),
                    ndmin=1,
                )

            else:
                Eg_guess = None

            for fixed_height in fixed_height_loop:

                save_name = (
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
                    + "_3J_spd.txt"
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
                        fixed_bandgaps=fixed_bandgaps,
                        plot=False,
                        return_archipelagos=True
                    )

                    champion_effs = result["champion_eff"]
                    champion_pops = result["champion_pop"]

                    final_populations = result["archipelagos"]

                    np.savetxt(
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
                        + "_3J_spd.txt",
                        champion_effs,
                    )
                    np.savetxt(
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
                        + "_3J_spd.txt",
                        champion_pops,
                    )
                    np.save(
                        "results/final_pop_"
                        + type
                        + str(n_peaks)
                        + "_"
                        + str(n_junctions)
                        + "_"
                        + str(fixed_height)
                        + str(max_height)
                        + "_"
                        + str(base)
                        + "_3J_spd.npy",
                        final_populations,
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
                        + type
                        + str(n_peaks)
                        + "_"
                        + str(n_junctions)
                        + "_"
                        + str(fixed_height)
                        + str(max_height)
                        + "_"
                        + str(base)
                        + "_3J_spd.txt"
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
                        + "_3J_spd.txt"
                    )

                    max_effs[i1] = champion_effs

                    for l1 in range(len(color_XYZ)):
                        # best_Eg[i1, l1] = champion_pops[l1, -n_junctions:] if n_junctions > 0 else fixed_bandgaps
                        champion_pop_array[i1, l1] = champion_pops[l1]

    # unconst_1j = np.loadtxt("results/champion_eff_" + type + str(2) + '_' + str(
    #                     1) + '_' + 'True' + str(max_height) + '_' + str(base) + '.txt')
    # unconst_2j = np.loadtxt("results/champion_eff_" + type + str(2) + '_' + str(
    #                     2) + '_' + 'True' + str(max_height) + '_' + str(base) + '.txt')
    # unconst_3j = np.loadtxt("results/champion_eff_" + type + str(2) + '_' + str(
    #     3) + '_' + 'True' + str(max_height) + '_' + str(base) + '.txt')
    #
    # unconst = [unconst_1j, unconst_2j, unconst_3j]
    #
    # black_cell_eff = np.array([33.8, 45.9, 51.8, 55.5, 57.8, 59.7])
    # unconst_xr = []
    #
    # for k1, unc in enumerate(unconst):
    #     eff_xr_col = xr.DataArray(data=unc[:18], dims=['color'],
    #                               coords={'color': Y_cols})
    #
    #     eff_xr_col = eff_xr_col.sortby('color', ascending=False)
    #     eff_xr_col = eff_xr_col.assign_coords(color=col_names.data)
    #
    #     eff_xr_bw = xr.DataArray(data=unc[18:].tolist() + [black_cell_eff[k1]], dims=['color'],
    #                              coords={'color': color_names[18:].tolist() + ['Black']})
    #
    #     unconst_xr.append(xr.concat([eff_xr_col, eff_xr_bw], dim='color'))
    #
    # pal = sns.color_palette("husl", 3)
    # shapes = ['o', '+', '^', '.', '*', "v", "s", "x"]
    #
    # fig, (ax, ax2) = plt.subplots(1,2, figsize=(10,4))
    #
    # black_Eg = [[1.12], [1.72918698, 1.12], [2.0054986, 1.4983, 1.12]]
    #
    #
    # for i1 in n_junc_loop:
    #     eff_xr_col = xr.DataArray(data=max_effs[i1, :18].flatten(), dims=['color'],
    #                               coords={'color': Y_cols})
    #
    #     eff_xr_col = eff_xr_col.sortby('color', ascending=False)
    #     eff_xr_col = eff_xr_col.assign_coords(color=col_names.data)
    #
    #     black_cell_eff = getPmax(black_Eg[i1], photon_flux_cell[1], wl_cell, interval) / 10
    #     print(black_cell_eff)
    #
    #     eff_xr_bw = xr.DataArray(data=max_effs[i1,18:].flatten().tolist() + [black_cell_eff], dims=['color'],
    #                              coords={'color': color_names[18:].tolist() + ['Black']})
    #
    #     eff_xr = xr.concat([eff_xr_col, eff_xr_bw], dim='color')
    #
    #     ax.plot(eff_xr.color.data, eff_xr.data, marker=shapes[i1], linestyle='none',
    #             alpha=0.5, label=str(i1+1) + " junctions", color=pal[i1],
    #             markersize=7)
    #
    #     ax.plot(eff_xr.color.data, unconst_xr[i1], marker=shapes[i1], mfc='none',
    #              color='k', alpha=0.5, linestyle='none', markersize=5)
    #
    #     chp = np.vstack(champion_pop_array[i1]).astype(float)
    #
    #     pop_xr_col = xr.DataArray(data=chp[:18], dims=['color', 'x'],
    #                             coords={'color': Y_cols})
    #
    #     pop_xr_col = pop_xr_col.sortby('color', ascending=False)
    #     pop_xr_col = pop_xr_col.assign_coords(color=col_names.data)
    #
    #     pop_xr_bw = xr.DataArray(
    #         data=chp[18:],
    #         dims=['color', 'x'],
    #         coords={'color': color_names[18:]})
    #
    #     pop_xr = xr.concat([pop_xr_col, pop_xr_bw], dim='color')
    #
    #     I_arr = np.zeros((len(eff_xr.color.data), i1+1))
    #
    #     for l1, lab in enumerate(col_names_all):
    #         pop = pop_xr.sel(color=lab)
    #         spec = gen_spectrum_ndip(pop.data, n_peaks, wl_cell)
    #         Egs = fixed_bandgaps + pop[-i1:].data.tolist() if i1 > 0 else fixed_bandgaps
    #         Egs = -np.sort(-np.array(Egs))
    #         _, I_arr[l1] = getIVmax(Egs, (1 - spec) * photon_flux_cell[1], wl_cell, interval)
    #
    #     I_arr[-1] = getIVmax(black_Eg[i1], photon_flux_cell[1], wl_cell, interval)[1]
    #
    #
    #
    #     if i1 ==2 :
    #         for l1 in range(i1+1):
    #             ax2.plot(eff_xr.color.data, I_arr[:, l1], marker=shapes[l1], color=pal[l1])
    #
    # ax.legend()
    #
    # ax.xaxis.set_ticks(ax.get_xticks())
    # ax.set_xticklabels(eff_xr.color.data, rotation=45, ha='right', rotation_mode='anchor')
    # ax.set_ylabel("Efficiency (%)")
    #
    # plt.tight_layout()
    # plt.show()

    # plotting for triple junction

    black_cell_eff = (
        getPmax(fixed_bandgaps, photon_flux_cell[1], wl_cell, interval, upperE=1240/min(wl_cell)) / 10
    )

    eff_xr = make_sorted_xr(max_effs[0], color_names, black_cell_eff)

    pop_xr = make_sorted_xr(champion_pops, color_names)

    pal = sns.color_palette("husl", 3)

    fig, ax = plt.subplots(1)

    ax2 = ax.twinx()

    J1_c = np.zeros(len(color_XYZ) + 1)
    J2_c = np.zeros(len(color_XYZ) + 1)
    J3_c = np.zeros(len(color_XYZ) + 1)

    for k1 in range(len(color_XYZ)):
        spec = gen_spectrum_ndip(pop_xr[k1].data, n_peaks=2, wl=wl_cell)
        _, Is = getIVmax(
            fixed_bandgaps, (1 - spec) * photon_flux_cell[1], wl_cell, interval
        )
        J1_c[k1] = Is[0]
        J2_c[k1] = Is[1]
        J3_c[k1] = Is[2]

    _, Is = getIVmax(fixed_bandgaps, photon_flux_cell[1], wl_cell, interval)
    J1_c[-1] = Is[0]
    J2_c[-1] = Is[1]
    J3_c[-1] = Is[2]
    ax2.plot(
        eff_xr.color,
        J1_c / 10,
        label="InGaP",
        color=pal[0],
        marker=shapes[1],
        alpha=0.5,
        linestyle="--",
    )
    ax2.plot(
        eff_xr.color,
        J2_c / 10,
        label="GaAs",
        color=pal[0],
        marker=shapes[2],
        alpha=0.5,
        linestyle="--",
    )
    ax2.plot(
        eff_xr.color,
        J3_c / 10,
        label="Ge",
        color=pal[0],
        marker="s",
        alpha=0.5,
        linestyle="--",
    )

    ax.plot(
        eff_xr.color.data,
        eff_xr,
        marker=shapes[0],
        linestyle="none",
        markersize=6,
        color="k",
    )
    ax.xaxis.set_ticks(ax.get_xticks())
    ax.set_xticklabels(
        eff_xr.color.data, rotation=45, ha="right", rotation_mode="anchor"
    )
    ax.set_ylabel("Efficiency (%)")
    ax.arrow(
        4.5,
        41,
        -4.1,
        0,
        head_width=0.7,
        head_length=0.5,
        overhang=0.2,
        color="k",
        alpha=0.6,
    )
    ax2.arrow(
        22,
        15,
        1.75,
        0,
        head_width=0.7,
        head_length=0.5,
        overhang=0.2,
        color=pal[0],
        alpha=0.6,
    )
    ax2.set_ylabel(r"$J_{max}$ per junction")
    ax2.legend(loc=(0.8, 0.1))
    ax.set_ylim(12, 44)
    ax2.set_ylim(0, 35)
    apply_formatting(
        ax, color_labels=eff_xr.color.data, n_colors=len(eff_xr.color.data)
    )
    ax2.yaxis.label.set_color(pal[0])  # setting up Y-axis label color to blue
    ax2.yaxis.set_minor_locator(tck.AutoMinorLocator())
    ax2.tick_params(direction="in", which="both", top=True, right=True, colors=pal[0])
    ax2.spines["right"].set_color(pal[0])  # setting up Y-axis tick color to red
    add_colour_patches(ax, 0.75, eff_xr.color.data)
    plt.tight_layout()
    plt.show()
