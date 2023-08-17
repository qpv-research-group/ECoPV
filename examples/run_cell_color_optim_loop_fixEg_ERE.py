from ecopv.main_optimization import (
    load_colorchecker,
    multiple_color_cells,
    cell_optimization,
)
from ecopv.plot_utilities import add_colour_patches, apply_formatting, sRGB_color_list
from ecopv.optimization_functions import getPmax, getIVmax
from ecopv.spectrum_functions import gen_spectrum_ndip
import numpy as np
from solcore.light_source import LightSource
import matplotlib.pyplot as plt
import pygmo as pg
from os import path
import xarray as xr
import seaborn as sns

from matplotlib import rc

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})

force_rerun = True

col_thresh = 0.004  # for a wavelength interval of 0.1, minimum achievable color error will be (very rough estimate!) ~ 0.001.
# This is the maximum allowed fractional error in X, Y, or Z colour coordinates.

acceptable_eff_change = 1e-4  # how much can the efficiency (in %) change between iteration sets? Stop when have reached
# col_thresh and efficiency change is less than this.

n_trials = 8  # number of islands which will run concurrently
interval = 0.1  # wavelength interval (in nm)
wl_cell = np.arange(
    300, 4000, interval
)  # wavelengths used for cell calculations (range of wavelengths in AM1.5G solar
# spectrum. For calculations relating to colour perception, only the visible range (380-780 nm) will be used.

initial_iters = 100  # number of initial evolutions for the archipelago
add_iters = 100  # additional evolutions added each time if color threshold/convergence condition not met
# every color will run a minimum of initial_iters + add_iters evolutions before ending!

max_trials_col = (
    6 * add_iters
)  # how many population evolutions happen before giving up if there are no populations
# which meet the color threshold

R_type = "sharp"  # "sharp" for rectangular dips or "gauss" for gaussians
fixed_height = True  # fixed height peaks (will be at the value of max_height) if True, or peak height is an optimization
# variable if False
light_source_name = "AM1.5g"
j01_method = "perfect_R"

max_height = (
    1  # maximum height of reflection peaks; fixed at this value of fixed_height = True
)
base = 0  # baseline fixed reflection (fixed at this value for both fixed_height = True and False).

n_junc_loop = [0, 1, 2]  # loop through these numbers of junctions

n_peak_loop = [2]  # loop through these numbers of reflection peaks

ERE_top = 0.1 # external radiative efficiency of the top (non-Si) junctions

patch_width = 0.75

(
    color_names,
    color_XYZ,
) = (
    load_colorchecker(source="BabelColor", output_coords="XYZ",
                      illuminant=light_source_name)
)  # load the names and XYZ coordinates of the 24 default Babel colors

Y = np.hstack((color_XYZ[:, 1], [0]))
Y_cols = Y[:18]
col_names = xr.DataArray(data=color_names[:18], dims=["Y"], coords={"Y": Y_cols})
col_names = col_names.sortby("Y", ascending=True)
col_names_all = col_names.data.tolist() + ['Black'] + color_names[18:].tolist()

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
linetypes = ['-', '--']

loop_n = 0

save_path = path.join(path.dirname(path.abspath(__file__)), "results")

fixed_height_loop = [True]

# fixed_bandgaps = [1.90, 1.44, 0.67]
fixed_bandgaps = [1.12]

for n_junctions in n_junc_loop:

    if n_junctions > 0:

        save_loc = save_path + "/champion_pop_{}juncs_{}spec_Si_ERE.txt".format(
            n_junctions, light_source_name
        )

        if not path.exists(save_loc) or force_rerun:

            p_init = cell_optimization(
                n_junctions,
                photon_flux_cell,
                power_in=light_source.power_density,
                eta_ext=[ERE_top]*n_junctions + [0.02],
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
                    + "/champion_pop_{}juncs_{}spec_Si_ERE".format(
                        n_junctions, light_source_name
                    ),
                    ndmin=1,
                )

            else:
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
                    + "_Si_ERE.txt"
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

                    rad_eff_int = [1.6/100] + [ERE_top]*n_junctions
                    rad_eff_int = rad_eff_int[::-1]
                    print(rad_eff_int)

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
                        power_in=light_source.power_density,
                        return_archipelagos=True,
                        j01_method=j01_method,
                        rad_eff=rad_eff_int,
                    )

                    champion_effs = result["champion_eff"]
                    champion_pops = result["champion_pop"]

                    final_populations = result["archipelagos"]

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
                        + "_Si_ERE.txt",
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
                        + "_Si_ERE.txt",
                        champion_pops,
                    )
                    np.save(
                        "results/final_pop_"
                        + R_type
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
                        + R_type
                        + str(n_peaks)
                        + "_"
                        + str(n_junctions)
                        + "_"
                        + str(fixed_height)
                        + str(max_height)
                        + "_"
                        + str(base)
                        + "_Si_ERE.txt"
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
                        + "_Si_ERE.txt"
                    )

                    max_effs[i1] = champion_effs

                    for l1 in range(len(color_XYZ)):
                        # best_Eg[i1, l1] = champion_pops[l1, -n_junctions:] if n_junctions > 0 else fixed_bandgaps
                        champion_pop_array[i1, l1] = champion_pops[l1]

    unconst_1j = np.loadtxt(
        "results/champion_eff_"
        + R_type
        + str(2)
        + "_"
        + str(1)
        + "_"
        + "True"
        + str(max_height)
        + "_"
        + str(base)
        + "_" + j01_method + light_source_name + ".txt",
    )
    unconst_2j = np.loadtxt(
        "results/champion_eff_"
        + R_type
        + str(2)
        + "_"
        + str(2)
        + "_"
        + "True"
        + str(max_height)
        + "_"
        + str(base)
        + "_" + j01_method + light_source_name + ".txt",
    )
    unconst_3j = np.loadtxt(
        "results/champion_eff_"
        + R_type
        + str(2)
        + "_"
        + str(3)
        + "_"
        + "True"
        + str(max_height)
        + "_"
        + str(base)
        + "_" + j01_method + light_source_name + ".txt",
    )

    unconst = [unconst_1j, unconst_2j, unconst_3j]

    unconst_ERE = []

    for nj in [1, 2, 3]:
        unconst_ERE.append(np.loadtxt(
            "results/champion_eff_"
            + R_type
            + str(2)
            + "_"
            + str(nj)
            + "_"
            + "True"
            + str(max_height)
            + "_"
            + str(base)
            + "_" + j01_method + light_source_name + "ERE.txt",
        ))

    black_cell_eff = np.array([33.8, 45.9, 51.8, 55.5, 57.8, 59.7])
    black_cell_eff_ERE = np.array([30.213, 41.853, 48.017])
    unconst_xr = []
    unconst_xr_ERE = []

    for k1, unc in enumerate(unconst):
        eff_xr_col = xr.DataArray(
            data=unc[:18], dims=["color"], coords={"color": Y_cols}
        )

        eff_xr_col = eff_xr_col.sortby("color", ascending=True)
        eff_xr_col = eff_xr_col.assign_coords(color=col_names.data)

        eff_xr_bw = xr.DataArray(
            data=unc[18:].tolist() + [black_cell_eff[k1]],
            dims=["color"],
            coords={"color": color_names[18:].tolist() + ["Black"]},
        )

        eff_xr_bw.data = eff_xr_bw.data[::-1]
        eff_xr_bw.coords["color"] = eff_xr_bw.coords["color"][::-1]

        unconst_xr.append(xr.concat([eff_xr_col, eff_xr_bw], dim="color"))


        eff_xr_col = xr.DataArray(
            data=unconst_ERE[k1][:18], dims=["color"], coords={"color": Y_cols}
        )

        eff_xr_col = eff_xr_col.sortby("color", ascending=True)
        eff_xr_col = eff_xr_col.assign_coords(color=col_names.data)

        eff_xr_bw = xr.DataArray(
            data=unconst_ERE[k1][18:].tolist() + [black_cell_eff_ERE[k1]],
            dims=["color"],
            coords={"color": color_names[18:].tolist() + ["Black"]},
        )

        eff_xr_bw.data = eff_xr_bw.data[::-1]
        eff_xr_bw.coords["color"] = eff_xr_bw.coords["color"][::-1]

        unconst_xr_ERE.append(xr.concat([eff_xr_col, eff_xr_bw], dim="color"))

    pal = sns.color_palette("husl", 3)
    # pal = np.array([pal[1], pal[0], pal[2]])
    shapes = ["o", "*", "^", ".", "*", "v", "s", "x"]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    black_Eg = [[1.12], [1.73015, 1.12], [2.006471, 1.49885, 1.12]]

    for i1 in n_junc_loop:
        eff_xr_col = xr.DataArray(
            data=max_effs[i1, :18].flatten(), dims=["color"], coords={"color": Y_cols}
        )

        eff_xr_col = eff_xr_col.sortby("color", ascending=True)
        eff_xr_col = eff_xr_col.assign_coords(color=col_names.data)

        black_cell_eff = getPmax(black_Eg[i1], photon_flux_cell[1], wl_cell, interval,
                    upperE=1240/min(wl_cell), method="no_R",
                    rad_eff=[ERE_top]*i1 + [1.6/100]) / 10

        eff_xr_bw = xr.DataArray(
            data=max_effs[i1, 18:].flatten().tolist() + [black_cell_eff],
            dims=["color"],
            coords={"color": color_names[18:].tolist() + ["Black"]},
        )

        eff_xr_bw.data = eff_xr_bw.data[::-1]
        eff_xr_bw.coords["color"] = eff_xr_bw.coords["color"][::-1]

        eff_xr = xr.concat([eff_xr_col, eff_xr_bw], dim="color")

        ax.plot(
            eff_xr.color.data,
            eff_xr.data,
            marker=shapes[i1],
            linestyle="none",
            alpha=0.75,
            label=str(i1 + 1) + " junctions",
            color=pal[i1],
            markersize=9,
        )

        ax.plot(
            eff_xr.color.data,
            unconst_xr[i1],
            marker=shapes[i1],
            mfc="none",
            color="k",
            alpha=0.5,
            linestyle="none",
            markersize=7,
        )

        print("difference", 100*(unconst_xr[i1].data - eff_xr.data)/unconst_xr[i1].data)

        chp = np.vstack(champion_pop_array[i1]).astype(float)

        pop_xr_col = xr.DataArray(
            data=chp[:18], dims=["color", "x"], coords={"color": Y_cols}
        )

        pop_xr_col = pop_xr_col.sortby("color", ascending=True)
        pop_xr_col = pop_xr_col.assign_coords(color=col_names.data)

        pop_xr_bw = xr.DataArray(
            data=chp[18:][::-1], dims=["color", "x"], coords={"color": color_names[
                                                                       18:][::-1]}
        )

        # pop_xr = xr.concat([pop_xr_col, pop_xr_bw], dim="color")

        bw_data = xr.DataArray(data=np.array([[0, 0, 0, 0] + black_Eg[i1][:-1][
                                                             ::-1]]), \
                               dims=["color", "x"], coords={"color": ["Black"]})


        pop_xr = xr.concat([pop_xr_col, bw_data, pop_xr_bw], dim="color")

        I_arr = np.zeros((len(eff_xr.color.data), i1 + 1))

        for l1, lab in enumerate(col_names_all):
            pop = pop_xr.sel(color=lab)

            if pop[0] != 0:
                print(pop)
                spec = gen_spectrum_ndip(pop.data, n_peaks, wl_cell)
                Egs = fixed_bandgaps + pop[-i1:].data.tolist() if i1 > 0 else fixed_bandgaps
                Egs = -np.sort(-np.array(Egs))

                I_arr[l1] = getIVmax(
                    Egs, (1 - spec) * photon_flux_cell[1], wl_cell, interval, pop,
                    n_peaks=2, rad_eff=[ERE_top]*n_junctions + [1.6/100],
                )[1]

            else:
                I_arr[l1] = getIVmax(black_Eg[i1], photon_flux_cell[1], wl_cell,
                                    interval,
                                     rad_eff=[ERE_top] * n_junctions + [1.6 / 100],
                             method="no_R")[1]

        if i1 > 0:
            for l1 in range(1, i1 + 1):
                print("l1", l1)
                ax2.plot(
                    eff_xr.color.data,
                    #I_arr[:, l1] / 10,
                    pop_xr[:, -l1:],
                    marker=shapes[l1],
                    color=pal[i1],
                    linestyle=linetypes[i1-1]
                )

                print(pop_xr[:, -l1:])

    ax.legend(loc=(0.02, 0.05))
    # ax2.legend(["Junction 1", "Junction 2", "Junction 3"])
    # ax2.legend(["2 junctions", "3 junctions"])
    ax.xaxis.set_ticks(ax.get_xticks())
    ax.set_xticklabels(
        eff_xr.color.data, rotation=45, ha="right", rotation_mode="anchor"
    )
    ax2.set_xticklabels(
        eff_xr.color.data, rotation=45, ha="right", rotation_mode="anchor"
    )
    ax.set_ylabel("Efficiency (%)")
    # ax2.set_ylabel(r"J$_{sc}$ (mA/cm$^2$)")
    ax2.set_ylabel(r"Optimal $E_g$ (eV)")
    ax2.text(6, 1.49, r"3J cell - middle cell $E_g$")
    ax2.text(6, 1.71, r"2J cell - top cell $E_g$")
    ax2.text(6, 1.965, r"3J cell - top cell $E_g$")
    ax.set_ylim(17,)
    ax2.set_ylim(1.35,)

    apply_formatting(ax, color_labels=eff_xr.color.data)
    apply_formatting(ax2, color_labels=eff_xr.color.data)

    rgb_colors = sRGB_color_list(order="sorted", include_black=True)
    add_colour_patches(ax, patch_width, eff_xr.color.data, rgb_colors,
                       color_coords='rgb')
    add_colour_patches(ax2, patch_width, eff_xr.color.data, rgb_colors,
                       color_coords='rgb')

    plt.tight_layout()
    plt.show()


    # New plot

    label = ['1 junction', '2 junctions', '3 junctions']

    fig, axes = plt.subplots(3, 1, figsize=(5.5, 6.5), sharex=True)

    for i1 in n_junc_loop:
        eff_xr_col = xr.DataArray(
            data=max_effs[i1, :18].flatten(), dims=["color"], coords={"color": Y_cols}
        )

        eff_xr_col = eff_xr_col.sortby("color", ascending=True)
        eff_xr_col = eff_xr_col.assign_coords(color=col_names.data)

        black_cell_eff = getPmax(black_Eg[i1], photon_flux_cell[1], wl_cell, interval,
                                 upperE=1240 / min(wl_cell), method="no_R",
                                 rad_eff=[ERE_top] * i1 + [1.6 / 100]) / 10

        eff_xr_bw = xr.DataArray(
            data=max_effs[i1, 18:].flatten().tolist() + [black_cell_eff],
            dims=["color"],
            coords={"color": color_names[18:].tolist() + ["Black"]},
        )

        eff_xr_bw.data = eff_xr_bw.data[::-1]
        eff_xr_bw.coords["color"] = eff_xr_bw.coords["color"][::-1]

        eff_xr = xr.concat([eff_xr_col, eff_xr_bw], dim="color")

        axes[i1].plot(
            eff_xr.color.data,
            eff_xr.data,
            mfc='none',
            marker=shapes[0],
            color=pal[0],
        )

        axes[i1].plot(
            eff_xr.color.data,
            unconst_xr_ERE[i1],
            mfc='none',
            marker=shapes[1],
            color=pal[1],
        )

        axes[i1].plot(
            eff_xr.color.data,
            unconst_xr[i1],
            mfc='none',
            marker=shapes[2],
            color=pal[2],
        )

        axes[i1].grid('both')
        axes[i1].set_ylim(0.9*np.min(eff_xr.data), 1.1*np.max(unconst_xr[i1]))
        axes[i1].text(0.1, 1.02*np.max(unconst_xr[i1]), label[i1])
        apply_formatting(axes[i1], n_colors=len(eff_xr.color.data))
        axes[i1].set_ylabel("Efficiency (%)")

    axes[1].legend(['Constrained $E_g$, limit ERE', 'Unconstrained $E_g$, limit ERE',
                    'Unconstrained $E_g$, ERE = 100%'],
                   labelspacing=0.1)

    apply_formatting(axes[2], color_labels=eff_xr.color.data)
    plt.tight_layout()
    add_colour_patches(axes[2], patch_width, eff_xr.color.data, rgb_colors,
                       color_coords='rgb')
    plt.subplots_adjust(wspace=0, hspace=0.05)
    plt.show()



    # New plot - fractional loss

    label = ['1 junction', '2 junctions', '3 junctions']

    fig, axes = plt.subplots(3, 1, figsize=(5.5, 6.5), sharex=True)

    for i1 in n_junc_loop:
        eff_xr_col = xr.DataArray(
            data=max_effs[i1, :18].flatten(), dims=["color"], coords={"color": Y_cols}
        )

        eff_xr_col = eff_xr_col.sortby("color", ascending=True)
        eff_xr_col = eff_xr_col.assign_coords(color=col_names.data)

        black_cell_eff = getPmax(black_Eg[i1], photon_flux_cell[1], wl_cell, interval,
                                 upperE=1240 / min(wl_cell), method="no_R",
                                 rad_eff=[ERE_top] * i1 + [1.6 / 100]) / 10

        eff_xr_bw = xr.DataArray(
            data=max_effs[i1, 18:].flatten().tolist() + [black_cell_eff],
            dims=["color"],
            coords={"color": color_names[18:].tolist() + ["Black"]},
        )

        eff_xr_bw.data = eff_xr_bw.data[::-1]
        eff_xr_bw.coords["color"] = eff_xr_bw.coords["color"][::-1]

        eff_xr = xr.concat([eff_xr_col, eff_xr_bw], dim="color")

        axes[i1].plot(
            eff_xr.color.data,
            eff_xr.data/unconst_xr_ERE[i1].data,
            mfc='none',
            marker=shapes[i1],
            color=pal[0],
        )

        axes[i1].plot(
            eff_xr.color.data,
            unconst_xr_ERE[i1]/unconst_xr_ERE[i1].data,
            mfc='none',
            marker=shapes[i1],
            color=pal[1],
        )

        axes[i1].grid('both')
        # axes[i1].set_ylim(0.9*np.min(eff_xr.data), 1.1*np.max(unconst_xr[i1]))
        # axes[i1].text(0.1, 1.02*np.max(unconst_xr[i1]), label[i1])
        apply_formatting(axes[i1], n_colors=len(eff_xr.color.data))
        axes[i1].set_ylabel("Efficiency (%)")

    axes[1].legend(['Constrained $E_g$, limit ERE', 'Unconstrained $E_g$, limit ERE',
                    'Unconstrained $E_g$, ERE = 100%'],
                   labelspacing=0.1)

    apply_formatting(axes[2], color_labels=eff_xr.color.data)
    plt.tight_layout()
    add_colour_patches(axes[2], patch_width, eff_xr.color.data, rgb_colors,
                       color_coords='rgb')
    plt.subplots_adjust(wspace=0, hspace=0.05)
    plt.show()



    label = ['1 junction', '2 junctions', '3 junctions']

    fig, axes = plt.subplots(3, 1, figsize=(5.5, 6.5), sharex=True)

    eff_xr_list = []

    for i1 in n_junc_loop:
        eff_xr_col = xr.DataArray(
            data=max_effs[i1, :18].flatten(), dims=["color"], coords={"color": Y_cols}
        )

        eff_xr_col = eff_xr_col.sortby("color", ascending=True)
        eff_xr_col = eff_xr_col.assign_coords(color=col_names.data)

        black_cell_eff = getPmax(black_Eg[i1], photon_flux_cell[1], wl_cell, interval,
                                 upperE=1240 / min(wl_cell), method="no_R",
                                 rad_eff=[ERE_top] * i1 + [1.6 / 100]) / 10

        eff_xr_bw = xr.DataArray(
            data=max_effs[i1, 18:].flatten().tolist() + [black_cell_eff],
            dims=["color"],
            coords={"color": color_names[18:].tolist() + ["Black"]},
        )

        eff_xr_bw.data = eff_xr_bw.data[::-1]
        eff_xr_bw.coords["color"] = eff_xr_bw.coords["color"][::-1]

        eff_xr = xr.concat([eff_xr_col, eff_xr_bw], dim="color")

        eff_xr_list.append(eff_xr)

        if i1 > 0:
            axes[i1].plot(
                eff_xr.color.data,
                eff_xr.data/eff_xr_list[0].data,
                mfc='none',
                marker=shapes[i1],
                color=pal[0],
            )


            axes[i1].grid('both')
            # axes[i1].set_ylim(0.9*np.min(eff_xr.data), 1.1*np.max(unconst_xr[i1]))
            # axes[i1].text(0.1, 1.02*np.max(unconst_xr[i1]), label[i1])
            apply_formatting(axes[i1], n_colors=len(eff_xr.color.data))
            axes[i1].set_ylabel("Efficiency (%)")

    axes[1].legend(['Constrained $E_g$, limit ERE', 'Unconstrained $E_g$, limit ERE',
                    'Unconstrained $E_g$, ERE = 100%'],
                   labelspacing=0.1)

    apply_formatting(axes[2], color_labels=eff_xr.color.data)
    plt.tight_layout()
    add_colour_patches(axes[2], patch_width, eff_xr.color.data, rgb_colors,
                       color_coords='rgb')
    plt.subplots_adjust(wspace=0, hspace=0.05)
    plt.show()

