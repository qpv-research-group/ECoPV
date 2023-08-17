from ecopv.main_optimization import (
    multiple_color_cells,
)
from colormath.color_objects import sRGBColor, xyYColor
from solcore.light_source import LightSource
from os import path
from cycler import cycler
from ecopv.plot_utilities import *
from matplotlib.patches import Rectangle

force_rerun = False
calc = True

col_thresh = 0.004  # for a wavelength interval of 0.1, minimum achievable color error will be (very rough estimate!) ~ 0.001.
# This is the maximum allowed fractional error in X, Y, or Z colour coordinates.

acceptable_eff_change = 1e-4  # how much can the efficiency (in %) change between iteration sets? Stop when have reached
# col_thresh and efficiency change is less than this.

n_trials = 6  # number of islands which will run concurrently in parallel
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

R_type = "sharp"  # "sharp" for rectangular dips or "gauss" for gaussians
fixed_height = True  # fixed height peaks (will be at the value of max_height) if True, or peak height is an optimization
# variable if False
light_source_name = "AM1.5g"

max_height = (
    1  # maximum height of reflection peaks; fixed at this value of fixed_height = True
)
base = 0  # baseline fixed reflection (fixed at this value for both fixed_height = True and False).

n_junc_loop = [1, 2, 3, 4, 5]  # loop through these numbers of junctions

inds = [10, 12, 14, 18]

color_names, color_xyY = load_colorchecker(
    output_coords="xyY", illuminant="AM1.5g"
)  # load the names and XYZ coordinates of the 24 default Babel colors
color_names_pre = color_names[inds]
color_xyY = color_xyY[inds]

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

shapes = ["+", "o", "^", ".", "*", "v", "s", "x"]

loop_n = 0

# precalculate optimal bandgaps for junctions:
save_path = path.join(path.dirname(path.abspath(__file__)), "results")

# for n_junctions in n_junc_loop:
#
#     save_loc = save_path + "/champion_pop_{}juncs_{}spec.txt".format(
#         n_junctions, light_source_name
#     )
#
#     if not path.exists(save_loc) or force_rerun:
#
#         p_init = cell_optimization(
#             n_junctions,
#             photon_flux_cell,
#             power_in=light_source.power_density,
#             eta_ext=1,
#         )
#
#         prob = pg.problem(p_init)
#         algo = pg.algorithm(
#             pg.de(
#                 gen=1000,
#                 F=1,
#                 CR=1,
#             )
#         )
#
#         pop = pg.population(prob, 20 * n_junctions)
#         pop = algo.evolve(pop)
#
#         champion_pop = np.sort(pop.champion_x)
#
#         np.savetxt(
#             save_path
#             + "/champion_pop_{}juncs_{}spec.txt".format(n_junctions, light_source_name),
#             champion_pop,
#         )


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
                    + R_type
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

                if not path.exists(save_loc) or force_rerun:

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
                        plot=False,
                        power_in=light_source.power_density,
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
                        + R_type
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
                        + R_type
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
                        + R_type
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

                else:
                    print("Result already exists")
                    champion_effs = np.loadtxt(
                                                "results/champion_eff_Y_"
                        + color_suff[j1]
                        + "_"
                        + R_type
                        + str(n_peaks)
                        + "_"
                        + str(n_junctions)
                        + "_"
                        + str(fixed_height)
                        + str(max_height)
                        + "_"
                        + str(base)
                        + "_spd.txt")

                    champion_pops = np.loadtxt("results/champion_pop_Y_"
                        + color_suff[j1]
                        + "_"
                        + R_type
                        + str(n_peaks)
                        + "_"
                        + str(n_junctions)
                        + "_"
                        + str(fixed_height)
                        + str(max_height)
                        + "_"
                        + str(base)
                        + "_spd.txt")

                    champion_effs_array[i1, j1, poss_colors] = champion_effs

                    for ind, l1 in enumerate(poss_colors):
                        champion_pops_array[i1, j1, l1] = champion_pops[ind]

                    poss_colors = np.delete(poss_colors, np.where(champion_effs == 0))


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

    color_labels = ["YellowGreen", "Blue", "Red", "Greyscale"]

    shapes = ["o", "+", "^", "*", "v", "."]
    linestyle = ["-", "--", "-.", ":", "-", "--"]

    color_sRGB_array = np.zeros((len(Y_values), len(color_xyY), 3))

    for j1, Y in enumerate(Y_values):
        xyY_coords = color_xyY
        color_xyY[:, 2] = Y
        color_sRGB_array[j1] = np.array(
            [convert_color(xyYColor(*x), sRGBColor).get_value_tuple() for x in color_xyY])

    color_sRGB_array[color_sRGB_array > 1] = 1
    pal = color_sRGB_array[int(len(color_sRGB_array)/2)] # use actual colours!
    cols = cycler("color", pal)
    params = {"axes.prop_cycle": cols}
    plt.rcParams.update(params)

    fig, (ax1, ax4, ax2, ax3) = plt.subplots(4, 1,
                                        gridspec_kw={"height_ratios": [1, 0.1, 1, 0.2],
                                                   "wspace": 0, "hspace": 0.4,
                                                },
                                        figsize=(5.5, 10.5))

    deltaY = np.diff(Y_values)[0]

    for j1, Y in enumerate(Y_values):
        for j2 in range(len(color_xyY)):
            print(j1, j2)
            ax3.add_patch(
                Rectangle(
                    xy=(Y - 0.2*deltaY, j2),
                    width=0.8*deltaY,
                    height=0.8,
                    facecolor=color_sRGB_array[j1, j2],
                )
            )
    ax3.set_yticks([0.5, 1.5, 2.5, 3.5])
    ax3.set_yticklabels(['YellowGreen', 'Blue', 'Red', 'Greyscale'])
    ax3.set_ylim(0, 4)
    # ax3.set_xlim(0, len(Y_values))


    for i1, n_junc in enumerate(n_junc_loop):
        plot_data = np.insert(champion_effs_array[i1], 0, black_cell_eff[i1], axis=0)

        ax1.plot(
            Y_plot, plot_data, linestyle=linestyle[i1], marker=shapes[i1], mfc="none"
        )

        ax4.plot(
            0,
            1,
            "k",
            linestyle=linestyle[i1],
            marker=shapes[i1],
            mfc="none",
            label=str(n_junc),
        )

    ax4_b = ax4.twinx()
    for j1 in range(len(color_xyY)):
        ax4_b.plot(0, 1, "-o", mfc="none", color=pal[j1], label=color_labels[j1])


    ax1.grid(axis="both", color="0.8")
    # ax1.set_xlabel("Luminance (Y)")
    ax1.set_ylabel("Efficiency (%)")
    ax1.set_xlim(-0.03, 1)
    ax1.set_ylim(23, 58)

    ax4.legend(title="Junctions:", bbox_to_anchor=(-.05, -1.7), loc=(0, 0), ncol=3)#, frameon=False)
    ax4.set_xlim(-2, -1)

    ax4_b.legend(title="Colour:", bbox_to_anchor=(0.45, -1.7), loc=(0, 0), ncol=2)#, frameon=False)
    ax4.axis('off')
    ax4_b.axis('off')

    plt.tight_layout()
    # fig.savefig("fig5.pdf", bbox_inches="tight")
    # plt.show()

    # fig, ax = plt.subplots()
    for i1, n_junc in enumerate(n_junc_loop):
        plot_data = np.insert(
            champion_effs_array[n_junc - 1], 0, black_cell_eff[n_junc - 1], axis=0
        )
        ax2.plot(
            Y_plot,
            plot_data / np.nanmax(plot_data),
            linestyle=linestyle[i1],
            marker=shapes[i1],
            mfc="none",
        )

    #     ax.plot(
    #         0,
    #         2,
    #         "k",
    #         linestyle=linestyle[i1],
    #         marker=shapes[i1],
    #         mfc="none",
    #         label=str(n_junc),
    #     )
    #
    # ax1 = ax.twinx()
    # for j1 in range(len(color_xyY)):
    #     ax1.plot(-100, -100, "-o", mfc="none", color=pal[j1], label=color_labels[j1])

    # ax1.set_ylim(1, 1.1)
    # ax1.get_yaxis().set_visible(False)
    ax1.set_xlabel("Luminance (Y)")
    ax2.grid(axis="both", color="0.8")
    ax2.set_xlabel("Luminance (Y)")
    ax2.set_ylabel("Normalized efficiency")
    ax2.set_xlim(-0.03, 1)
    ax2.set_ylim(0.64, 1.02)

    # ax2.legend(title="Junctions:", bbox_to_anchor=(1.22, 0.8), frameon=False)
    # plt.tight_layout()
    # ax1.legend(title="Colour:", bbox_to_anchor=(1.25, 0.4), frameon=False)

    ax3.axis("off")
    plt.tight_layout()


    plt.show()

