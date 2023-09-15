from ecopv.main_optimization import multiple_color_cells
from colormath.color_objects import xyYColor
from solcore.light_source import LightSource
from os import path
from ecopv.plot_utilities import *
from colour import wavelength_to_XYZ
import os
from time import time

from matplotlib import rc
rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})

# in such a large number of runs, get one or two values which are not converged. Try
# to avoid this by enforcing a minimum efficiency? A point should always be better than
# the one diagonally below to the left of it (i.e. x-1, y-1 coordinates).

force_rerun = False
Ys = [0.25, 0.5, 0.75]
# Ys = [0.75]
wl_vis = np.linspace(360, 780, 500)

XYZ = wavelength_to_XYZ(wl_vis)

sumXYZ = np.sum(XYZ, axis=1)

xg = XYZ[:, 0] / sumXYZ
yg = XYZ[:, 1] / sumXYZ

xs = np.arange(np.min(xg), np.max(xg), 0.02)
ys = np.arange(np.min(yg), np.max(yg), 0.02)

is_inside = np.full((len(xs), len(ys)), False)

peak = np.argmax(yg)

left_edge = [xg[:peak], yg[:peak]]
right_edge = [xg[peak:], yg[peak:]]

# now check if the points are inside the gamut defined by the spectral locus

for j, yc in enumerate(ys):
    left_y = np.argmin(np.abs(left_edge[1] - yc))
    right_y = np.argmin(np.abs(right_edge[1] - yc))

    left_x = left_edge[0][left_y]
    right_x = right_edge[0][right_y]
    is_inside[np.all((xs > left_x, xs < right_x), axis=0), j] = True

# eliminate everything below the line of purples:

# equation for line of purples:

slope = (yg[-1] - yg[0]) / (xg[-1] - xg[0])
c = yg[0] - slope * xg[0]

for j, yc in enumerate(ys):
    above = yc > slope * xs + c
    is_inside[:, j] = np.all((above, is_inside[:, j]), axis=0)

print("Number of points inside gamut:", np.sum(is_inside))

col_thresh = 0.004  # for a wavelength interval of 0.1, minimum achievable color
# error will be (very rough estimate!) ~ 0.001.
# This is the maximum allowed fractional error in X, Y, or Z colour coordinates.

acceptable_eff_change = 1e-4  # how much can the efficiency (in %) change between iteration sets? Stop when have reached
# col_thresh and efficiency change is less than this.

n_trials = 8  # number of islands which will run concurrently in parallel
interval = 0.1  # wavelength interval (in nm)
wl_cell = np.arange(
    300, 4000, interval
)  # wavelengths used for cell calculations (range of wavelengths in AM1.5G solar
# spectrum. For calculations relating to colour perception, only the visible range (
# 380-730 nm) will be used.

n_peaks = 2

iters_multiplier = 50  # additional evolutions added each time if color threshold/convergence condition not met
# every color will run a minimum of initial_iters + add_iters evolutions before ending!

max_trials_col = 750  # how many population evolutions happen before giving up if there are no populations
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

n_junctions = 1  # loop through these numbers of junctions

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

# fixed_bandgaps = [1.90, 1.44, 0.67]
fixed_bandgaps = [1.12]

label_top = ["(a) ", "(b) ", "(c) "]
label_bottom = ["(d) ", "(e) ", "(f) "]

if __name__ == "__main__":
    start = time()
    # Need this __main__ construction because otherwise the parallel running of the different islands (n_trials) may throw an error

    fig, (axs, axs2, axs3) = plt.subplots(3, len(Ys), figsize=(len(Ys) * 4, 8),
                                    gridspec_kw={'height_ratios': [1, 1, 0.1]})
    if len(Ys) == 1:
        axs = [axs]

    for i1, Y in enumerate(Ys):

        if os.path.exists(
            save_path + "/pop_gamut_Y_{}_{}_{}_ERE.npy".format(Y, n_junctions, "Si")
        ) and not force_rerun:
            print("Load existing result")

            pop_array = np.load(
                save_path + "/pop_gamut_Y_{}_{}_{}_ERE.npy".format(Y, n_junctions, "Si")
            )
            eff_array = np.load(
                save_path + "/eff_gamut_Y_{}_{}_{}_ERE.npy".format(Y, n_junctions, "Si")
            )

        else:
            col_possible_str = save_path + "/possible_colours_Y_{}.txt".format(Y)

            print("Found existing file for possible colours")
            is_possible = np.loadtxt(col_possible_str)
            print("Possible colours:", np.sum(is_possible))

            best_population = np.load(
                save_path + "/possible_colours_Y_{}_populations.npy".format(
                Y))
            # print(is_possible)
            eff_array = np.zeros((len(xs), len(ys)))
            pop_array = np.zeros((len(xs), len(ys), 4 + n_junctions))

            for j1, x in enumerate(xs):
                for k1, y in enumerate(ys):
                    if is_inside[j1, k1] and is_possible[j1, k1]:
                        XYZ = np.array(
                            [
                                convert_color(
                                    xyYColor(x, y, Y), XYZColor
                                ).get_value_tuple()
                            ]
                        )
                        print(XYZ)
                        seed_pop = best_population[j1, k1]

                        if j1 > 0 and k1 > 0:
                            minimum_eff = [0.99*eff_array[j1-1, k1-1]]
                        else:
                            minimum_eff = [0]

                        print("minimum eff:", minimum_eff)


                        result = multiple_color_cells(
                            XYZ,
                            [str(j1) + "_" + str(k1)],
                            photon_flux_cell,
                            n_peaks=n_peaks,
                            n_junctions=n_junctions,
                            R_type=R_type,
                            fixed_height=fixed_height,
                            n_trials=n_trials,
                            iters_multiplier=iters_multiplier,
                            col_thresh=col_thresh,
                            col_cutoff=0.05,
                            acceptable_eff_change=acceptable_eff_change,
                            max_trials_col=max_trials_col,
                            base=base,
                            max_height=max_height,
                            Eg_black=np.array([1.74]),
                            fixed_bandgaps=fixed_bandgaps,
                            plot=False,
                            power_in=light_source.power_density,
                            return_archipelagos=True,
                            j01_method=j01_method,
                            seed_population=np.array([seed_pop]),
                            rad_eff=[0.1, 0.016],
                            n_reset=2,
                        )

                        champion_effs = result["champion_eff"]
                        champion_pops = result["champion_pop"]

                        print("champion effs", champion_effs)

                        eff_array[j1, k1] = champion_effs
                        pop_array[j1, k1, :] = champion_pops
                        if np.max(champion_effs) == 0:
                            print("SETTING TO 0")
                            is_possible[j1, k1] = False

            print("sum:", np.sum(is_possible))

            np.save(
                save_path + "/pop_gamut_Y_{}_{}_{}_ERE.npy".format(Y, n_junctions, "Si"),
                pop_array,
            )
            np.save(
                save_path + "/eff_gamut_Y_{}_{}_{}_ERE.npy".format(Y, n_junctions, "Si"),
                eff_array,
            )

        width = np.diff(xs)[0]
        height = np.diff(ys)[0]

        eff_mask = eff_array > 0

        Egs = pop_array[:, :, -1]
        Egs[~eff_mask] = np.nan
        Egs[Egs < 1.12] = np.nan

        standard_illuminant = [0.3128, 0.3290, Y]
        XYZ = convert_color(xyYColor(*standard_illuminant), XYZColor)
        s_i_RGB = np.array(convert_color(XYZ, sRGBColor).get_value_tuple())
        s_i_RGB[s_i_RGB > 1] = 1

        axs[i1].set_facecolor(s_i_RGB)
        wl_gamut_plot(axs[i1], xg, yg)


        for j1, x in enumerate(xs):
            for k1, y in enumerate(ys):

                if ~np.isnan(eff_array[j1, k1]) and eff_array[j1, k1] > 0:
                    XYZ = convert_color(xyYColor(x, y, Y), XYZColor)
                    RGB = np.array(convert_color(XYZ, sRGBColor).get_value_tuple())

                    RGB[RGB > 1] = 1
                    #     plt.plot(x, y, '.')
                    axs[i1].add_patch(
                        Rectangle(
                            xy=(x - width / 2, y - height / 2),
                            width=width,
                            height=height,
                            facecolor=RGB,
                        )
                    )
                    # ax.text(x, y, str(round(eff_array[j1, k1], 2)), color='k', ha='center', va='center')
                #
                # plt.plot(x, y, 'k.')

        # levels = np.arange(np.ceil(np.min(eff_array[eff_array>0]))+2, 1.01*np.floor(np.nanmax(eff_array[eff_array>0])), 1)

        lower_end = np.ceil(0.98 * np.max(eff_array) - 2)
        upper_end = np.ceil(np.max(eff_array) - 1)
        levels = np.arange(lower_end, upper_end, 0.5)

        within_99_p = 0.99 * np.max(eff_array)
        # print(np.min(eff_array[eff_array>0]), np.max(eff_array))
        cs = axs[i1].contour(xs, ys, eff_array.T, levels=levels, colors="k", alpha=0.7)

        cs2 = axs[i1].contour(
            xs, ys, eff_array.T, levels=[within_99_p], colors="firebrick",
            linestyles="dashed"
        )

        label_loc = []

        for seg in cs.allsegs:
            x_lower = seg[0][:, 0][seg[0][:, 1] < 0.6]
            y_lower = seg[0][:, 1][seg[0][:, 1] < 0.6]

            close_x = np.argmin(np.abs(x_lower - 0.35))
            close_y = y_lower[close_x]
            label_loc.append([0.35, close_y])

        max_loc = np.unravel_index(np.argmax(eff_array), eff_array.shape)
        axs[i1].plot(xs[max_loc[0]], ys[max_loc[1]], "o", markersize=8,
                   color="firebrick",
                markerfacecolor="none", markeredgewidth=2)

        if Y < 0.75:
            axs[i1].text(xs[max_loc[0]] + 0.015, ys[max_loc[1]] - 0.025,
                    np.round(eff_array[max_loc], 1),
                    ha="right", va="top", fontsize=14, color="firebrick", weight="bold")

        else:
            axs[i1].text(xs[max_loc[0]] + 0.035, ys[max_loc[1]] + 0.02,
                    np.round(eff_array[max_loc], 1),
                    ha="left", va="center", fontsize=14, color="firebrick",
                    weight="bold")

        axs[i1].clabel(cs, inline=1, fontsize=12, manual=label_loc)
        axs[i1].set_title(label_top[i1] + "Y = " + str(Y), loc="left")
        axs[i1].yaxis.set_minor_locator(tck.AutoMinorLocator())
        axs[i1].xaxis.set_minor_locator(tck.AutoMinorLocator())
        axs[i1].grid(axis="both", color="0.4", alpha=0.5)
        axs[i1].tick_params(direction="in", which="both", top=True, right=True)
        axs[i1].set_axisbelow(True)

        wl_gamut_plot(axs2[i1], xg, yg)
        print(np.nanmin(Egs), np.nanmax(Egs))
        c = axs2[i1].pcolor(xs, ys, Egs.T, vmin=1.54, vmax=1.75, cmap="inferno_r")
        axs3[i1].axis("off")

        axs2[i1].set_title(label_bottom[i1], loc="left")

        if i1 == 1:
            fig.colorbar(c, ax=axs3[i1], orientation="horizontal", fraction=1,
                         ticks=np.arange(1.54, 1.75, 0.05))
            axs3[i1-1].text(1.1, 0.7, "Bandgap (eV)", ha="right", va="center")

        axs2[i1].grid(axis="both", color="0.4", alpha=0.5)


    fig.savefig("efficiency_colour_gamut.pdf", bbox_inches="tight")
    plt.tight_layout()
    plt.show()

    print("TIME: ", time() - start)

