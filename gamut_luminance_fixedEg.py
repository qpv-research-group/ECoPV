from color_cell_optimization import load_babel, multiple_color_cells, cell_optimization, getIVtandem
import numpy as np
from colormath.color_conversions import convert_color
from colormath.color_objects import xyYColor, XYZColor
from solcore.light_source import LightSource
import matplotlib.pyplot as plt
import pygmo as pg
from os import path
from cycler import cycler
from plot_utilities import *
from colour import wavelength_to_XYZ
import os

Ys = [0.25, 0.5, 0.75]

wl_vis = np.linspace(360, 780, 500)

XYZ = wavelength_to_XYZ(wl_vis)

sumXYZ = np.sum(XYZ, axis=1)

xg = XYZ[:,0]/sumXYZ
yg = XYZ[:,1]/sumXYZ

xs = np.arange(np.min(xg), np.max(xg), 0.01)
ys = np.arange(np.min(yg), np.max(yg), 0.01)

is_inside = np.full((len(xs), len(ys)), False)

peak = np.argmax(yg)

left_edge = [xg[:peak], yg[:peak]]
right_edge = [xg[peak:], yg[peak:]]


# now check if the points are inside the gamut defined by the spectral locus

for j, yc in enumerate(ys):
    left_y = np.argmin(np.abs(left_edge[1]-yc))
    right_y = np.argmin(np.abs(right_edge[1]-yc))

    left_x = left_edge[0][left_y]
    right_x = right_edge[0][right_y]
    is_inside[np.all((xs > left_x, xs < right_x), axis=0), j] = True

# eliminate everything below the line of purples:

# equation for line of purples:

slope = (yg[-1] - yg[0])/(xg[-1] - xg[0])
c = yg[0] - slope*xg[0]

for j, yc in enumerate(ys):
    above = yc > slope*xs + c
    is_inside[:, j] = np.all((above, is_inside[:,j]), axis=0)

# plt.figure()
# plt.plot(xg, yg)
# for j1, x in enumerate(xs):
#     for k1, y in enumerate(ys):
#         if is_inside[j1, k1]:
#             plt.plot(x, y, "o", color="black")
#
# plt.show()
print("Inside gamut:", np.sum(is_inside))

col_thresh = 0.004 # for a wavelength interval of 0.1, minimum achievable color error will be (very rough estimate!) ~ 0.001.
# This is the maximum allowed fractional error in X, Y, or Z colour coordinates.

acceptable_eff_change = 1e-4 # how much can the efficiency (in %) change between iteration sets? Stop when have reached
# col_thresh and efficiency change is less than this.

n_trials = 10 # number of islands which will run concurrently in parallel
interval = 0.1 # wavelength interval (in nm)
wl_cell = np.arange(300, 4000, interval) # wavelengths used for cell calculations (range of wavelengths in AM1.5G solar
# spectrum. For calculations relating to colour perception, only the visible range (380-780 nm) will be used.

n_peaks = 2

initial_iters = 100 # number of initial evolutions for the archipelago
add_iters = 100 # additional evolutions added each time if color threshold/convergence condition not met
# every color will run a minimum of initial_iters + add_iters evolutions before ending!

max_trials_col = 3*add_iters  # how many population evolutions happen before giving up if there are no populations
                              # which meet the color threshold

type = "sharp" # "sharp" for rectangular dips or "gauss" for gaussians
fixed_height = True # fixed height peaks (will be at the value of max_height) if True, or peak height is an optimization
# variable if False
light_source_name = "AM1.5g"

max_height = 1 # maximum height of reflection peaks; fixed at this value of fixed_height = True
base = 0 # baseline fixed reflection (fixed at this value for both fixed_height = True and False).

n_junctions = 0 # loop through these numbers of junctions

# Use AM1.5G spectrum:
light_source = LightSource(
    source_type="standard", version=light_source_name, x=wl_cell, output_units="photon_flux_per_nm")

photon_flux_cell = np.array(light_source.spectrum(wl_cell))

photon_flux_color = photon_flux_cell[:, np.all((photon_flux_cell[0] >= 380, photon_flux_cell[0] <= 780), axis=0)]

shapes = ['+', 'o', '^', '.', '*', "v", "s", "x"]

loop_n = 0

# precalculate optimal bandgaps for junctions:
save_path = path.join(path.dirname(path.abspath(__file__)), "results")


fixed_bandgaps = [1.90, 1.44, 0.67]


if __name__ == "__main__":
# Need this __main__ construction because otherwise the parallel running of the different islands (n_trials) may throw an error

    fig, axs = plt.subplots(1,len(Ys),figsize=(len(Ys)*3.5, 4))
    if len(Ys) == 1:
        axs = [axs]

    for i1, Y in enumerate(Ys):

        if os.path.exists(save_path + "/pop_gamut_Y_{}_{}_spd.npy".format(Y, n_junctions)):
            print("Load existing result")

            pop_array = np.load(save_path + "/pop_gamut_Y_{}_{}_spd.npy".format(Y, n_junctions))
            eff_array = np.load(save_path + "/eff_gamut_Y_{}_{}_spd.npy".format(Y, n_junctions))

        else:
            col_possible_str = save_path + "/possible_colours_Y_{}_spd.txt".format(Y)

            if os.path.exists(save_path + "/possible_colours_Y_{}_spd.txt".format(Y)):
                is_possible = np.loadtxt(col_possible_str)
                print("Possible colours:", np.sum(is_possible))

            else:

                if i1 == 0:
                    is_possible = np.full((len(xs), len(ys)), True)
                    is_possible[is_inside == False] = False

                else:
                    col_possible_str_pr = save_path + "/possible_colours_Y_{}_spd.txt".format(Ys[i1 - 1])
                    is_possible = np.loadtxt(col_possible_str_pr)

            # print(is_possible)
            eff_array = np.zeros((len(xs), len(ys)))
            pop_array = np.zeros((len(xs), len(ys), 4+n_junctions))
        # for i1, n_junctions in enumerate(n_junc_loop):
            for j1, x in enumerate(xs):
                for k1, y in enumerate(ys):
                    if is_inside[j1, k1] and is_possible[j1, k1]:
                        XYZ = np.array([convert_color(xyYColor(x, y, Y), XYZColor).get_value_tuple()])

                        result = multiple_color_cells(XYZ, [str(j1) + '_' + str(k1)], photon_flux_cell,
                                                       n_peaks=n_peaks, n_junctions=n_junctions,
                                                  type=type, fixed_height=fixed_height,
                                                  n_trials=n_trials, initial_iters=initial_iters, add_iters=add_iters,
                                                  col_thresh=col_thresh, acceptable_eff_change=acceptable_eff_change,
                                                  max_trials_col=max_trials_col, base=base, max_height=max_height,
                                                  fixed_bandgaps=fixed_bandgaps,
                                                  plot=False)

                        champion_effs = result["champion_eff"]
                        champion_pops = result["champion_pop"]

                        print("champion effs", champion_effs)

                        eff_array[j1, k1] = champion_effs
                        pop_array[j1, k1, :] = champion_pops
                        if np.max(champion_effs) == 0:
                            print("SETTING TO 0")
                            is_possible[j1, k1] = False

            print("sum:", np.sum(is_possible))

            np.savetxt(col_possible_str, is_possible)
            np.save(save_path + "/pop_gamut_Y_{}_{}_spd.npy".format(Y, n_junctions), pop_array)
            np.save(save_path + "/eff_gamut_Y_{}_{}_spd.npy".format(Y, n_junctions), eff_array)

        width = np.diff(xs)[0]
        height = np.diff(ys)[0]

        standard_illuminant = [0.3128, 0.3290, Y]
        XYZ = convert_color(xyYColor(*standard_illuminant), XYZColor)
        s_i_RGB = np.array(convert_color(XYZ, sRGBColor).get_value_tuple())
        s_i_RGB[s_i_RGB > 1] = 1

        label_wls = np.arange(440, 660, 20)

        XYZlab = wavelength_to_XYZ(label_wls)

        sumXYZlab = np.sum(XYZlab, axis=1)

        xgl = XYZlab[:, 0] / sumXYZlab
        ygl = XYZlab[:, 1] / sumXYZlab

        tick_orig = np.zeros((len(label_wls),2))
        tick_dir = np.zeros((len(label_wls),2))
        # create ticks
        for m1, lwl in enumerate(label_wls):
            p0 = wavelength_to_XYZ(lwl)
            p1 = wavelength_to_XYZ(lwl-1)
            p2 = wavelength_to_XYZ(lwl+1)

            p0 = p0 / np.sum(p0)
            p1 = p1 / np.sum(p1)
            p2 = p2 / np.sum(p2)

            m = np.array([p2[0]-p1[0], p2[1]-p1[1]])
            mp = np.array([-m[1], m[0]])
            mp = mp/np.linalg.norm(mp)
            # b = p1[1] + (1/m) * p1[0]

            tick_orig[m1] = p0[:2]
            tick_dir[m1] = p0[:2] + 0.02*mp



        ax = axs[i1]
        ax.set_aspect('equal')
        ax.set_facecolor(s_i_RGB)
        ax.plot(xg, yg, 'k')
        ax.plot([xg[0], xg[-1]], [yg[0], yg[-1]], 'k')

        for m1, lwl in enumerate(label_wls):
            ax.plot([tick_orig[m1, 0], tick_dir[m1, 0]], [tick_orig[m1, 1], tick_dir[m1, 1]], '-k')

            if lwl > 520:
                ax.text(*tick_dir[m1], str(lwl))

            elif lwl == 520:
                ax.text(*tick_dir[m1], str(lwl), horizontalalignment='center')

            else:
                ax.text(*tick_dir[m1], str(lwl),
                         horizontalalignment='right', verticalalignment='center')

        ax.set_xlim(-0.09, 0.8)
        ax.set_ylim(-0.07, 0.9)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        print(np.nanmin(eff_array[eff_array>0]), np.nanmax(eff_array[eff_array>0]))
        for j1, x in enumerate(xs):
            for k1, y in enumerate(ys):

                if ~np.isnan(eff_array[j1, k1]) and eff_array[j1, k1] > 0:
                    XYZ = convert_color(xyYColor(x, y, Y), XYZColor)
                    RGB = np.array(convert_color(XYZ, sRGBColor).get_value_tuple())

                    RGB[RGB > 1] = 1
                #     plt.plot(x, y, '.')
                    ax.add_patch(
                        Rectangle(xy=(x - width / 2, y - height / 2), width=width,
                                  height=height,
                                  facecolor=RGB)
                    )
                    # ax.text(x, y, str(round(eff_array[j1, k1], 2)), color='k', ha='center', va='center')
                #
                # plt.plot(x, y, 'k.')

        levels = np.linspace(np.min(eff_array[eff_array>0])+2, np.max(eff_array[eff_array>0]), 5)
        levels = np.floor(levels)

        if Y == 0.25:
            levels = [31, 36, 39, 41, 42]

        elif Y == 0.5:
            levels = [22, 31, 35, 37, 38]

        print(levels)

        if len(levels) > 5:
            levels = levels[-5:]

        cs = ax.contour(xs, ys, eff_array.T,
                        levels=levels,
                        colors='k', alpha=0.5)

        label_loc = []

        for i1, seg in enumerate(cs.allsegs):
            x_lower = seg[0][:, 0][seg[0][:, 1] < 0.55]
            y_lower = seg[0][:, 1][seg[0][:, 1] < 0.55]

            close_x = np.argmin(np.abs(x_lower - 0.4-i1*0.02))
            close_y = y_lower[close_x]
            label_loc.append([0.4-i1*0.02, close_y])

        ax.clabel(cs, inline=1, fontsize=10, manual=label_loc)
        ax.set_title("Y = " + str(Y))
        ax.yaxis.set_minor_locator(tck.AutoMinorLocator())
        ax.xaxis.set_minor_locator(tck.AutoMinorLocator())
        ax.grid(axis="both", color='0.4', alpha=0.5)
        ax.tick_params(direction='in', which='both', top=True, right=True)
        #ax.set_axisbelow(True)
    

    plt.tight_layout()
    plt.show()

