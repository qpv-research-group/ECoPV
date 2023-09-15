from ecopv.main_optimization import load_colorchecker, single_color_cell, color_function_mobj
from ecopv.spectrum_functions import make_spectrum_ndip, spec_to_XYZ, load_cmf
import numpy as np
from solcore.light_source import LightSource
import matplotlib.pyplot as plt
from time import time
import pygmo as pg
from os import path
from pygmo.core import fast_non_dominated_sorting
from colormath.color_objects import sRGBColor, XYZColor
from colormath.color_conversions import convert_color
from copy import deepcopy
from matplotlib import rc
from solcore.constants import h, c

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})

col_cutoff = 0.05


def plot_non_dominated_fronts(points, marker='o', comp=[0, 1], axes=None,
                              color=None, linecolor='k', mfc=None,
                              markersize=4, **kwargs):
    """
    Plots the nondominated fronts of a set of points. Makes use of :class:`~pygmo.fast_non_dominated_sorting` to
    compute the non dominated fronts.
    """


    if color is None:
        color = ['k']*len(points)


    if mfc is None:
        mfc = ['none']*len(points)

    fronts, _, _, _ = fast_non_dominated_sorting(points)

    # We define the colors of the fronts (grayscale from black to white)
    alpha = np.linspace(1, 0.1, len(fronts))

    if axes is None:
        axes = plt.axes()

    for ndr, front in enumerate(fronts):

        # We plot the fronts
        # First compute the points coordinates
        x = [points[idx][comp[0]] for idx in front]
        y = [-100*points[idx][comp[1]] for idx in front]
        # Then sort them by the first objective
        tmp = [(a, b) for a, b in zip(x, y)]
        tmp = sorted(tmp, key=lambda k: k[0])
        # Now plot using step
        axes.step([c[0] for c in tmp], [c[1]
                                        for c in tmp],
                  color=linecolor,
                  where='post',
                  # alpha=alpha[ndr],
                  alpha=0.5,
                  linestyle='--')

        # We plot the points
        for idx in front:
            axes.plot(points[idx][comp[0]], -100*points[idx][
                comp[1]], marker=marker,
                      # alpha=alpha[ndr],
                      color=color[idx],
                      markersize=markersize,
                      mfc=mfc[idx], **kwargs)

    return axes


col_thresh = 0.004  # for a wavelength interval of 0.1, minimum achievable color error deltaXYZ will be ~ 0.001.
# (deltaXYZ = maximum fractional error in X, Y, Z colour coordinates)
acceptable_eff_change = (
    1e-4  # how much can the efficiency (in %) change between iteration sets?
)
n_trials = 1  # number of islands which will run concurrently
interval = 0.1  # wavelength interval (in nm)
wl_cell = np.arange(280, 4000, interval)  # wavelengths

add_iters = 500 # additional evolutions added each time if color
# threshold/convergence condition not met
# every color will run a minimum of initial_iters + add_iters evolutions before ending!

R_type = "sharp"  # "sharp" for rectangular dips or "gauss" for gaussians
fixed_height = True  # fixed height peaks (will be at the value of max_height) or not

max_height = 1  # maximum height of reflection peaks
base = 0  # baseline fixed reflection

n_junctions = 6  # number of junctions in the cell

n_peaks = 2  # number of reflection peaks

color_names, color_XYZ = load_colorchecker()  # load the 24 default ColorChecker colors
color_names = [color_names[2]]#, color_names[14]]
color_XYZ = [color_XYZ[2]]#, color_XYZ[14]]

# BlueSky maximum efficiency:
found_max = 55.53376

# Define the incident photon flux. This should be a 2D array with the first row being the wavelengths and the second row
# being the photon flux at each wavelength. The wavelengths should be in nm and the photon flux in photons/m^2/s/nm.
light_source = LightSource(
        source_type="standard",
        version="AM1.5g",
        x=wl_cell,
        output_units="photon_flux_per_nm",
    )

light_source_name = "AM1.5g"

photon_flux_cell = np.array(
    light_source.spectrum(wl_cell)
)

# Use only the visible range of wavelengths (380-780 nm) for color calculations:
photon_flux_color = photon_flux_cell[
    :, np.all((photon_flux_cell[0] >= 380, photon_flux_cell[0] <= 730), axis=0)
]

wl_col = np.arange(380, 730, interval)

cmf = load_cmf(wl_col)

solar_spec_color = LightSource(
        source_type="standard",
        version="AM1.5g",
        x=wl_col,
        output_units="power_density_per_nm",
    ).spectrum(wl_col)[1]

shapes = ["x", "o", "^", ".", "*", "v", "s", "+"]

loop_n = 0

illuminant = h*c*photon_flux_color[1]/ (wl_col * 1e-9)

save_path = path.join(path.dirname(path.abspath(__file__)), "results")

if __name__ == "__main__":

    placeholder_obj = make_spectrum_ndip(
        n_peaks=n_peaks, R_type=R_type, fixed_height=fixed_height
    )
    n_params = placeholder_obj.n_spectrum_params + n_junctions
    pop_size = n_params * 10

    Eg_black = np.loadtxt(
        save_path
        + "/champion_pop_{}juncs_{}spec.txt".format(
            n_junctions, light_source_name
        ),
        ndmin=1,
    )

    # x_vals_start = np.loadtxt("results/pareto_plot_pop.txt")


    RGB_finalpop = np.zeros((pop_size, 3))

    for j1, target_col in enumerate(color_XYZ):

        spectrum_obj = make_spectrum_ndip(
            n_peaks=n_peaks,
            target=target_col,
            R_type=R_type,
            fixed_height=fixed_height,
        )

        internal_run = single_color_cell(
            spectrum_function=spectrum_obj.spectrum_function
        )

        p_init = color_function_mobj(
            n_peaks,
            n_junctions,
            target_col,
            col_cutoff,
            photon_flux_cell,
            illuminant,
            spectrum_obj.spectrum_function,
            light_source.power_density,
            spectrum_obj.get_bounds(),
            Eg_black,
        )

        udp = pg.problem(p_init)

        algo = pg.algorithm(pg.moead(gen=1, CR=1, F=0.5, preserve_diversity=True))

        archi = pg.archipelago(n=n_trials, algo=algo, prob=udp, pop_size=pop_size)

        # for i1 in range(len(x_vals_start)):
        #     pop = archi[0].get_population()
        #     pop.set_x(i1, x_vals_start[i1])
        #     archi[0].set_population(pop)

        RGB_initpop = np.zeros((pop_size, 3))

        for i1, xs in enumerate(archi[0].get_population().get_x()):
            spec = placeholder_obj.spectrum_function(xs, n_peaks=n_peaks, wl=wl_col)
            XYZ_finalpop = spec_to_XYZ(spec, solar_spec_color, cmf, interval)
            color_xyz_t = XYZColor(*XYZ_finalpop)
            RGB_initpop[i1, :] = convert_color(color_xyz_t, sRGBColor).get_value_tuple()

        RGB_initpop[RGB_initpop > 1] = 1

        fig, (ax, ax3) = plt.subplots(1, 2, figsize=(10, 3))

        if j1 == 0:
            f_vals_start = archi[0].get_population().get_f()
            plot_non_dominated_fronts(f_vals_start, axes=ax, color=RGB_initpop,
                                      linecolor='k')

        best_eff = np.zeros(add_iters)
        best_overall_eff = np.zeros(add_iters)\

        best_eff[0] = 0
        best_overall_eff[0] = -100*np.min(f_vals_start[:, 1])

        for i1 in range(1, add_iters):
            print(i1)

            archi = internal_run.run(
                target=target_col,
                col_cutoff=col_cutoff,
                photon_flux=photon_flux_cell,
                illuminant=illuminant,
                n_peaks=n_peaks,
                n_gaps=n_junctions,
                popsize=pop_size,
                gen=1,
                n_trials=n_trials,
                power_in=light_source.power_density,
                spectrum_bounds=spectrum_obj.get_bounds(),
                Eg_black=Eg_black,
                archi=archi,
                base=base,
                max_height=max_height,
            )

            f_vals = archi[0].get_population().get_f()
            x_vals = archi[0].get_population().get_x()
            sln = f_vals[:, 0] < col_thresh

            best_overall_eff[i1] = -100*np.min(f_vals[:, 1])

            if np.sum(sln) > 0:

                best_acc_ind = np.argmin(f_vals[sln, 1])
                new_eff = -100*f_vals[sln, 1][best_acc_ind]
                if new_eff > max(best_eff):
                    best_eff[i1] = new_eff

                else:
                    best_eff[i1] = max(best_eff)

            else:
                best_eff[i1] = max(best_eff)


        # back-calculate colors:
        XYZ_finalpop = np.zeros((len(x_vals), 3))

        for i1, xs in enumerate(x_vals):
            spec = spectrum_obj.spectrum_function(xs, n_peaks=n_peaks, wl=wl_col)
            XYZ_finalpop[i1] = spec_to_XYZ(spec, solar_spec_color, cmf, interval)
            color_xyz_t = XYZColor(*XYZ_finalpop[i1])
            RGB_finalpop[i1, :] = np.clip(convert_color(color_xyz_t,
                                                 sRGBColor).get_value_tuple(),
                                          a_min=0, a_max=1)


        plot_non_dominated_fronts(f_vals, axes=ax,
                                  color=RGB_finalpop,
                                  mfc=RGB_finalpop,
                                  # linecolor=RGB_finalpop[-1],
                                  # linecolor=[0.8, 0.5, 0.5],
                                  markeredgewidth=0.5,
                                  markersize=6,
                                  )


        ax.set_xlabel(r"Colour deviation, max(|$\Delta XYZ$|)")
        ax.set_ylabel("Cell efficiency (%)")

        left, bottom, width, height = [0.13, 0.8, 0.15, 0.15]
        # ax2 = fig.add_axes([left, bottom, width, height])
        #
        # ax2.set_facecolor((0.7, 0.7, 0.7))

        # f_vals_thresh = f_vals[f_vals[:, 0] < 0.051]
        # RGB_finalpop_thresh = RGB_finalpop[f_vals[:, 0] < 0.051]
        #
        # plot_non_dominated_fronts(f_vals_thresh, axes=ax2, color=RGB_finalpop_thresh,
        #                           mfc=RGB_finalpop_thresh,
        #                           linecolor=RGB_finalpop_thresh[-1])

        # for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
        #     label.set_fontsize(8)

        ax.grid(axis="both")
        # ax2.grid(axis="both")
        ax3.grid(axis="both")
        # ax.set_xlim(0, np.max(f_vals_start[:, 0] + 0.01))
        # ax.set_xlim(0, 1.02)
        # ax.set_ylim(-100*(np.max(f_vals_start[:, 1]) + 0.01), 34.3)
        # ax.set_ylim(16, 34.3)
        # ax3.set_ylim(16, 34.3)
        # ax2.set_xlim(-0.001, 0.01)
        # ax2.set_ylim(29, 31)
        # ax2.axvline(0.004, linestyle='--', color='k', alpha=0.6)
        ax.set_facecolor((0.98, 0.97, 0.95))

        best_eff[best_eff == 0] = np.nan

        ax3.plot(best_eff, color=RGB_finalpop[0], linewidth=1.5, label=f"Best {color_names[0]} cell")
        # ax3.plot(best_overall_eff, 'k--', alpha=0.5, linewidth=1.5, label="Best overall cell")
        ax3.axhline(found_max, linestyle='--', color='k', alpha=0.7)
        ax3.legend()
        ax3.set_xlabel("Generation")
        ax3.set_ylabel("Efficiency (%)")
        ax3.set_xlim(0, add_iters-1)

        plt.tight_layout()
        plt.show()

        plt.figure()
        plt.plot(-f_vals[:,1], np.abs(XYZ_finalpop[:,0] - target_col[0])/target_col[0], 'o', label='X')
        plt.plot(-f_vals[:,1], np.abs(XYZ_finalpop[:,1] - target_col[1])/target_col[1], 'o', label='Y')
        plt.plot(-f_vals[:,1], np.abs(XYZ_finalpop[:,2] - target_col[2])/target_col[2], 'o', label='Z')
        plt.legend()
        plt.title('Fractional difference in X/Y/Z')
        plt.xlabel("Efficiency of vector")
        plt.show()
        #
        # sumXYZ = np.sum(XYZ_finalpop, axis=1)
        #
        # plt.figure()
        # plt.plot(-f_vals[:,1], XYZ_finalpop[:,0], 'o', label='X')
        # plt.plot(-f_vals[:,1], XYZ_finalpop[:,1], 'o',  label='Y')
        # plt.plot(-f_vals[:,1], XYZ_finalpop[:,2], 'o',  label='Z')
        # plt.legend()
        # plt.title('Values of X/Y/Z')
        # plt.xlabel("Efficiency of vector")
        # plt.show()
        #
        # plt.figure()
        # plt.plot(-f_vals[:,1], XYZ_finalpop[:,0]/sumXYZ, 'o', label='x')
        # plt.plot(-f_vals[:,1], XYZ_finalpop[:,1]/sumXYZ, 'o', label='y')
        # plt.plot(-f_vals[:,1], XYZ_finalpop[:,1], 'o', label='Y')
        # plt.legend()
        # plt.title('Values of xyY')
        # plt.xlabel("Efficiency of vector")
        # plt.show()