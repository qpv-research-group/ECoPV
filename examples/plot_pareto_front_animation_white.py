from ecopv.main_optimization import load_colorchecker, single_color_cell, color_function_mobj
from ecopv.spectrum_functions import make_spectrum_ndip, spec_to_XYZ, load_cmf
import numpy as np
from solcore.light_source import LightSource
import matplotlib.pyplot as plt
import pygmo as pg
from os import path
from pygmo.core import fast_non_dominated_sorting
from colormath.color_objects import sRGBColor, XYZColor
from colormath.color_conversions import convert_color
from matplotlib import rc
from solcore.constants import h, c
from imageio.v3 import imread
from imageio import get_writer
from matplotlib.patches import Rectangle

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"], "size": 14})

# Use smaller population size than actual results for clarity!

def plot_non_dominated_fronts(points, marker='o', comp=[0, 1], axes=None,
                              color=None, linecolor='k', mfc='none',
                              markersize=4, **kwargs):
    """
    Plots the nondominated fronts of a set of points. Makes use of :class:`~pygmo.fast_non_dominated_sorting` to
    compute the non dominated fronts.
    """


    if color is None:
        color = ['k']*len(points)

    # if mfc == 'none':
    #     mfc = ['none']*len(points)


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


# function takes frame as an input

col_thresh = 0.01  # for a wavelength interval of 0.1, minimum achievable color error
# deltaXYZ will be ~ 0.001.
# (deltaXYZ = maximum fractional error in X, Y, Z colour coordinates)
acceptable_eff_change = (
    1e-4  # how much can the efficiency (in %) change between iteration sets?
)
n_trials = 1  # number of islands which will run concurrently
interval = 0.1  # wavelength interval (in nm)
wl_cell = np.arange(280, 4000, interval)  # wavelengths

add_iters = 150  # additional evolutions added each time if color
# threshold/convergence condition not met
# every color will run a minimum of initial_iters + add_iters evolutions before ending!

max_trials_col = (
    5 * add_iters
)  # how many population evolutions happen before giving up if there are no populations
# which meet the color threshold

R_type = "sharp"  # "sharp" for rectangular dips or "gauss" for gaussians
fixed_height = True  # fixed height peaks (will be at the value of max_height) or not

max_height = 1  # maximum height of reflection peaks
base = 0  # baseline fixed reflection

n_junctions = 1  # number of junctions in the cell

n_peaks = 2  # number of reflection peaks

color_names, color_XYZ = load_colorchecker()  # load the 24 default ColorChecker colors

color_names = [color_names[-6]]
color_XYZ = [color_XYZ[-6]]

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


illuminant = h*c*photon_flux_color[1]/ (wl_col * 1e-9)

save_path = path.join(path.dirname(path.abspath(__file__)), "results")



placeholder_obj = make_spectrum_ndip(
    n_peaks=n_peaks, R_type=R_type, fixed_height=fixed_height
)
n_params = placeholder_obj.n_spectrum_params + n_junctions
pop_size = n_params * 10

# Need this because otherwise the parallel running of the different islands (n_trials) may throw an error


# Run for the selected peak shape, with both fixed and non-fixed height
Eg_black = np.loadtxt(
    save_path
    + "/champion_pop_{}juncs_{}spec.txt".format(
        n_junctions, light_source_name
    ),
    ndmin=1,
)

best_eff = []
overall_best_eff = []

j1 = 0
target_col = color_XYZ[j1]

if __name__ == "__main__":

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


    for k1 in range(add_iters):
        print("iteration", k1)
        f_vals = archi[0].get_population().get_f()
        x_vals = archi[0].get_population().get_x()

        RGB_finalpop = np.zeros((len(x_vals), 3))
        # back-calculate colors:

        archi = internal_run.run(
            target_col,
            photon_flux_cell,
            illuminant,
            n_peaks,
            n_junctions,
            pop_size,
            gen=1,
            n_trials=n_trials,
            power_in=light_source.power_density,
            spectrum_bounds=spectrum_obj.get_bounds(),
            Eg_black=Eg_black,
            archi=archi,
            base=base,
            max_height=max_height,
        )


        for i1, xs in enumerate(x_vals):
            spec = spectrum_obj.spectrum_function(xs, n_peaks=n_peaks, wl=wl_col)
            XYZ_finalpop = spec_to_XYZ(spec, solar_spec_color, cmf, interval)
            color_xyz_t = XYZColor(*XYZ_finalpop)
            RGB_finalpop[i1, :] = np.clip(convert_color(color_xyz_t,
                                                        sRGBColor).get_value_tuple(),
                                          a_min=0, a_max=1)

        fig, (ax, ax2) = plt.subplots(1,2, figsize=(9, 4))
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
        ax.grid(axis="both")

        ax.set_xlim(0, 1.02)

        ax.set_ylim(20, 34.3)

        ax.set_facecolor((0.98, 0.97, 0.95))

        sln = f_vals[:, 0] < col_thresh

        if np.sum(sln) > 0:

            best_acc_ind = np.argmin(f_vals[sln, 1])
            new_eff = -100*f_vals[sln, 1][best_acc_ind]
            if new_eff > best_eff[-1]:
                best_eff.append(new_eff)

            else:
                best_eff.append(best_eff[-1])

            ax.add_patch(
                Rectangle(
                    xy=(0.05, 31.5),
                    width=0.15,
                    height=2,
                    facecolor=RGB_finalpop[sln][best_acc_ind],
                    edgecolor='k',
                )
            )

        else:
            if k1 == 0:
                best_eff.append(0)
            else:
                best_eff.append(best_eff[-1])

        overall_best_eff.append(-100*np.min(f_vals[:, 1]))

        ax2.plot(best_eff, color='k', linewidth=1.5, label="Best coloured cell")
        ax2.plot(overall_best_eff, 'k--', alpha=0.5, linewidth=1.5, label="Best overall cell")
        ax2.plot(k1, best_eff[-1], 'ro')
        ax2.plot(k1, overall_best_eff[-1], 'ko', alpha=0.5)
        ax2.set_xlim(0, add_iters)
        ax2.set_ylim(20, 34.3)

        ax2.legend(loc="lower right")

        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Efficiency (%)")

        plt.tight_layout()
        plt.savefig('results/' + str(k1) +'.png')

    n_frames = np.ones(add_iters)
    n_frames[0] = 10
    n_frames[-1] = 10

    with get_writer('white.mp4', mode='I') as writer:
        for k1 in range(add_iters):

            for _ in range(int(n_frames[k1])):

                image = imread('results/' + str(k1) +'.png')
                writer.append_data(image)

    writer.close()

    # import glob
    # import os
    #
    # writer = get_writer('test.mp4', fps=5)
    # for file in glob.glob(os.path.join('results', f'{k1}*.png')):
    #     im = imread(file)
    #     writer.append_data(im)
    # writer.close()