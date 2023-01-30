import pygmo as pg
import numpy as np
from time import time
from colour.difference import delta_E_CIE2000
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import Sequence, Callable, Tuple

from colormath.color_objects import sRGBColor, XYZColor
from colormath.color_conversions import convert_color
from solcore.constants import h, c
import pathlib
from os.path import join
from optimization_functions import reorder_peaks, getPmax
from spectrum_functions import load_cmf, spec_to_XYZ, convert_xyY_to_XYZ, convert_xyY_to_Lab, convert_XYZ_to_Lab, make_spectrum_ndip, gen_spectrum_ndip, delta_XYZ

hc = h * c

current_path = pathlib.Path(__file__).parent.resolve()


# TODO:
# - write docstrings
# - allow passing other argeumtns to DE and MOAED using kwargs
# - allow plotting of Pareto front again
# - consistent spelling colour/color and optimi(z/s)ation

def load_colorchecker(output_coords: str = "XYZ") -> Tuple[np.ndarray, np.ndarray]:
    """Load the colorchecker data from the csv file and return is as an array of coordinates

    :param output_coords: The color space to return the data in. Can be "XYZ", "xyY", or "Lab"

    :return: a list of color names, array of color coordinates

    """

    color_names = np.array(
        [
            "DarkSkin",
            "LightSkin",
            "BlueSky",
            "Foliage",
            "BlueFlower",
            "BluishGreen",
            "Orange",
            "PurplishBlue",
            "ModerateRed",
            "Purple",
            "YellowGreen",
            "OrangeYellow",
            "Blue",
            "Green",
            "Red",
            "Yellow",
            "Magenta",
            "Cyan",
            "White-9-5",
            "Neutral-8",
            "Neutral-6-5",
            "Neutral-5",
            "Neutral-3-5",
            "Black-2",
        ]
    )

    color_xyY = np.loadtxt(join(current_path, "data", "paper_colors.csv"), skiprows=1, usecols=[2,3,4], delimiter=',')

    if output_coords == "xyY":
        return color_names, color_xyY

    elif output_coords == "XYZ":
        color_XYZ = np.array([convert_xyY_to_XYZ(x) for x in color_xyY])
        return color_names, color_XYZ

    elif output_coords == "Lab":
        color_Lab = np.array([convert_xyY_to_Lab(x) for x in color_xyY])
        return color_names, color_Lab

    else:
        raise ValueError("output_coords must be one of xyY, XYZ, or Lab")


def multiple_color_cells(
    color_XYZ: np.ndarray,
    color_names: Sequence[str],
    photon_flux: np.ndarray,
    n_peaks: int = 2,
    n_junctions: int = 1,
    type: str = "sharp",
    fixed_height: bool = True,
    n_trials: int = 10,
    initial_iters: int = 100,
    add_iters: int = 100,
    col_thresh: float = 0.004,
    acceptable_eff_change=1e-4,
    max_trials_col: int = None,
    base: float = 0,
    max_height: float = 1,
    Eg_black: list = None,
    plot: bool = True,
    fixed_bandgaps: list = None,
    power_in: float = 1000,
    return_archipelagos: bool = False,
) -> dict:

    """Optimize color and efficiency of multiple colored cells using pygmo2's moaed (multi-objective differential evolution)
    implementation, by calling single_color_cell() for each target colour.

    :param color_XYZ: numpy array of XYZ color coordinates of the target colors. Dimensions (n_colors, 3).
    :param color_names: names of the target colors. List of strings of length n_colors.
    :param photon_flux: incident photon flux. 2D numpy array with the first row being the wavelengths and the second row
                        being the photon flux at each wavelength. The wavelengths should be in nm and the photon flux in photons/m^2/s/nm.
    :param n_peaks: number of peaks in the spectrum
    :param n_junctions: number of junctions in the cell
    :param type: type of spectrum, "sharp" or "gauss" currently implemented
    :param fixed_height: whether to fix the height of the reflection peaks to max_height (True) or allow it to vary (False)
    :param n_trials: number of islands (separate threads) which will run concurrently
    :param initial_iters: number of evolutions of the initial population
    :param add_iters: number of additional evolutions per optimization loop
    :param col_thresh: maximum acceptable value for deltaXYZ (maximum fractional error in X, Y, Z coordinates)
    :param acceptable_eff_change: maximum acceptable change in efficiency between optimization loops; if the change is smaller,
            the optimization ends
    :param max_trials_col: maximum number of evolutions to try to find a color within col_thresh before giving up.
    :param base: base reflection (0-1)
    :param max_height: maximum height of reflection spectrum (0-1)
    :param Eg_black: bandgap of the black body spectrum
    :param plot: whether to plot the results
    :param fixed_bandgaps: fixed bandgaps of the junctions
    :param power_in: power in of the cell
    :param return_archipelagos: whether to return the archipelagos (pygmo2 objects) at the end of the optimization (these
                                can be very large objects!)

    :return: results from the optimization in a dictionary with elements "champion_eff" (maximum cell efficiencies for
            each color), "champion_pop" (the champion population which maximizes the efficiency while staying within the
            allowed col_thresh) and, if requested, "archipelagos", which contains the final population of the n_trials islands
            for each color being optimized (very large objects!)
    """

    placeholder_obj = make_spectrum_ndip(
        n_peaks=n_peaks, type=type, fixed_height=fixed_height
    )
    n_params = placeholder_obj.n_spectrum_params + n_junctions
    pop_size = n_params * 10

    cmf = load_cmf(photon_flux[0])
    interval = np.diff(photon_flux[0])[0]

    if max_trials_col is None:
        max_trials_col = 5 * initial_iters

    mean_sd_effs = np.empty((len(color_XYZ), 4))

    all_converged = False

    conv = np.array([False] * len(color_XYZ))

    color_indices = np.arange(len(color_XYZ))

    champion_eff = np.zeros(len(color_XYZ))
    champion_pop = np.empty((len(color_XYZ), n_params))

    archipelagos = [None] * len(color_XYZ)

    color_XYZ_found = [None] * len(color_XYZ)

    iters_needed = np.zeros(len(color_XYZ))

    current_iters = initial_iters

    start_time = time()

    while not all_converged:

        start = time()
        print("Add iters:", current_iters)

        for k1 in color_indices:

            spectrum_obj = make_spectrum_ndip(
                n_peaks=n_peaks,
                target=color_XYZ[k1],
                type=type,
                fixed_height=fixed_height,
            )

            internal_run = single_color_cell(
                spectrum_function=spectrum_obj.spectrum_function
            )

            iters_needed[k1] += current_iters

            archi = internal_run.run(
                color_XYZ[k1],
                photon_flux,
                n_peaks,
                n_junctions,
                pop_size,
                current_iters,
                n_trials=n_trials,
                power_in=power_in,
                spectrum_bounds=spectrum_obj.get_bounds(),
                Eg_black=Eg_black,
                archi=archipelagos[k1],
                base=base,
                max_height=max_height,
                fixed_bandgaps=fixed_bandgaps,
            )

            archipelagos[k1] = archi

            all_fs = np.stack(
                [archi[j1].get_population().get_f() for j1 in range(n_trials)]
            )
            all_xs = np.stack(
                [archi[j1].get_population().get_x() for j1 in range(n_trials)]
            )

            sln = all_fs[:, :, 0] < col_thresh

            best_acc_ind = np.array(
                [
                    np.argmin(x[sln[i1], 1]) if len(x[sln[i1]]) > 0 else 0
                    for i1, x in enumerate(all_fs)
                ]
            )
            best_acc_eff = np.array(
                [
                    -np.min(x[sln[i1], 1]) if len(x[sln[i1]]) > 0 else 0
                    for i1, x in enumerate(all_fs)
                ]
            )
            best_acc_pop = np.array(
                [
                    x[sln[i1]][best_acc_ind[i1]]
                    if len(x[sln[i1]]) > 0
                    else [0] * n_params
                    for i1, x in enumerate(all_xs)
                ]
            )

            max_eff_acc = best_acc_eff[best_acc_eff > 0] * 100
            best_acc_pop = best_acc_pop[best_acc_eff > 0]

            if len(max_eff_acc) > 0:

                print(
                    color_names[k1],
                    np.round(np.max(max_eff_acc), 3),
                    np.round(np.mean(max_eff_acc), 3),
                    np.round(np.std(max_eff_acc), 6),
                )

                ch_eff = np.max(max_eff_acc)
                ch_eff_ind = np.argmax(max_eff_acc)

                ch_pop = best_acc_pop[ch_eff_ind]

                mean_sd_effs[k1] = [
                    np.min(max_eff_acc),
                    ch_eff,
                    np.mean(max_eff_acc),
                    np.std(max_eff_acc),
                ]

                delta_eta = ch_eff - champion_eff[k1]

                if delta_eta >= acceptable_eff_change:
                    champion_eff[k1] = ch_eff
                    champion_pop[k1] = ch_pop
                    print(np.round(delta_eta, 5), "delta eff - New champion efficiency")

                else:
                    print("No change/worse champion efficiency")
                    conv[k1] = True

                    spec = internal_run.spec_func(
                        champion_pop[k1],
                        n_peaks,
                        photon_flux[0],
                        base=base,
                        max_height=max_height,
                    )

                    color_XYZ_found[k1] = spec_to_XYZ(
                        spec,
                        hc * photon_flux[1] / (photon_flux[0] * 1e-9),
                        cmf,
                        interval,
                    )

                    if plot:
                        spec = internal_run.spec_func(
                            champion_pop[k1],
                            n_peaks,
                            photon_flux[0],
                            base=base,
                            max_height=max_height,
                        )
                        plot_outcome(spec, photon_flux, color_XYZ[k1], color_names[k1])
                        plt.xlim(300, 1000)
                        plt.show()

                    print(
                        "Champion pop:",
                        champion_pop[k1],
                        "width limits:",
                        spectrum_obj.get_bounds(),
                    )

            else:
                print(
                    color_names[k1],
                    "no acceptable populations",
                    np.min(all_fs[:, :, 0]),
                )
                if iters_needed[k1] >= max_trials_col:
                    flat_x = all_xs.reshape(-1, all_xs.shape[-1])
                    flat_f = all_fs.reshape(-1, all_fs.shape[-1])
                    best_col = np.argmin(flat_f[:, 0])
                    champion_pop[k1] = flat_x[best_col]
                    print(
                        "Cannot reach target color - give up. Minimum color deviation: "
                        + str(np.round(np.min(flat_f[:, 0]), 5))
                    )
                    print(
                        "Champion pop (best color):",
                        champion_pop[k1],
                        "width limits:",
                        spectrum_obj.get_bounds(),
                    )
                    conv[k1] = True

                    spec = internal_run.spec_func(
                        champion_pop[k1],
                        n_peaks,
                        photon_flux[0],
                        base=base,
                        max_height=max_height,
                    )

                    color_XYZ_found[k1] = spec_to_XYZ(
                        spec,
                        h * c * photon_flux[1] / (photon_flux[0] * 1e-9),
                        cmf,
                        interval,
                    )

                    if plot:
                        spec = internal_run.spec_func(
                            champion_pop[k1],
                            n_peaks,
                            photon_flux[0],
                            base=base,
                            max_height=max_height,
                        )
                        plot_outcome(
                            spec,
                            hc * photon_flux,
                            color_XYZ[k1],
                            color_names[k1] + " (target not reached)",
                        )
                        plt.xlim(300, 1000)
                        plt.show()

        time_taken = time() - start

        color_indices = np.where(~conv)[0]
        print(
            len(color_indices),
            "color(s) are still above acceptable std. dev. threshold. Took",
            time_taken,
            "s",
        )

        if len(color_indices) == 0:
            print("All colors are converged")
            all_converged = True

        else:
            # n_iters = n_iters + 200
            print("Running for another", add_iters, "iterations")
            current_iters = add_iters

    print("TOTAL TIME:", time() - start_time)
    champion_pop = np.array(
        [reorder_peaks(x, n_peaks, n_junctions, fixed_height) for x in champion_pop]
    )

    color_Lab_found = [convert_XYZ_to_Lab(x) for x in color_XYZ_found]
    color_Lab_target = [convert_XYZ_to_Lab(x) for x in color_XYZ]

    delta_E = [delta_E_CIE2000(x, y) for x, y in zip(color_Lab_found, color_Lab_target)]
    print("Delta E*:", delta_E, np.max(delta_E))

    if return_archipelagos:
        return {
            "champion_eff": champion_eff,
            "champion_pop": champion_pop,
            "archipelagos": archipelagos,
        }

    else:
        return {
            "champion_eff": champion_eff,
            "champion_pop": champion_pop,
        }


class single_color_cell:

    """Class to create object to run the optimization for a single coloured cell, combining colour calculation with
    the electrical model."""
    def __init__(
        self, fix_height: bool = True, spectrum_function: Callable = gen_spectrum_ndip, plot_pareto: bool = False
    ):
        """Initialise the object.

        :param fix_height: If True, the height of the peaks is fixed to the maximum height of the spectrum, if False
                            the height is a variable in the optimization
        :param spectrum_function: Function to generate the spectrum from population (made by make_spectrum_ndip)
        """

        self.fix_height = fix_height
        self.spec_func = spectrum_function
        self.plot_pareto = plot_pareto
        pass

    def run(
        self,
        target,
        photon_flux,
        n_peaks=2,
        n_gaps=1,
        popsize=80,
        gen=1000,
        n_trials=10,
        power_in=1000,
        spectrum_bounds=None,
        Eg_black=None,
        archi=None,
        fixed_bandgaps=None,
        **kwargs
    ):

        p_init = color_function_mobj(
            n_peaks,
            n_gaps,
            target,
            photon_flux,
            self.spec_func,
            power_in,
            spectrum_bounds,
            Eg_black,
            fixed_bandgaps,
            **kwargs
        )

        udp = pg.problem(p_init)
        algo = pg.algorithm(pg.moead(gen=gen, CR=1, F=1, preserve_diversity=True))
        # decomposition="bi"))#, preserve_diversity=True, decomposition="bi"))

        if archi is None:
            archi = pg.archipelago(n=n_trials, algo=algo, prob=udp, pop_size=popsize)

        archi.evolve()

        archi.wait()

        return archi


class color_optimization_only:
    def __init__(self, fix_height=True, spectrum_function=gen_spectrum_ndip):
        self.fix_height = fix_height
        self.spec_func = spectrum_function

    def run(
        self,
        target,
        photon_flux,
        n_peaks=2,
        popsize=40,
        gen=1000,
        n_trials=10,
        ftol=1e-6,
        archi=None,
    ):

        p_init = color_optimization(n_peaks, target, photon_flux, self.spec_func)

        udp = pg.problem(p_init)
        algo = pg.algorithm(pg.de(gen=gen, CR=1, F=1, ftol=ftol))

        if archi is None:
            archi = pg.archipelago(n=n_trials, algo=algo, prob=udp, pop_size=popsize)

        archi.evolve()

        archi.wait()

        best_delta = np.min(archi.get_champions_f())
        return best_delta


class color_function_mobj:
    def __init__(
        self,
        n_peaks,
        n_juncs,
        tg,
        photon_flux,
        spec_func=gen_spectrum_ndip,
        power_in=1000,
        spectrum_bounds=[],
        Eg_black=None,
        fixed_bandgaps=None,
        **kwargs
    ):

        self.n_peaks = n_peaks
        self.n_juncs = n_juncs
        self.target_color = tg
        self.c_bounds = [380, 780]
        self.spec_func = spec_func
        self.dim = len(spectrum_bounds[0]) + n_juncs
        self.bounds_passed = spectrum_bounds
        self.Eg_black = Eg_black

        self.cell_wl = photon_flux[0]
        self.col_wl = self.cell_wl[
            np.all(
                [self.cell_wl >= self.c_bounds[0], self.cell_wl <= self.c_bounds[1]],
                axis=0,
            )
        ]
        self.solar_flux = photon_flux[1]
        self.solar_spec = (
            hc
            * self.solar_flux[
                np.all([self.cell_wl >= 380, self.cell_wl <= 780], axis=0)
            ]
            / (self.col_wl * 1e-9)
        )
        self.incident_power = power_in
        self.interval = np.round(np.diff(self.cell_wl)[0], 6)
        self.cmf = load_cmf(self.col_wl)
        self.add_args = kwargs

        if fixed_bandgaps is not None:
            self.fixed_bandgaps = fixed_bandgaps

        else:
            self.fixed_bandgaps = []

    def calculate(self, x):

        Egs = x[-self.n_juncs :] if self.n_juncs > 0 else np.array([])
        Egs = Egs.tolist() + self.fixed_bandgaps
        Egs = -np.sort(-np.array(Egs))

        # T = 298 currently fixed!

        R_spec = self.spec_func(x, self.n_peaks, wl=self.col_wl, **self.add_args)

        XYZ = np.array(spec_to_XYZ(R_spec, self.solar_spec, self.cmf, self.interval))
        delta = delta_XYZ(self.target_color, XYZ)

        R_spec_cell = self.spec_func(x, self.n_peaks, wl=self.cell_wl, **self.add_args)

        flux = self.solar_flux * (1 - R_spec_cell)

        eta = getPmax(Egs, flux, self.cell_wl, self.interval) / self.incident_power

        return delta, eta

    def fitness(self, x):
        delta, eff = self.calculate(x)

        return [delta, -eff]

    def get_bounds(self):

        # Limits for n junctions should be pre-calculated for a black cell using cell_optimization

        if self.Eg_black is not None:
            lower_lim = []
            upper_lim = []
            for i1 in range(self.n_juncs):
                lower_lim.append(0.6 * self.Eg_black[i1])
                upper_lim.append(1.2 * self.Eg_black[i1])

            Eg_bounds = [lower_lim, upper_lim]

        else:
            Eg_bounds = [[0.5] * self.n_juncs, [4.0] * self.n_juncs]

        bounds = (
            self.bounds_passed[0] + Eg_bounds[0],
            self.bounds_passed[1] + Eg_bounds[1],
        )

        return bounds

    def get_name(self):
        return "Combined colour and efficiency optimization"

    def get_extra_info(self):
        return "\tDimensions: " + str(self.dim)

    def get_nobj(self):
        return 2


class color_optimization:
    def __init__(self, n_peaks, tg, photon_flux, spec_func=gen_spectrum_ndip):

        self.n_peaks = n_peaks
        self.target_color = tg
        self.spec_func = spec_func
        self.dim = n_peaks * 2

        self.col_wl = photon_flux[0]
        self.solar_spec = hc * photon_flux[1] / (self.col_wl * 1e-9)
        self.interval = np.round(np.diff(self.col_wl)[0], 6)
        self.cmf = load_cmf(self.col_wl)

    def fitness(self, x):

        R_spec = self.spec_func(x, self.n_peaks, wl=self.col_wl)
        XYZ = np.array(spec_to_XYZ(R_spec, self.solar_spec, self.cmf, self.interval))
        delta = delta_XYZ(self.target_color, XYZ)

        return [delta]

    def get_bounds(self):

        bounds = ([425, 500, 0, 0], [476, 625, 150, 160])

        return bounds

    def get_name(self):
        return "Colour optimization only"

    def get_extra_info(self):
        return "\tDimensions: " + str(self.dim)

    def get_nobj(self):
        return 1


class cell_optimization:
    """Class to create object for pygmo2 for the optimization of the bandgaps of a cell for a given incident photon flux (ignoring colour, i.e. a black cell)."""
    def __init__(
        self, n_juncs: int, photon_flux: np.ndarray, power_in: float = 1000.0, eta_ext: float = 1.0, fixed_bandgaps: Sequence=[]
    ):
        """Initializes the object for cell optimization.

        :param n_juncs: Number of junctions in the cell
        :param photon_flux: Photon flux in the cell; first row is wavelengths in nm, second row is flux (in units of photons/s/m^2/nm)
        :param power_in: Incident power in the cell (in units of W/m^2)
        :param eta_ext: External quantum efficiency of the cell (0-1)
        :param fixed_bandgaps: List of fixed bandgaps (in eV) for the cell (optional); these bandgaps are fixed and will not be optimized.
                                These will be in addition to n_juncs which will be optimized, so you can specify e.g. n_juncs = 2 and
                                fixed_bandgaps = [1.5] to optimize two junctions and fix the bandgap of the third junction to 1.5 eV.

        """

        self.n_juncs = n_juncs

        self.cell_wl = photon_flux[0]
        self.solar_flux = photon_flux[1]
        self.E_limits = [1240 / np.max(self.cell_wl), 1240 / np.min(self.cell_wl)]
        self.incident_power = power_in
        self.interval = np.round(np.diff(self.cell_wl)[0], 6)
        self.dim = n_juncs
        self.eta_ext = eta_ext
        self.fixed_bandgaps = fixed_bandgaps

    def fitness(self, x):
        x = x.tolist() + self.fixed_bandgaps
        Egs = -np.sort(-np.array(x))  # [0]

        eta = (
            getPmax(Egs, self.solar_flux, self.cell_wl, self.interval, self.eta_ext)
            / self.incident_power
        )

        return [-eta]

    def get_bounds(self):

        Eg_bounds = [[self.E_limits[0]] * self.dim, [self.E_limits[1]] * self.dim]

        return Eg_bounds

    def get_name(self):
        return "Cell optimization only"

    def get_extra_info(self):
        return "\tDimensions: " + str(self.dim)

    def get_nobj(self):
        return 1


def plot_outcome(spec: np.ndarray, photon_flux_cell: np.ndarray, target: np.ndarray, name: str, Egs: Sequence=None, ax=None):
    """Function to plot the outcome (reflection spectrum, target and found colour) of a combined cell efficiency/colour
    optimization, or a colour-only optimization.

    :param spec: reflection spectrum
    :param photon_flux_cell: photon flux of the cell; first row is wavelengths, second row is photon flux (in photons/s/m^2/nm)
    :param target: target colour XYZ coordinates
    :param name: plot title
    :param Egs: list of bandgaps of the cell (optional)
    :param ax: matplotlib axis to plot on (optional)
    """

    cmf = load_cmf(photon_flux_cell[0])
    interval = np.round(np.diff(photon_flux_cell[0])[0], 6)

    found_xyz = spec_to_XYZ(
        spec, hc * photon_flux_cell[1] / (photon_flux_cell[0] * 1e-9), cmf, interval
    )
    color_xyz_f = XYZColor(*found_xyz)
    color_xyz_t = XYZColor(*target)
    color_srgb_f = convert_color(color_xyz_f, sRGBColor)
    color_srgb_t = convert_color(color_xyz_t, sRGBColor)

    color_srgb_f = [
        color_srgb_f.clamped_rgb_r,
        color_srgb_f.clamped_rgb_g,
        color_srgb_f.clamped_rgb_b,
    ]
    color_srgb_t = [
        color_srgb_t.clamped_rgb_r,
        color_srgb_t.clamped_rgb_g,
        color_srgb_t.clamped_rgb_b,
    ]

    if ax is None:
        fig, ax = plt.subplots()
    ax.set_prop_cycle(color=["red", "green", "blue"])
    ax.fill_between(photon_flux_cell[0], 1, 1 - spec, color="black", alpha=0.3)
    ax.plot(photon_flux_cell[0], cmf / np.max(cmf))
    ax.plot(
        photon_flux_cell[0],
        photon_flux_cell[1] / np.max(photon_flux_cell[1]),
        "-k",
        alpha=0.5,
    )

    ax.set_title(name)

    ax.add_patch(
        Rectangle(xy=(800, 0.4), width=100, height=0.1, facecolor=color_srgb_t)
    )

    ax.add_patch(
        Rectangle(xy=(800, 0.3), width=100, height=0.1, facecolor=color_srgb_f)
    )

    if Egs is not None:
        for Eg in Egs:
            ax.axvline(x=1240 / Eg)

    return ax
