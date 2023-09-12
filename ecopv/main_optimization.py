import pygmo as pg
import numpy as np
from time import time
from colour.difference import delta_E_CIE2000
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import Sequence, Callable, Tuple
from solcore.light_source import LightSource

from colormath.color_objects import sRGBColor, XYZColor
from colormath.color_conversions import convert_color
from solcore.constants import h, c
import pathlib
from os.path import join, dirname
from ecopv.optimization_functions import reorder_peaks, getPmax
from ecopv.spectrum_functions import (
    load_cmf,
    load_D50,
    load_D65,
    spec_to_XYZ,
    convert_XYZ_to_Lab,
    convert_xyY_to_XYZ,
    convert_xyY_to_Lab,
    make_spectrum_ndip,
    gen_spectrum_ndip,
    delta_XYZ,
    XYZ_from_pop_dips
)

hc = h * c

current_path = pathlib.Path(__file__).parent.resolve()


# TODO:
# - write docstrings
# - allow passing other argeumtns to DE and MOAED using kwargs
# - allow plotting of Pareto front again
# - consistent spelling colour/color and optimi(z/s)ation


def load_colorchecker(output_coords: str = "XYZ",
                      source: str = "BabelColor",
                      illuminant = "AM1.5g",) -> Tuple[np.ndarray, np.ndarray]:
    """Load the colorchecker data from the csv file and return is as an array of coordinates

    :param output_coords: The color space to return the data in. Can be "XYZ" or "xyY"
    :param source: The source of the data. Can be "BabelColor" (default)  or
        "1J_paper". The BabelColor data uses reflectance spectra from
        https://babelcolor.com/index_htm_files/ColorChecker_RGB_and_spectra.zip
        (April 2012) to calculate colour coordinates depending on the illuminant. The
        1J data is the xyY data from Table 1 in https://doi.org/10.1039/c8ee03161d,
        so setting the illuminant has no effect.
    :param illuminant: The illuminant to use for the XYZ conversion. Can be "AM1.5g",
         "D50", or a temperature (will be interpreted as the temperature of a
         Sun-like black body). Alternatively, pass an array for any
         other illuminant with the relative spectral power distribution at 5 nm
         intervals from 380 nm to 730 nm (inclusive of start and endpoint). This only
         has an effect is source="BabelColor".

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

    if source == "BabelColor":

        color_R = np.loadtxt(join(dirname(current_path), "ecopv", "data",
                        "ColorChecker_R_BabelColor.csv"), delimiter=',',
                             encoding='utf-8-sig')
        # Measured reflectance of the ColorChecker colours. Downlaoded from:
        # https://babelcolor.com/colorchecker-2.htm
        # Based on measurements of 30 different ColorChecker cards.

        # Convert these reflectance values to XYZ using the chosen illuminant,
        # and then to xyY if necessary.

        wl = color_R[0]
        R = color_R[1:]

        cmf = load_cmf(wl)

        if illuminant == "AM1.5g":

            illum_array = np.array(
                LightSource(
                    source_type="standard",
                    version="AM1.5g",
                    x=wl,
                    output_units="power_density_per_nm",
                ).spectrum(wl)
            )[1]

        elif type(illuminant) == int or type(illuminant) == float:

            illum_array = np.array(
                LightSource(
                    source_type="black body",
                    x=wl,
                    output_units="power_density_per_nm",
                    entendue="Sun",
                    T=illuminant,
                ).spectrum(wl)
            )[1]

        elif illuminant == "D50":

            illum_array = load_D50(wl)

        elif illuminant == "D65":

            illum_array = load_D65(wl)

        else:
            illum_array = illuminant # assume it's an array
            if len(illum_array) != len(wl):
                raise ValueError("Illuminant array must be same length as wavelength array")

        XYZ = np.zeros((len(R), 3))

        for i1, R in enumerate(R):
            XYZ[i1] = spec_to_XYZ(R, illum_array, cmf, 5)

        if output_coords == "XYZ":
            return color_names, XYZ

        elif output_coords == "xyY":
            x = XYZ[:, 0] / np.sum(XYZ, axis=1)
            y = XYZ[:, 1] / np.sum(XYZ, axis=1)
            Y = XYZ[:, 1]
            color_xyY = np.array([x, y, Y]).T
            return color_names, color_xyY

        else:
            raise ValueError("output_coords must be one of xyY or XYZ")

    else:
        color_xyY = np.loadtxt(
            join(current_path, "data", "paper_colors.csv"),
            skiprows=1,
            usecols=[2, 3, 4],
            delimiter=",",
        )

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
    pop_size: int = None,
    R_type: str = "sharp",
    fixed_height: bool = True,
    n_trials: int = 10,
    iters_multiplier: int = 50,
    col_thresh: float = 0.004,
    col_cutoff: float = None,
    acceptable_eff_change=1e-4,
    max_trials_col: int = None,
    base: float = 0,
    max_height: float = 1,
    Eg_black: list = None,
    plot: bool = True,
    fixed_bandgaps: list = None,
    power_in: float = 1000,
    return_archipelagos: bool = False,
    return_convergence_info: bool = False,
    j01_method: str = "perfect_R",
    minimum_eff: Sequence[float] = None,
    seed_population: np.ndarray = None,
    illuminant: str = "AM1.5g",
    reinsert_optimal_Eg: float = 0,
    DE_options: dict = None,
    **kwargs,
) -> dict:

    """Optimize color and efficiency of multiple colored cells using pygmo2's moaed (multi-objective differential evolution)
    implementation, by calling single_color_cell() for each target colour.

    :param color_XYZ: numpy array of XYZ color coordinates of the target colors. Dimensions (n_colors, 3).
    :param color_names: names of the target colors. List of strings of length n_colors.
    :param photon_flux: incident photon flux. 2D numpy array with the first row being the wavelengths and the second row
                        being the photon flux at each wavelength. The wavelengths should be in nm and the photon flux in photons/m^2/s/nm.
    :param n_peaks: number of peaks in the spectrum
    :param n_junctions: number of junctions in the cell
    :param pop_size: population size for each island (thread) in the optimization
    :param R_type: type of spectrum, "sharp" or "gauss" currently implemented
    :param fixed_height: whether to fix the height of the reflection peaks to max_height (True) or allow it to vary (False)
    :param n_trials: number of islands (separate threads) which will run concurrently
    :param iters: number of additional evolutions per optimization loop
    :param col_thresh: maximum acceptable value for deltaXYZ (maximum fractional error in X, Y, Z coordinates)
    :param col_cutoff: if the color error is larger than this, color_function_mobj will set the efficiency to zero to
            restrict the Pareto front to only colors with colour error smaller than col_cutoff, rather than keeping the
            full Pareto front up to a deltaXYZ of 1.
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
    :param j01_method: method to use for calculating dark current. Can be "perfect_R"
       (default), "no_R" or "numerical_R".
    :param minimum_eff: minimum efficiency of the cell. If the efficiency is lower than
        this, the optimization will continue even if the other conditions are met. This
        could be e.g. an efficiency obtained from optimizing a cell with the same colour
        and fewer junctions.
    :param seed_population: population to seed the optimization with. This could be
        e.g. the result of the optimization with fewer junctions. If only a partial
        population is provided, the first n elements of the decision vector will be
        set.
    :param illuminant: illuminant to use for calculating the XYZ coordinates of the
        colours from the spectrum. Can be "AM1.5g", "D50" or "BB" (5778K black body). Can
        also pass an array of the same length as the spectrum with the spectral power
        density at each wavelength.
    :param DE_options: dictionary of options to pass to the pygmo2 differential evolution. See pygmo2
        documentation for details.
    :param kwargs: additional arguments to pass to color_function_mobj (including rad_eff)

    :return: results from the optimization in a dictionary with elements "champion_eff" (maximum cell efficiencies for
            each color), "champion_pop" (the champion population which maximizes the efficiency while staying within the
            allowed col_thresh) and, if requested, "archipelagos", which contains the final population of the n_trials islands
            for each color being optimized (very large objects!)
    """

    placeholder_obj = make_spectrum_ndip(
        n_peaks=n_peaks, R_type=R_type, fixed_height=fixed_height
    )
    n_params = placeholder_obj.n_spectrum_params + n_junctions
    pop_size = 10*n_params if pop_size is None else pop_size

    # cmf = load_cmf(photon_flux[0])
    interval = np.diff(photon_flux[0])[0]

    if max_trials_col is None:
        max_trials_col = 1000

    if minimum_eff is None:
        minimum_eff = np.zeros(len(color_XYZ))

    if seed_population is None:
        seed_population = [None] * len(color_XYZ)

    if col_cutoff is None:
        col_cutoff = 1 # keep the whole Pareto front

    wl_visible = photon_flux[0][np.all([photon_flux[0] >= 380, photon_flux[0] <=
                                        730], axis=0)]

    cmf_visible = load_cmf(wl_visible)

    if illuminant == "AM1.5g":

        illuminant = np.array(
            LightSource(
                source_type="standard",
                version="AM1.5g",
                x=wl_visible,
                output_units="power_density_per_nm",
            ).spectrum(wl_visible)[1]
        )
        # print("Loaded AM1.5g")

    elif illuminant == "BB":

        illuminant = np.array(
            LightSource(
                source_type="black body",
                x=wl_visible,
                output_units="power_density_per_nm",
                entendue="Sun",
                T=5778,
            ).spectrum(wl_visible)[1]
        )

    elif illuminant == "D50":

        illuminant = load_D50(wl_visible)

    elif illuminant == "D65":

            illuminant = load_D65(wl_visible)

    else:
        if len(illuminant) != len(wl_visible):
            raise ValueError("Illuminant array must be same length as photon flux "
                             "array")


    mean_sd_effs = np.empty((len(color_XYZ), 4))

    all_converged = False

    conv = np.array([False] * len(color_XYZ))

    color_indices = np.arange(len(color_XYZ))

    champion_eff = np.zeros(len(color_XYZ))
    champion_pop = np.empty((len(color_XYZ), n_params))

    archipelagos = [None] * len(color_XYZ)

    color_XYZ_found = [None] * len(color_XYZ)

    to_reset = [None] * len(color_XYZ)

    iters_needed = np.zeros(len(color_XYZ))

    current_iters = iters_multiplier*n_params

    best_eta_or_deltaXYZ = [[] for _ in range(len(color_XYZ))]

    start_time = time()

    while not all_converged:

        start = time()
        print(f"Not all converged - run {current_iters} more generations:")

        for k1 in color_indices:

            spectrum_obj = make_spectrum_ndip(
                n_peaks=n_peaks,
                target=color_XYZ[k1],
                R_type=R_type,
                fixed_height=fixed_height,
            )
            # the bounds are generated when calling make_spectrum_ndip depending on the target colour

            internal_run = single_color_cell(
                spectrum_function=spectrum_obj.spectrum_function
            )

            iters_needed[k1] += current_iters

            archi = internal_run.run(
                color_XYZ[k1],
                col_cutoff,
                photon_flux,
                illuminant,
                n_peaks,
                n_junctions,
                pop_size,
                current_iters,
                n_trials=n_trials,
                power_in=power_in,
                spectrum_bounds=spectrum_obj.get_bounds(), # spectrum_bounds which are sent to single_color_cell and then to color_function_mobj
                Eg_black=Eg_black,
                archi=archipelagos[k1],
                base=base,
                max_height=max_height,
                fixed_bandgaps=fixed_bandgaps,
                j01_method=j01_method,
                seed_pop=seed_population[k1],
                DE_options=DE_options,
                reinsert_optimal_Eg=reinsert_optimal_Eg,
                to_reset=to_reset[k1],
                **kwargs,
            )

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

            low_to_high = np.argsort(best_acc_eff)
            to_reset[k1] = low_to_high[:(n_trials // 5)]
            # reset worst 1/5 of populations

            # save best efficiency and population on each iteration
            # with open(f"{color_names[k1]}_order.txt", "a") as myfile:
            #     if iters_needed[k1] == current_iters:
            #         myfile.write("# New trial \n")
            #     myfile.write(' '.join(map(str, low_to_high)) + " \n")
            #
            # with open(f"{color_names[k1]}_efficiency.txt", "a") as myfile:
            #     if iters_needed[k1] == current_iters:
            #         myfile.write("# New trial \n")
            #     myfile.write(' '.join(map(str, best_acc_eff)) + " \n")
            #
            # with open(f"{color_names[k1]}_pops.txt", "a") as myfile:
            #     if iters_needed[k1] == current_iters:
            #         myfile.write("# New trial \n")
            #     for pop_list in best_acc_pop:
            #         myfile.write(' '.join(map(str, pop_list)) + " \n")

            max_eff_acc = best_acc_eff[best_acc_eff > 0] * 100
            best_acc_pop = best_acc_pop[best_acc_eff > 0]

            print('Island with best pop:', np.argmax(best_acc_eff))

            # if iters_needed[k1] == current_iters:
            #     print("Setting all populations to best")
            #     isl_best_pop = archi[np.argmax(best_acc_eff)].get_population()
            #     for isl in archi:
            #         isl.set_population(isl_best_pop)

            archipelagos[k1] = archi

            if len(max_eff_acc) > 0:

                print(
                    color_names[k1], "- max. efficiency:",
                    np.round(np.max(max_eff_acc), 3),
                    np.sum(sln),
                )

                ch_eff = np.max(max_eff_acc)
                ch_eff_ind = np.argmax(max_eff_acc)

                best_eta_or_deltaXYZ[k1].append(ch_eff)

                ch_pop = best_acc_pop[ch_eff_ind]

                print(ch_pop)

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
                    print(np.round(delta_eta, 5), "change in efficiency (new champion efficiency)")

                else:
                    print("No further improvement in champion efficiency", np.round(ch_pop, 3))

                    if ch_eff < minimum_eff[k1]:
                        print("Minimum efficiency not met - keep trying", ch_pop,
                              minimum_eff[k1])
                        # if ch_eff < champion_eff[k1]:
                        #     print("New efficiency worse than current champion, "
                        #           "reset population")
                        archipelagos[k1] = None

                    else:
                        conv[k1] = True

                        spec = internal_run.spec_func(
                            champion_pop[k1],
                            n_peaks,
                            wl_visible,
                            base=base,
                            max_height=max_height,
                        )

                        color_XYZ_found[k1] = spec_to_XYZ(
                            spec,
                            illuminant,
                            cmf_visible,
                            interval,
                        )

                        if plot:

                            plot_outcome(spec, photon_flux, color_XYZ[k1], illuminant,
                                         color_names[k1])
                            plt.xlim(300, 1000)
                            plt.show()

                        # print(
                        #     "Champion pop:",
                        #     champion_pop[k1],
                        #     "width limits:",
                        #     spectrum_obj.get_bounds(),
                        # )

            else:
                print(
                    color_names[k1],"- no acceptable populations. Minimum colour deviation: ",
                    np.min(all_fs[:, :, 0]),
                )
                best_eta_or_deltaXYZ[k1].append(np.min(all_fs[:, :, 0]))

                if iters_needed[k1] >= max_trials_col:
                    flat_x = all_xs.reshape(-1, all_xs.shape[-1])
                    flat_f = all_fs.reshape(-1, all_fs.shape[-1])
                    best_col = np.argmin(flat_f[:, 0])
                    champion_pop[k1] = flat_x[best_col]
                    print(
                        "Cannot reach target color - give up. Minimum colour deviation: "
                        + str(np.round(np.min(flat_f[:, 0]), 5))
                    )
                    # print(
                    #     "Champion pop (best color):",
                    #     champion_pop[k1],
                    #     # "width limits:",
                    #     # spectrum_obj.get_bounds(),
                    # )
                    conv[k1] = True

                    spec = internal_run.spec_func(
                        champion_pop[k1],
                        n_peaks,
                        wl_visible,
                        base=base,
                        max_height=max_height,
                    )

                    color_XYZ_found[k1] = XYZ_from_pop_dips(champion_pop[k1], n_peaks,
                                                            photon_flux, interval)

                    if plot:
                        spec = internal_run.spec_func(
                            champion_pop[k1],
                            n_peaks,
                            wl_visible,
                            base=base,
                            max_height=max_height,
                        )
                        plot_outcome(
                            spec,
                            hc * photon_flux,
                            color_XYZ[k1],
                            illuminant,
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


    print("TOTAL TIME:", time() - start_time)
    champion_pop = np.array(
        [reorder_peaks(x, n_peaks, n_junctions, fixed_height) for x in champion_pop]
    )

    color_Lab_found = [convert_XYZ_to_Lab(x) for x in color_XYZ_found]
    color_Lab_target = [convert_XYZ_to_Lab(x) for x in color_XYZ]

    delta_E = [delta_E_CIE2000(x, y) for x, y in zip(color_Lab_found, color_Lab_target)]
    # print("Delta E*:", delta_E, np.max(delta_E))

    results = {
            "champion_eff": champion_eff,
            "champion_pop": champion_pop
    }

    if return_archipelagos:
        results["archipelagos"] = archipelagos

    if return_convergence_info:
        results["convergence_info"] = best_eta_or_deltaXYZ

    return results


class single_color_cell:

    """Class to create object to run the optimization for a single coloured cell, combining colour calculation with
    the electrical model."""

    def __init__(
        self,
        fix_height: bool = True,
        spectrum_function: Callable = gen_spectrum_ndip,
        plot_pareto: bool = False,
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
        col_cutoff,
        photon_flux,
        illuminant,
        n_peaks=2,
        n_gaps=1,
        popsize=80,
        gen=1000,
        n_trials=10,
        power_in=1000.0,
        spectrum_bounds=None, # spectrum_bounds which will be passed to color_function_mobj
        Eg_black=None,
        archi=None,
        fixed_bandgaps=None,
        j01_method="perfect_R",
        seed_pop=None,
        DE_options=None,
        reinsert_optimal_Eg=0,
        to_reset=None,
        **kwargs
    ):

        if DE_options is None:
            DE_options = {}

        p_init = color_function_mobj(
            n_peaks,
            n_gaps,
            target,
            col_cutoff,
            photon_flux,
            illuminant,
            self.spec_func,
            power_in,
            spectrum_bounds,
            Eg_black,
            fixed_bandgaps,
            j01_method,
            **kwargs
        )

        F = DE_options.pop('F') if 'F' in DE_options.keys() else 1
        CR = DE_options.pop('CR') if 'CR' in DE_options.keys() else 1
        preserve_diversity = DE_options.pop('preserve_diversity') if 'preserve_diversity' in DE_options.keys() else True

        if archi is None:
            udp = pg.problem(p_init)
            algo = pg.algorithm(pg.moead(gen=gen, CR=CR, F=F, preserve_diversity=preserve_diversity, **DE_options))
            archi = pg.archipelago(n=n_trials, algo=algo, prob=udp, pop_size=popsize)#, t=pg.fully_connected())

            if seed_pop is not None:

                for isl in archi:

                    population = isl.get_population()
                    for i in range(popsize//2):
                        current_x = population.get_x()[i]
                        current_x[:len(seed_pop)] = seed_pop

                        population.set_x(i, current_x)

                    isl.set_population(population)

        if reinsert_optimal_Eg:
            # re-set bandgaps to black cell optimal Eg while keeping colour peaks the same
            # for some members of the population

            for isl in archi:

                population = isl.get_population()

                reset = np.random.randint(0, popsize, size=int(popsize*reinsert_optimal_Eg))

                # # best_col_ind = np.argmin(isl.get_population().get_f()[:, 0])
                # best_col_ind = np.where(isl.get_population().get_f()[:, 0] > 0.004)[0][-1]
                # # print('Current best', population.get_f()[best_col_ind -1])
                # current_x = population.get_x()[best_col_ind]
                # current_x[-n_gaps:] = Eg_black
                #
                # population.set_x(best_col_ind, current_x)
                #
                # print(population.get_f()[best_col_ind])
                # print(population.get_x()[best_col_ind])

                for i in reset:
                    current_x = population.get_x()[i]
                    current_x[-n_gaps:] = Eg_black

                    population.set_x(i, current_x)

                isl.set_population(population)

        if to_reset is not None:
            udp = pg.problem(p_init)
            for isl_ind in to_reset:
                print("reset island", isl_ind)
                archi[isl_ind].set_population(pg.population(prob=udp, size=popsize))

        archi.evolve()

        # print(archi.get_migration_log())

        archi.wait()

        return archi


class color_optimization_only:
    def __init__(self, fix_height=True, spectrum_function=gen_spectrum_ndip):
        self.fix_height = fix_height
        self.spec_func = spectrum_function

    def run(
        self,
        target,
        illuminant,
        n_peaks=2,
        popsize=40,
        gen=1000,
        n_trials=10,
        ftol=1e-6,
        archi=None,
        DE_options=None, # other arguments for pygmo de
    ):

        if DE_options is None:
            DE_options = {}

        F = DE_options.pop('F') if 'F' in DE_options.keys() else 1
        CR = DE_options.pop('CR') if 'CR' in DE_options.keys() else 1

        p_init = color_optimization(n_peaks, target, illuminant, self.spec_func)

        udp = pg.problem(p_init)
        algo = pg.algorithm(pg.de(gen=gen, CR=CR, F=F, ftol=ftol))

        if archi is None:
            archi = pg.archipelago(n=n_trials, algo=algo, prob=udp, pop_size=popsize)

        archi.evolve()

        archi.wait()

        best_delta = np.min(archi.get_champions_f())
        best_population = archi.get_champions_x()[np.argmin(archi.get_champions_f())]
        return best_delta, best_population


class color_function_mobj:
    def __init__(
        self,
        n_peaks,
        n_juncs,
        tg,
        col_cutoff,
        photon_flux,
        illuminant,
        spec_func=gen_spectrum_ndip,
        power_in=1000,
        spectrum_bounds=[], # spectrum_bounds is passed to the object here - these are only the bounds describing the
            # reflection peaks, not the bandgap. This is a tuple (or list) of lists with the format:
            # ([lower bound of peak 1 centre, lower bound of peak 2 centre, lower bound of peak 1 width, lower bound of peak 2 width],
            # [upper bound of peak 1 centre, upper bound of peak 2 centre, upper bound of peak 1 width, upper bound of peak 2 width])
            # Similar for more than two peaks.
        Eg_black=None,
        fixed_bandgaps=None,
        j01_method="perfect_R",
        **kwargs
    ):

        self.n_peaks = n_peaks
        self.n_juncs = n_juncs
        self.target_color = tg
        self.cutoff = col_cutoff
        self.c_bounds = [380, 730]
        self.spec_func = spec_func
        self.dim = len(spectrum_bounds[0]) + n_juncs
        self.bounds_passed = spectrum_bounds
        self.Eg_black = Eg_black

        self.cell_wl = photon_flux[0]
        self.upperE = 1240/min(self.cell_wl)
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
                np.all([self.cell_wl >= 380, self.cell_wl <= 730], axis=0)
            ]
            / (self.col_wl * 1e-9)
        )
        self.incident_power = power_in
        self.interval = np.round(np.diff(self.cell_wl)[0], 6)
        self.cmf = load_cmf(self.col_wl)
        self.add_args = kwargs
        self.j01_method = j01_method

        self.illuminant = illuminant

        if 'rad_eff' in kwargs.keys():
            self.rad_eff = kwargs.pop('rad_eff')

        else:
            self.rad_eff = [1]*n_juncs

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

        XYZ = np.array(spec_to_XYZ(R_spec, self.illuminant, self.cmf, self.interval))
        delta = delta_XYZ(self.target_color, XYZ)
        R_spec_cell = self.spec_func(x, self.n_peaks, wl=self.cell_wl, **self.add_args)

        flux = self.solar_flux * (1 - R_spec_cell)

        eta = getPmax(Egs, flux, self.cell_wl, self.interval,
                      x, upperE=self.upperE,
                      method=self.j01_method,
                      n_peaks=self.n_peaks,
                      rad_eff=self.rad_eff,
                      ) / self.incident_power

        if delta > self.cutoff:
            eta = 0

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
                lower_lim.append(np.max([0.6 * self.Eg_black[i1], self.Eg_black[i1] - 0.55]))
                upper_lim.append(np.min([1.2 * self.Eg_black[i1], self.Eg_black[i1] + 0.2]))
                # upper_lim.append(1.03*self.Eg_black[i1])
            Eg_bounds = [lower_lim, upper_lim]

        else:
            Eg_bounds = [[0.5] * self.n_juncs, [4.0] * self.n_juncs]

        bounds = (
            self.bounds_passed[0] + Eg_bounds[0],
            self.bounds_passed[1] + Eg_bounds[1],
        )
        # self.bounds_passed is the spectrum_bounds argument which was passed to the object, and contains the information
        # about the bounds for the reflection peaks. Eg_bounds is the bounds for the bandgaps. Overall a list of lists is
        # generated: [list of lower bounds, list of upper bounds]

        return bounds

    def get_name(self):
        return "Combined colour and efficiency optimization"

    def get_extra_info(self):
        return "\tDimensions: " + str(self.dim)

    def get_nobj(self):
        return 2


class color_optimization:
    def __init__(self, n_peaks, tg, illuminant, spec_func=gen_spectrum_ndip):

        self.n_peaks = n_peaks
        self.target_color = tg
        self.spec_func = spec_func
        self.dim = n_peaks * 2

        self.col_wl = illuminant[0]
        # self.solar_spec = hc * photon_flux[1] / (self.col_wl * 1e-9)
        self.interval = np.round(np.diff(self.col_wl)[0], 6)
        self.cmf = load_cmf(self.col_wl)
        # if illuminant == "D50":
        #     self.illuminant = load_D50(self.col_wl)
        #
        # else:
        self.illuminant = illuminant[1]

    def fitness(self, x):

        R_spec = self.spec_func(x, self.n_peaks, wl=self.col_wl)
        XYZ = np.array(spec_to_XYZ(R_spec, self.illuminant, self.cmf, self.interval))
        delta = delta_XYZ(self.target_color, XYZ)

        return [delta]

    def get_bounds(self):

        bounds = ([425, 500, 0, 0], [476, 625, 200, 200])

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
        self,
        n_juncs: int,
        photon_flux: np.ndarray,
        power_in: float = 1000.0,
        eta_ext: list[float] = 1.0,
        fixed_bandgaps: Sequence = [],
        j01_method: str = "no_R",
        Eg_limits: Sequence = None,
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
        self.upperE = 1240 / np.min(self.cell_wl)
        self.solar_flux = photon_flux[1]

        if Eg_limits is None:
            self.E_limits = [1240 / np.max(self.cell_wl), 1240 / np.min(self.cell_wl)]

        else:
            self.E_limits = Eg_limits

        self.incident_power = power_in
        self.interval = np.round(np.diff(self.cell_wl)[0], 6)
        self.dim = n_juncs
        self.eta_ext = eta_ext
        self.fixed_bandgaps = fixed_bandgaps
        self.j01_method = j01_method

    def fitness(self, x):
        x = x.tolist() + self.fixed_bandgaps
        Egs = -np.sort(-np.array(x))  # [0]
        eta = (
            getPmax(Egs, self.solar_flux, self.cell_wl, self.interval,
                    x, self.eta_ext,
                    self.upperE, method=self.j01_method)
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


def plot_outcome(
    spec: np.ndarray,
    photon_flux_cell: np.ndarray,
    target: np.ndarray,
    illuminant,
    name: str,
    Egs: Sequence = None,
    ax=None,
):
    """Function to plot the outcome (reflection spectrum, target and found colour) of a combined cell efficiency/colour
    optimization, or a colour-only optimization.

    :param spec: reflection spectrum
    :param photon_flux_cell: photon flux of the cell; first row is wavelengths, second row is photon flux (in photons/s/m^2/nm)
    :param target: target colour XYZ coordinates
    :param name: plot title
    :param Egs: list of bandgaps of the cell (optional)
    :param ax: matplotlib axis to plot on (optional)
    :param illuminant: illuminant to use for the conversion to sRGB (optional)
    """

    wl_visible = photon_flux_cell[0][np.all([photon_flux_cell[0] >= 380,
                                               photon_flux_cell[0] <= 730],
                                              axis=0)]
    cmf = load_cmf(wl_visible)
    interval = np.round(np.diff(photon_flux_cell[0])[0], 6)
    found_xyz = spec_to_XYZ(
        spec, illuminant, cmf, interval
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
    ax.fill_between(wl_visible, 1, 1 - spec, color="black", alpha=0.3)
    ax.plot(wl_visible, cmf / np.max(cmf))
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
