from main_optimization import (
    load_colorchecker,
    multiple_color_cells,
    cell_optimization,
)
import numpy as np
from solcore.light_source import LightSource
import matplotlib.pyplot as plt
import pygmo as pg
from os import path

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
light_source_name = "bb"

max_height = (
    1  # maximum height of reflection peaks; fixed at this value of fixed_height = True
)
base = 0  # baseline fixed reflection (fixed at this value for both fixed_height = True and False).

n_junc_loop = [1, 2, 3, 4, 5, 6]  # loop through these numbers of junctions

n_peak_loop = [2]  # loop through these numbers of reflection peaks

fixed_height_loop = [True]

(
    color_names,
    color_XYZ,
) = (
    load_colorchecker()
)  # load the names and XYZ coordinates of the 24 default Babel colors

# Use AM1.5G spectrum:
light_source = LightSource(
    source_type="black body",
    x=wl_cell,
    output_units="photon_flux_per_nm",
    entendue="Sun",
    T=5778,
)

photon_flux_cell = np.array(light_source.spectrum(wl_cell))

photon_flux_color = photon_flux_cell[
    :, np.all((photon_flux_cell[0] >= 380, photon_flux_cell[0] <= 780), axis=0)
]

shapes = ["+", "o", "^", ".", "*", "v", "s", "x"]

loop_n = 0

# precalculate optimal bandgaps for junctions:
save_path = path.join(path.dirname(path.abspath(__file__)), "../results")

for n_junctions in n_junc_loop:

    save_loc = save_path + "/champion_pop_{}juncs_{}spec.txt".format(
        n_junctions, light_source_name
    )

    if not path.exists(save_loc):

        p_init = cell_optimization(
            n_junctions,
            photon_flux_cell,
            power_in=light_source.power_density,
            eta_ext=1,
        )

        prob = pg.problem(p_init)
        algo = pg.algorithm(
            pg.de(
                gen=2000,
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


if __name__ == "__main__":
    # Need this __main__ construction because otherwise the parallel running of the different islands (n_trials) may throw an error

    for n_peaks in n_peak_loop:
        for n_junctions in n_junc_loop:
            Eg_guess = np.loadtxt(
                save_path
                + "/champion_pop_{}juncs_{}spec.txt".format(
                    n_junctions, light_source_name
                ),
                ndmin=1,
            )

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
                    + light_source_name
                    + "_spd.txt"
                )

                if not path.exists(save_name):
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
                        power_in=light_source.power_density,
                        plot=False,
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
                        + light_source_name
                        + "_spd.txt",
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
                        + light_source_name
                        + "_spd.txt",
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
                        + light_source_name
                        + "_spd.npy",
                        final_populations,
                    )

                else:
                    print("Existing saved result found")
