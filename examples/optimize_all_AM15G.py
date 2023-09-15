from ecopv.main_optimization import (
    load_colorchecker,
    multiple_color_cells,
    cell_optimization,
)
import numpy as np
from solcore.light_source import LightSource
import matplotlib.pyplot as plt
import pygmo as pg
from os import path

force_rerun = False # if True, will not load previous results but re-run and replace existing results
force_rerun_ideal = False # if True, will re-calculate black cell optimal bandgaps (used to determine bounds for
    # coloured cell optimization) even if they have already been calculated and saved

col_thresh = 0.004  # for a wavelength interval of 0.1, minimum achievable color error will be (very rough estimate!) ~ 0.001.
# This is the maximum allowed fractional error in X, Y, or Z colour coordinates.

acceptable_eff_change = 1e-4  # how much can the efficiency (in %) change between iteration sets? Stop when have reached
# col_thresh and efficiency change is less than this.

n_trials = 10  # number of islands which will run concurrently in parallel
interval = 0.1  # wavelength interval (in nm)
wl_cell = np.arange(
    300, 4000, interval
)  # wavelengths used for cell calculations (range of wavelengths in AM1.5G solar
# spectrum. For calculations relating to colour perception, only the visible range (380-730 nm) will be used.

iters_multiplier = 50  # 50 seems good. run iters_multiplier * n_params iterations per batch for each colour until convergence
# conditions are met

max_trials_col = 500
# how many population evolutions happen before giving up if there are no populations which meet the color threshold

R_type = "sharp"  # "sharp" for rectangular dips or "gauss" for gaussians
fixed_height = True  # fixed height peaks (will be at the value of max_height) if True, or peak height is an optimization
# variable if False
light_source_name = "AM1.5g"
j01_method = "perfect_R"

max_height = 1
# maximum height of reflection peaks; fixed at this value of if fixed_height = True

base = 0
# baseline fixed reflection (fixed at this value for both fixed_height = True and False).

n_junc_loop = [1, 2, 3, 4, 5, 6]  # loop through these numbers of junctions
n_peak_loop = [2, 3]  # loop through these numbers of reflection peaks

color_names, color_XYZ = load_colorchecker(illuminant="AM1.5g", output_coords="XYZ")
# load the names and XYZ coordinates of the 24 default Babel colors

# Use AM1.5G spectrum for cell calculations:
light_source = LightSource(
    source_type='standard',
    version=light_source_name,
    x=wl_cell,
    output_units="photon_flux_per_nm",
)

photon_flux_cell = np.array(light_source.spectrum(wl_cell))

# precalculate optimal bandgaps for N junctions for a black cell:
save_path = path.join(path.dirname(path.abspath(__file__)), "results")

for n_junctions in n_junc_loop:

    save_loc = save_path + "/champion_pop_{}juncs_{}spec.txt".format(
        n_junctions, light_source_name
    )

    if not path.exists(save_loc) or force_rerun_ideal:

        p_init = cell_optimization(
            n_junctions,
            photon_flux_cell,
            power_in=light_source.power_density,
            eta_ext=1,
            Eg_limits=[0.55, 2.4]
        )

        prob = pg.problem(p_init)
        algo = pg.algorithm(
            pg.de(
                gen=500*n_junctions,
                F=1,
                CR=1,
                ftol=0,
                xtol=0,
            )
        )

        pop = pg.population(prob, 30 * n_junctions)
        pop = algo.evolve(pop)

        champion_pop = np.sort(pop.champion_x)

        print(n_junctions, pop.problem.get_fevals()/(30*n_junctions),
              -pop.champion_f*100)

        np.savetxt(
            save_path
            + "/champion_pop_{}juncs_{}spec.txt".format(
                n_junctions, light_source_name
            ),
            champion_pop,
        )


if __name__ == "__main__":
    # Need this __main__ construction because otherwise the parallel running of the different islands (n_trials) may throw an error

    for n_peaks in n_peak_loop:
        for n_junctions in n_junc_loop:

            champion_bandgaps = np.zeros((len(color_names), n_junctions))

            Eg_guess = np.loadtxt(
                save_path
                + "/champion_pop_{}juncs_{}spec.txt".format(
                    n_junctions, light_source_name
                ),
                ndmin=1,
            )

            save_loc = (
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
                + "_"
                + j01_method + light_source_name + "_2.txt"
            )

            if not path.exists(save_loc) or force_rerun:

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
                    Eg_black=Eg_guess,
                    plot=False,
                    power_in=light_source.power_density,
                    return_archipelagos=False,
                    j01_method=j01_method,
                    illuminant=light_source_name,
                    # DE_options={},
                    n_reset=2,
                )

                champion_effs = result["champion_eff"]
                champion_pops = result["champion_pop"]
                champion_bandgaps = champion_pops[:, -n_junctions:]

                # final_populations = result["archipelagos"]

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
                    + "_" + j01_method + light_source_name + "_2.txt",
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
                    + "_" + j01_method + light_source_name + "_2.txt",
                    champion_pops,
                )
                # np.save(
                #     "results/final_pop_"
                #     + R_type
                #     + str(n_peaks)
                #     + "_"
                #     + str(n_junctions)
                #     + "_"
                #     + str(fixed_height)
                #     + str(max_height)
                #     + "_"
                #     + str(base)
                #     + "_"  + j01_method + light_source_name + ".npy",
                #     final_populations,
                # )

            else:

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
                    + "_"  + j01_method + light_source_name + "_2.txt",
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
                    + "_"  + j01_method + light_source_name + "_2.txt",
                )
                champion_bandgaps = champion_pops[:, -n_junctions:]


            plt.figure()
            plt.plot(
                color_names,
                champion_effs,
                marker='+',
                mfc="none",
                linestyle="none",
                label="Power density",
            )

            plt.xticks(rotation=45)
            plt.legend()
            plt.ylabel("Efficiency (%)")
            # plt.title("Pop:" + str(pop_size) + "Iters:" + str(n_iters) + "Time:" + str(time_taken))
            plt.title("Peaks:" + str(n_peaks) + "Junctions:" + str(n_junctions))
            plt.tight_layout()
            plt.show()

            plt.figure()
            plt.plot(
                color_names,
                champion_bandgaps,
                marker='+',
                mfc="none",
                linestyle="none",
                label="Power density",
            )

            plt.xticks(rotation=45)
            plt.legend()
            plt.ylabel("Bandgap (eV)")
            plt.title("Peaks:" + str(n_peaks) + "Junctions:" + str(n_junctions))
            plt.tight_layout()
            plt.show()
