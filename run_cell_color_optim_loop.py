from color_cell_optimization import load_babel, multiple_color_cells
import numpy as np
from solcore.light_source import LightSource
import matplotlib.pyplot as plt

col_thresh = 0.004 # for a wavelength interval of 0.1, minimum achievable color error will be ~ 0.001. Fractional error in X, Y, Z.
acceptable_eff_change = 1e-4 # how much can the efficiency (in %) change between iteration sets?
n_trials = 8 # number of islands which will run concurrently
interval = 0.1 # wavelength interval (in nm)
wl_cell = np.arange(300, 4000, interval) # wavelengths

initial_iters = 100 # number of initial evolutions for the archipelago
add_iters = 100 # additional evolutions added each time if color threshold/convergence condition not met
# every color will run a minimum of initial_iters + add_iters evolutions before ending!

max_trials_col = 3*add_iters # how many population evolutions happen before giving up if there are no populations
                            # which meet the color threshold

type = "sharp" # "sharp" for rectangular dips or "gauss" for gaussians
fixed_height = False # fixed height peaks (will be at the value of max_height) or not

max_height = 1 # maximum height of reflection peaks
base = 0 # baseline fixed reflection

n_junc_loop = [1, 2, 3, 4, 5, 6]

n_peak_loop = [1, 2, 3, 4]

color_names, color_XYZ = load_babel() # 24 default Babel colors
# color_names = color_names[:5]
# color_XYZ = color_XYZ[:5]

photon_flux_cell = np.array(LightSource(
    source_type="standard", version="AM1.5g", x=wl_cell, output_units="photon_flux_per_nm"
).spectrum(wl_cell))

photon_flux_color = photon_flux_cell[:, np.all((photon_flux_cell[0] >= 380, photon_flux_cell[0] <= 780), axis=0)]

shapes = ['+', 'o', '^', '.', '*', "v", "s", "x"]

loop_n = 0

if __name__ == "__main__":
# Need this because otherwise the parallel running of the different islands (n_trials) may throw an error

    for n_peaks in n_peak_loop:
        for n_junctions in n_junc_loop:
            for fixed_height in [True, False]:
                print(n_peaks, "peaks,", n_junctions, "junctions,", "fixed height:", fixed_height)
                result = multiple_color_cells(color_XYZ, color_names, photon_flux_cell,
                                               n_peaks=n_peaks, n_junctions=n_junctions,
                                          type=type, fixed_height=fixed_height,
                                          n_trials=n_trials, initial_iters=initial_iters, add_iters=add_iters,
                                          col_thresh=col_thresh, acceptable_eff_change=acceptable_eff_change,
                                          max_trials_col=max_trials_col, base=base, max_height=max_height,
                                          plot=False)

                champion_effs = result["champion_eff"]
                champion_pops = result["champion_pop"]

                final_populations = result["archipelagos"]

                np.savetxt("results/champion_eff_tcheb_" + type + str(n_peaks) + '_' + str(
                    n_junctions) + '_' + str(fixed_height) + str(max_height) + '_' + str(base)+'.txt', champion_effs)
                np.savetxt("results/champion_pop_tcheb_" + type + str(n_peaks) + '_' + str(
                    n_junctions) + '_' + str(fixed_height) + str(max_height) + '_' + str(base) +'.txt', champion_pops)
                np.save("results/final_pop_tcheb_" + type + str(n_peaks) + '_' + str(
                    n_junctions) + '_' + str(fixed_height) + str(max_height) + '_' + str(base)+'.npy', final_populations)

            # plt.figure()
            # plt.plot(color_names, champion_effs[0], marker=shapes[0], mfc='none', linestyle='none', label="Fixed")
            # plt.plot(color_names, champion_effs[1], marker=shapes[1], mfc='none', linestyle='none', label="Not fixed")
            #
            # plt.xticks(rotation=45)
            # plt.legend()
            # plt.ylabel("Efficiency (%)")
            # # plt.title("Pop:" + str(pop_size) + "Iters:" + str(n_iters) + "Time:" + str(time_taken))
            # plt.tight_layout()
            # plt.show()
            #
            # plt.figure()
            # plt.plot(color_names, champion_bandgaps[0], marker=shapes[0], mfc='none', linestyle='none', label="Fixed")
            # plt.plot(color_names, champion_bandgaps[1], marker=shapes[1], mfc='none', linestyle='none', label="Not fixed")
            #
            # plt.xticks(rotation=45)
            # plt.legend()
            # plt.ylabel("Bandgap (eV)")
            # # plt.title("Pop:" + str(pop_size) + "Iters:" + str(n_iters) + "Time:" + str(time_taken))
            # plt.tight_layout()
            # plt.show()

