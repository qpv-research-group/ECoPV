from color_cell_optimization import load_babel, single_color, make_spectrum_ndip, cell_optimization
import numpy as np
from solcore.light_source import LightSource
import matplotlib.pyplot as plt
import pygmo as pg

col_thresh = 0.004 # for a wavelength interval of 0.1, minimum achievable colour error will be ~ 0.001. Fractional error in X, Y, Z.
n_trials = 3 # number of islands which will run concurrently
interval = 0.1 # wavelength interval (in nm)
wl_col = np.arange(380, 780, interval) # wavelengths
wl_cell = np.arange(300, 4000, interval) # wavelengths

initial_iters = 100 # number of initial evolutions for the archipelago
add_iters = 100 # additional evolutions added each time if colour threshold/convergence condition not met
# every colour will run a minimum of initial_iters + add_iters evolutions before ending!

max_trials_col = 5*add_iters # how many population evolutions happen before giving up if there are no populations
                            # which meet the colour threshold

type = "sharp" # "sharp" for rectangular dips or "gauss" for gaussians
fixed_height = False # fixed height peaks (will be at the value of max_height) or not

max_height = 0.9 # maximum height of reflection peaks
base = 0.15 # baseline fixed reflection

color_names, color_XYZ = load_babel() # 24 default Babel colours
color_names = color_names
color_XYZ = color_XYZ

photon_flux = np.array(LightSource(
    source_type="standard", version="AM1.5g", x=wl_col, output_units="photon_flux_per_nm"
).spectrum(wl_col))

photon_flux_cell = np.array(LightSource(
    source_type="standard", version="AM1.5g", x=wl_cell, output_units="photon_flux_per_nm"
).spectrum(wl_cell))

shapes = ['+', 'o', '^', '.', '*']

n_peaks = 2

n_junctions = 1

loop_n = 0

if __name__ == "__main__": # Need this because otherwise the parallel running of the different islands (n_trials) may throw an error

    plt.figure(figsize=(7.5, 3))

    for fixed_height in [True, False]:

        spec_obj = make_spectrum_ndip(n_peaks=n_peaks, type=type, fixed_height=fixed_height)

        champ_pops = np.empty((len(color_XYZ),spec_obj.n_spectrum_params))
        champ_fs = np.zeros(len(color_XYZ))

        champ_pops_cell = np.empty((len(color_XYZ),spec_obj.n_spectrum_params))
        champ_fs_cell = np.zeros(len(color_XYZ))

        for k1, target_col in enumerate(color_XYZ):
            spec_obj = make_spectrum_ndip(n_peaks=n_peaks, target=target_col, type=type, fixed_height=fixed_height)
            n_params = spec_obj.n_spectrum_params
            pop_size = n_params * 10
            obj = single_color(fix_height=fixed_height, spectrum_function=spec_obj.spectrum_function)

            archi = obj.run(target_col, photon_flux, n_peaks, pop_size, max_trials_col, n_trials,
                    spec_obj.get_bounds())

            champions = archi.get_champions_f()
            champion_pops = archi.get_champions_x()

            overall_champion = np.argmin(champions)
            champ_pops[k1] = champion_pops[overall_champion]
            champ_fs[k1] = champions[overall_champion]

            # Now we have the colour; generate incident spectrum for a cell and optimise junctions.
            #
            # R_spec = spec_obj.spectrum_function(champ_pops[k1], n_peaks, photon_flux_cell[0], max_height=max_height, base=base)
            # inc_light = (1-R_spec)*photon_flux_cell[1]
            #
            # p_init = cell_optimization(n_junctions, [photon_flux_cell[0], inc_light], power_in=1000, eta_ext=1)
            #
            # udp = pg.problem(p_init)
            # algo = pg.algorithm(pg.de(gen=max_trials_col, CR=1, F=1))
            #
            # archi_cell = pg.archipelago(n=n_trials, algo=algo, prob=udp, pop_size=n_junctions*10)
            #
            # archi_cell.evolve()
            #
            # archi_cell.wait()
            #
            # champions_cell = archi_cell.get_champions_f()
            # champion_pops_cell = archi_cell.get_champions_x()
            #
            # overall_champion_cell = np.argmin(champions_cell)
            # champ_pops_cell[k1] = champion_pops_cell[overall_champion_cell]
            # champ_fs_cell[k1] = champions_cell[overall_champion_cell]

        # plt.plot(color_names, -100*champ_fs_cell, 'o', mfc='none')
        # plt.show()
        #
        # plt.plot(color_names, champ_pops_cell, 'o', mfc='none')
        # plt.show()