from ecopv.main_optimization import load_colorchecker, multiple_color_cells
import numpy as np
from solcore.light_source import LightSource
import matplotlib.pyplot as plt
from time import time
import os

col_thresh = 0.004  # for a wavelength interval of 0.1, minimum achievable color error deltaXYZ will be ~ 0.001.
# (deltaXYZ = maximum fractional error in X, Y, Z colour coordinates)
acceptable_eff_change = (
    1e-4  # how much can the efficiency (in %) change between iteration sets?
)
n_trials = 10  # number of islands which will run concurrently
interval = 0.1  # wavelength interval (in nm)
wl_cell = np.arange(280, 4000, interval)  # wavelengths

initial_iters = 100  # number of initial evolutions for the archipelago
add_iters = 100  # additional evolutions added each time if color threshold/convergence condition not met
# every color will run a minimum of initial_iters + add_iters evolutions before ending!

max_trials_col = 5 * add_iters
# how many population evolutions happen before giving up if there are no populations
# which meet the color threshold

R_type = "sharp"  # "sharp" for rectangular dips or "gauss" for gaussians
fixed_height = True  # fixed height peaks (will be at the value of max_height) or not
j01_method = "perfect_R"

max_height = 1  # maximum height of reflection peaks
base = 0  # baseline fixed reflection

n_junctions = 1  # number of junctions in the cell

n_peaks = 2  # number of reflection peaks

color_names, color_XYZ = load_colorchecker()  # load the 24 default ColorChecker colors

single_junction_data = np.loadtxt(os.path.join(os.path.dirname(os.path.dirname(
    __file__)), 'ecopv', 'data',
                   'paper_colors.csv'), skiprows=1, delimiter=',',
    usecols=np.arange(2,10))
# Define the incident photon flux. This should be a 2D array with the first row being the wavelengths and the second row
# being the photon flux at each wavelength. The wavelengths should be in nm and the photon flux in photons/m^2/s/nm.
photon_flux_cell = np.array(
    LightSource(
        source_type="standard",
        version="AM1.5g",
        x=wl_cell,
        output_units="photon_flux_per_nm",
    ).spectrum(wl_cell)
)

# Use only the visible range of wavelengths (380-780 nm) for color calculations:
photon_flux_color = photon_flux_cell[
    :, np.all((photon_flux_cell[0] >= 380, photon_flux_cell[0] <= 780), axis=0)
]

shapes = ["+", "o", "^", ".", "*", "v", "s", "x"]

loop_n = 0

if __name__ == "__main__":

    start = time()
    # Need this because otherwise the parallel running of the different islands (n_trials) may throw an error

    fig1 = plt.figure(1, figsize=(7.5, 3))

    # Run for the selected peak shape, with both fixed and non-fixed height
    result = multiple_color_cells(
        color_XYZ,
        color_names,
        photon_flux_cell,
        n_peaks=n_peaks,
        n_junctions=n_junctions,
        R_type=R_type,
        fixed_height=fixed_height,
        n_trials=n_trials,
        initial_iters=initial_iters,
        add_iters=add_iters,
        col_thresh=col_thresh,
        acceptable_eff_change=acceptable_eff_change,
        max_trials_col=max_trials_col,
        base=base,
        max_height=max_height,
        plot=True,
        j01_method=j01_method,
    )

    plt.figure()
    plt.plot(
        color_names,
        result["champion_eff"],
        marker=shapes[0],
        mfc="none",
        linestyle="none",
    )

    plt.plot(color_names, single_junction_data[:,3],
             marker=shapes[1], mfc='none', linestyle='none')

    plt.xticks(rotation=45)
    plt.legend()
    plt.ylabel("Efficiency (%)")
    # plt.title("Pop:" + str(pop_size) + "Iters:" + str(n_iters) + "Time:" + str(time_taken))
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(
        color_names,
        result["champion_pop"][:, -n_junctions:],
        marker=shapes[0],
        mfc="none",
        linestyle="none",
    )

    plt.plot(color_names, single_junction_data[:,7],
             marker=shapes[1], mfc='none', linestyle='none')

    plt.xticks(rotation=45)
    plt.legend()
    plt.ylabel("Bandgap (eV)")
    plt.tight_layout()
    plt.show()

    # champion_pop = np.array([reorder_peaks(x, n_peaks) for x in champion_pop])
    # np.save("results/champion_eff_tcheb_adaptpopsize_gauss_vheight2" + str(n_peaks) + '_' + str(n_junctions) + '_' + str(ntest), champion_eff)
    # np.save("results/champion_pop_tcheb_adaptpopsize_gauss_vheight2" + str(n_peaks) + '_' + str(n_junctions) + '_' + str(ntest), champion_pop)
    # np.save("results/niters_tcheb_adaptpopsize_gauss_vheight2" + str(n_peaks) + '_' + str(n_junctions) + '_' + str(ntest), iters_needed)
    print(time() - start)