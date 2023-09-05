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
import pandas as pd
from matplotlib import rc

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})


force_rerun = False

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

single_J_result = pd.read_csv("../ecopv/data/paper_colors.csv")

initial_iters = 100  # number of initial evolutions for the archipelago
add_iters = 100  # additional evolutions added each time if color threshold/convergence condition not met
# every color will run a minimum of initial_iters + add_iters evolutions before ending!

max_trials_col = 3 * add_iters
# how many population evolutions happen before giving up if there are no populations
# which meet the color threshold

R_type = "sharp"  # "sharp" for rectangular dips or "gauss" for gaussians
fixed_height = True  # fixed height peaks (will be at the value of max_height) if True, or peak height is an optimization
# variable if False
light_source_name = "AM1.5g"
j01_method = "perfect_R"

max_height = 1
# maximum height of reflection peaks; fixed at this value of if fixed_height = True

base = 0
# baseline fixed reflection (fixed at this value for both fixed_height = True and False).

n_junctions = 1  # loop through these numbers of junctions
n_peaks = 2  # loop through these numbers of reflection peaks

color_names, color_XYZ = load_colorchecker(illuminant="AM1.5g", output_coords="XYZ",
                                           source="1J_paper")

# Use AM1.5G spectrum for cell calculations:
light_source = LightSource(
    source_type="standard",
    version=light_source_name,
    x=wl_cell,
    output_units="photon_flux_per_nm",
)

photon_flux_cell = np.array(light_source.spectrum(wl_cell))

# precalculate optimal bandgaps for junctions:
save_path = path.join(path.dirname(path.abspath(__file__)), "results")

if __name__ == "__main__":
    # Need this __main__ construction because otherwise the parallel running of the different islands (n_trials) may throw an error

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
        + j01_method + light_source_name + "_1JXYZ.txt"
    )

    if not path.exists(save_loc) or force_rerun:

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
        Eg_black=Eg_guess,
        plot=False,
        power_in=light_source.power_density,
        return_archipelagos=False,
        j01_method=j01_method,
        illuminant=light_source_name,
        )

        champion_effs = result["champion_eff"]
        champion_pops = result["champion_pop"]
        champion_bandgaps = champion_pops[:, -n_junctions:]

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
            + "_" + j01_method + light_source_name + "_1JXYZ.txt",
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
            + "_" + j01_method + light_source_name + "_1JXYZ.txt",
            champion_pops,
        )

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
            + "_"  + j01_method + light_source_name + "_1JXYZ.txt",
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
            + "_" + j01_method + light_source_name + "_1JXYZ.txt",
        )
        champion_bandgaps = champion_pops[:, -n_junctions:]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4))
    ax1.plot(
        color_names, single_J_result["eta"], "ok", mfc="none", label="Previous results"
    )
    ax1.plot(color_names, champion_effs, "xr", label="MODE results")

    ax1.set_xticklabels(labels=color_names, rotation=45, ha='right')
    ax1.legend()
    ax1.set_ylabel("Efficiency (%)")

    ax2.plot(
        color_names, single_J_result["Eg"], "ok", mfc="none", label="Previous results"
    )
    ax2.plot(color_names, champion_bandgaps[:,0], "xr", label="MODE results")

    ax2.set_xticklabels(labels=color_names, rotation=45,ha='right')
    ax2.legend()
    ax2.set_ylabel("Bandgap (eV)")

    plt.tight_layout()
    plt.show()
