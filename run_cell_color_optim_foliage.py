from color_cell_optimization import load_babel, multiple_color_cells, reorder_peaks
import numpy as np
from solcore.light_source import LightSource
import matplotlib.pyplot as plt
from colormath.color_objects import xyYColor, XYZColor
from colormath.color_conversions import convert_color
import seaborn as sns

col_thresh = 0.006 # for a wavelength interval of 0.1, minimum achievable color error will be ~ 0.001. Fractional error in X, Y, Z.
acceptable_eff_change = 1e-4 # how much can the efficiency (in %) change between iteration sets?
n_trials = 3 # number of islands which will run concurrently
interval = 0.1 # wavelength interval (in nm)
wl_cell = np.arange(300, 4000, interval) # wavelengths

initial_iters = 100 # number of initial evolutions for the archipelago
add_iters = 100 # additional evolutions added each time if color threshold/convergence condition not met
# every color will run a minimum of initial_iters + add_iters evolutions before ending!

max_trials_col = 5*add_iters # how many population evolutions happen before giving up if there are no populations
                            # which meet the color threshold

type = "sharp" # "sharp" for rectangular dips or "gauss" for gaussians
fixed_height = False # fixed height peaks (will be at the value of max_height) or not

max_height = 1 # maximum height of reflection peaks
base = 0 # baseline fixed reflection

n_junctions = 3

n_peaks = 2

color_names, color_xyY = load_babel(output_coords="xyY") # 24 default Babel colors
# color_names = color_names[:5]
# color_XYZ = color_XYZ[:5]
foliage = color_xyY[3]

foliage_Y = np.linspace(0.04, 0.93, 12)

foliage_xyY = np.ones((len(foliage_Y), 3))

foliage_xyY[:,0] = foliage[0]
foliage_xyY[:,1] = foliage[1]
foliage_xyY[:,2] = foliage_Y

foliage_XYZ = np.array([convert_color(xyYColor(*x), XYZColor).get_value_tuple() for x in foliage_xyY])

color_names = [str(np.round(x, 2)) for x in foliage_Y]
color_XYZ = foliage_XYZ

photon_flux_cell = np.array(LightSource(
    source_type="standard", version="AM1.5g", x=wl_cell, output_units="photon_flux_per_nm"
).spectrum(wl_cell))

photon_flux_color = photon_flux_cell[:, np.all((photon_flux_cell[0] >= 380, photon_flux_cell[0] <= 780), axis=0)]

shapes = ['+', 'o', '^', '.', '*', "v", "s", "x"]

loop_n = 0

if __name__ == "__main__": # Need this because otherwise the parallel running of the different islands (n_trials) may throw an error

    fig1 = plt.figure(1, figsize=(7.5, 3))

    # champion_effs = np.empty((2, len(color_XYZ)))
    # champion_bandgaps = np.empty((2, len(color_XYZ), n_junctions))

    # Run for the selected peak shape, with both fixed and non-fixed height
    for fixed_height in [True]:
        result = multiple_color_cells(color_XYZ, color_names, photon_flux_cell,
                                       n_peaks=n_peaks, n_junctions=n_junctions,
                                  type=type, fixed_height=fixed_height,
                                  n_trials=n_trials, initial_iters=initial_iters, add_iters=add_iters,
                                  col_thresh=col_thresh, acceptable_eff_change=acceptable_eff_change,
                                  max_trials_col=max_trials_col, base=base, max_height=max_height,
                                  plot=True)

        champion_effs = result["champion_eff"]
        champion_bandgaps = result["champion_pop"][:, -n_junctions:]
        champion_pop = result["champion_pop"]
        loop_n += 1


        plt.figure()
        plt.plot(color_names, champion_effs, marker=shapes[0], mfc='none', linestyle='none', label="Fixed height")

        plt.xticks(rotation=45)
        plt.legend()
        plt.ylabel("Efficiency (%)")
        # plt.title("Pop:" + str(pop_size) + "Iters:" + str(n_iters) + "Time:" + str(time_taken))
        plt.tight_layout()
        plt.show()

        plt.figure()
        plt.plot(color_names, champion_bandgaps, marker=shapes[0], mfc='none', linestyle='none', label="Fixed height")

        plt.xticks(rotation=45)
        plt.legend()
        plt.ylabel("Bandgap (eV)")
        # plt.title("Pop:" + str(pop_size) + "Iters:" + str(n_iters) + "Time:" + str(time_taken))
        plt.tight_layout()
        plt.show()


        champion_pop = np.array([reorder_peaks(x, n_peaks, n_junctions) for x in champion_pop])
        np.save("results/foliage_effs_" + str(n_peaks) + '_' + str(n_junctions), champion_effs)
        np.save("results/foliage_pops_" + str(n_peaks) + '_' + str(n_junctions), champion_pop)
        # np.save("results/niters_tcheb_adaptpopsize_gauss_vheight2" + str(n_peaks) + '_' + str(n_junctions) + '_' + str(ntest), iters_needed)

J1_eff = np.load("results/foliage_effs_" + str(n_peaks) + '_' + '1'+'.npy')
J1_pop = np.load("results/foliage_pops_" + str(n_peaks) + '_' + '1'+'.npy')

J2_eff = np.load("results/foliage_effs_" + str(n_peaks) + '_' + '2'+'.npy')
J2_pop = np.load("results/foliage_pops_" + str(n_peaks) + '_' + '2'+'.npy')

J3_eff = np.load("results/foliage_effs_" + str(n_peaks) + '_' + '3'+'.npy')
J3_pop = np.load("results/foliage_pops_" + str(n_peaks) + '_' + '3'+'.npy')

cols = sns.color_palette("husl", 3)

plt.figure()
plt.plot(color_names, J1_eff, 'o-', mfc='none', label="1 junction",
         color=cols[0])
plt.plot(color_names, J2_eff, 'o-', mfc='none', label="2 junctions",
         color=cols[1])
plt.plot(color_names, J3_eff, 'o-', mfc='none', label="3 junctions",
         color=cols[2])

plt.xticks(rotation=45)
plt.legend()
plt.ylabel("Efficiency (%)")
plt.xlabel("Luminance (Y)")
# plt.title("Pop:" + str(pop_size) + "Iters:" + str(n_iters) + "Time:" + str(time_taken))
plt.tight_layout()
plt.show()

fix, (ax1, ax2, ax3) = plt.subplots(3, figsize=(5, 6))
ax1.plot(color_names, J1_pop[:, -1], 'o-', mfc='none', label="1 junction",
         color=cols[0])


ax2.plot(color_names, J2_pop[:, -1], 'o-', mfc='none', label="1 junction",
         color=cols[1])
ax2.plot(color_names, J2_pop[:, -2], 'o-', mfc='none', label="1 junction",
         color=cols[1])

ax3.plot(color_names, J3_pop[:, -1], 'o-', mfc='none', label="1 junction",
         color=cols[2])
ax3.plot(color_names, J3_pop[:, -2], 'o-', mfc='none', label="1 junction",
         color=cols[2])
ax3.plot(color_names, J3_pop[:, -3], 'o-', mfc='none', label="1 junction",
         color=cols[2])


ax1.set_ylabel("Bandgap (eV)")
ax2.set_ylabel("Bandgaps (eV)")
ax3.set_ylabel("Bandgaps (eV)")

ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

ax2.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

ax3.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

plt.xlabel("Luminance (Y)")
# plt.title("Pop:" + str(pop_size) + "Iters:" + str(n_iters) + "Time:" + str(time_taken))
plt.tight_layout()
plt.show()