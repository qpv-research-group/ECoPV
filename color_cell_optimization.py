import pygmo as pg
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.special import lambertw
from solcore.constants import kb, q, h, c
import numpy as np
from scipy.interpolate import interp1d

from colormath.color_objects import LabColor, XYZColor, sRGBColor, xyYColor, SpectralColor
from colormath.color_conversions import convert_color

from time import time

k = kb/q
h_eV = h/q
e = np.exp(1)
T = 298
kbT = k*T
pref = ((2*np.pi* q)/(h_eV**3 * c**2))* kbT
hc = h*c

def load_babel(output_coords="XYZ"):
    color_names = np.array([
        "DarkSkin", "LightSkin", "BlueSky", "Foliage", "BlueFlower", "BluishGreen",
        "Orange", "PurplishBlue", "ModerateRed", "Purple", "YellowGreen", "OrangeYellow",
        "Blue", "Green", "Red", "Yellow", "Magenta", "Cyan", "White-9-5", "Neutral-8",
        "Neutral-6-5", "Neutral-5", "Neutral-3-5", "Black-2"
    ])

    single_J_result = pd.read_csv("data/paper_colors.csv")

    color_xyY = np.array(single_J_result[['x', 'y', 'Y']])

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



def load_cmf(wl):
    cmf = np.loadtxt('data/cmf.txt')
    cmf_new = np.zeros((len(wl), 3))  # Interpolating cmf data to be in 1nm intervals

    # this could be done in one go
    intfunc = interp1d(cmf[:, 0], cmf[:, 1], fill_value=(0, 0), bounds_error=False)
    cmf_new[:, 0] = intfunc(wl)
    intfunc = interp1d(cmf[:, 0], cmf[:, 2], fill_value=(0, 0), bounds_error=False)
    cmf_new[:, 1] = intfunc(wl)
    intfunc = interp1d(cmf[:, 0], cmf[:, 3], fill_value=(0, 0), bounds_error=False)
    cmf_new[:, 2] = intfunc(wl)

    return cmf_new


def convert_xyY_to_Lab(xyY_list):
    xyY = xyYColor(*xyY_list)
    lab = convert_color(xyY, LabColor)
    return lab.get_value_tuple()


def convert_xyY_to_XYZ(xyY_list):
    xyY = xyYColor(*xyY_list)
    lab = convert_color(xyY, XYZColor)
    return lab.get_value_tuple()


def convert_XYZ_to_Lab(XYZ_list):
    XYZ = XYZColor(*XYZ_list)
    lab = convert_color(XYZ, LabColor)
    return lab.get_value_tuple()


def gen_spectrum_ndip(pop, n_peaks, wl, max_height=1, base=0):  # center and width in nm

    centres = pop[:n_peaks]
    widths = pop[n_peaks:2 * n_peaks]

    # centres and widths should be np arrays
    spectrum = np.ones_like(wl) * base

    lower = centres - widths / 2
    upper = centres + widths / 2

    for i in range(len(centres)):
        spectrum[np.all((wl >= lower[i], wl <= upper[i]), axis=0)] += max_height

    # possible peaks are overlapping; R can't be more than peak value

    spectrum[spectrum > max_height] = max_height

    return spectrum


def gen_spectrum_ndip_varyheight(pop, n_peaks, wl, max_height=1, base=0):  # center and width in nm

    # centres and widths should be np arrays
    spectrum = np.ones_like(wl) * base
    centres = pop[:n_peaks]
    widths = pop[n_peaks:2 * n_peaks]
    heights = pop[2 * n_peaks:3 * n_peaks]

    lower = centres - widths / 2
    upper = centres + widths / 2

    for i in range(len(centres)):
        spectrum[np.all((wl >= lower[i], wl <= upper[i]), axis=0)] += heights[i]

    # possible peaks are overlapping; R can't be more than peak value

    spectrum[spectrum > max_height] = max_height

    return spectrum


def gen_spectrum_ngauss(pop, n_peaks, wl, max_height=1, base=0):  # center and width in nm
    centres = pop[:n_peaks]
    widths = pop[n_peaks:2 * n_peaks]

    spectrum = np.zeros_like(wl)

    for i in range(len(centres)):
        spectrum += np.exp(-(wl - centres[i]) ** 2 / (2 * widths[i] ** 2))

    return base + (max_height - base) * spectrum / max(spectrum)


def gen_spectrum_ngauss_varyheight(pop, n_peaks, wl, max_height=1, base=0):  # center and width in nm

    centres = pop[:n_peaks]
    widths = pop[n_peaks:2 * n_peaks]
    heights = pop[2 * n_peaks:3 * n_peaks]

    spectrum = np.zeros_like(wl)

    for i in range(len(centres)):
        spectrum += heights[i] * np.exp(-(wl - centres[i]) ** 2 / (2 * widths[i] ** 2))

    return base + (max(heights) - base) * spectrum / max(spectrum)


class make_spectrum_ndip:

    def __init__(self, n_peaks=2, target=np.array([0, 0, 0]), type="sharp", fixed_height=True, w_bounds=None,
                 h_bounds=[0.01, 1]):

        self.c_bounds = [380, 780]
        self.fixed_height = fixed_height
        self.n_peaks = n_peaks

        if w_bounds is None:
            if fixed_height:
                self.w_bounds = [0, np.max([120/n_peaks, (350/n_peaks) * target[1]])]

            else:
                self.w_bounds = [0, 400]

        else:
            self.w_bounds = w_bounds

        if type == "sharp":
            if fixed_height:
                self.n_spectrum_params = 2 * n_peaks
                self.spectrum_function = gen_spectrum_ndip

            else:
                self.h_bounds = h_bounds
                self.n_spectrum_params = 3 * n_peaks
                self.spectrum_function = gen_spectrum_ndip_varyheight

        elif type == "gauss":
            if fixed_height:
                self.n_spectrum_params = 2 * n_peaks
                self.spectrum_function = gen_spectrum_ngauss

            else:
                self.h_bounds = h_bounds
                self.n_spectrum_params = 3 * n_peaks
                self.spectrum_function = gen_spectrum_ngauss_varyheight

    def get_bounds(self):

        if self.fixed_height:
            return ([self.c_bounds[0]] * self.n_peaks +
                    [self.w_bounds[0]] * self.n_peaks,
                    [self.c_bounds[1]] * self.n_peaks +
                    [self.w_bounds[1]] * self.n_peaks)

        else:
            return ([self.c_bounds[0]] * self.n_peaks +
                    [self.w_bounds[0]] * self.n_peaks +
                    [self.h_bounds[0]] * self.n_peaks,
                    [self.c_bounds[1]] * self.n_peaks +
                    [self.w_bounds[1]] * self.n_peaks +
                    [self.h_bounds[1]] * self.n_peaks)

        # Ref https://scipython.com/blog/converting-a-spectrum-to-a-colour/


def spec_to_XYZ(spec, solar_spec, cmf, interval):
    # insert the name of the column as a string in brackets

    Ymax = np.sum(interval * cmf[:, 1] * solar_spec)
    X = np.sum(interval * cmf[:, 0] * solar_spec * spec)
    Y = np.sum(interval * cmf[:, 1] * solar_spec * spec)
    Z = np.sum(interval * cmf[:, 2] * solar_spec * spec)

    if Ymax == 0:
        return (X, Y, Z)

    else:
        X = X / Ymax
        Y = Y / Ymax
        Z = Z / Ymax
        XYZ = (X, Y, Z)

        return XYZ


def delta_E_CIE2000(Lab1, Lab2):
    """
    Returns the difference :math:`\Delta E_{ab}` between two given *CIE Lab*
    *array_like* colors using CIE 2000 recommendation.
    Parameters
    ----------
    Lab1 : array_like, (3,)        *CIE Lab* *array_like* color 1.
    Lab2 : array_like, (3,)        *CIE Lab* *array_like* color 2.
    Returns
    -------
    numeric:        color difference :math:`\Delta E_{ab}`.
    Ref: Lindbloom, B. (2009). Delta E (CIE 2000). Retrieved February 24,
            2014, from http://brucelindbloom.com/Eqn_DeltaE_CIE2000.html
    """

    L1, a1, b1 = np.ravel(Lab1)
    L2, a2, b2 = np.ravel(Lab2)

    kL = 1
    kC = 1
    kH = 1

    l_bar_prime = 0.5 * (L1 + L2)

    c1 = np.sqrt(a1 * a1 + b1 * b1)
    c2 = np.sqrt(a2 * a2 + b2 * b2)

    c_bar = 0.5 * (c1 + c2)
    c_bar7 = np.power(c_bar, 7)

    g = 0.5 * (1 - np.sqrt(c_bar7 / (c_bar7 + 25 ** 7)))

    a1_prime = a1 * (1 + g)
    a2_prime = a2 * (1 + g)
    c1_prime = np.sqrt(a1_prime * a1_prime + b1 * b1)
    c2_prime = np.sqrt(a2_prime * a2_prime + b2 * b2)
    c_bar_prime = 0.5 * (c1_prime + c2_prime)

    h1_prime = (np.arctan2(b1, a1_prime) * 180) / np.pi
    if h1_prime < 0:
        h1_prime += 360

    h2_prime = (np.arctan2(b2, a2_prime) * 180) / np.pi
    if h2_prime < 0.0:
        h2_prime += 360

    h_bar_prime = (0.5 * (h1_prime + h2_prime + 360)
                   if np.fabs(h1_prime - h2_prime) > 180 else
                   0.5 * (h1_prime + h2_prime))

    t = (1 - 0.17 * np.cos(np.pi * (h_bar_prime - 30) / 180) +
         0.24 * np.cos(np.pi * (2 * h_bar_prime) / 180) +
         0.32 * np.cos(np.pi * (3 * h_bar_prime + 6) / 180) -
         0.20 * np.cos(np.pi * (4 * h_bar_prime - 63) / 180))

    if np.fabs(h2_prime - h1_prime) <= 180:
        delta_h_prime = h2_prime - h1_prime
    else:
        delta_h_prime = (h2_prime - h1_prime + 360
                         if h2_prime <= h1_prime else
                         h2_prime - h1_prime - 360)

    delta_L_prime = L2 - L1
    delta_C_prime = c2_prime - c1_prime
    delta_H_prime = (2 * np.sqrt(c1_prime * c2_prime) *
                     np.sin(np.pi * (0.5 * delta_h_prime) / 180))

    sL = 1 + ((0.015 * (l_bar_prime - 50) * (l_bar_prime - 50)) /
              np.sqrt(20 + (l_bar_prime - 50) * (l_bar_prime - 50)))
    sC = 1 + 0.045 * c_bar_prime
    sH = 1 + 0.015 * c_bar_prime * t

    delta_theta = (30 * np.exp(-((h_bar_prime - 275) / 25) *
                               ((h_bar_prime - 275) / 25)))

    c_bar_prime7 = c_bar_prime ** 7

    rC = np.sqrt(c_bar_prime7 / (c_bar_prime7 + 25 ** 7))
    rT = -2 * rC * np.sin(np.pi * (2 * delta_theta) / 180)

    return np.sqrt(
        (delta_L_prime / (kL * sL)) * (delta_L_prime / (kL * sL)) +
        (delta_C_prime / (kC * sC)) * (delta_C_prime / (kC * sC)) +
        (delta_H_prime / (kH * sH)) * (delta_H_prime / (kH * sH)) +
        (delta_C_prime / (kC * sC)) * (delta_H_prime / (kH * sH)) * rT)


def delta_XYZ(target, col):
    dXYZ = np.abs(target - col) / target

    return max(dXYZ)


def multiple_color_cells(color_XYZ, color_names, photon_flux, n_peaks=2, n_junctions=1, type="sharp", fixed_height="True",
                     n_trials=10, initial_iters=100, add_iters=100, col_thresh=0.004, acceptable_eff_change=1e-4,
                     max_trials_col=None, base=0, max_height=1, Eg_black=None, plot=True, fixed_bandgaps=None, power_in=1000):

    placeholder_obj = make_spectrum_ndip(n_peaks=n_peaks, type=type, fixed_height=fixed_height)
    n_params = placeholder_obj.n_spectrum_params + n_junctions
    pop_size = n_params * 10
    # print(n_peaks, 'peaks,', n_junctions, 'junctions', 'Population size:', pop_size)

    cmf = load_cmf(photon_flux[0])
    interval = np.diff(photon_flux[0])[0]

    if max_trials_col is None:
        max_trials_col = 5*initial_iters

    # width_bounds = [None]*len(color_XYZ)

    mean_sd_effs = np.empty((len(color_XYZ), 4))

    all_converged = False

    conv = np.array([False] * len(color_XYZ))

    color_indices = np.arange(len(color_XYZ))

    champion_eff = np.zeros(len(color_XYZ))
    champion_pop = np.empty((len(color_XYZ), n_params))

    archipelagos = [None] * len(color_XYZ)

    color_XYZ_found = [None] * len(color_XYZ)

    iters_needed = np.zeros(len(color_XYZ))

    # n_fronts = np.zeros((len(color_XYZ), n_trials))

    current_iters = initial_iters

    start_time = time()

    while not all_converged:

        start = time()
        print("Add iters:", current_iters)

        for k1 in color_indices:

            spectrum_obj = make_spectrum_ndip(n_peaks=n_peaks, target=color_XYZ[k1], type=type,
                                              fixed_height=fixed_height)

            internal_run = single_color_cell(plot_pareto=False, spectrum_function=spectrum_obj.spectrum_function)

            iters_needed[k1] += current_iters

            archi = internal_run.run(color_XYZ[k1], photon_flux,
                                     n_peaks, n_junctions, pop_size,
                                     current_iters, n_trials=n_trials, power_in=power_in,
                                     spectrum_bounds=spectrum_obj.get_bounds(),
                                     Eg_black=Eg_black,
                                     archi=archipelagos[k1],
                                     base=base,
                                     max_height=max_height,
                                     fixed_bandgaps=fixed_bandgaps)

            archipelagos[k1] = archi

            all_fs = np.stack([archi[j1].get_population().get_f() for j1 in range(n_trials)])
            all_xs = np.stack([archi[j1].get_population().get_x() for j1 in range(n_trials)])

            sln = all_fs[:, :, 0] < col_thresh

            # acc_fs = all_fs[sln]

            best_acc_ind = np.array(
                [np.argmin(x[sln[i1], 1]) if len(x[sln[i1]]) > 0 else 0 for i1, x in enumerate(all_fs)])
            best_acc_eff = np.array(
                [-np.min(x[sln[i1], 1]) if len(x[sln[i1]]) > 0 else 0 for i1, x in enumerate(all_fs)])
            best_acc_pop = np.array(
                [x[sln[i1]][best_acc_ind[i1]] if len(x[sln[i1]]) > 0 else [0] * n_params for i1, x in
                 enumerate(all_xs)])

            # all_acc_eff = -all_fs[sln, 1] * 100

            # plt.scatter([color_names[k1]]*n_trials, best_acc_eff * 100, color=colors, facecolors='none')

            max_eff_acc = best_acc_eff[best_acc_eff > 0] * 100
            best_acc_pop = best_acc_pop[best_acc_eff > 0]

            if len(max_eff_acc) > 0:

                print(color_names[k1], np.round(np.max(max_eff_acc), 3),
                      np.round(np.mean(max_eff_acc), 3), np.round(np.std(max_eff_acc), 6))

                ch_eff = np.max(max_eff_acc)
                ch_eff_ind = np.argmax(max_eff_acc)

                ch_pop = best_acc_pop[ch_eff_ind]
                # if not hasattr(ch_pop, "shape"):
                #     print("weird population", ch_eff, ch_eff_ind, max_eff_acc, best_acc_pop)
                mean_sd_effs[k1] = [np.min(max_eff_acc), ch_eff, np.mean(max_eff_acc), np.std(max_eff_acc)]

                delta_eta = ch_eff - champion_eff[k1]

                if delta_eta >= acceptable_eff_change:
                    champion_eff[k1] = ch_eff
                    champion_pop[k1] = ch_pop
                    print(np.round(delta_eta, 5), "delta eff - New champion efficiency")

                else:
                    print("No change/worse champion efficiency")
                    conv[k1] = True

                    spec = internal_run.spec_func(champion_pop[k1], n_peaks, photon_flux[0],
                                                  base=base, max_height=max_height)

                    color_XYZ_found[k1] = spec_to_XYZ(spec, hc*photon_flux[1]/(photon_flux[0]*1e-9), cmf, interval)

                    if plot:
                        spec = internal_run.spec_func(champion_pop[k1], n_peaks, photon_flux[0],
                                                      base=base, max_height=max_height)
                        plot_outcome(spec, photon_flux, color_XYZ[k1], color_names[k1])
                        plt.xlim(300,1000)
                        plt.show()

                    print("Champion pop:", champion_pop[k1], "width limits:", spectrum_obj.get_bounds())


            else:
                print(color_names[k1], "no acceptable populations", np.min(all_fs[:, :, 0]))
                if iters_needed[k1] >= max_trials_col:
                    flat_x = all_xs.reshape(-1, all_xs.shape[-1])
                    flat_f = all_fs.reshape(-1, all_fs.shape[-1])
                    best_col = np.argmin(flat_f[:, 0])
                    champion_pop[k1] = flat_x[best_col]
                    print("Cannot reach target color - give up. Minimum color deviation: " + str(
                        np.round(np.min(flat_f[:, 0]), 5)))
                    print("Champion pop (best color):", champion_pop[k1], "width limits:", spectrum_obj.get_bounds())
                    conv[k1] = True

                    spec = internal_run.spec_func(champion_pop[k1], n_peaks, photon_flux[0],
                                                  base=base, max_height=max_height)

                    color_XYZ_found[k1] = spec_to_XYZ(spec, h * c * photon_flux[1] / (photon_flux[0] * 1e-9), cmf,
                                                      interval)

                    if plot:
                        spec = internal_run.spec_func(champion_pop[k1], n_peaks, photon_flux[0],
                                                      base=base, max_height=max_height)
                        plot_outcome(spec, hc*photon_flux, color_XYZ[k1], color_names[k1] + " (target not reached)")
                        plt.xlim(300, 1000)
                        plt.show()

        time_taken = time() - start

        color_indices = np.where(~conv)[0]
        print(len(color_indices), "color(s) are still above acceptable std. dev. threshold. Took", time_taken, "s")

        if len(color_indices) == 0:
            print("All colors are converged")
            all_converged = True

        else:
            # n_iters = n_iters + 200
            print("Running for another", add_iters, "iterations")
            current_iters = add_iters

    print("TOTAL TIME:", time() - start_time)
    champion_pop = np.array([reorder_peaks(x, n_peaks, n_junctions, fixed_height) for x in champion_pop])

    color_Lab_found = [convert_XYZ_to_Lab(x) for x in color_XYZ_found]
    color_Lab_target = [convert_XYZ_to_Lab(x) for x in color_XYZ]

    delta_E = [delta_E_CIE2000(x, y) for x, y in zip(color_Lab_found, color_Lab_target)]
    print('Delta E*:', delta_E, np.max(delta_E))

    return {"champion_eff": champion_eff, "champion_pop": champion_pop, "archipelagos": archipelagos}


def multiple_colors(color_XYZ, color_names, photon_flux, n_peaks=2, type="sharp", fixed_height="True",
                     n_trials=10, initial_iters=100, add_iters=100, col_thresh=0.004, acceptable_eff_change=1e-4,
                     max_trials_col=None, base=0, max_height=1, plot=True,
                     electrical_calc=False, n_junctions=1, photon_flux_cell=None):

    placeholder_obj = make_spectrum_ndip(n_peaks=n_peaks, type=type, fixed_height=fixed_height)
    n_params = placeholder_obj.n_spectrum_params
    pop_size = n_params * 10
    # print(n_peaks, 'peaks,', n_junctions, 'junctions', 'Population size:', pop_size)

    if max_trials_col is None:
        max_trials_col = 5*initial_iters

    # width_bounds = [None]*len(color_XYZ)

    mean_sd_effs = np.empty((len(color_XYZ), 4))

    all_converged = False

    conv = np.array([False] * len(color_XYZ))

    color_indices = np.arange(len(color_XYZ))

    champion_eff = np.ones(len(color_XYZ))*100
    champion_pop = np.empty((len(color_XYZ), n_params))

    archipelagos = [None] * len(color_XYZ)

    iters_needed = np.zeros(len(color_XYZ))

    # n_fronts = np.zeros((len(color_XYZ), n_trials))

    current_iters = initial_iters

    start_time = time()

    while not all_converged:

        start = time()
        # print("Add iters:", current_iters)

        for k1 in color_indices:

            spectrum_obj = make_spectrum_ndip(n_peaks=n_peaks, target=color_XYZ[k1], type=type,
                                              fixed_height=fixed_height)

            internal_run = single_color(spectrum_function=spectrum_obj.spectrum_function)

            iters_needed[k1] += current_iters

            archi = internal_run.run(color_XYZ[k1], photon_flux,
                                     n_peaks, pop_size,
                                     current_iters, n_trials=n_trials,
                                     spectrum_bounds=spectrum_obj.get_bounds(),
                                     archi=archipelagos[k1],
                                     base=base,
                                     max_height=max_height)

            archipelagos[k1] = archi

            all_fs = np.stack([archi[j1].get_population().get_f() for j1 in range(n_trials)])
            all_xs = np.stack([archi[j1].get_population().get_x() for j1 in range(n_trials)])

            sln = all_fs[:, :, 0] < col_thresh

            # acc_fs = all_fs[sln]

            best_acc_ind = np.array(
                [np.argmin(x[sln[i1], 1]) if len(x[sln[i1]]) > 0 else 0 for i1, x in enumerate(all_fs)])
            best_acc_eff = np.array(
                [np.min(x[sln[i1], 1]) if len(x[sln[i1]]) > 0 else 0 for i1, x in enumerate(all_fs)])
            best_acc_pop = np.array(
                [x[sln[i1]][best_acc_ind[i1]] if len(x[sln[i1]]) > 0 else [0] * n_params for i1, x in
                 enumerate(all_xs)])

            # all_acc_eff = -all_fs[sln, 1] * 100

            # plt.scatter([color_names[k1]]*n_trials, best_acc_eff * 100, color=colors, facecolors='none')

            max_eff_acc = best_acc_eff[best_acc_eff > 0] * 100
            best_acc_pop = best_acc_pop[best_acc_eff > 0]

            # print(np.max(all_fs[:,1]), np.min(all_fs[:,1]))

            if len(max_eff_acc) > 0:

                # print(color_names[k1], np.round(np.max(max_eff_acc), 3),
                #       np.round(np.mean(max_eff_acc), 3), np.round(np.std(max_eff_acc), 6))

                ch_eff = np.max(max_eff_acc)
                ch_eff_ind = np.argmax(max_eff_acc)

                ch_pop = best_acc_pop[ch_eff_ind]
                # if not hasattr(ch_pop, "shape"):
                #     print("weird population", ch_eff, ch_eff_ind, max_eff_acc, best_acc_pop)
                mean_sd_effs[k1] = [np.min(max_eff_acc), ch_eff, np.mean(max_eff_acc), np.std(max_eff_acc)]

                delta_eta = champion_eff[k1] - ch_eff
                print(champion_eff[k1], ch_eff)

                if delta_eta >= acceptable_eff_change:
                    champion_eff[k1] = ch_eff
                    champion_pop[k1] = ch_pop
                    # print(np.round(delta_eta, 5), "delta eff - New champion efficiency")

                else:
                    # print("No change/worse champion efficiency")
                    conv[k1] = True

                    if plot:
                        spec = internal_run.spec_func(champion_pop[k1], n_peaks, photon_flux[0],
                                                      base=base, max_height=max_height)
                        plot_outcome(spec, photon_flux, color_XYZ[k1], color_names[k1])
                    # print("Champion pop:", champion_pop[k1], "width limits:", spectrum_obj.get_bounds())


            else:
                # print(color_names[k1], "no acceptable populations", np.min(all_fs[:, :, 0]))
                if iters_needed[k1] >= max_trials_col:
                    flat_x = all_xs.reshape(-1, all_xs.shape[-1])
                    flat_f = all_fs.reshape(-1, all_fs.shape[-1])
                    best_col = np.argmin(flat_f[:, 0])
                    champion_pop[k1] = flat_x[best_col]
                    print("Cannot reach target color - give up. Minimum color deviation: " + str(
                        np.round(np.min(flat_f[:, 0]), 5)))
                    # print("Champion pop (best color):", champion_pop[k1], "width limits:", spectrum_obj.get_bounds())
                    conv[k1] = True

                    if plot:
                        spec = internal_run.spec_func(champion_pop[k1], n_peaks, photon_flux[0],
                                                      base=base, max_height=max_height)
                        plot_outcome(spec, photon_flux, color_XYZ[k1], color_names[k1] + " (target not reached)")

        time_taken = time() - start

        color_indices = np.where(~conv)[0]
        # print(len(color_indices), "color(s) are still above acceptable std. dev. threshold. Took", time_taken, "s")

        if len(color_indices) == 0:
            # print("All colors are converged")
            all_converged = True

        else:
            # n_iters = n_iters + 200
            # print("Running for another", add_iters, "iterations")
            current_iters = add_iters

    # print("TOTAL TIME:", time() - start_time)

    # champion_pop = np.array([reorder_peaks(x, n_peaks, 0, fixed_height) for x in champion_pop])

    if electrical_calc:

        eta_max = np.empty(len(color_XYZ))
        pop_max = np.empty((len(color_XYZ), n_junctions))

        for k1 in range(len(color_XYZ)):
            spec = placeholder_obj.spectrum_function(champion_pop[k1], n_peaks, photon_flux_cell[0],
                                          base=base, max_height=max_height)

            print(champion_pop[k1], np.max(spec))

            p_init = cell_optimization(n_junctions,
                                       [photon_flux_cell[0], (1-spec)*photon_flux_cell[1]],
                                       eta_ext=1)

            prob = pg.problem(p_init)
            algo = pg.algorithm(pg.de(gen=1000, F=1, CR=1))

            pop = pg.population(prob, 20 * n_junctions)
            pop = algo.evolve(pop)

            eta_max[k1] = -pop.champion_f*100
            pop_max[k1] = pop.champion_x


        return {"champion_eff": champion_eff, "champion_pop": champion_pop, "archipelagos": archipelagos,
                "eta_max": eta_max, "pop_max": pop_max}

    else:

        return {"champion_eff": champion_eff, "champion_pop": champion_pop, "archipelagos": archipelagos}



class single_color_cell:

    def __init__(self, plot_pareto=False, fix_height=True, spectrum_function=gen_spectrum_ndip):
        self.plot_pareto = plot_pareto
        self.fix_height = fix_height
        self.spec_func = spectrum_function
        pass

    def run(self, target, photon_flux, n_peaks=2, n_gaps=1, popsize=80, gen=1000,
            n_trials=10, power_in=1000, spectrum_bounds=None, Eg_black=None, archi=None,
            fixed_bandgaps=None, **kwargs):

        p_init = color_function_mobj(n_peaks, n_gaps, target,
                                      photon_flux, self.spec_func,
                                      power_in, spectrum_bounds, Eg_black,
                                      fixed_bandgaps,
                                      **kwargs)

        udp = pg.problem(p_init)
        algo = pg.algorithm(pg.moead(gen=gen, CR=1, F=1,
                                     preserve_diversity=True))
                                     # decomposition="bi"))#, preserve_diversity=True, decomposition="bi"))

        if archi is None: archi = pg.archipelago(n=n_trials, algo=algo, prob=udp, pop_size=popsize)

        archi.evolve()

        archi.wait()

        return archi


class color_optimization_only:

    def __init__(self, fix_height=True, spectrum_function=gen_spectrum_ndip):
        self.fix_height = fix_height
        self.spec_func = spectrum_function

    def run(self, target, photon_flux, n_peaks=2, popsize=40, gen=1000,
            n_trials=10, ftol=1e-6, archi=None):

        p_init = color_optimization(n_peaks, target,
                                      photon_flux, self.spec_func)

        udp = pg.problem(p_init)
        algo = pg.algorithm(pg.de(gen=gen, CR=1, F=1, ftol=ftol))

        if archi is None: archi = pg.archipelago(n=n_trials, algo=algo, prob=udp, pop_size=popsize)

        archi.evolve()

        archi.wait()

        best_delta = np.min(archi.get_champions_f())
        return best_delta


class single_color:

    def __init__(self, fix_height=True, spectrum_function=gen_spectrum_ndip):
        self.fix_height = fix_height
        self.spec_func = spectrum_function
        pass

    def run(self, target, photon_flux, n_peaks=2, popsize=80, gen=1000,
            n_trials=10, spectrum_bounds=None,
            # ftol=1e-6,
            archi=None, **kwargs):

        # p_init = color_optimization(n_peaks, target,
        #                               photon_flux, self.spec_func, spectrum_bounds,
        #                               **kwargs)
        #
        # udp = pg.problem(p_init)
        # algo = pg.algorithm(pg.de(gen=gen, CR=1, F=1, ftol=ftol))

        p_init = color_optimization_min_photons(n_peaks, target,
                                    photon_flux, self.spec_func, spectrum_bounds,
                                    **kwargs)

        udp = pg.problem(p_init)
        algo = pg.algorithm(pg.moead(gen=gen, CR=1, F=1,
                                     preserve_diversity=True))

        if archi is None: archi = pg.archipelago(n=n_trials, algo=algo, prob=udp, pop_size=popsize)

        archi.evolve()

        archi.wait()

        return archi


def reorder_peaks(pop, n_peaks, n_junctions, fixed_height=True):
    peaks = pop[:n_peaks]
    bandgaps = pop[-n_junctions:] if n_junctions > 0 else np.array([])
    sorted_widths = np.array([x for _, x in sorted(zip(peaks, pop[n_peaks:2*n_peaks]))])

    if not fixed_height:
        sorted_heights = np.array([x for _, x in sorted(zip(peaks, pop[2*n_peaks:3 * n_peaks]))])

    peaks.sort()
    bandgaps.sort()

    if fixed_height:
        return np.hstack((peaks, sorted_widths, bandgaps))

    else:
        return np.hstack((peaks, sorted_widths,  sorted_heights, bandgaps))


def XYZ_from_pop_dips(pop, n_peaks, photon_flux, interval):
    cs = pop[:n_peaks]
    ws = pop[n_peaks:2*n_peaks]

    cmf = load_cmf(photon_flux[0])
    # T = 298

    R_spec = gen_spectrum_ndip(cs, ws, wl=photon_flux[0])
    XYZ = np.array(spec_to_XYZ(R_spec, hc*photon_flux[1]/(photon_flux[0]*1e-9), cmf, interval))

    return XYZ

def getPmax(egs, flux, wl, interval, rad_eff=1):
    # Since we need previous Eg info have to iterate the Jsc array
    jscs = np.empty_like(egs)  # Quick way of defining jscs with same dimensions as egs
    j01s = np.empty_like(egs)

    upperE = 4.14
    for i, eg in enumerate(egs):
        j01s[i] = (pref/rad_eff) * (eg ** 2 + 2 * eg * (kbT) + 2 * (kbT) ** 2) * np.exp(-(eg) / (kbT))
        jscs[i] = q * np.sum(flux[np.all((wl < 1240 / eg, wl > 1240 / upperE), axis=0)]) * interval
        # plt.figure()
        # plt.plot(wl_cell[np.all((wl_cell < 1240 / eg, wl_cell > 1240 / upperE), axis=0)], flux[np.all((wl_cell < 1240 / eg, wl_cell > 1240 / upperE), axis=0)])
        # plt.show()
        upperE = eg

    Vmaxs = (kbT * (lambertw(e * (jscs / j01s)) - 1)).real
    Imaxs = jscs - j01s * np.exp(Vmaxs / (kbT))

    # Find the minimum Imaxs
    minImax = np.amin(Imaxs)

    #   Find tandem voltage

    vsubcell = kbT * np.log((jscs - minImax) / j01s)
    vTandem = np.sum(vsubcell)

    return vTandem * minImax


def getIVmax(egs, flux, wl, interval, rad_eff=1):
    # Since we need previous Eg info have to iterate the Jsc array
    jscs = np.empty_like(egs)  # Quick way of defining jscs with same dimensions as egs
    j01s = np.empty_like(egs)

    upperE = 4.14
    for i, eg in enumerate(egs):
        j01s[i] = (pref/rad_eff) * (eg ** 2 + 2 * eg * (kbT) + 2 * (kbT) ** 2) * np.exp(-(eg) / (kbT))
        jscs[i] = q * np.sum(flux[np.all((wl < 1240 / eg, wl > 1240 / upperE), axis=0)]) * interval
        # plt.figure()
        # plt.plot(wl_cell[np.all((wl_cell < 1240 / eg, wl_cell > 1240 / upperE), axis=0)], flux[np.all((wl_cell < 1240 / eg, wl_cell > 1240 / upperE), axis=0)])
        # plt.show()
        upperE = eg

    Vmaxs = (kbT * (lambertw(e * (jscs / j01s)) - 1)).real
    Imaxs = jscs - j01s * np.exp(Vmaxs / (kbT))

    # Find the minimum Imaxs
    minImax = np.amin(Imaxs)

    #   Find tandem voltage

    vsubcell = kbT * np.log((jscs - minImax) / j01s)
    vTandem = np.sum(vsubcell)

    return Vmaxs, Imaxs

def getIVtandem(egs, flux, wl, interval, rad_eff=1):
    # Since we need previous Eg info have to iterate the Jsc array
    jscs = np.empty_like(egs)  # Quick way of defining jscs with same dimensions as egs
    j01s = np.empty_like(egs)

    upperE = 4.14
    for i, eg in enumerate(egs):
        j01s[i] = (pref/rad_eff) * (eg ** 2 + 2 * eg * (kbT) + 2 * (kbT) ** 2) * np.exp(-(eg) / (kbT))
        jscs[i] = q * np.sum(flux[np.all((wl < 1240 / eg, wl > 1240 / upperE), axis=0)]) * interval
        # plt.figure()
        # plt.plot(wl_cell[np.all((wl_cell < 1240 / eg, wl_cell > 1240 / upperE), axis=0)], flux[np.all((wl_cell < 1240 / eg, wl_cell > 1240 / upperE), axis=0)])
        # plt.show()
        upperE = eg

    Vmaxs = (kbT * (lambertw(e * (jscs / j01s)) - 1)).real
    Imaxs = jscs - j01s * np.exp(Vmaxs / (kbT))

    # Find the minimum Imaxs
    minImax = np.amin(Imaxs)

    #   Find tandem voltage

    vsubcell = kbT * np.log((jscs - minImax) / j01s)
    vTandem = np.sum(vsubcell)

    return vTandem, np.min(Imaxs)


class color_function_mobj:
    def __init__(self, n_peaks, n_juncs, tg, photon_flux, spec_func=gen_spectrum_ndip, power_in=1000,
                 spectrum_bounds=[], Eg_black=None, fixed_bandgaps=None, **kwargs):

        self.n_peaks = n_peaks
        self.n_juncs = n_juncs
        self.target_color = tg
        self.c_bounds = [380, 780]
        self.spec_func = spec_func
        self.dim = len(spectrum_bounds[0]) + n_juncs
        self.bounds_passed = spectrum_bounds
        self.Eg_black = Eg_black

        self.cell_wl = photon_flux[0]
        self.col_wl = self.cell_wl[np.all([self.cell_wl >= self.c_bounds[0], self.cell_wl <= self.c_bounds[1]], axis=0)]
        self.solar_flux = photon_flux[1]
        self.solar_spec = hc*self.solar_flux[np.all([self.cell_wl >= 380, self.cell_wl <= 780], axis=0)]/(self.col_wl*1e-9)
        self.incident_power = power_in
        self.interval = np.round(np.diff(self.cell_wl)[0], 6)
        self.cmf = load_cmf(self.col_wl)
        self.add_args = kwargs

        if fixed_bandgaps is not None:
            self.fixed_bandgaps = fixed_bandgaps

        else:
            self.fixed_bandgaps = []

    def calculate(self, x):

        Egs = x[-self.n_juncs:] if self.n_juncs > 0 else np.array([])
        Egs = Egs.tolist() + self.fixed_bandgaps
        Egs = -np.sort(-np.array(Egs))

        # T = 298 currently fixed!

        R_spec = self.spec_func(x, self.n_peaks, wl=self.col_wl, **self.add_args)

        XYZ = np.array(spec_to_XYZ(R_spec, self.solar_spec, self.cmf, self.interval))
        delta = delta_XYZ(self.target_color, XYZ)

        R_spec_cell = self.spec_func(x, self.n_peaks, wl=self.cell_wl, **self.add_args)

        flux = self.solar_flux * (1 - R_spec_cell)

        eta = getPmax(Egs, flux, self.cell_wl, self.interval)/self.incident_power

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
            Eg_bounds = [[0.5]*self.n_juncs,
                         [4.0]*self.n_juncs]

        bounds = (self.bounds_passed[0] +
                 Eg_bounds[0],
                self.bounds_passed[1] +
                 Eg_bounds[1])

        return bounds

    def get_name(self):
        return "color and efficiency optimization function"

    def get_extra_info(self):
        return "\tDimensions: " + str(self.dim)

    def get_nobj(self):
        return 2

class color_optimization:
    def __init__(self, n_peaks, tg, photon_flux, spec_func=gen_spectrum_ndip):

        self.n_peaks = n_peaks
        self.target_color = tg
        self.spec_func = spec_func
        self.dim = n_peaks*2

        self.col_wl = photon_flux[0]
        self.solar_spec = hc * photon_flux[1] / (
                    self.col_wl * 1e-9)
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
        return "Color optimization function"

    def get_extra_info(self):
        return "\tDimensions: " + str(self.dim)

    def get_nobj(self):
        return 1

class color_optimization_min_photons:
    def __init__(self, n_peaks, tg, photon_flux, spec_func=gen_spectrum_ndip,
                 bounds=[], **kwargs):

        self.n_peaks = n_peaks
        self.target_color = tg
        self.spec_func = spec_func
        self.dim = len(bounds[0])
        self.bounds_passed = bounds

        self.col_wl = photon_flux[0]
        self.solar_spec = hc*photon_flux[1]/(photon_flux[0]*1e-9)
        self.interval = np.round(np.diff(self.col_wl)[0], 6)
        self.cmf = load_cmf(self.col_wl)
        self.add_args = kwargs
        self.avg_flux = np.mean(self.solar_spec)

    def fitness(self, x):

        R_spec = self.spec_func(x, self.n_peaks, wl=self.col_wl, **self.add_args)

        XYZ = np.array(spec_to_XYZ(R_spec, self.solar_spec, self.cmf, self.interval))
        delta = delta_XYZ(self.target_color, XYZ)

        n_reflected = np.mean(R_spec*self.solar_spec)/self.avg_flux

        # print(self.target_color, XYZ, delta, n_reflected)

        return [delta, n_reflected]

    def get_bounds(self):

        bounds = (self.bounds_passed[0],
                self.bounds_passed[1])

        return bounds

    def get_name(self):
        return "Color optimization function"

    def get_extra_info(self):
        return "\tDimensions: " + str(self.dim)

    def get_nobj(self):
        return 2

class cell_optimization:
    def __init__(self, n_juncs, photon_flux, power_in=1000, eta_ext=1, fixed_bandgaps=[]):

        self.n_juncs = n_juncs

        self.cell_wl = photon_flux[0]
        self.solar_flux = photon_flux[1]
        self.E_limits = [1240/np.max(self.cell_wl), 1240/np.min(self.cell_wl)]
        self.incident_power = power_in
        self.interval = np.round(np.diff(self.cell_wl)[0], 6)
        self.dim = n_juncs
        self.eta_ext = eta_ext
        self.fixed_bandgaps = fixed_bandgaps


    def fitness(self, x):
        x = x.tolist() + self.fixed_bandgaps
        Egs = -np.sort(-np.array(x))  # [0]

        eta = getPmax(Egs, self.solar_flux, self.cell_wl, self.interval, self.eta_ext) / self.incident_power

        return [-eta]

    def get_bounds(self):

        # j1 = [0.7, 1.4]
        # j2 = [0.9, 1.8]
        # j3 = [1.1, 2]
        # j4 = [1.3, 2.2]
        # j5 = [1.5, 2.4]
        #
        # lims = [j1, j2, j3, j4, j5]
        #
        # lower_bounds = []
        # upper_bounds = []
        #
        # for k1 in range(self.dim):
        #     if k1 < 5:
        #         lower_bounds.append(lims[k1][0])
        #         upper_bounds.append(lims[k1][1])
        #
        #     else:
        #         lower_bounds.append(1.5)
        #         upper_bounds.append(3)

        Eg_bounds = [[self.E_limits[0]] * self.dim,
                     [self.E_limits[1]] * self.dim]

        return Eg_bounds

    def get_name(self):
        return "Cell-only optimization"

    def get_extra_info(self):
        return "\tDimensions: " + str(self.dim)

    def get_nobj(self):
        return 1


def plot_outcome(spec, photon_flux_cell, target, name, Egs=None, ax=None):

    cmf = load_cmf(photon_flux_cell[0])
    interval = np.round(np.diff(photon_flux_cell[0])[0], 6)

    found_xyz = spec_to_XYZ(spec, hc*photon_flux_cell[1]/(photon_flux_cell[0]*1e-9), cmf, interval)
    color_xyz_f = XYZColor(*found_xyz)
    color_xyz_t = XYZColor(*target)
    color_srgb_f = convert_color(color_xyz_f, sRGBColor)
    color_srgb_t = convert_color(color_xyz_t, sRGBColor)

    color_srgb_f = [color_srgb_f.clamped_rgb_r, color_srgb_f.clamped_rgb_g, color_srgb_f.clamped_rgb_b]
    color_srgb_t = [color_srgb_t.clamped_rgb_r, color_srgb_t.clamped_rgb_g, color_srgb_t.clamped_rgb_b]


    if ax is None: fig, ax = plt.subplots()
    ax.set_prop_cycle(color=['red', 'green', 'blue'])
    ax.fill_between(photon_flux_cell[0], 1, 1 - spec, color='black', alpha=0.3)
    ax.plot(photon_flux_cell[0], cmf / np.max(cmf))
    ax.plot(photon_flux_cell[0], photon_flux_cell[1] / np.max(photon_flux_cell[1]), '-k',
             alpha=0.5)

    # plt.xlim(300, 1000)
    ax.set_title(name)

    ax.add_patch(
        Rectangle(xy=(800, 0.4), width=100,
                  height=0.1,
                  facecolor=color_srgb_t)
    )

    ax.add_patch(
        Rectangle(xy=(800, 0.3), width=100,
                  height=0.1,
                  facecolor=color_srgb_f)
    )

    if Egs is not None:
        for Eg in Egs:
            ax.axvline(x=1240/Eg)

    return ax


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors

#
# def wavelength_to_rgb(wavelength, gamma=0.8):
#     ''' taken from http://www.noah.org/wiki/Wavelength_to_RGB_in_Python
#     This converts a given wavelength of light to an
#     approximate RGB color value. The wavelength must be given
#     in nanometers in the range from 380 nm through 750 nm
#     (789 THz through 400 THz).
#
#     Based on code by Dan Bruton
#     http://www.physics.sfasu.edu/astro/color/spectra.html
#     Additionally alpha value set to 0.5 outside range
#     '''
#     wavelength = float(wavelength)
#     if wavelength >= 380 and wavelength <= 750:
#         A = 1.
#     else:
#         A=0.5
#     if wavelength < 380:
#         wavelength = 380.
#     if wavelength >750:
#         wavelength = 750.
#     if wavelength >= 380 and wavelength <= 440:
#         attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
#         R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
#         G = 0.0
#         B = (1.0 * attenuation) ** gamma
#     elif wavelength >= 440 and wavelength <= 490:
#         R = 0.0
#         G = ((wavelength - 440) / (490 - 440)) ** gamma
#         B = 1.0
#     elif wavelength >= 490 and wavelength <= 510:
#         R = 0.0
#         G = 1.0
#         B = (-(wavelength - 510) / (510 - 490)) ** gamma
#     elif wavelength >= 510 and wavelength <= 580:
#         R = ((wavelength - 510) / (580 - 510)) ** gamma
#         G = 1.0
#         B = 0.0
#     elif wavelength >= 580 and wavelength <= 645:
#         R = 1.0
#         G = (-(wavelength - 645) / (645 - 580)) ** gamma
#         B = 0.0
#     elif wavelength >= 645 and wavelength <= 750:
#         attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
#         R = (1.0 * attenuation) ** gamma
#         G = 0.0
#         B = 0.0
#     else:
#         R = 0.0
#         G = 0.0
#         B = 0.0
#     return (R,G,B,A)


def wavelength_to_rgb(wavelengths, gamma=0.8):
    ''' taken from http://www.noah.org/wiki/Wavelength_to_RGB_in_Python
    This converts a given wavelength of light to an
    approximate RGB color value. The wavelength must be given
    in nanometers in the range from 380 nm through 750 nm
    (789 THz through 400 THz).

    Based on code by Dan Bruton
    http://www.physics.sfasu.edu/astro/color/spectra.html
    Additionally alpha value set to 0.5 outside range
    '''

    RGBA = np.zeros((len(wavelengths), 4))

    for wavelength in wavelengths:
        wavelength = float(wavelength)
        if wavelength >= 380 and wavelength <= 750:
            A = 1.
        else:
            A = 0.5
        if wavelength < 380:
            wavelength = 380.
        if wavelength > 750:
            wavelength = 750.
        if 380 <= wavelength <= 440:
            attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
            R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
            G = 0.0
            B = (1.0 * attenuation) ** gamma
        elif 440 <= wavelength <= 490:
            R = 0.0
            G = ((wavelength - 440) / (490 - 440)) ** gamma
            B = 1.0
        elif 490 <= wavelength <= 510:
            R = 0.0
            G = 1.0
            B = (-(wavelength - 510) / (510 - 490)) ** gamma
        elif 510 <= wavelength <= 580:
            R = ((wavelength - 510) / (580 - 510)) ** gamma
            G = 1.0
            B = 0.0
        elif 580 <= wavelength <= 645:
            R = 1.0
            G = (-(wavelength - 645) / (645 - 580)) ** gamma
            B = 0.0
        elif 645 <= wavelength <= 750:
            attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
            R = (1.0 * attenuation) ** gamma
            G = 0.0
            B = 0.0
        else:
            R = 0.0
            G = 0.0
            B = 0.0

        RGBA[wavelengths == wavelength] = (R, G, B, A)

    return RGBA