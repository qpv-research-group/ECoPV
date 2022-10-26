import pygmo as pg
from Spectrum_Functions_de import gen_spectrum_2dip, gen_spectrum_ndip, spec_to_xyz, delta_E_CIE2000, delta_XYZ
import numpy as np
from solcore.light_source import LightSource
from colormath.color_objects import LabColor, XYZColor, sRGBColor, xyYColor
from colormath.color_conversions import convert_color
import matplotlib.pyplot as plt
import pandas as pd
from skimage import io
from joblib import Parallel, delayed
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.special import lambertw
from solcore.constants import kb, q, h, c
from time import time
import cProfile
import pstats


k = kb/q
h_eV = h/q
e = 2.718281828459045
T = 298
kbT = k*T
pref = ((2*np.pi* q)/(h_eV**3 * c**2))* kbT

col_thresh = 0.01

n_trials = 10

def convert_xyY_to_Lab(xyY_list):

    xyY = xyYColor(*xyY_list)
    lab = convert_color(xyY, LabColor)
    return lab.get_value_tuple()

def convert_xyY_to_XYZ(xyY_list):
    xyY = xyYColor(*xyY_list)
    lab = convert_color(xyY, XYZColor)
    return lab.get_value_tuple()

def reorder_peaks(pop, n_peaks):
    peaks = pop[:n_peaks]
    bandgaps = pop[2*n_peaks:]
    sorted_widths = np.array([x for _, x in sorted(zip(peaks, pop[n_peaks:]))])
    peaks.sort()
    bandgaps.sort()
    return np.hstack((peaks, sorted_widths, bandgaps))

def getPmax(egs, flux):
    # Since we need previous eg info have to iterate the Jsc array
    jscs = np.empty_like(egs)  # Quick way of defining jscs with same dimensions as egs
    j01s = np.empty_like(egs)

    upperE = 4.14
    for i, eg in enumerate(egs):
        j01s[i] = pref * (eg ** 2 + 2 * eg * (kbT) + 2 * (kbT) ** 2) * np.exp(-(eg) / (kbT))
        jscs[i] = q * np.sum(flux[np.all((wl_cell < 1240 / eg, wl_cell > 1240 / upperE), axis=0)]) * interval
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

interval=0.1 # interval between each two wavelength points, 0.02 needed for low dE values
wl=np.arange(380,780+interval,interval)

wl_cell=np.arange(300, 4000, interval)

light = LightSource(
    source_type="standard", version="AM1.5g", x=wl_cell, output_units="photon_flux_per_nm"
).spectrum()[1]

photon_flux =  LightSource(
    source_type="standard", version="AM1.5g", x=wl, output_units="photon_flux_per_nm"
)

photon_flux_norm = photon_flux.spectrum(wl)[1]/max(photon_flux.spectrum(wl)[1])

photon_flux_cell = LightSource(
    source_type="standard", version="AM1.5g", x=wl_cell, output_units="photon_flux_per_nm"
).spectrum(wl_cell)[1]

solar_spec_col = LightSource(
    source_type="standard", version="AM1.5g", x=wl, output_units="photon_flux_per_nm"
).spectrum(wl)[1]

class two_dip_colour_function_mobj:
    def __init__(self, n_peaks, n_juncs, tg):
        self.dim = n_peaks*2 + n_juncs
        self.n_peaks = n_peaks
        self.n_juncs = n_juncs
        self.target_color = tg
        self.c_bounds = [380, 780]
        self.w_bounds = [0, 150]

    def calculate(self, x):
        # center = x[0]
        # center2 = x[1]
        # width = x[2]
        # width2 = x[3]
        # Eg = x[4]

        profile = cProfile.Profile()
        profile.enable()
        cs = x[:self.n_peaks]
        ws = x[self.n_peaks:2*self.n_peaks]
        Egs = -np.sort(-x[2*self.n_peaks:]) #[0]

        # T = 298

        R_spec = gen_spectrum_ndip(cs, ws)
        XYZ=np.array(spec_to_xyz(R_spec, solar_spec_col))
        delta = delta_XYZ(self.target_color, XYZ)
        # XYZ=XYZColor(XYZ[0],XYZ[1],XYZ[2])
        # Lab=convert_color(XYZ, LabColor)
        # delta=delta_E_CIE2000((Lab.lab_l,Lab.lab_a,Lab.lab_b),self.target_color)
        # delta = delta_XYZ(self.target_color, XYZ)

        # n_available = np.sum((1 - R_spec) * photon_flux_norm)/np.sum(photon_flux_norm)
        R_spec_cell = gen_spectrum_ndip(cs, ws, wl=wl_cell)

        # Establish an interpolation function to allow integration over arbitrary limits
        # solarfluxInterpolate = InterpolatedUnivariateSpline(wl_cell, light * (1 - R_spec_cell), k=1)
        #
        # Jsc = q * solarfluxInterpolate.integral(300, 1240 / Eg)
        # Jsc = q * int_flux

        flux = light * (1 - R_spec_cell)
        # Jsc = q*np.sum(flux[wl_cell < 1240/Eg])*interval
        #
        # J01 = pref * (Eg**2 + 2*Eg*(kbT) + 2*(kbT)**2)*np.exp(-(Eg)/(kbT))
        #
        # # Vmax = (kbT*(lambertw(np.exp(1)*(Jsc/J01))-1)).real
        # Vmax = (kbT * (lambertw(e * (Jsc / J01)) - 1)).real
        # Imax = Jsc - J01*np.exp(Vmax/(kbT))
        # eta = Vmax*Imax/1000

        eta = getPmax(Egs, flux)/1000
        # print(eta)

        profile.disable()
        ps = pstats.Stats(profile)
        ps.print_stats()

        return delta, eta

    def fitness(self, x):
        delta, eff = self.calculate(x)

        return [delta, -eff]

    def plot(self, x):

        delta_E, R_spec = self.calculate(x)

        plt.figure()
        plt.plot(wl, R_spec)
        plt.title(str(delta_E))
        plt.show()

    def get_bounds(self):
        return ([self.c_bounds[0]] * self.n_peaks + [self.w_bounds[0]] * self.n_peaks + self.n_juncs*[0.9],
                [self.c_bounds[1]] * self.n_peaks + [self.w_bounds[1]] * self.n_peaks + self.n_juncs*[2.2])

    def get_name(self):
        return "Two-dip colour generation function"

    def get_extra_info(self):
        return "\tDimensions: " + str(self.dim)

    def get_nobj(self):
        return 2


# target_color = [0.11781738, 0.1029, 0.05100166]
#
# udp = pg.problem(two_dip_colour_function_mobj(5, tg=target_color))
# algo = pg.algorithm(pg.maco(gen=2000))
#
# pop = pg.population(prob = udp, size = 100)
# ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(pop.get_f())
#
# ax = pg.plot_non_dominated_fronts(pop.get_f())
# plt.show()
# pop = algo.evolve(pop)
#
# ax = pg.plot_non_dominated_fronts(pop.get_f())
# plt.show()
#
# ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(pop.get_f())


color_names =(
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
    "Black-2"
    )

single_J_result = pd.read_csv("paper_colours.csv")

color_xyY = np.array(single_J_result[['x', 'y', 'Y']])

color_XYZ = np.array([convert_xyY_to_XYZ(x) for x in color_xyY])


def make_new_population(prob, popsize, load_prev, i):
    pop = pg.population(prob=prob, size=popsize)

    if load_prev > 0:
        print("Loading previous population", color_names[i])
        acc_pop = np.loadtxt('results/' + color_names[i] + '_' + str(j1) + '_' + str(load_prev-1) + '.txt')

        for l1, ipop in enumerate(acc_pop):
            pop.set_x(l1, ipop)

    return pop



def internal_run(target, i1, n_peaks=2, n_gaps=1, popsize=80, gen=1000, load_previous=0):

    p_init = two_dip_colour_function_mobj(n_peaks, n_gaps, target)
    udp = pg.problem(p_init)
    algo = pg.algorithm(pg.moead(gen=gen))#, preserve_diversity=True, decomposition="bi"))

    pop = make_new_population(udp, popsize, load_previous, i1)
    # ax = pg.plot_non_dominated_fronts(pop.get_f())
    # plt.show()
    # # fits, vectors = pop.get_f(), pop.get_x()
    # for i in range(10):
    #     pop = algo.evolve(pop)
    #     print(pop.champion_x)
    #     print(pop.champion_f)

    found_col = False

    n_tries = 0

    while not found_col:

        # print(n_tries)

        pop = algo.evolve(pop)
        a = pop.get_f()

        acc = a[a[:, 0] < col_thresh]

        ndf, _, _, _ = pg.fast_non_dominated_sorting(pop.get_f())

        # print(color_names[i1], len(ndf))
        n_tries += 1

        if len(acc) >= 1: # and len(ndf) == 1:

            # if len(ndf) > 1:
            #     pg.plot_non_dominated_fronts(pop.get_f())
            #     plt.title(color_names[i1])
            #     plt.show()

            ndf, _, _, _ = pg.fast_non_dominated_sorting(pop.get_f())
            print(color_names[i1], len(ndf), len(acc), n_tries)
            found_col = True
            acc_pop = pop.get_x()[a[:, 0] < col_thresh]

            # reorder peaks if not in ascending order

            acc_pop = np.stack([reorder_peaks(x, n_peaks) for x in acc_pop])

            best_index = np.argmin(acc[:, 1])

            best_pop = acc_pop[best_index]

            # XYZ = XYZColor(*target)
            # rgb_target = convert_color(XYZ, sRGBColor)
            # rgb_target_list = np.array(rgb_target.get_value_tuple()) * 255
            #
            # XYZ = spec_to_xyz(
            #     gen_spectrum_ndip(best_pop[:n_peaks], best_pop[n_peaks:2*n_peaks]), solar_spec_col)
            # XYZ = XYZColor(XYZ[0], XYZ[1], XYZ[2])
            # rgb = convert_color(XYZ, sRGBColor)
            # rgb_list = np.array(rgb.get_value_tuple()) * 255
            #
            # palette = np.array([[rgb_target_list, rgb_list]], dtype=int)
            #
            # io.imshow(palette)
            # plt.title(color_names[i1])
            # plt.show()

            # print(color_names[i1], acc[best_index])

            # np.savetxt('results/' + color_names[i1] + '_' + str(j1) + '_' + str(load_previous) + '.txt', acc_pop)


        elif n_tries >= 3:
            print(color_names[i1], ": MAKING NEW POPULATION", str(np.min(pop.get_f()[:,0])),
                  len(ndf))
            #
            # if len(ndf) > 1:
            #     pg.plot_non_dominated_fronts(pop.get_f())
            #     plt.title(color_names[i1])
            #     plt.show()

            # sometimes hangs - just make a new population and try again
            # pop = pg.population(prob=udp, size=popsize)
            pop = make_new_population(udp, popsize, load_previous, i1)
            n_tries = 0



    return [-acc[best_index][1], *best_pop]


n_peaks = 2
n_gaps = 2


internal_run(color_XYZ[0], 0, n_peaks, n_gaps, 30, 10, 0)

#
# compare_results = np.zeros((len(color_XYZ), n_trials, n_peaks*2 + n_gaps + 1))
#
# start = time()
#
# for j1 in range(n_trials):
#
#     eff_result_par = Parallel(n_jobs=-1)(delayed(internal_run)
#                                          (color_XYZ[i1], i1, n_peaks, n_gaps, 80, 300, 0) for i1 in range(len(color_XYZ)))
#     compare_results[:, j1] = np.stack(eff_result_par)
#
# print(time()-start)
#
# #single_J_result = pd.read_csv("paper_colours.csv")
#
# plt.figure(figsize=(8,3))
# plt.plot(color_names, compare_results[:,:,0], 'o', mfc='none')
# plt.plot(color_names, single_J_result['eta']/100, 'or', mfc='none')
# # plt.ylim(750,)
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()
# #
# # plt.figure(figsize=(8,3))
# # plt.plot(color_names, compare_results[:,:,1], 'o', mfc='none')
# # # plt.ylim(750,)
# # plt.xticks(rotation=45)
# # plt.tight_layout()
# # plt.show()
#
#
#
# max_eff_index = np.argmax(compare_results[:,:, 0], 1)
# Eg_max_eta = [compare_results[i1,max_eff_index[i1],2*n_peaks+1] for i1 in range(len(color_XYZ))]
#
# plt.figure(figsize=(8,3))
# plt.plot(color_names, compare_results[:,:, 2*n_peaks+1], 'o', mfc='none')
# plt.plot(color_names, Eg_max_eta, 'ko', mfc='none')
# # plt.ylim(750,)
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()
#
# fig, axs = plt.subplots(len(color_XYZ)//2, 2, figsize=(4, 12))
# axs = axs.flatten()
#
# for i1, name in enumerate(color_names):
#
#     final_pop_col = []
#     for j1 in range(n_trials):
#
#         final_pop = np.loadtxt('results/' + name + '_' + str(j1) + '.txt')
#         final_pop_col.append(final_pop[:,4])
#
#     final_pop_col = np.hstack(final_pop_col)
#     counts, bins = np.histogram(final_pop_col)
#     axs[i1].stairs(counts, bins)
#
# plt.tight_layout()
# plt.show()

# col_thresh = 0.01
# start = time()
#
# compare_results_2 = np.zeros((len(color_XYZ), n_trials, 6))
#
# for j1 in range(n_trials):
#
#     eff_result_par = Parallel(n_jobs=-1)(delayed(internal_run)
#                                          (color_XYZ[i1], i1, 80, 1000, 1) for i1 in range(len(color_XYZ)))
#     compare_results_2[:, j1] = np.stack(eff_result_par)
#
# print(time()-start)
#
#
# plt.figure(figsize=(8,3))
# plt.plot(color_names, compare_results_2[:,:,0], 'o', mfc='none')
# plt.plot(color_names, single_J_result['eta']/100, 'or', mfc='none')
# # plt.ylim(750,)
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()
#
# col_thresh = 0.005
# start = time()
#
# compare_results_3 = np.zeros((len(color_XYZ), n_trials, 6))
#
# for j1 in range(n_trials):
#
#     eff_result_par = Parallel(n_jobs=-1)(delayed(internal_run)
#                                          (color_XYZ[i1], i1, 80, 1000, 2) for i1 in range(len(color_XYZ)))
#     compare_results_3[:, j1] = np.stack(eff_result_par)
#
# print(time()-start)
#
# plt.figure(figsize=(8,3))
# plt.plot(color_names, compare_results_3[:,:,0], 'o', mfc='none')
# plt.plot(color_names, single_J_result['eta']/100, 'or', mfc='none')
# # plt.ylim(750,)
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()
# for j1 in range(n_trials):
#
#     for i1, target in enumerate(color_lab):
#
#         p_init = two_dip_colour_function_mobj(4, tg=target)
#         udp = pg.problem(p_init)
#         algo = pg.algorithm(pg.moead(gen=2000))
#         pop = pg.population(prob=udp, size=50)
#         pop = algo.evolve(pop)
#
#         # ax = pg.plot_non_dominated_fronts(pop.get_f())
#         # plt.show()
#         # # fits, vectors = pop.get_f(), pop.get_x()
#         # for i in range(10):
#         #     pop = algo.evolve(pop)
#         #     print(pop.champion_x)
#         #     print(pop.champion_f)
#
#         a = pop.get_f()
#
#         acc = a[a[:, 0] < col_thresh]
#
#         if len(acc) >=1:
#             acc_pop = pop.get_x()[a[:,0] < col_thresh]
#
#             best_index = np.argmin(acc[:,1])
#
#             best_pop = acc_pop[best_index]
#
#             Lab = LabColor(*target)
#             rgb_target = convert_color(Lab, sRGBColor)
#             rgb_target_list = np.array(rgb_target.get_value_tuple())*255
#
#             XYZ = spec_to_xyz(gen_spectrum_2dip(best_pop[0],best_pop[2],peak1=1,center2=best_pop[1],width2=best_pop[3],peak2=1),
#                             solar_spec_col)
#             XYZ = XYZColor(XYZ[0],XYZ[1],XYZ[2])
#             rgb = convert_color(XYZ, sRGBColor)
#             rgb_list = np.array(rgb.get_value_tuple())*255
#
#
#             palette = np.array([[rgb_target_list, rgb_list]], dtype=int)
#
#             io.imshow(palette)
#             plt.title(color_names[i1])
#             plt.show()
#
#             print(color_names[i1], acc[best_index])
#
#             compare_results[i1, j1] = -acc[best_index][1]
#
#         else:
#             print('Did not find population which met threshold for ' + color_names[i1])

