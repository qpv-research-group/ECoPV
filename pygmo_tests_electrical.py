import pygmo as pg
from Spectrum_Function import gen_spectrum_2dip, spec_to_xyz, delta_E_CIE2000
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

k = kb/q
h_eV = h/q
e = 2.718281828459045
T = 298
kbT = k*T
pref = ((2*np.pi* q)/(h_eV**3 * c**2))* kbT

def convert_xyY_to_Lab(xyY_list):

    xyY = xyYColor(*xyY_list)
    lab = convert_color(xyY, LabColor)
    return lab.get_value_tuple()

def reorder_peaks(pop, n_peaks):
    peaks = pop[:n_peaks]
    bandgaps = pop[2*n_peaks:]
    sorted_widths = np.array([x for _, x in sorted(zip(peaks, pop[n_peaks:]))])
    peaks.sort()
    bandgaps.sort()
    return np.hstack((peaks, sorted_widths, bandgaps))


interval=0.5 # interval between each two wavelength points, 0.02 needed for low dE values
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
    def __init__(self, dim, tg):
        self.dim = dim
        self.target_color = tg
        self.c_bounds = [380, 780]
        self.w_bounds = [0, 150]

    def calculate(self, x):
        center = x[0]
        center2 = x[1]
        width = x[2]
        width2 = x[3]
        Eg = x[4]

        # T = 298

        R_spec = gen_spectrum_2dip(center,width,1,center2, width2,1,base=0)
        XYZ=np.array(spec_to_xyz(R_spec, solar_spec_col))
        XYZ=XYZColor(XYZ[0],XYZ[1],XYZ[2])
        Lab=convert_color(XYZ, LabColor)
        delta=delta_E_CIE2000((Lab.lab_l,Lab.lab_a,Lab.lab_b),self.target_color)
        # delta = delta_XYZ(self.target_color, XYZ)

        # n_available = np.sum((1 - R_spec) * photon_flux_norm)/np.sum(photon_flux_norm)
        R_spec_cell = gen_spectrum_2dip(center, width, 1, center2, width2, 1, wl=wl_cell, base=0)

        flux = light * (1 - R_spec_cell)
        Jsc = q*np.sum(flux[wl_cell < 1240/Eg])*interval

        # Establish an interpolation function to allow integration over arbitrary limits
        # solarfluxInterpolate = InterpolatedUnivariateSpline(wl_cell, light * (1 - R_spec_cell), k=1)
        #
        # Jsc = q * solarfluxInterpolate.integral(300, 1240 / Eg)
        # Jsc = q * int_flux

        J01 = pref * (Eg**2 + 2*Eg*(kbT) + 2*(kbT)**2)*np.exp(-(Eg)/(kbT))

        # Vmax = (kbT*(lambertw(np.exp(1)*(Jsc/J01))-1)).real
        Vmax = (kbT * (lambertw(e * (Jsc / J01)) - 1)).real
        Imax = Jsc - J01*np.exp(Vmax/(kbT))
        eta = Vmax*Imax/1000

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
        return ([self.c_bounds[0]] * 2 + [self.w_bounds[0]] *2 + [1],
                [self.c_bounds[1]] * 2 + [self.w_bounds[1]] *2 + [1.5])

    def get_name(self):
        return "Two-dip colour generation function"

    def get_extra_info(self):
        return "\tDimensions: " + str(self.dim)

    def get_nobj(self):
        return 2


# target_color_lab = [55.2600000000000,-38.3400000000000,31.3700000000000]
#
# udp = pg.problem(two_dip_colour_function_mobj(5, tg=target_color_lab))
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

color_lab = np.array([convert_xyY_to_Lab(x) for x in color_xyY])

col_thresh = 0.25

n_trials = 2


def internal_run(target, i1, popsize=80):

    p_init = two_dip_colour_function_mobj(5, tg=target)
    udp = pg.problem(p_init)
    algo = pg.algorithm(pg.moead(gen=500))#, preserve_diversity=True, decomposition="bi"))
    pop = pg.population(prob=udp, size=popsize)

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

        print(color_names[i1], len(ndf))
        n_tries += 1

        if len(acc) >= 1:# and len(ndf) == 1:

            # if len(ndf) > 1:
            #     pg.plot_non_dominated_fronts(pop.get_f())
            #     plt.title(color_names[i1])
            #     plt.show()

            ndf, _, _, _ = pg.fast_non_dominated_sorting(pop.get_f())
            print(color_names[i1], len(ndf), len(acc), n_tries)
            found_col = True
            acc_pop = pop.get_x()[a[:, 0] < col_thresh]

            best_index = np.argmin(acc[:, 1])

            best_pop = acc_pop[best_index]

            # reorder peaks if not in ascending order

            acc_pop = np.stack([reorder_peaks(x, 2) for x in acc_pop])

            # Lab = LabColor(*target)
            # rgb_target = convert_color(Lab, sRGBColor)
            # rgb_target_list = np.array(rgb_target.get_value_tuple()) * 255
            #
            # XYZ = spec_to_xyz(
            #     gen_spectrum_2dip(best_pop[0], best_pop[2], peak1=1, center2=best_pop[1],
            #                       width2=best_pop[3], peak2=1),
            #     solar_spec_col)
            # XYZ = XYZColor(XYZ[0], XYZ[1], XYZ[2])
            # rgb = convert_color(XYZ, sRGBColor)
            # rgb_list = np.array(rgb.get_value_tuple()) * 255
            #
            # palette = np.array([[rgb_target_list, rgb_list]], dtype=int)
            #
            # io.imshow(palette)
            # plt.title(color_names[i1])
            # plt.show()

            #print(color_names[i1], acc[best_index])
            np.savetxt('results/' + color_names[i1] + '_' + str(j1) + '.txt', acc_pop)

        elif n_tries >= 4:
            print("MAKING NEW POPULATION")
            # sometimes hangs - just make a new population and try again
            pop = pg.population(prob=udp, size=popsize)


    return [-acc[best_index][1], *best_pop]

    # else:
    #     print('Did not find population which met threshold for ' + str(target))
    #     return 0

compare_results = np.zeros((len(color_lab), n_trials, 6))

start = time()

for j1 in range(n_trials):

    eff_result_par = Parallel(n_jobs=-1)(delayed(internal_run)
                                         (color_lab[i1], i1) for i1 in range(len(color_lab)))
    compare_results[:, j1] = np.stack(eff_result_par)

print(time()-start)

plt.figure(figsize=(8,3))
plt.plot(color_names, compare_results[:,:,0], 'o', mfc='none')
# plt.ylim(750,)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,3))
plt.plot(color_names, compare_results[:,:,4], 'o', mfc='none')
# plt.ylim(750,)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



max_eff_index = np.argmax(compare_results[:,:, 0], 1)
Eg_max_eta = [compare_results[i1,max_eff_index[i1],5] for i1 in range(len(color_lab))]

plt.figure(figsize=(8,3))
plt.plot(color_names, compare_results[:,:, 5], 'o', mfc='none')
plt.plot(color_names, Eg_max_eta, 'ko', mfc='none')
# plt.ylim(750,)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(len(color_lab)//2, 2, figsize=(4, 12))
axs = axs.flatten()

for i1, name in enumerate(color_names):

    final_pop_col = []
    for j1 in range(n_trials):

        final_pop = np.loadtxt('results/' + name + '_' + str(j1) + '.txt')
        final_pop_col.append(final_pop[:,4])

    final_pop_col = np.hstack(final_pop_col)
    counts, bins = np.histogram(final_pop_col)
    axs[i1].stairs(counts, bins)

plt.tight_layout()
plt.show()





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

