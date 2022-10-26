import pygmo as pg
from Spectrum_Functions_de import gen_spectrum_2dip, spec_to_xyz, delta_E_CIE2000
import numpy as np
from solcore.light_source import LightSource
from colormath.color_objects import LabColor, XYZColor, sRGBColor, xyYColor
from colormath.color_conversions import convert_color
import matplotlib.pyplot as plt
import pandas as pd
from skimage import io
from joblib import Parallel, delayed

def convert_xyY_to_Lab(xyY_list):

    xyY = xyYColor(*xyY_list)
    lab = convert_color(xyY, LabColor)
    return lab.get_value_tuple()

interval=0.2 # interval between each two wavelength points, 0.02 needed for low dE values
wl=np.arange(380,780+interval,interval)

wl_cell=np.arange(300, 4000, 0.5)

light = LightSource(
    source_type="standard", version="AM1.5g", x=wl_cell, output_units="photon_flux_per_nm"
)

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

class sphere_function:
    def __init__(self, dim):
        self.dim = dim

    def fitness(self, x):
        return [sum(x*x)]

    def get_bounds(self):
        return ([-1] * self.dim, [1] * self.dim)

    def get_name(self):
        return "Sphere Function"

    def get_extra_info(self):
        return "\tDimensions: " + str(self.dim)


class two_dip_colour_function:
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

        R_spec = gen_spectrum_2dip(center,width,1,center2, width2,1,base=0)
        XYZ=np.array(spec_to_xyz(R_spec, solar_spec_col))
        XYZ=XYZColor(XYZ[0],XYZ[1],XYZ[2])
        Lab=convert_color(XYZ, LabColor)
        delta=delta_E_CIE2000((Lab.lab_l,Lab.lab_a,Lab.lab_b),self.target_color)
        # delta = delta_XYZ(self.target_color, XYZ)

        return delta, R_spec

    def fitness(self, x):
        delta, R_spec = self.calculate(x)
        return [delta]

    def plot(self, x):

        delta_E, R_spec = self.calculate(x)

        plt.figure()
        plt.plot(wl, R_spec)
        plt.title(str(delta_E))
        plt.show()

    def get_bounds(self):
        return ([self.c_bounds[0]] * 2 + [self.w_bounds[0]] *2, [self.c_bounds[1]] * 2 + [self.w_bounds[1]] *2)

    def get_name(self):
        return "Two-dip colour generation function"

    def get_extra_info(self):
        return "\tDimensions: " + str(self.dim)


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

        R_spec = gen_spectrum_2dip(center,width,1,center2, width2,1,base=0)
        XYZ=np.array(spec_to_xyz(R_spec, solar_spec_col))
        XYZ=XYZColor(XYZ[0],XYZ[1],XYZ[2])
        Lab=convert_color(XYZ, LabColor)
        delta=delta_E_CIE2000((Lab.lab_l,Lab.lab_a,Lab.lab_b),self.target_color)
        # delta = delta_XYZ(self.target_color, XYZ)

        return delta, R_spec

    def fitness(self, x):
        delta, R_spec = self.calculate(x)
        n_available = np.sum((1 - R_spec) * photon_flux_norm)/np.sum(photon_flux_norm)
        return [delta, -n_available]

    def plot(self, x):

        delta_E, R_spec = self.calculate(x)

        plt.figure()
        plt.plot(wl, R_spec)
        plt.title(str(delta_E))
        plt.show()

    def get_bounds(self):
        return ([self.c_bounds[0]] * 2 + [self.w_bounds[0]] *2, [self.c_bounds[1]] * 2 + [self.w_bounds[1]] *2)

    def get_name(self):
        return "Two-dip colour generation function"

    def get_extra_info(self):
        return "\tDimensions: " + str(self.dim)

    def get_nobj(self):
        return 2



target_color_lab = [55.2600000000000,-38.3400000000000,31.3700000000000]

prob = pg.problem(two_dip_colour_function(4, tg=target_color_lab))

#pygmo.de(gen=1, F=0.8, CR=0.9, variant=2, ftol=1e-06, xtol=1e-06, seed=random)

# algo = pg.algorithm(pg.bee_colony(gen=1000, limit=20))
# pop = pg.population(prob,20)
# pop = algo.evolve(pop)
# print(pop.champion_x)
# print(pop.champion_f)
#
#
# two_dip_colour_function(4, tg=target_color_lab).plot(pop.champion_x)
#
#
# algo = pg.algorithm(pg.de(gen=1000, ftol=0.5))
# algo.set_verbosity(100)
# pop = pg.population(prob,20)
# pop = algo.evolve(pop)
# print(pop.champion_x)
# print(pop.champion_f)

# uda = algo.extract(pg.de)
# uda.get_log()
#
# two_dip_colour_function(4, tg=target_color_lab).plot(pop.champion_x)


# udp = pg.problem(two_dip_colour_function_mobj(4, tg=target_color_lab))
# algo = pg.algorithm(pg.moead(gen=500))
#
# pop = pg.population(prob = udp, size = 200)
# ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(pop.get_f())
#
# ax = pg.plot_non_dominated_fronts(pop.get_f())
# plt.show()
# print(pg.pareto_dominance(*pop.get_f().T))
# pop = algo.evolve(pop)
#
# print(pg.pareto_dominance(*pop.get_f().T))
# ax = pg.plot_non_dominated_fronts(pop.get_f())
# plt.show()
#
# ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(pop.get_f())


# # fits, vectors = pop.get_f(), pop.get_x()
# for i in range(10):
#     pop = algo.evolve(pop)
#     print(pop.champion_x)
#     print(pop.champion_f)

# a = pop.get_f()
#
# col_thresh = 0.5
#
# acc = a[a[:,0] < col_thresh]

# plt.figure()
# plt.plot(a[:,0]/np.max(a[:,0]))
# plt.plot(a[:,1]/np.max([a[:,1]]))
# plt.show()

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

n_trials = 3

compare_results = np.zeros((len(color_lab), n_trials))

def internal_run(target, i1):
    p_init = two_dip_colour_function_mobj(4, tg=target)
    udp = pg.problem(p_init)
    algo = pg.algorithm(pg.moead(gen=200))#, preserve_diversity=True, decomposition="bi"))
    pop = pg.population(prob=udp, size=50)
    #pop = algo.evolve(pop)

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

        if len(acc) >= 1 and len(ndf) == 1:
            # ax = pg.plot_non_dominated_fronts(pop.get_f())
            # plt.title(color_names[i1])
            # plt.show()

            ndf, _, _, _ = pg.fast_non_dominated_sorting(pop.get_f())
            print(color_names[i1], len(ndf), len(acc))
            found_col = True
            acc_pop = pop.get_x()[a[:, 0] < col_thresh]

            best_index = np.argmin(acc[:, 1])

            best_pop = acc_pop[best_index]

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

    return -acc[best_index][1]

    # else:
    #     print('Did not find population which met threshold for ' + str(target))
    #     return 0



for j1 in range(n_trials):

    eff_result_par = Parallel(n_jobs=-1)(delayed(internal_run)
                                         (color_lab[i1], i1) for i1 in range(len(color_lab)))
    compare_results[:, j1] = eff_result_par

plt.figure()
plt.plot(color_names, compare_results, 'o')
# plt.ylim(750,)
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