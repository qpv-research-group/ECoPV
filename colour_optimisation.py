import pygmo as pg
from Spectrum_Function import gen_spectrum_ndip, spec_to_xyz, load_cmf, delta_XYZ
import numpy as np

import matplotlib.pyplot as plt

from scipy.special import lambertw
from solcore.constants import kb, q, h, c
# import cProfile
# import pstats

k = kb/q
h_eV = h/q
e = np.exp(1)
T = 298
kbT = k*T
pref = ((2*np.pi* q)/(h_eV**3 * c**2))* kbT

class single_colour:

    def __init__(self):
        pass

    def run(self, target, colour_name, colour_threshold, photon_flux, n_peaks=2, n_gaps=1, popsize=80, gen=1000, max_tries=3):

        p_init = two_dip_colour_function_mobj(n_peaks, n_gaps, target, photon_flux, 1000)
        udp = pg.problem(p_init)
        algo = pg.algorithm(pg.moead(gen=gen))#, preserve_diversity=True, decomposition="bi"))

        pop = pg.population(prob=udp, size=popsize)

        found_col = False

        n_tries = 0

        while not found_col:

            # print(n_tries)

            pop = algo.evolve(pop)
            a = pop.get_f()

            acc = a[a[:, 0] < colour_threshold]

            ndf, _, _, _ = pg.fast_non_dominated_sorting(pop.get_f())

            # print(color_names[i1], len(ndf))
            n_tries += 1

            if len(acc) >= 1: # and len(ndf) == 1:

                # if len(ndf) > 1:
                #     pg.plot_non_dominated_fronts(pop.get_f())
                #     plt.title(color_names[i1])
                #     plt.show()

                ndf, _, _, _ = pg.fast_non_dominated_sorting(pop.get_f())
                # print(colour_name, len(ndf), len(acc), n_tries)
                found_col = True
                acc_pop = pop.get_x()[a[:, 0] < colour_threshold]

                # reorder peaks if not in ascending order

                acc_pop = np.stack([reorder_peaks(x, n_peaks) for x in acc_pop])

                best_index = np.argmin(acc[:, 1])

                best_pop = acc_pop[best_index]

            elif n_tries >= max_tries:
                print(colour_name, ": MAKING NEW POPULATION", str(np.min(pop.get_f()[:,0])),
                      len(ndf))

                pop = pg.population(prob=udp, size=popsize)
                n_tries = 0

        self.pop = pop

        return ([-acc[best_index][1], *best_pop], pop.problem.get_fevals())


def reorder_peaks(pop, n_peaks):
    peaks = pop[:n_peaks]
    bandgaps = pop[2*n_peaks:]
    sorted_widths = np.array([x for _, x in sorted(zip(peaks, pop[n_peaks:]))])
    peaks.sort()
    bandgaps.sort()
    return np.hstack((peaks, sorted_widths, bandgaps))

def XYZ_from_pop_dips(pop, n_peaks, inc_spec, interval):
    cs = pop[:n_peaks]
    ws = pop[n_peaks:2*n_peaks]

    cmf = load_cmf(inc_spec[0])
    # T = 298

    R_spec = gen_spectrum_ndip(cs, ws, wl=inc_spec[0])
    XYZ = np.array(spec_to_xyz(R_spec, inc_spec[1], cmf, interval))

    return XYZ

def getPmax(egs, flux, wl, interval):
    # Since we need previous Eg info have to iterate the Jsc array
    jscs = np.empty_like(egs)  # Quick way of defining jscs with same dimensions as egs
    j01s = np.empty_like(egs)

    upperE = 4.14
    for i, eg in enumerate(egs):
        j01s[i] = pref * (eg ** 2 + 2 * eg * (kbT) + 2 * (kbT) ** 2) * np.exp(-(eg) / (kbT))
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

class two_dip_colour_function_mobj:
    def __init__(self, n_peaks, n_juncs, tg, photon_flux, power_in=1000):
        self.dim = n_peaks*2 + n_juncs
        self.n_peaks = n_peaks
        self.n_juncs = n_juncs
        self.target_color = tg
        self.c_bounds = [380, 780]
        self.w_bounds = [0, 150]
        self.cell_wl = photon_flux[0]
        self.col_wl = self.cell_wl[np.all([self.cell_wl >= self.c_bounds[0], self.cell_wl <= self.c_bounds[1]], axis=0)]
        self.solar_flux = photon_flux[1]
        self.solar_spec = self.solar_flux[np.all([self.cell_wl >= 380, self.cell_wl <= 780], axis=0)]
        self.incident_power = power_in
        self.interval = np.round(np.diff(self.cell_wl)[0], 6)
        self.cmf = load_cmf(self.col_wl)

    def calculate(self, x):
        # center = x[0]
        # center2 = x[1]
        # width = x[2]
        # width2 = x[3]
        # Eg = x[4]

        # profile = cProfile.Profile()
        # profile.enable()
        cs = x[:self.n_peaks]
        ws = x[self.n_peaks:2*self.n_peaks]
        Egs = -np.sort(-x[2*self.n_peaks:]) #[0]

        # T = 298

        R_spec = gen_spectrum_ndip(cs, ws, wl=self.col_wl)
        XYZ=np.array(spec_to_xyz(R_spec, self.solar_spec, self.cmf, self.interval))
        delta = delta_XYZ(self.target_color, XYZ)

        R_spec_cell = gen_spectrum_ndip(cs, ws, wl=self.cell_wl)

        # Establish an interpolation function to allow integration over arbitrary limits
        # solarfluxInterpolate = InterpolatedUnivariateSpline(wl_cell, light * (1 - R_spec_cell), k=1)
        #
        # Jsc = q * solarfluxInterpolate.integral(300, 1240 / Eg)
        # Jsc = q * int_flux

        flux = self.solar_flux * (1 - R_spec_cell)
        # Jsc = q*np.sum(flux[wl_cell < 1240/Eg])*interval
        #
        # J01 = pref * (Eg**2 + 2*Eg*(kbT) + 2*(kbT)**2)*np.exp(-(Eg)/(kbT))
        #
        # # Vmax = (kbT*(lambertw(np.exp(1)*(Jsc/J01))-1)).real
        # Vmax = (kbT * (lambertw(e * (Jsc / J01)) - 1)).real
        # Imax = Jsc - J01*np.exp(Vmax/(kbT))
        # eta = Vmax*Imax/1000

        eta = getPmax(Egs, flux, self.cell_wl, self.interval)/self.incident_power

        # profile.disable()
        # ps = pstats.Stats(profile)
        # ps.print_stats()

        return delta, eta

    def fitness(self, x):
        delta, eff = self.calculate(x)

        return [delta, -eff]

    def plot(self, x):

        delta_E, R_spec = self.calculate(x)

        plt.figure()
        plt.plot(self.col_wl, R_spec)
        plt.title(str(delta_E))
        plt.show()

    def get_bounds(self):

        # Limits for n junctions based on limits for a black cell from https://doi.org/10.1016/0927-0248(96)00015-3

        if self.n_juncs == 1:
            Eg_bounds = [[1.13-0.3],
                         [1.13+0.3]]

        elif self.n_juncs == 2:
            Eg_bounds = [[0.94-0.3, 1.64-0.3],
                         [0.94+0.3, 1.64+0.3]]

        elif self.n_juncs == 3:
            Eg_bounds = [[0.71-0.3, 1.16-0.3, 1.83-0.3],
                         [0.71+0.3, 1.16+0.3, 1.83+0.3]]

        elif self.n_juncs == 4:
            Eg_bounds = [[0.53-0.3, 1.13-0.3, 1.55-0.3, 2.13-0.3],
                         [0.53+0.3, 1.13+0.3, 1.55+0.3, 2.13+0.3]]

        else:
            Eg_bounds = [[0.5]*self.n_juncs,
                         [4.0]*self.n_juncs]

        return ([self.c_bounds[0]] * self.n_peaks + [self.w_bounds[0]] * self.n_peaks + Eg_bounds[0],
                [self.c_bounds[1]] * self.n_peaks + [self.w_bounds[1]] * self.n_peaks + Eg_bounds[1])

    def get_name(self):
        return "Two-dip colour generation function"

    def get_extra_info(self):
        return "\tDimensions: " + str(self.dim)

    def get_nobj(self):
        return 2



def make_new_population(prob, popsize, load_prev=0, trial_n=None, colour_name=None):
    pop = pg.population(prob=prob, size=popsize)

    if load_prev > 0:
        print("Loading previous population", colour_name)
        acc_pop = np.loadtxt('results/' + colour_name + '_' + str(trial_n) + '_' + str(load_prev-1) + '.txt')

        for l1, ipop in enumerate(acc_pop):
            pop.set_x(l1, ipop)

    return pop

