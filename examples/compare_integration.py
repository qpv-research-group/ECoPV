# compare integration methods: just calculation area of rectangles vs. trapezoid rule

import numpy as np
from scipy.special import lambertw
from solcore.constants import kb, q, h, c
from solcore.light_source import LightSource
from time import time


def indefinite_integral_J0(eg1, eg2):
    # eg1 is lower Eg, eg2 is higher Eg
    p1 = -(eg1 ** 2 + 2 * eg1 * (kbT) + 2 * (kbT) ** 2)* np.exp(-(eg1) / (kbT))
    p2 = -(eg2 ** 2 + 2 * eg2 * (kbT) + 2 * (kbT) ** 2)* np.exp(-(eg2) / (kbT))

    return (pref * kbT)*(p2 - p1)/rad_eff

interval = 0.1
wl = np.arange(1240/4.43, 4000, interval)

k = kb / q
h_eV = h / q
e = np.exp(1)
T = 298
kbT = k * T
pref = ((2 * np.pi * q) / (h_eV**3 * c**2))

pref_wl = 1e27*2*np.pi*q*c
wl_exp_const = 1e9*h*c/(kb*T)

egs = [2, 1.5, 1.1]
rad_eff = 1

photon_flux_cell = np.array(
    LightSource(
        source_type="standard",
        version="AM1.5g",
        x=wl,
        output_units="photon_flux_per_nm",
    ).spectrum(wl)
)

flux = photon_flux_cell[1]

# Since we need previous Eg info have to iterate the Jsc array

jscs = np.empty_like(egs)  # Quick way of defining jscs with same dimensions as egs
j01s = np.empty_like(egs)
jscs_trapz = np.empty_like(egs)  # Quick way of defining jscs with same dimensions as egs
j01s_all = np.empty_like(egs)
j01s_wl = np.empty_like(egs)
j01s_wl_rect = np.empty_like(egs)

R = np.random.rand(len(wl))
# R = R*0
flux = flux * (1 - R)

start = time()
j = 0
while j < 1000:
    upperE = 4.43
    for i, eg in enumerate(egs):
        j01s_all[i] = (
                (pref * kbT / rad_eff)
                * (eg ** 2 + 2 * eg * (kbT) + 2 * (kbT) ** 2)
                * np.exp(-(eg) / (kbT))
        )



        wl_inds = np.all((wl < 1240 / eg, wl > 1240 / upperE), axis=0)
        wl_slice = wl[np.all((wl < 1240 / eg, wl > 1240 / upperE), axis=0)]
        j01s[i] = (pref_wl/rad_eff) * np.trapz((1-R[wl_inds])*np.exp(-wl_exp_const/wl_slice)/(wl_slice)**4, wl_slice)

        # j01s[i] = (pref_wl/rad_eff) * np.sum((1-R[wl_inds])*np.exp(-wl_exp_const/wl_slice)/(wl_slice)**4)*interval

        jscs[i] = q * np.trapz(flux[np.all((wl < 1240 / eg, wl > 1240 / upperE), axis=0)], dx=interval)

        # plt.figure()
        # plt.plot(wl_cell[np.all((wl_cell < 1240 / eg, wl_cell > 1240 / upperE), axis=0)], flux[np.all((wl_cell < 1240 / eg, wl_cell > 1240 / upperE), axis=0)])
        # plt.show()
        upperE = eg

    Vmaxs_all = kbT * (lambertw(e * (jscs / j01s_all)) - 1)
    Vmaxs = (kbT * (lambertw(e * (jscs / j01s)) - 1)).real
    Imaxs_all = jscs - j01s_all * np.exp(Vmaxs_all / (kbT))
    Imaxs = jscs - j01s * np.exp(Vmaxs / (kbT))

    j += 1


print(time()-start, j01s)
print(Vmaxs, Vmaxs_all)
print(Imaxs, Imaxs_all)

#     jscs[i] = (
#             q
#             * np.sum(flux[np.all((wl < 1240 / eg, wl > 1240 / upperE), axis=0)])
#             * interval
#     )
#
#     jscs_trapz[i] = q*np.trapz(flux[np.all((wl < 1240 / eg, wl > 1240 / upperE), axis=0)], wl[np.all((wl < 1240 / eg, wl > 1240 / upperE), axis=0)])
#
#     upperE = eg
#
# Vmaxs = (kbT * (lambertw(e * (jscs / j01s)) - 1)).real
# Imaxs = jscs - j01s * np.exp(Vmaxs / (kbT))
#
# print("rectangle sum Jscs:", jscs)
# print("trapz Jscs:", jscs_trapz)
#
# print("rectangle sum J01s:", j01s)
# print("trapz J01s:", j01s_trapz)
# print("wl J01s:", j01s_wl)
# print("wl J01s rec:", j01s_wl_rect)
# print(np.array(j01s_wl)/np.array(j01s))

from ecopv.optimization_functions import db_cell_calculation_numericalR, \
    db_cell_calculation_noR, db_cell_calculation_perfectR
from ecopv.spectrum_functions import gen_spectrum_ndip
import matplotlib.pyplot as plt

x = np.array([450, 575, 50, 100])
x = np.array([450, 575, 800, 75, 50, 100])
egs = [1.1]

R_array = gen_spectrum_ndip(x, 3, wl, 1, 0)

plt.figure()
plt.plot(1240/wl, R_array)
for eg in egs:
    plt.axvline(eg, color='r')
plt.show()

flux_with_R = flux * (1-R_array)

res_numerical = db_cell_calculation_numericalR(egs, flux_with_R, wl, interval, 1,
                                              4.43, x, n_peaks=3)

res_noR = db_cell_calculation_noR(egs, flux_with_R, wl, interval, 1, 4.43, x)
res_perfectR = db_cell_calculation_perfectR(egs, flux_with_R, wl, interval, 1,
                                            4.43, x, n_peaks=3)