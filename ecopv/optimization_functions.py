import numpy as np
from scipy.special import lambertw
from solcore.constants import kb, q, h, c
from typing import Sequence
from ecopv.spectrum_functions import gen_spectrum_ndip

k = kb / q
h_eV = h / q
e = np.exp(1)
T = 298
kbT = k * T
pref = ((2 * np.pi * q) / (h_eV**3 * c**2)) * kbT

pref_wl = 1e27*2*np.pi*q*c
wl_exp_const = 1e9*h*c/(kb*T)


def reorder_peaks(pop, n_peaks, n_junctions, fixed_height=True):
    peaks = pop[:n_peaks]
    bandgaps = pop[-n_junctions:] if n_junctions > 0 else np.array([])
    sorted_widths = np.array(
        [x for _, x in sorted(zip(peaks, pop[n_peaks : 2 * n_peaks]))]
    )

    if not fixed_height:
        sorted_heights = np.array(
            [x for _, x in sorted(zip(peaks, pop[2 * n_peaks : 3 * n_peaks]))]
        )

    peaks.sort()
    bandgaps.sort()

    if fixed_height:
        return np.hstack((peaks, sorted_widths, bandgaps))

    else:
        return np.hstack((peaks, sorted_widths, sorted_heights, bandgaps))


def db_cell_calculation_noR(
    egs: Sequence[float],
    flux: np.ndarray,
    wl: np.ndarray,
    interval: float,
    rad_eff: int = 1,
    upperE: float = 4.43,
    *args, # this is in case a different db_cell_calculation method has additional
                # arguments
) -> tuple:

    """Calculates recombination current density, current due to illumination, voltages
    at the maximum power point and currents at the maximum power point of a
    multi-junction solar cell in the detailed-balance limit. Ignores the effect of
    reflected photons on the recombination current (dark current).

    :param egs: Bandgaps of the subcells in eV, order from highest to lowest
    :param flux: Flux of the solar spectrum in W/m^2/nm
    :param wl: Wavelengths of the solar spectrum in nm
    :param interval: Wavelength interval of the solar spectrum in nm
    :param rad_eff: Radiative efficiency of the cell (0-1)
    :param upperE: upper limit (in eV) for integrating over photon flux
    """

    # Since we need previous Eg info have to iterate the Jsc array
    jscs = np.empty_like(egs, dtype=float)  # Quick way of defining jscs with same
    # dimensions as egs
    j01s = np.empty_like(egs, dtype=float)

    for i, eg in enumerate(egs):

        j01s[i] = (
            (pref / rad_eff[i])
            * (eg**2 + 2 * eg * (kbT) + 2 * (kbT) ** 2)
            * np.exp(-(eg) / (kbT))
        )
        jscs[i] = (
            q
            * np.sum(flux[np.all((wl < 1240 / eg, wl > 1240 / upperE), axis=0)])
            * interval
        )

        upperE = eg

    Vmaxs = (kbT * (lambertw(e * (jscs / j01s)) - 1)).real
    Imaxs = jscs - j01s * np.exp(Vmaxs / (kbT))

    return j01s, jscs, Vmaxs, Imaxs

def indefinite_integral_J0(eg1, eg2):
    # eg1 is lower Eg, eg2 is higher Eg
    p1 = -(eg1 ** 2 + 2 * eg1 * (kbT) + 2 * (kbT) ** 2)* np.exp(-(eg1) / (kbT))
    p2 = -(eg2 ** 2 + 2 * eg2 * (kbT) + 2 * (kbT) ** 2)* np.exp(-(eg2) / (kbT))

    return (p2 - p1)

def db_cell_calculation_perfectR(
    egs: Sequence[float],
    flux: np.ndarray,
    wl: np.ndarray,
    interval: float,
    rad_eff: int = 1,
    upperE: float = 4.43,
    x: Sequence[float] = None,
    n_peaks: int = 2,
    *args, # this is in case a different db_cell_calculation method has additional args
) -> tuple:

    """Calculates recombination current density, current due to illumination, voltages
    at the maximum power point and currents at the maximum power point of a
    multi-junction solar cell in the detailed-balance limit. Takes into account the
    effect of reflected photons on the recombination current (dark current)
    for rectangular reflection peaks using the analytical form of the integral.

    :param egs: Bandgaps of the subcells in eV, order from highest to lowest
    :param flux: Flux of the solar spectrum in W/m^2/nm
    :param wl: Wavelengths of the solar spectrum in nm
    :param interval: Wavelength interval of the solar spectrum in nm
    :param rad_eff: Radiative efficiency of the cell (0-1)
    :param upperE: upper limit (in eV) for integrating over photon flux
    :param x: List of parameters for the reflection peaks
    :param n_peaks: Number of reflection peaks
    """

    # limits are in terms of energy, but R bands are defined in terms of wavelength!

    # limits = np.array([
    #     [0, 1240/(x[1]+x[3]/2)], # infinite wavelength to upper end of upper band
    #     [1240/(x[1]-x[3]/2), 1240/(x[0]+x[2]/2)], # between the R bands
    #     [1240/(x[0]-x[2]/2), upperE] # below the lower band
    # ])

    x = reorder_peaks(x, n_peaks, len(x) - 2*n_peaks)

    limits = [[0, 1240/(x[n_peaks-1]+x[2*n_peaks-1]/2)]]

    if n_peaks > 1:

        for i1 in np.arange(1, n_peaks):
            limits.append([1240/(x[n_peaks-i1]-x[2*n_peaks-i1]/2), 1240/(x[
                  n_peaks-(i1+1)]+x[2*n_peaks-(i1+1)]/2)])

    limits.append([1240/(x[0]-x[n_peaks]/2), upperE])

    limits = np.array(limits)

    # if we have more than 2 peaks, some peaks can overlap. Remove superfluous limits
    # which cause incorrect integration limits
    overlapping = np.where(limits[:, 0] > limits[:, 1])[0]
    limits[overlapping + 1, 0] = limits[overlapping, 1]
    limits = np.delete(limits, overlapping, axis=0)
    # if len(limits) < 4:
    #     print('after', limits)

    # Can still have issues with > 3 peaks!
    # # if we have more than 2 peaks, some peaks can overlap. Remove superfluous limits
    # can also have partial overlap
    # o_ind = np.where(limits[1:, 0] < limits[:-1, 1])[0]
    # if len(o_ind) > 0:
    #     print(limits)
    #     limits[o_ind, 1] = limits[o_ind + 1, 1]
    #     limits = np.delete(limits, o_ind + 1, axis=0)
    #     print(limits, x)

    limits[limits < 0] = 0 # set negative values to 0
    limits[limits > upperE] = upperE # set values above upperE to upperE

    # Since we need previous Eg info have to iterate the Jsc array
    jscs = np.zeros_like(egs, dtype=float)  # Quick way of defining jscs with same
    # dimensions as egs
    j01s = np.zeros_like(egs, dtype=float)

    for i, eg in enumerate(egs):

        # check where Eg is in the limits array. Only care about bands below Eg
        loop_limits = limits[limits[:,1] > eg]
        loop_limits[0, 0] = np.max([eg, loop_limits[0,0]])

        for lims in loop_limits:
            j01s[i] += (pref/rad_eff[i])*indefinite_integral_J0(lims[0], lims[1])

        jscs[i] = (
            q
            * np.sum(flux[np.all((wl < 1240 / eg, wl > 1240 / upperE), axis=0)])
            * interval
        )

        upperE = eg

    Vmaxs = (kbT * (lambertw(e * (jscs / j01s)) - 1)).real
    Imaxs = jscs - j01s * np.exp(Vmaxs / (kbT))

    return j01s, jscs, Vmaxs, Imaxs

def db_cell_calculation_numericalR(
    egs: Sequence[float],
    flux: np.ndarray,
    wl: np.ndarray,
    interval: float,
    rad_eff: int = 1,
    upperE: float = 4.43,
    x: Sequence[float] = None,
    n_peaks: int = 2,
    *args, # this is in case a different db_cell_calculation method has additional args
) -> tuple:

    """Calculates recombination current density, current due to illumination, voltages
    at the maximum power point and currents at the maximum power point of a
    multi-junction solar cell in the detailed-balance limit. Takes into accoun the
    effect of reflected photons on the recombination current (dark current) by
    using numerical integration (this is much slower than the
    db_cell_calculation_perfectR method).

    :param egs: Bandgaps of the subcells in eV, order from highest to lowest
    :param flux: Flux of the solar spectrum in W/m^2/nm
    :param wl: Wavelengths of the solar spectrum in nm
    :param interval: Wavelength interval of the solar spectrum in nm
    :param rad_eff: Radiative efficiency of the cell (0-1)
    :param upperE: upper limit (in eV) for integrating over photon flux
    :param x: List of parameters for the reflection peaks
    :param n_peaks: Number of reflection peaks
    """

    # Since we need previous Eg info have to iterate the Jsc array
    jscs = np.empty_like(egs, dtype=float)  # Quick way of defining jscs with same dimensions as egs
    j01s = np.zeros_like(egs, dtype=float)

    R = gen_spectrum_ndip(x, n_peaks, wl, 1, 0)

    upperE_j01 = upperE

    for i, eg in enumerate(egs):

        wl_inds = np.all((wl < 1240 / eg, wl > 1240 / upperE_j01), axis=0)
        wl_slice = wl[wl_inds]
        # print("numerical", eg, np.sum(wl_inds))
        j01s[i] = (pref_wl / rad_eff[i]) * np.trapz(
            (1 - R[wl_inds]) * np.exp(-wl_exp_const / wl_slice) / (wl_slice) ** 4,
            wl_slice)

        jscs[i] = (
            q
            * np.sum(flux[np.all((wl < 1240 / eg, wl > 1240 / upperE), axis=0)])
            * interval
        )

        upperE = eg

    Vmaxs = (kbT * (lambertw(e * (jscs / j01s)) - 1)).real
    Imaxs = jscs - j01s * np.exp(Vmaxs / (kbT))

    return j01s, jscs, Vmaxs, Imaxs


def getPmax(
    egs: Sequence[float],
    flux: np.ndarray,
    wl: np.ndarray,
    interval: float,
    x: Sequence[float] = None,
    rad_eff: float = 1,
    upperE: float = 4.43,
    method: str = "perfect_R",
    n_peaks: int = 2,
) -> float:
    """Calculates the maximum power (in W) of a multi-junction solar cell in the detailed-balance limit.

    :param egs: Bandgaps of the subcells in eV, order from highest to lowest
    :param flux: Flux of the solar spectrum in W/m^2/nm
    :param wl: Wavelengths of the solar spectrum in nm
    :param interval: Wavelength interval of the solar spectrum in nm
    :param x: population vector defining the reflection peaks
    :param rad_eff: Radiative efficiency of the cell (0-1)
    :param upperE: upper limit (in eV) for integrating over photon flux
    :param method: Method for calculating the maximum power point. Current options are
        "perfect_R" and "no_R": in the first case, integration by parts is used to
        calculate the recombination current (dark current) accurately for rectangular
        reflection peaks, in the second case the reflection peaks are ignored when
        calculating the recombination current. In any case, reflection peaks are taken
        into account when calculating the Jsc (light-generated current).
    :param n_peaks: Number of peaks in the reflection spectrum

    """

    if type(rad_eff) == float or type(rad_eff) == int:
        rad_eff = np.full_like(egs, rad_eff, dtype=float)
    elif len(rad_eff) == 0:
        rad_eff = np.full_like(egs, 1, dtype=float)


    db_cell_calculation = {
        'perfect_R': db_cell_calculation_perfectR,
        'no_R': db_cell_calculation_noR,
        'numerical_R': db_cell_calculation_numericalR,
    }

    j01s, jscs, Vmaxs, Imaxs = db_cell_calculation[method](egs, flux, wl, interval,
                                                        rad_eff, upperE, x, n_peaks)

    # Find the minimum Imaxs
    minImax = np.amin(Imaxs)

    # Find tandem voltage

    if np.any(minImax > jscs) or np.any(j01s < 0):
        # vsubcell = kbT * np.log((jscs - minImax) / j01s)

        # vTandem = np.sum(vsubcell)
        # print("vTandem", vTandem, np.isnan(vTandem), vTandem*(~np.isnan(vTandem)))
        # print(vsubcell, jscs, minImax, j01s)
        # print(vTandem*(~np.isnan(vTandem))*np.all(vsubcell > 0))
        return 0

    else:
        vsubcell = kbT * np.log((jscs - minImax) / j01s)

        vTandem = np.sum(vsubcell)

        return vTandem * minImax


def getIVmax(
    egs: Sequence[float],
    flux: np.ndarray,
    wl: np.ndarray,
    interval: float,
    x: Sequence[float] = None,
    rad_eff: int = 1,
    upperE: float = 4.43,
    method: str = "perfect_R",
    n_peaks: int = 2,
) -> tuple:
    """Calculates the voltages and currents of each junction in a multi-junction cell at the maximum power point
     in the detailed-balance limit.

    :param egs: Bandgaps of the subcells in eV, order from highest to lowest
    :param flux: Flux of the solar spectrum in W/m^2/nm
    :param wl: Wavelengths of the solar spectrum in nm
    :param interval: Wavelength interval of the solar spectrum in nm
    :param x: population vector defining the reflection peaks
    :param rad_eff: Radiative efficiency of the cell (0-1)
    :param upperE: upper limit (in eV) for integrating over photon flux
    :param method: Method for calculating the maximum power point. Current options are
        "perfect_R" and "no_R": in the first case, integration by parts is used to
        calculate the recombination current (dark current) accurately for rectangular
        reflection peaks, in the second case the reflection peaks are ignored when
        calculating the recombination current. In any case, reflection peaks are taken
        into account when calculating the Jsc (light-generated current).
    :param n_peaks: Number of peaks in the reflection spectrum

    """

    db_cell_calculation = {
        'perfect_R': db_cell_calculation_perfectR,
        'no_R': db_cell_calculation_noR
    }

    _, _, Vmaxs, Imaxs = db_cell_calculation[method](egs, flux, wl, interval,
                                                        rad_eff, upperE, x, n_peaks)


    return Vmaxs, Imaxs


def getIVtandem(
    egs: Sequence[float],
    flux: np.ndarray,
    wl: np.ndarray,
    interval: float,
    x: Sequence[float],
    rad_eff: int = 1,
    upperE: float = 4.43,
    method: str = "perfect_R",
    n_peaks: int = 2,
) -> tuple:

    """Calculates the overall voltage and current at the maximum power point of multi-junction cell
     in the detailed-balance limit.

    :param egs: Bandgaps of the subcells in eV, order from highest to lowest
    :param flux: Flux of the solar spectrum in W/m^2/nm
    :param wl: Wavelengths of the solar spectrum in nm
    :param interval: Wavelength interval of the solar spectrum in nm
    :param x: population vector defining the reflection peaks
    :param rad_eff: Radiative efficiency of the cell (0-1)
    :param upperE: upper limit (in eV) for integrating over photon flux
    :param method: Method for calculating the maximum power point. Current options are
        "perfect_R" and "no_R": in the first case, integration by parts is used to
        calculate the recombination current (dark current) accurately for rectangular
        reflection peaks, in the second case the reflection peaks are ignored when
        calculating the recombination current. In any case, reflection peaks are taken
        into account when calculating the Jsc (light-generated current).
    :param n_peaks: Number of peaks in the reflection spectrum

    """

    db_cell_calculation = {
        'perfect_R': db_cell_calculation_perfectR,
        'no_R': db_cell_calculation_noR
    }

    j01s, jscs, Vmaxs, Imaxs = db_cell_calculation[method](egs, flux, wl, interval,
                                                         rad_eff, upperE, x, n_peaks)

    # Find the minimum Imaxs
    minImax = np.amin(Imaxs)

    #   Find tandem voltage

    vsubcell = kbT * np.log((jscs - minImax) / j01s)
    vTandem = np.sum(vsubcell)

    return vTandem, np.min(Imaxs)
