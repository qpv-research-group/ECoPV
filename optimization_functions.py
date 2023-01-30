import numpy as np
from scipy.special import lambertw
from solcore.constants import kb, q, h, c
from typing import Sequence

k = kb / q
h_eV = h / q
e = np.exp(1)
T = 298
kbT = k * T
pref = ((2 * np.pi * q) / (h_eV**3 * c**2)) * kbT


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

def db_cell_calculation(egs: Sequence[float], flux: np.ndarray, wl: np.ndarray, interval: float, rad_eff: int = 1) -> tuple:

    """Calculates recombination current density, current due to illumination, voltages at the maximum power point
    and currents at the maximum power point of a multi-junction solar cell in the detailed-balance limit.

    :param egs: Bandgaps of the subcells in eV, order from highest to lowest
    :param flux: Flux of the solar spectrum in W/m^2/nm
    :param wl: Wavelengths of the solar spectrum in nm
    :param interval: Wavelength interval of the solar spectrum in nm
    :param rad_eff: Radiative efficiency of the cell (0-1)
    """

    # Since we need previous Eg info have to iterate the Jsc array
    jscs = np.empty_like(egs)  # Quick way of defining jscs with same dimensions as egs
    j01s = np.empty_like(egs)

    upperE = 4.14
    for i, eg in enumerate(egs):
        j01s[i] = (
            (pref / rad_eff)
            * (eg**2 + 2 * eg * (kbT) + 2 * (kbT) ** 2)
            * np.exp(-(eg) / (kbT))
        )
        jscs[i] = (
            q
            * np.sum(flux[np.all((wl < 1240 / eg, wl > 1240 / upperE), axis=0)])
            * interval
        )
        # plt.figure()
        # plt.plot(wl_cell[np.all((wl_cell < 1240 / eg, wl_cell > 1240 / upperE), axis=0)], flux[np.all((wl_cell < 1240 / eg, wl_cell > 1240 / upperE), axis=0)])
        # plt.show()
        upperE = eg

    Vmaxs = (kbT * (lambertw(e * (jscs / j01s)) - 1)).real
    Imaxs = jscs - j01s * np.exp(Vmaxs / (kbT))

    return j01s, jscs, Vmaxs, Imaxs


def getPmax(egs: Sequence[float], flux: np.ndarray, wl: np.ndarray, interval: float, rad_eff: int = 1) -> float:
    """Calculates the maximum power (in W) of a multi-junction solar cell in the detailed-balance limit.

    :param egs: Bandgaps of the subcells in eV, order from highest to lowest
    :param flux: Flux of the solar spectrum in W/m^2/nm
    :param wl: Wavelengths of the solar spectrum in nm
    :param interval: Wavelength interval of the solar spectrum in nm
    :param rad_eff: Radiative efficiency of the cell (0-1)"""

    j01s, jscs, Vmaxs, Imaxs = db_cell_calculation(egs, flux, wl, interval, rad_eff)

    # Find the minimum Imaxs
    minImax = np.amin(Imaxs)

    # Find tandem voltage

    vsubcell = kbT * np.log((jscs - minImax) / j01s)
    vTandem = np.sum(vsubcell)

    return vTandem * minImax


def getIVmax(egs: Sequence[float], flux: np.ndarray, wl: np.ndarray, interval: float, rad_eff: int = 1) -> tuple:
    """Calculates the voltages and currents of each junction in a multi-junction cell at the maximum power point
     in the detailed-balance limit.

    :param egs: Bandgaps of the subcells in eV, order from highest to lowest
    :param flux: Flux of the solar spectrum in W/m^2/nm
    :param wl: Wavelengths of the solar spectrum in nm
    :param interval: Wavelength interval of the solar spectrum in nm
    :param rad_eff: Radiative efficiency of the cell (0-1)
    """
    _, _, Vmaxs, Imaxs = db_cell_calculation(egs, flux, wl, interval, rad_eff)

    return Vmaxs, Imaxs


def getIVtandem(egs: Sequence[float], flux: np.ndarray, wl: np.ndarray, interval: float, rad_eff: int = 1) -> tuple:

    """Calculates the overall voltage and current at the maximum power point of multi-junction cell
     in the detailed-balance limit.

    :param egs: Bandgaps of the subcells in eV, order from highest to lowest
    :param flux: Flux of the solar spectrum in W/m^2/nm
    :param wl: Wavelengths of the solar spectrum in nm
    :param interval: Wavelength interval of the solar spectrum in nm
    :param rad_eff: Radiative efficiency of the cell (0-1)
    """

    j01s, jscs, Vmaxs, Imaxs = db_cell_calculation(egs, flux, wl, interval, rad_eff)

    # Find the minimum Imaxs
    minImax = np.amin(Imaxs)

    #   Find tandem voltage

    vsubcell = kbT * np.log((jscs - minImax) / j01s)
    vTandem = np.sum(vsubcell)

    return vTandem, np.min(Imaxs)
