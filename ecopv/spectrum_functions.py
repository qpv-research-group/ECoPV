import numpy as np
from typing import Sequence, Tuple
import pathlib
from scipy.interpolate import interp1d
from os.path import join

from colormath.color_objects import (
    LabColor,
    XYZColor,
    xyYColor,
)
from colormath.color_conversions import convert_color

current_path = pathlib.Path(__file__).parent.resolve()


def load_cmf(wl: np.ndarray) -> np.ndarray:
    """Load the CIE 1931 2 degree standard observer color matching functions and interpolate them to the given wavelengths"""

    cmf = np.loadtxt(join(current_path, "data", "cmf.txt"))
    cmf_new = np.zeros((len(wl), 3))

    intfunc = interp1d(cmf[:, 0], cmf[:, 1], fill_value=(0, 0), bounds_error=False)
    cmf_new[:, 0] = intfunc(wl)
    intfunc = interp1d(cmf[:, 0], cmf[:, 2], fill_value=(0, 0), bounds_error=False)
    cmf_new[:, 1] = intfunc(wl)
    intfunc = interp1d(cmf[:, 0], cmf[:, 3], fill_value=(0, 0), bounds_error=False)
    cmf_new[:, 2] = intfunc(wl)

    return cmf_new


def convert_xyY_to_Lab(xyY_list: Sequence[float]) -> Tuple[float, float, float]:

    xyY = xyYColor(*xyY_list)
    lab = convert_color(xyY, LabColor)
    return lab.get_value_tuple()


def convert_xyY_to_XYZ(xyY_list: Sequence[float]) -> Tuple[float, float, float]:

    xyY = xyYColor(*xyY_list)
    lab = convert_color(xyY, XYZColor)
    return lab.get_value_tuple()


def convert_XYZ_to_Lab(XYZ_list: Sequence[float]) -> Tuple[float, float, float]:
    XYZ = XYZColor(*XYZ_list)
    lab = convert_color(XYZ, LabColor)
    return lab.get_value_tuple()


def gen_spectrum_ndip(
    pop: np.ndarray,
    n_peaks: int,
    wl: np.ndarray,
    max_height: float = 1,
    base: float = 0,
) -> np.ndarray:  # center and width in nm
    """Generate a reflection spectrum with n_peaks rectangular peaks of fixed height.

    :param pop: The population vector, which contains the peak centres and widths. The first n_peaks elements are the centres,
                the next n_peaks elements are the widths.
    :param n_peaks: The number of peaks in the reflection spectrum
    :param wl: vector of wavelengths
    :param max_height: height of the reflection peaks (0-1)
    :param base: fixed reflection baseline (0-1)
    """

    centres = pop[:n_peaks]
    widths = pop[n_peaks : 2 * n_peaks]

    # centres and widths should be np arrays
    spectrum = np.ones_like(wl) * base

    lower = centres - widths / 2
    upper = centres + widths / 2

    for i in range(len(centres)):
        spectrum[np.all((wl >= lower[i], wl <= upper[i]), axis=0)] += max_height

    # possible peaks are overlapping; R can't be more than peak value

    spectrum[spectrum > max_height] = max_height

    return spectrum


def gen_spectrum_ndip_varyheight(
    pop: np.ndarray,
    n_peaks: int,
    wl: np.ndarray,
    max_height: float = 1,
    base: float = 00,
) -> np.ndarray:
    """Generate a reflection spectrum with n_peaks rectangular peaks of variable height.

    :param pop: The population vector, which contains the peak centres and widths. The first n_peaks elements are the centres,
                the next n_peaks elements are the widths, the last n_peaks elements  are the height
    :param n_peaks: The number of peaks in the reflection spectrum
    :param wl: vector of wavelengths
    :param max_height: height of the reflection peaks (0-1)
    :param base: fixed reflection baseline (0-1)
    """

    # centres and widths should be np arrays
    spectrum = np.ones_like(wl) * base
    centres = pop[:n_peaks]
    widths = pop[n_peaks : 2 * n_peaks]
    heights = pop[2 * n_peaks : 3 * n_peaks]

    lower = centres - widths / 2
    upper = centres + widths / 2

    for i in range(len(centres)):
        spectrum[np.all((wl >= lower[i], wl <= upper[i]), axis=0)] += heights[i]

    # possible peaks are overlapping; R can't be more than peak value

    spectrum[spectrum > max_height] = max_height

    return spectrum


def gen_spectrum_ngauss(
    pop: np.ndarray,
    n_peaks: int,
    wl: np.ndarray,
    max_height: float = 1,
    base: float = 0,
):
    """Generate a reflection spectrum with n_peaks Gaussian peaks of fixed height.

    :param pop: The population vector, which contains the peak centres and widths. The first n_peaks elements are the centres,
                the next n_peaks elements are the widths.
    :param n_peaks: The number of peaks in the reflection spectrum
    :param wl: vector of wavelengths
    :param max_height: height of the reflection peaks (0-1)
    :param base: fixed reflection baseline (0-1)
    """
    centres = pop[:n_peaks]
    widths = pop[n_peaks : 2 * n_peaks]

    spectrum = np.zeros_like(wl)

    for i in range(len(centres)):
        spectrum += np.exp(-((wl - centres[i]) ** 2) / (2 * widths[i] ** 2))

    return base + (max_height - base) * spectrum / max(spectrum)


def gen_spectrum_ngauss_varyheight(
    pop: np.ndarray,
    n_peaks: int,
    wl: np.ndarray,
    max_height: float = 1,
    base: float = 0,
):  # center and width in nm
    """Generate a reflection spectrum with n_peaks Gaussian peaks of variable height.

    :param pop: The population vector, which contains the peak centres and widths. The first n_peaks elements are the centres,
                the next n_peaks elements are the widths.
    :param n_peaks: The number of peaks in the reflection spectrum
    :param wl: vector of wavelengths
    :param max_height: height of the reflection peaks (0-1)
    :param base: fixed reflection baseline (0-1)
    """

    centres = pop[:n_peaks]
    widths = pop[n_peaks : 2 * n_peaks]
    heights = pop[2 * n_peaks : 3 * n_peaks]

    spectrum = np.zeros_like(wl)

    for i in range(len(centres)):
        spectrum += heights[i] * np.exp(
            -((wl - centres[i]) ** 2) / (2 * widths[i] ** 2)
        )

    return base + (max_height - base) * spectrum / max(spectrum)


class make_spectrum_ndip:
    """Class which creates an object containing necessary information about the type of reflection spectrum to be used
    in the optimization: number of peaks, bounds on peak centres and widths, and the function to generate the spectrum."""

    def __init__(
        self,
        n_peaks: int = 2,
        target: np.ndarray = np.array([0, 0, 0]),
        type: str = "sharp",
        fixed_height: bool = True,
        w_bounds: Sequence[float] = None,
        h_bounds: Sequence[float] = [0.01, 1],
    ):

        """Initialise the spectrum object.

        :param n_peaks: The number of peaks in the reflection spectrum
        :param target: The XYZ coordinates of the target colour, used to set the bounds on the peak widths
        :param type: The type of spectrum to be generated. Current options are "sharp" and "gauss"
        :param fixed_height: Whether the peaks should have a fixed height
        :param w_bounds: The bounds on the peak widths. If None, the bounds will be set automatically
        :param h_bounds: The bounds on the peak heights. Only used if fixed_height is False
        """

        self.c_bounds = [380, 780]
        self.fixed_height = fixed_height
        self.n_peaks = n_peaks

        if w_bounds is None:
            if fixed_height:
                self.w_bounds = [
                    0,
                    np.max([120 / n_peaks, (350 / n_peaks) * target[1]]),
                ]

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

        """Return the bounds on the population vector which contains the peak centres and widths (and heights, if relevant).
        This is in the format [list of lower bounds, list of upper bounds]. For each list, the first n_peaks elements
        are the centres, the next n_peaks elements are the widths. If fixed_height is False, the next n_peaks elements are the
        heights."""

        if self.fixed_height:
            return (
                [self.c_bounds[0]] * self.n_peaks + [self.w_bounds[0]] * self.n_peaks,
                [self.c_bounds[1]] * self.n_peaks + [self.w_bounds[1]] * self.n_peaks,
            )

        else:
            return (
                [self.c_bounds[0]] * self.n_peaks
                + [self.w_bounds[0]] * self.n_peaks
                + [self.h_bounds[0]] * self.n_peaks,
                [self.c_bounds[1]] * self.n_peaks
                + [self.w_bounds[1]] * self.n_peaks
                + [self.h_bounds[1]] * self.n_peaks,
            )


def spec_to_XYZ(
    spec: np.ndarray, solar_spec: np.ndarray, cmf: np.ndarray, interval: float
) -> Tuple[float, float, float]:
    """Convert an incident spectrum (spectral power distribution, W m-2 nm-1) to XYZ colour coordinates.

    :param spec: The reflectance spectrum
    :param solar_spec: The solar spectrum (W m-2 nm-1)
    :param cmf: The CIE colour matching functions at the same wavelengths as spec and solar_spec
    :param interval: The wavelength interval between each element of spec, solar_spec and cmf
    """

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


def delta_XYZ(target: np.ndarray, col: np.ndarray):
    """Calculate the maximum fractional difference in the X, Y, Z colours between two colours in XYZ space, relative
    to the target argument."""
    dXYZ = np.abs(target - col) / target

    return max(dXYZ)


def XYZ_from_pop_dips(pop, n_peaks, photon_flux, interval):
    cs = pop[:n_peaks]
    ws = pop[n_peaks : 2 * n_peaks]

    cmf = load_cmf(photon_flux[0])
    # T = 298

    R_spec = gen_spectrum_ndip(cs, ws, wl=photon_flux[0])
    XYZ = np.array(
        spec_to_XYZ(
            R_spec, hc * photon_flux[1] / (photon_flux[0] * 1e-9), cmf, interval
        )
    )

    return XYZ
