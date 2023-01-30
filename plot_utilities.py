import matplotlib.ticker as tck
from main_optimization import load_colorchecker
from matplotlib.patches import Rectangle
from matplotlib import rc
import numpy as np
from typing import Sequence
import matplotlib.pyplot as plt

from colormath.color_objects import sRGBColor, XYZColor
from colormath.color_conversions import convert_color
import xarray as xr

from spectrum_functions import load_cmf, spec_to_XYZ

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})

patch_width = 0.7

color_names, color_XYZ = load_colorchecker()

Y = np.hstack((color_XYZ[:, 1], [0]))
default_Y_cols = Y[:18]

col_names_default = xr.DataArray(
    data=color_names[:18], dims=["Y"], coords={"Y": default_Y_cols}
)
col_names_default = col_names_default.sortby("Y", ascending=False)

color_XYZ_xr = xr.DataArray(
    color_XYZ[:18],
    dims=["color", "XYZ"],
    coords={"color": color_XYZ[:18, 1], "XYZ": ["X", "Y", "Z"]},
)

color_XYZ_xr = color_XYZ_xr.sortby("color", ascending=False)
color_XYZ_bw = xr.DataArray(
    color_XYZ[18:],
    dims=["color", "XYZ"],
    coords={"color": color_XYZ[18:, 1], "XYZ": ["X", "Y", "Z"]},
)
color_XYZ_xr = xr.concat([color_XYZ_xr, color_XYZ_bw], dim="color")


def wavelength_to_rgb(wavelengths, gamma=0.8):
    """
    Taken from http://www.noah.org/wiki/Wavelength_to_RGB_in_Python
    This converts a given wavelength of light to an
    approximate RGB color value. The wavelength must be given
    in nanometers in the range from 380 nm through 750 nm
    (789 THz through 400 THz).

    Based on code by Dan Bruton
    http://www.physics.sfasu.edu/astro/color/spectra.html
    Additionally alpha value set to 0.5 outside range

    :param wavelengths: list or array of wavelengths in nm
    :param gamma: gamma value for gamma correction
    """

    RGBA = np.zeros((len(wavelengths), 4))

    for wavelength in wavelengths:
        wavelength = float(wavelength)
        if wavelength >= 380 and wavelength <= 750:
            A = 1.0
        else:
            A = 0.5
        if wavelength < 380:
            wavelength = 380.0
        if wavelength > 750:
            wavelength = 750.0
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



def add_colour_patches(ax, width, labels, color_XYZ=color_XYZ_xr):
    # width is with an axis spacing of "1" for the x-axis colour labels

    ymin = ax.get_ylim()[0]
    ymax = ax.get_ylim()[1]
    h = ax.get_window_extent().height
    w = ax.get_window_extent().width

    h_p = width * w / len(labels)
    h_ax = h_p * (ymax - ymin) / h

    for l1, lab in enumerate(labels):

        if lab != "Black":

            target = color_XYZ[l1]

        else:
            target = [0, 0, 0]

        color_xyz_t = XYZColor(*target)
        color_srgb_t = convert_color(color_xyz_t, sRGBColor).get_value_tuple()

        ax.add_patch(
            Rectangle(
                xy=(l1 - (width / 2), ymin),
                width=width,
                height=h_ax,
                facecolor=color_srgb_t,
            )
        )


def apply_formatting(ax, color_labels=None, grid="both", n_colors=None):

    if n_colors is None:
        n_colors = len(color_labels)

    ax.yaxis.set_minor_locator(tck.AutoMinorLocator())
    ax.grid(axis=grid, color="0.9")

    ax.xaxis.set_ticks(np.arange(0, n_colors))
    ax.tick_params(direction="in", which="both", top=True, right=True)
    print(color_labels)

    if color_labels is not None:
        ax.set_xticklabels(
            color_labels, rotation=45, ha="right", rotation_mode="anchor"
        )
        ax.tick_params(direction="inout", axis="x", top=True, length=7)

    else:
        ax.set_xticklabels([])

    ax.set_xlim(-0.6, len(ax.get_xticks()) - 0.4)

    ax.set_axisbelow(True)


def make_sorted_xr(
    arr,
    color_names,
    append_black=None,
    Y_cols=default_Y_cols,
    col_names_sorted=col_names_default,
):
    if arr.ndim == 1:
        dims = ["color"]

    else:
        dims = ["color", "n"]

    eff_xr_col = xr.DataArray(data=arr[:18], dims=dims, coords={"color": Y_cols})

    eff_xr_col = eff_xr_col.sortby("color", ascending=False)
    eff_xr_col = eff_xr_col.assign_coords(color=col_names_sorted.data)

    if append_black is not None:
        eff_xr_bw = xr.DataArray(
            data=np.append(arr[18:], [append_black], axis=0),
            dims=dims,
            coords={"color": np.append(color_names[18:], "Black")},
        )

    else:
        eff_xr_bw = xr.DataArray(
            data=arr[18:], dims=dims, coords={"color": color_names[18:]}
        )

    eff_xr = xr.concat([eff_xr_col, eff_xr_bw], dim="color")

    return eff_xr
