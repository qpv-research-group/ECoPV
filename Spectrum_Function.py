import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
from colormath.color_objects import LabColor, XYZColor, sRGBColor, xyYColor
from colormath.color_conversions import convert_color


def load_cmf(wl):
    cmf = np.loadtxt('cmf.txt')
    cmf_new = np.zeros((len(wl), 3))  # Interpolating cmf data to be in 1nm intervals

    # this could be done in one go
    intfunc = interp1d(cmf[:,0], cmf[:, 1], fill_value="extrapolate")
    cmf_new[:, 0] = intfunc(wl)
    intfunc = interp1d(cmf[:,0], cmf[:, 2], fill_value="extrapolate")
    cmf_new[:, 1] = intfunc(wl)
    intfunc = interp1d(cmf[:,0], cmf[:, 3], fill_value="extrapolate")
    cmf_new[:, 2] = intfunc(wl)

    return cmf_new

# df = pd.read_excel("ASTMG173_split.xlsx", sheet_name=0)

"""Convert a spectrum to an xyz point.
The spectrum must be on the same grid of points as the colour-matching
function, self.cmf: 380-780 nm in 1 nm steps.
"""
# Trying to get Python cooridnates to match Matlab
# Insert complete path to the excel file and index of the worksheet


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

def gen_spectrum_ndip(centres, widths, peaks=1, wl=None, base=0): # center and width in nm

    # centres and widths should be np arrays
    spectrum = np.ones_like(wl)*base

    lower = centres - widths/2
    upper = centres + widths/2

    for i in range(len(centres)):
        spectrum[np.all((wl >= lower[i], wl <= upper[i]), axis=0)] = peaks

    # possible peaks are overlapping; R can't be more than peak value

    spectrum[spectrum > peaks] = peaks

    return spectrum


#Ref https://scipython.com/blog/converting-a-spectrum-to-a-colour/
def spec_to_xyz(spec, solar_spec, cmf, interval):
    # insert the name of the column as a string in brackets

    Ymax = np.sum(interval * cmf[:,1] * solar_spec)
    X = np.sum(interval * cmf[:,0] * solar_spec * spec)
    Y = np.sum(interval * cmf[:,1] * solar_spec * spec)
    Z = np.sum(interval * cmf[:,2] * solar_spec * spec)

    #

    if Ymax == 0:
        return (X,Y,Z)

    else:
        X = X/Ymax
        Y = Y/Ymax
        Z = Z/Ymax
        XYZ = (X,Y,Z)

        return XYZ


def delta_E_CIE2000(Lab1, Lab2):
    """
    Returns the difference :math:`\Delta E_{ab}` between two given *CIE Lab*
    *array_like* colours using CIE 2000 recommendation.
    Parameters
    ----------
    Lab1 : array_like, (3,)        *CIE Lab* *array_like* colour 1.
    Lab2 : array_like, (3,)        *CIE Lab* *array_like* colour 2.
    Returns
    -------
    numeric:        Colour difference :math:`\Delta E_{ab}`.
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

        # return np.sum(np.abs(target-col))/np.sum(target)
        dXYZ = np.abs(target - col) / target

        return max(dXYZ)