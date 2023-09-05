from ecopv.spectrum_functions import load_cmf, load_D50, load_D65,\
    spec_to_XYZ, \
    gen_spectrum_ndip

from ecopv.main_optimization import (
    load_colorchecker,
)
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import numpy as np
from os import path
from solcore.light_source import LightSource
from colormath.color_objects import (
    LabColor,
    XYZColor,
    xyYColor,
    sRGBColor,
)
from colormath.color_conversions import convert_color

# how many population evolutions happen before giving up if there are no populations
# which meet the color threshold

R_type = "sharp"  # "sharp" for rectangular dips or "gauss" for gaussians
fixed_height = True  # fixed height peaks (will be at the value of max_height) if True, or peak height is an optimization
# variable if False
light_source_name = "AM1.5g"
j01_method = "perfect_R"

max_height = 1
# maximum height of reflection peaks; fixed at this value of if fixed_height = True

base = 0
# baseline fixed reflection (fixed at this value for both fixed_height = True and False).


color_names, color_XYZ = load_colorchecker()
# load the names and XYZ coordinates of the 24 default Babel colors
start_ind = 0
end_ind = len(color_names)
color_names = color_names[start_ind:end_ind]
color_XYZ = color_XYZ[start_ind:end_ind]

# precalculate optimal bandgaps for junctions:
save_path = path.join(path.dirname(path.abspath(__file__)), "results")

n_peaks = 2
n_junctions = 1

# for some fixed reflectance spectrum, show what colors look like using either AM1.5G,
# a 5778K black body, or the D50 standard illuminant.

# Load the optimized parameters for each colour, calculate the R spectrum, calculate the
# XYZ coordinates using the different illuminants, and then convert to sRGB.
interval = 0.01
wl_cell = np.arange(
    300, 4000, interval
)

cmf = load_cmf(wl_cell)

D50 = load_D50(wl_cell)

D65 = load_D65(wl_cell)

AM15G = np.array(
    LightSource(
        source_type="standard",
        version="AM1.5g",
        x=wl_cell,
        output_units="power_density_per_nm",
    ).spectrum(wl_cell)
)[1]

BB = np.array(
    LightSource(
        source_type="black body",
        x=wl_cell,
        output_units="power_density_per_nm",
        entendue="Sun",
        T=5778,
    ).spectrum(wl_cell)
)[1]

illuminants = [D50, D65, AM15G, BB]

label = ["D50", "D65", "AM1.5g", 5778]

champion_pops = np.loadtxt(save_path +
                    "/champion_pop_"
                    + R_type
                    + str(n_peaks)
                    + "_"
                    + str(n_junctions)
                    + "_"
                    + str(fixed_height)
                    + str(max_height)
                    + "_"
                    + str(base)
                    + "_" + j01_method + ".txt",
                )

fig, ax = plt.subplots(1,1, figsize=(8, 1.3))

for j1, champion_pop in enumerate(champion_pops):

    for k1, illuminant in enumerate(illuminants):

        spec = gen_spectrum_ndip(champion_pop, 2, wl_cell, 1, 0)

        found_xyz = spec_to_XYZ(
            spec, illuminant, cmf, interval
        )
        print(j1, found_xyz)
        color_xyz_f = XYZColor(*found_xyz)
        color_srgb_f = convert_color(color_xyz_f, sRGBColor, target_illuminant='d65')

        color_srgb_f = [
            color_srgb_f.clamped_rgb_r,
            color_srgb_f.clamped_rgb_g,
            color_srgb_f.clamped_rgb_b,
        ]
        # print(color_srgb_f)

        ax.add_patch(
            Rectangle(
                xy=((j1, k1)),
                width=1,
                height=1,
                facecolor=color_srgb_f,
            )
        )

ax.text(-0.2, 0.45, "D50", fontsize=12,
        horizontalalignment='right', verticalalignment='center')
ax.text(-0.2, 1.45, "D65", fontsize=12,
        horizontalalignment='right', verticalalignment='center')
ax.text(-0.2, 2.45, "AM1.5G", fontsize=12,
        horizontalalignment='right', verticalalignment='center')
ax.text(-0.2, 3.45, "BB", fontsize=12,
        horizontalalignment='right', verticalalignment='center')
ax.set_ylim(0,4)
ax.set_xlim(0, 24)
ax.set_aspect('equal')

plt.axis('off')
plt.show()

x_rgb = np.array([0.64, 0.3, 0.15])
y_rgb = np.array([0.33, 0.6, 0.06])

X_rgb = x_rgb/y_rgb
Y_rgb = np.ones(3)
Z_rgb = (1-x_rgb-y_rgb)/y_rgb

XYZ_rgb_mat = np.vstack([X_rgb, Y_rgb, Z_rgb])

XYZ_rgb_mat_inv = np.linalg.inv(XYZ_rgb_mat)

for i1, illuminant in enumerate(illuminants):
    found_XYZ = np.array(spec_to_XYZ(
        1, illuminant, cmf, interval
    ))

    x = found_XYZ[0]/np.sum(found_XYZ)
    y = found_XYZ[1]/np.sum(found_XYZ)
    z = found_XYZ[2]/np.sum(found_XYZ)

    S_vec = np.matmul(XYZ_rgb_mat_inv, found_XYZ)

    M_mat = S_vec[None,:]*XYZ_rgb_mat

    # M is the matrix which converts RGB to XYZ. Invert for XYZ to RGB!

    M_mat_inv = np.linalg.inv(M_mat)

    # print(label[i1], found_XYZ)
    # # print(S_vec)
    # print(label[i1], M_mat)

def gamma_correction(x):

    if x <= 0.0031308:
        return 12.92*x
    else:
        return 1.055*x**(1/2.4) - 0.055

def convert_XYZ_to_sRGB(XYZ, illuminant, wl):
    cmf = load_cmf(wl)

    if illuminant == "D50":
        illum = load_D50(wl)

    elif illuminant == "D65":
        illum = load_D65(wl)

    elif illuminant == "AM1.5g":
        illum = np.array(
            LightSource(
                source_type="standard",
                version="AM1.5g",
                x=wl,
                output_units="power_density_per_nm",
            ).spectrum(wl)
        )[1]

    elif type(illuminant) == int or type(illuminant) == float:
        illum = np.array(
            LightSource(
                source_type="black body",
                x=wl,
                output_units="power_density_per_nm",
                entendue="Sun",
                T=illuminant,
            ).spectrum(wl)
        )[1]

    else: # assume illuminant is some arbitrary spectrum
        illum = illuminant

    x_rgb = np.array([0.64, 0.3, 0.15]) # from Wikipedia, x and y values for sRGB
    y_rgb = np.array([0.33, 0.6, 0.06])

    X_rgb = x_rgb / y_rgb
    Y_rgb = np.ones(3)
    Z_rgb = (1 - x_rgb - y_rgb) / y_rgb

    XYZ_rgb_mat = np.vstack([X_rgb, Y_rgb, Z_rgb])

    XYZ_rgb_mat_inv = np.linalg.inv(XYZ_rgb_mat)

    XYZ_W = np.array(spec_to_XYZ(
        1, illum, cmf, interval
    ))

    S_vec = np.matmul(XYZ_rgb_mat_inv, XYZ_W)

    M_mat = S_vec[None, :] * XYZ_rgb_mat

    # M is the matrix which converts RGB to XYZ. Invert for XYZ to RGB!

    M_mat_inv = np.linalg.inv(M_mat)

    RGB = np.matmul(M_mat_inv, XYZ)

    RGB_c = np.clip(RGB, 0, 1)

    # gamma correction
    sRGB = np.array([gamma_correction(x) for x in RGB_c])

    return sRGB

label_ilum = ["D50", "D65", "AM1.5g", 4000, 8000]

fig, ax = plt.subplots(1,1, figsize=(8, 1.3))

for j1, champion_pop in enumerate(champion_pops):

    for k1, lab in enumerate(label_ilum):
        color_names, color_XYZ = load_colorchecker(illuminant=lab)

        # spec = gen_spectrum_ndip(champion_pop, 2, wl_cell, 1, 0)
        #
        # found_xyz = spec_to_XYZ(
        #     spec, illuminant, cmf, interval
        # )
        found_xyz = color_XYZ[j1]
        # print(j1, found_xyz)
        color_srgb_f = convert_XYZ_to_sRGB(found_xyz, "D50", wl_cell)

        ax.add_patch(
            Rectangle(
                xy=((j1, k1)),
                width=1,
                height=1,
                facecolor=color_srgb_f,
            )
        )
        print(k1, color_srgb_f)

for j1, lab in enumerate(label_ilum):
    ax.text(-0.2, j1+0.45, lab, fontsize=12,
            horizontalalignment='right', verticalalignment='center')

ax.set_ylim(0,len(label_ilum))
ax.set_xlim(0, 24)
ax.set_aspect('equal')

plt.axis('off')
plt.show()


# check if the behaviour makes sense by using fixed XYZ values and using the D50
# and D65 illuminants in colormath.

color_names, color_XYZ = load_colorchecker()

fig, ax = plt.subplots(1,1, figsize=(8, 1.3))

for j1, XYZ in enumerate(color_XYZ):
    color_xyz_f = XYZColor(*XYZ)
    color_srgb_f_D50 = convert_color(color_xyz_f, sRGBColor,
                                     target_illuminant="d50")

    color_srgb_f_D50 = [
        color_srgb_f_D50.clamped_rgb_r,
        color_srgb_f_D50.clamped_rgb_g,
        color_srgb_f_D50.clamped_rgb_b,
    ]

    color_srgb_f_D65 = convert_color(color_xyz_f, sRGBColor,
                                     target_illuminant="e")

    color_srgb_f_D65 = [
        color_srgb_f_D65.clamped_rgb_r,
        color_srgb_f_D65.clamped_rgb_g,
        color_srgb_f_D65.clamped_rgb_b,
    ]

    ax.add_patch(
        Rectangle(
            xy=((j1, 0)),
            width=1,
            height=1,
            facecolor=color_srgb_f_D50,
        )
    )

    ax.add_patch(
        Rectangle(
            xy=((j1, 1)),
            width=1,
            height=1,
            facecolor=color_srgb_f_D65,
        )
    )

ax.set_ylim(0,2)
ax.set_xlim(0, 24)
ax.set_aspect('equal')

plt.axis('off')
plt.show()