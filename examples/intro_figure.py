import numpy as np
import matplotlib.pyplot as plt
from os.path import join, dirname
import pathlib
from solcore.light_source import LightSource
from ecopv.spectrum_functions import gen_spectrum_ndip, load_cmf, spec_to_XYZ
from ecopv.plot_utilities import *

from matplotlib import rc

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})

current_path = pathlib.Path(__file__).parent.resolve()

color_names = np.array(
    [
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
        "Black-2",
    ]
)

ind = np.where(color_names == "Magenta")[0][0]

color_R = np.loadtxt(join(dirname(current_path), "ecopv", "data",
                          "ColorChecker_R_BabelColor.csv"), delimiter=',',
                     encoding='utf-8-sig')
interval = 0.1
wl_cell = np.arange(300, 1200, interval)

light_source = LightSource(
    source_type="standard",
    version="AM1.5g",
    x=wl_cell,
    output_units="photon_flux_per_nm",
)

R_type = "sharp"
n_peaks = 2
n_junctions = 1
fixed_height = True
max_height = 1
base = 0
j01_method = "perfect_R"
light_source_name = "AM1.5g"

champion_pops = np.loadtxt(
    "results/champion_pop_"
    + R_type
    + str(n_peaks)
    + "_"
    + str(n_junctions)
    + "_"
    + str(fixed_height)
    + str(max_height)
    + "_"
    + str(base)
    + "_" + j01_method + light_source_name + ".txt",
)

pop = champion_pops[ind]
R_optim = gen_spectrum_ndip(pop, 2, wl_cell)

color_list = sRGB_color_list(order="unsort")

cmf = load_cmf(wl_cell)

wl_colors = np.arange(380, 780.1, 0.5)
RGBA = wavelength_to_rgb(wl_colors)

fig, (ax, ax2) = plt.subplots(2,1, figsize=(5, 5),
                              gridspec_kw={
                                  "height_ratios": [2, 1.2],
                                  "hspace": 0.2,
                                  "wspace": 0.05,
                              },
                              )

ax.plot(color_R[0], color_R[ind+1], '--', color=color_list[ind], label=r"$R_{real}$")
ax.set_xlim(min(color_R[0]), max(color_R[0]))
ax2.set_xlim(min(color_R[0]), max(color_R[0]))
ax.plot(wl_cell, R_optim, '-', color=color_list[ind], label=r"$R_{ideal}$")

ax.grid()

plt.tight_layout()

ax2.grid()
ax2.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Reflectance")

ax2.set_ylabel(r"SPD ($\times 10^{18}$ W m$^{-2}$ nm$^{-1}$)")
ax2.set_ylim(0, 5)
ax.set_ylim(0, 1.01)
ax.legend(loc="upper right")

ax.set_yticks([0, 0.25, 0.5, 0.75, 1])

ax.spines['left'].set_color(color_list[ind])
ax.yaxis.label.set_color(color_list[ind])
ax.tick_params(axis='y', colors=color_list[ind])
ax.xaxis.set_ticklabels([])

ax2.plot(wl_colors, light_source.spectrum(wl_colors)[1]/1e18, '-k',
         alpha=0.5)

for i1 in range(len(RGBA)):
    ax2.add_patch(
        Rectangle(
            xy=(wl_colors[i1], 0),
            height=light_source.spectrum(wl_colors)[1][i1]/1e18,
            width=0.5,
            facecolor=RGBA[i1, :3],
            alpha=RGBA[i1, 3],
        )
    )
plt.tight_layout()

ax_cmf = ax.twinx()

ax_cmf.set_ylim(0, 2.02)
ax_cmf.plot(wl_cell, cmf[:,0], '-.', color='r', alpha=0.7, label=r"$\bar{x}$")
ax_cmf.plot(wl_cell, cmf[:,1], '-.', color='g', alpha=0.7, label=r"$\bar{y}$")
ax_cmf.plot(wl_cell, cmf[:,2], '-.', color='b', alpha=0.7, label=r"$\bar{z}$")

# ax.plot(0, 0, '-k', label="AM1.5g")
# ax.plot(0, 0, '-', color=color_list[ind], label=r"$R_{ideal}$")
# ax.plot(0, 0, '--', color=color_list[ind], label=r"$R_{real}$")
ax_cmf.set_ylabel("Spectral sensitivity")
ax_cmf.set_yticks([0, 1, 2])

plt.show()


from matplotlib import rc


rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"], "size": 14})

params = np.array( [438.44042242,  610.9970763,   18.43156718,   67.74693666])
params_schr = np.array([ 440.13933368,  580.93374126])
R = gen_spectrum_ndip(params, 2, wl_cell)

R_schr = np.zeros_like(R)
R_schr[wl_cell <= params_schr[0]] = 1
R_schr[wl_cell >= params_schr[1]] = 1

illuminant = np.array(
    LightSource(
        source_type="standard",
        version="AM1.5g",
        x=wl_cell,
        output_units="power_density_per_nm",
    ).spectrum(wl_cell)[1]
)

XYZ_schr = spec_to_XYZ(R_schr, illuminant, cmf, interval)
XYZ = spec_to_XYZ(R, illuminant, cmf, interval)

color_xyz_t = XYZColor(*XYZ_schr)
color_srgb_t = convert_color(color_xyz_t, sRGBColor,
                             target_illuminant="d65")#.get_value_tuple()
color_srgb_t_schr = [
    color_srgb_t.clamped_rgb_r,
    color_srgb_t.clamped_rgb_g,
    color_srgb_t.clamped_rgb_b,
        ]

color_xyz_t = XYZColor(*XYZ)
color_srgb_t = convert_color(color_xyz_t, sRGBColor,
                             target_illuminant="d65")#.get_value_tuple()
color_srgb_t = [
    color_srgb_t.clamped_rgb_r,
    color_srgb_t.clamped_rgb_g,
    color_srgb_t.clamped_rgb_b,
        ]

fig, ax = plt.subplots(1,1, figsize=(7, 4))

ax.plot(wl_cell, cmf[:,0], '-.', color='r', alpha=0.7, label=r"$\bar{x}$")
ax.plot(wl_cell, cmf[:,1], '-.', color='g', alpha=0.7, label=r"$\bar{y}$")
ax.plot(wl_cell, cmf[:,2], '-.', color='b', alpha=0.7, label=r"$\bar{z}$")

# ax.plot(0, 0, '-k', label="AM1.5g")
# ax.plot(0, 0, '-', color=color_list[ind], label=r"$R_{ideal}$")
# ax.plot(0, 0, '--', color=color_list[ind], label=r"$R_{real}$")
# ax.grid()
ax2.set_ylim(0,1)
ax2.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Spectral sensitivity")
ax.set_ylim(0, 2.04)
ax2 = ax.twinx()
ax2.set_ylabel("Reflectance")
ax2.set_yticks([0, 0.25, 0.5, 0.75, 1])
ax2.plot(wl_cell, R_schr, '-k')
ax2.plot(wl_cell, R, '--r')
ax.legend(loc="upper right")
# ax2.set_yticks([0, 0.25, 0.5, 0.75, 1])
ax.set_xlim(380, 730)
ax.set_xlabel("Wavelength (nm)")
ax2.set_ylim(0,1.02)

ax.add_patch(
    Rectangle(
        xy=(670, 0.8),
        width=40,
        height=0.2,
        facecolor=color_srgb_t,
    )
)
ax.add_patch(
    Rectangle(
        xy=(670, 0.5),
        width=40,
        height=0.2,
        facecolor=color_srgb_t_schr,
    )
)

plt.tight_layout()
plt.show()
