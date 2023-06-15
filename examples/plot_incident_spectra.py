import numpy as np
from solcore.light_source import LightSource
import matplotlib.pyplot as plt
from matplotlib import rc
from ecopv.spectrum_functions import load_cmf, spec_to_XYZ
from ecopv.main_optimization import load_colorchecker
from os import path
from cycler import cycler
import seaborn as sns
from colormath.color_objects import sRGBColor, XYZColor
from colormath.color_conversions import convert_color

pal = sns.husl_palette(6, s=0.6)

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})

interval = 0.1  # wavelength interval (in nm)
wl_cell = np.arange(150, 4000, interval)  # wavelengths

# Define the incident photon flux. This should be a 2D array with the first row being the wavelengths and the second row
# being the photon flux at each wavelength. The wavelengths should be in nm and the photon flux in photons/m^2/s/nm.
photon_flux_AM15 = np.array(
    LightSource(
        source_type="standard",
        version="AM1.5g",
        x=wl_cell,
        output_units="photon_flux_per_nm",
    ).spectrum(wl_cell)
)

photon_flux_bb = np.array(LightSource(
    source_type="black body",
    x=wl_cell,
    output_units="photon_flux_per_nm",
    entendue="Sun",
    T=5778,
).spectrum(wl_cell))

fig, ax = plt.subplots(1,1)
ax.plot(wl_cell, photon_flux_AM15[1]/1e18, '-k', label="AM1.5G")
ax.plot(wl_cell, photon_flux_bb[1]/1e18, '--r', label="5778K black body")
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel(r"Photon flux ($\times 10^{18}$ photons m$^{-2}$s$^{-1}$ nm$^{-1}$)")
ax.grid(axis="both", color="0.9")
ax.legend()
plt.tight_layout()
ax.set_xlim(min(wl_cell), max(wl_cell))
ax.set_ylim(-0.05, 5.2)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from solcore.light_source import LightSource
import pathlib

wl_cell = np.linspace(300, 1800, 400)

SPD = np.array(
    LightSource(
        source_type="standard",
        version="AM1.5g",
        x=wl_cell,
        output_units="power_density_per_nm",
    ).spectrum(wl_cell)
)[1]

cmf = load_cmf(wl_cell)

SPD_bb = np.array(
    LightSource(
        source_type="black body",
        x=wl_cell,
        output_units="power_density_per_nm",
        entendue="Sun",
        T=5778,
    ).spectrum(wl_cell)
)[1]

current_path = pathlib.Path(__file__).parent.resolve()


D50 = np.loadtxt(path.join(path.dirname(current_path), "ecopv", "data",
                    "CIE_std_illum_D50.csv"), delimiter=',').T
D65 = np.loadtxt(path.join(path.dirname(current_path), "ecopv", "data",
                    "CIE_std_illum_D65.csv"), delimiter=',').T
pal2 = ["r", "g", "b"]
cols = cycler("color", pal2)
params = {"axes.prop_cycle": cols}
plt.rcParams.update(params)

fig, ax = plt.subplots(1,1)
ax.plot(wl_cell, SPD/max(SPD), '-', label="AM1.5G", color=pal[5])
ax.plot(wl_cell, SPD_bb/max(SPD_bb), '--', label="5778K black body", color="k")
ax.plot(D50[0], D50[1]/max(D50[1]), '-.', label="D50 illuminant", color=pal[1])
ax.plot(D50[0], D65[1]/max(D65[1]), '-.', label="D65 illuminant", color=pal[4])
ax.plot(wl_cell, cmf[:,0]/3, linestyle='dotted', alpha=0.8, label=r"$\bar{x}$/3")
ax.plot(wl_cell, cmf[:,1]/3, linestyle='dotted', alpha=0.8, label=r"$\bar{y}$/3")
ax.plot(wl_cell, cmf[:,2]/3, linestyle='dotted', alpha=0.8, label=r"$\bar{z}$/3")
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Normalised spectral power distribution")
ax.grid(axis="both", color="0.9")
ax.legend(loc="lower right")
plt.tight_layout()
ax.set_xlim(min(wl_cell), max(wl_cell))
ax.set_ylim(0, 1.02)
plt.show()


fig, ax = plt.subplots(1,1, figsize=(5, 2))
ax.plot(wl_cell, SPD/max(SPD), '-', label="AM1.5G", color=pal[2])
ax.plot(wl_cell, SPD_bb/max(SPD_bb), '--', label="5778K black body", color="k")
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Normalised SPD")
ax.grid(axis="both", color="0.9")
ax.legend(loc="upper right")
plt.tight_layout()
ax.set_xlim(min(wl_cell), max(wl_cell))
ax.axvline(1240/1.13, linestyle='--', color=pal[0])
ax.axvline(1240/1.34, linestyle='--', color=pal[0])
ax.text(1240/1.13, 0.7, "1.13 eV", rotation=90, va='center', ha='right')
ax.text(1240/1.34, 0.7, "1.34 eV", rotation=90, va='center', ha='right')
ax.set_ylim(0, 1.02)
plt.show()


Macbeth = np.loadtxt(path.join(path.dirname(current_path), "ecopv", "data",
                    "Macbeth_ColorChecker_R.csv"), delimiter=',', skiprows=1).T

MacBeth_2 = np.loadtxt(path.join(path.dirname(current_path), "ecopv", "data",
                    "ColorChecker_R_BabelColor.csv"), delimiter=',', encoding='utf-8-sig')

wl_M = Macbeth[0]
R_M = Macbeth[1:]

wl_M_2 = MacBeth_2[0]
R_M_2 = MacBeth_2[1:]

cmf_M = load_cmf(wl_M_2)

AM15M = np.array(
    LightSource(
        source_type="standard",
        version="AM1.5g",
        x=wl_cell,
        output_units="power_density_per_nm",
    ).spectrum(wl_M_2)[1])

XYZ_R = np.zeros((len(R_M_2), 3))

color_names, _ = load_colorchecker()

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(8, 7))

for i1, R in enumerate(R_M_2):
    XYZ_R[i1] = spec_to_XYZ(R, AM15M, cmf_M, 5)

    color_xyz_f = XYZColor(*XYZ_R[i1])
    color_srgb_f = convert_color(color_xyz_f, sRGBColor,
                                 target_illuminant="d65")
    # d65 is native illuminant of sRGB (but actually specifying illuminant doesn't do
    # anything...)
    print(color_srgb_f)

    color_srgb_f = [
        color_srgb_f.clamped_rgb_r,
        color_srgb_f.clamped_rgb_g,
        color_srgb_f.clamped_rgb_b,
    ]

    if i1 < 6:
        # ax1.plot(wl_M, R, color=color_srgb_f, label=color_names[i1])
        ax1.plot(wl_M_2, R_M_2[i1], color=color_srgb_f, label=color_names[i1])

    if i1 >= 6 and i1 < 12:
        # ax2.plot(wl_M, R, color=color_srgb_f, label=color_names[i1])
        ax2.plot(wl_M_2, R_M_2[i1], color=color_srgb_f, label=color_names[i1])

    if i1 >= 12 and i1 < 18:
        # ax3.plot(wl_M, R, color=color_srgb_f, label=color_names[i1])
        ax3.plot(wl_M_2, R_M_2[i1], color=color_srgb_f, label=color_names[i1])

    if i1 >= 18 and i1 < 24:
        # ax4.plot(wl_M, R, color=color_srgb_f, label=color_names[i1])
        ax4.plot(wl_M_2, R_M_2[i1], color=color_srgb_f, label=color_names[i1])

for ax in [ax1, ax2, ax3, ax4]:
    ax.set_xlim(380, 730)
    ax.set_ylim(0, 1)
    ax.grid("both")
    ax.legend(fontsize=8)

for ax in [ax1, ax3]:
    ax.set_ylabel("Reflectance")

for ax in [ax3, ax4]:
    ax.set_xlabel("Wavelength (nm)")

plt.show()

np.savetxt(path.join(path.dirname(current_path), "ecopv", "data",
                     "Macbeth_XYZ_from_R.txt"), XYZ_R)
