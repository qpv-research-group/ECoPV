import numpy as np
from solcore.light_source import LightSource
import matplotlib.pyplot as plt
from matplotlib import rc
from ecopv.spectrum_functions import load_cmf
from os import path
from cycler import cycler
import seaborn as sns

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

wl_cell = np.linspace(300, 800, 100)

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
pal2 = ["r", "g", "b"]
cols = cycler("color", pal2)
params = {"axes.prop_cycle": cols}
plt.rcParams.update(params)

fig, ax = plt.subplots(1,1)
ax.plot(wl_cell, SPD/max(SPD), '-', label="AM1.5G", color=pal[5])
ax.plot(wl_cell, SPD_bb/max(SPD_bb), '--', label="5778K black body", color="k")
ax.plot(D50[0], D50[1]/max(D50[1]), '-.', label="D50 illuminant", color=pal[1])
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