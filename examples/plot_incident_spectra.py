import numpy as np
from solcore.light_source import LightSource
import matplotlib.pyplot as plt
from matplotlib import rc

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