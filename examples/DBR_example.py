from rayflare.transfer_matrix_method import tmm_structure
from solcore import material, si
import matplotlib.pyplot as plt
from solcore.structure import Layer
import numpy as np
from rayflare.options import default_options
from solcore.light_source import LightSource
from spectrum_functions import spec_to_XYZ, load_cmf
from optimization_functions import getPmax
from plot_utilities import plot_outcome
from solcore.constants import h, c

interval = 0.01
wavelengths = np.arange(300, 1200, interval) * 1e-9
opts = default_options()

opts.wavelengths = wavelengths
opts.pol = "s"

cmf = load_cmf(wavelengths * 1e9)

# Use AM1.5G spectrum:
light_source = LightSource(
    source_type="black body",
    x=wavelengths * 1e9,
    output_units="photon_flux_per_nm",
    entendue="Sun",
    T=5778,
)

photon_flux_cell = np.array(light_source.spectrum(wavelengths * 1e9))

photon_flux_color = photon_flux_cell[
    :, np.all((photon_flux_cell[0] >= 380, photon_flux_cell[0] <= 780), axis=0)
]

Egs = np.linspace(1, 1.8, 100)
plt.figure()
for rad_eff in [0.01, 0.1, 1]:
    eff = np.zeros_like(Egs)

    for i1, eg in enumerate(Egs):

        eff[i1] = getPmax(
            [eg], photon_flux_cell[1], wavelengths * 1e9, interval, rad_eff
        )

    argm = np.argmax(eff)
    print(Egs[argm])

    plt.plot(Egs, eff, label=str(rad_eff))
plt.legend()
plt.show()

# define the materials
SiN = material("Si3N4")()
TiO2 = material("TiO2b")()
Si = material("Si")()
Ag = material("Ag")()
Air = material("Air")()

plt.figure()
plt.plot(wavelengths * 1e9, SiN.n(wavelengths), label="SiN")
plt.plot(wavelengths * 1e9, TiO2.n(wavelengths), label="TiO2")
plt.legend()
plt.show()

target_wavelength = 575 * 1e-9
target_wavelength_2 = 450 * 1e-9
n_DBR_reps = 2

ARC_layer = Layer(width=si("75nm"), material=SiN)

cell_layer = Layer(width=si("300um"), material=Si)

DBR_layers = [
    Layer(3 * target_wavelength / (4 * TiO2.n(target_wavelength)), material=TiO2),
    Layer(3 * target_wavelength / (4 * SiN.n(target_wavelength)), material=SiN),
] * n_DBR_reps

DBR_layers_2 = [
    Layer(target_wavelength_2 / (4 * TiO2.n(target_wavelength_2)), material=TiO2),
    Layer(target_wavelength_2 / (4 * SiN.n(target_wavelength_2)), material=SiN),
] * n_DBR_reps


struct = tmm_structure(
    [ARC_layer] + DBR_layers_2 + DBR_layers, incidence=Air, transmission=Si
)

RAT = struct.calculate(opts)

plt.figure()
plt.plot(wavelengths * 1e9, RAT["R"], label="R")
plt.show()

XYZ = spec_to_XYZ(RAT["R"], h * c * photon_flux_cell[1] / wavelengths, cmf, interval)

plot_outcome(RAT["R"], photon_flux_cell, XYZ, "test")
plt.show()
