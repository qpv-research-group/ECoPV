from colour.difference import delta_E_CIE2000
from colormath.color_objects import sRGBColor, XYZColor
from colormath.color_conversions import convert_color
import numpy as np
from ecopv.spectrum_functions import XYZ_from_pop_dips, convert_XYZ_to_Lab
from solcore.light_source import LightSource
from ecopv.main_optimization import load_colorchecker

n_junc_loop = [1, 2, 3, 4, 5, 6]
n_peak_loop = [2, 3]

interval = 0.1  # wavelength interval (in nm)
wl = np.arange(350, 750, interval)

R_type = "sharp"  # "sharp" for rectangular dips or "gauss" for gaussians
fixed_height = True  # fixed height peaks (will be at the value of max_height) or not
j01_method = "perfect_R"
fixed_height = True
max_height = 1
base = 0
light_source_name = "AM1.5g"

color_names, color_XYZ = load_colorchecker()  # 24 default Babel colors

photon_flux = np.array(
    LightSource(
        source_type="standard",
        version="AM1.5g",
        x=wl,
        output_units="photon_flux_per_nm",
    ).spectrum(wl)
)

for j1, n_junctions in enumerate(n_junc_loop):
    for i1, n_peaks in enumerate(n_peak_loop):

        champion_effs = np.loadtxt(
            "results/champion_eff_"
            + R_type
            + str(n_peaks)
            + "_"
            + str(n_junctions)
            + "_"
            + str(fixed_height)
            + str(max_height)
            + "_"
            + str(base) + "_"  + j01_method + light_source_name
            + "_2.txt"
        )
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
            + str(base) + "_"  + j01_method + light_source_name
            + "_2.txt"
        )

        color_XYZ_found = [XYZ_from_pop_dips(x, n_peaks, photon_flux, interval) for x in champion_pops]

        color_Lab_found = [convert_XYZ_to_Lab(x) for x in color_XYZ_found]
        color_Lab_target = [convert_XYZ_to_Lab(x) for x in color_XYZ]

        delta_E = [delta_E_CIE2000(x, y) for x, y in zip(color_Lab_found, color_Lab_target)]
        print(np.max(delta_E))
        print(delta_E)