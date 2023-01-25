from color_cell_optimization import load_babel, make_spectrum_ndip, \
    load_cmf, spec_to_XYZ, wavelength_to_rgb, getIVmax, \
    gen_spectrum_ndip
from colormath.color_objects import sRGBColor, XYZColor
from colormath.color_conversions import convert_color

import numpy as np
from solcore.light_source import LightSource
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.patches import Rectangle
from cycler import cycler
import xarray as xr

from plot_utilities import *
from colour.plotting import plot_RGB_chromaticities_in_chromaticity_diagram_CIE1931, \
plot_chromaticity_diagram_CIE1931, render
from colour import XYZ_to_xy, xyY_to_XYZ, XYZ_to_RGB



import matplotlib.pyplot as plt

# Plotting the *CIE 1931 Chromaticity Diagram*.
# The argument *standalone=False* is passed so that the plot doesn't get
# displayed and can be used as a basis for other plots.
plot_chromaticity_diagram_CIE1931(standalone=False, show_diagram_colours=False)

plt.title("")
# Plotting the *CIE xy* chromaticity coordinates.
x, y = 0.4, 0.3
plt.plot(x, y, "o-", color="white")

# Annotating the plot.
# plt.annotate(
#     patch_sd.name.title(),
#     xy=xy,
#     xytext=(-50, 30),
#     textcoords="offset points",
#     arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=-0.2"),
# )

# Displaying the plot.

xs = np.linspace(0.2, 0.7, 20)
ys = np.linspace(0.25, 0.7, 30)

width = np.diff(xs)[0]
height = np.diff(ys)[0]

Y = 0.8

ax = plt.gca()

for x in xs:
    for y in ys:

        xyY = [x, y, Y]
        XYZ = XYZColor(*xyY_to_XYZ(xyY))
        RGB = np.array(convert_color(XYZ, sRGBColor).get_value_tuple())
        print(RGB)
        RGB[RGB > 1] = 1

        ax.add_patch(
            Rectangle(xy=(x-width/2,y-height/2), width=width,
                      height=height,
                      facecolor=RGB)
        )


render(
    standalone=True,
    limits=(-0.1, 0.9, -0.1, 0.9),
    x_tighten=True,
    y_tighten=True,
)