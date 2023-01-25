import matplotlib.ticker as tck
from colormath.color_objects import sRGBColor, XYZColor
from colormath.color_conversions import convert_color
import xarray as xr
from color_cell_optimization import load_babel
from matplotlib.patches import Rectangle
from matplotlib import rc
import numpy as np
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#rc('font',**{'family':'serif','serif':['Times']})
#rc('text', usetex=True)

patch_width = 0.7

color_names, color_XYZ = load_babel()

Y = np.hstack((color_XYZ[:, 1], [0]))
default_Y_cols = Y[:18]

col_names_default = xr.DataArray(data=color_names[:18], dims=['Y'],
                         coords={'Y': default_Y_cols})
col_names_default = col_names_default.sortby("Y", ascending=False)

color_XYZ_xr = xr.DataArray(color_XYZ[:18], dims=["color", "XYZ"], coords={"color": color_XYZ[:18, 1],
                                                                           "XYZ": ["X", "Y", "Z"]})

color_XYZ_xr = color_XYZ_xr.sortby("color", ascending=False)
color_XYZ_bw = xr.DataArray(color_XYZ[18:], dims=["color", "XYZ"], coords={"color": color_XYZ[18:, 1],
                                                                           "XYZ": ["X", "Y", "Z"]})
color_XYZ_xr = xr.concat([color_XYZ_xr, color_XYZ_bw], dim="color")

def add_colour_patches(ax, width, labels, color_XYZ=color_XYZ_xr):
     # width is with an axis spacing of "1" for the x-axis colour labels

    ymin = ax.get_ylim()[0]
    ymax = ax.get_ylim()[1]
    h = ax.get_window_extent().height
    w = ax.get_window_extent().width

    h_p = width*w/len(labels)
    h_ax = h_p*(ymax-ymin)/h

    for l1, lab in enumerate(labels):

        if lab != "Black":

            target = color_XYZ[l1]

        else:
            target = [0,0,0]

        color_xyz_t = XYZColor(*target)
        color_srgb_t = convert_color(color_xyz_t, sRGBColor).get_value_tuple()

        ax.add_patch(
            Rectangle(xy=(l1-(width/2), ymin), width=width,
                      height=h_ax,
                      facecolor=color_srgb_t)
        )


def apply_formatting(ax, color_labels=None, grid='both', n_colors=None):

    if n_colors is None:
        n_colors = len(color_labels)

    ax.yaxis.set_minor_locator(tck.AutoMinorLocator())
    ax.grid(axis=grid, color='0.9')

    ax.xaxis.set_ticks(np.arange(0, n_colors))
    ax.tick_params(direction='in', which='both', top=True, right=True)
    print(color_labels)

    if color_labels is not None:
        ax.set_xticklabels(color_labels, rotation=45, ha='right', rotation_mode='anchor')
        ax.tick_params(direction='inout', axis='x', top=True, length=7)

    else:
        ax.set_xticklabels([])

    ax.set_xlim(-0.6, len(ax.get_xticks())-0.4)

    ax.set_axisbelow(True)


def make_sorted_xr(arr, color_names, append_black=None, Y_cols=default_Y_cols, col_names_sorted=col_names_default):
    if arr.ndim == 1:
        dims = ['color']

    else:
        dims = ['color', 'n']

    eff_xr_col = xr.DataArray(data=arr[:18], dims=dims,
                              coords={'color': Y_cols})

    eff_xr_col = eff_xr_col.sortby('color', ascending=False)
    eff_xr_col = eff_xr_col.assign_coords(color=col_names_sorted.data)

    if append_black is not None:
        eff_xr_bw = xr.DataArray(data=np.append(arr[18:], [append_black], axis=0),
                                 dims=dims,
                                 coords={'color': np.append(color_names[18:], "Black")})

    else:
        eff_xr_bw = xr.DataArray(data=arr[18:],
                                 dims=dims,
                                 coords={'color': color_names[18:]})

    eff_xr = xr.concat([eff_xr_col, eff_xr_bw], dim='color')

    return eff_xr

