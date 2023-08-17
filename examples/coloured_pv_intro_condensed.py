import numpy as np
import matplotlib.pyplot as plt
from solcore.light_source import LightSource
from ECoPV.ecopv.spectrum_functions import spec_to_XYZ, load_cmf
from colormath.color_objects import sRGBColor, XYZColor
from colormath.color_conversions import convert_color
from matplotlib.patches import Rectangle
import seaborn as sns
from matplotlib import rc

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})

cols = sns.color_palette('Set2', 6)[-3:]

max_plot_wl = 1850

E1 = 1.7
E2 = 1.2
E3 = 0.7

interval = 1
wl = np.arange(250, 2500.1, interval)
AM15G = LightSource(source_type='standard', version='AM1.5g', x=wl, output_units='photon_flux_per_nm')

AM15G_spd = LightSource(source_type='standard', version='AM1.5g', x=wl, output_units='power_density_per_nm').spectrum()[1]

R = np.zeros_like(wl)

R_c_1 = 450
R_c_2 = 650

R_w_1 = 50
R_w_2 = 90

R[np.all((wl < R_c_1 + R_w_1/2, wl > R_c_1 - R_w_1/2), axis=0)] = 1
R[np.all((wl < R_c_2 + R_w_2/2, wl > R_c_2 - R_w_2/2), axis=0)] = 1

cmf = load_cmf(wl)
XYZ = spec_to_XYZ(R, AM15G_spd, cmf, interval)

color_xyz_t = XYZColor(*XYZ)
color_srgb_t = convert_color(color_xyz_t, sRGBColor,
                             target_illuminant="d65")  # .get_value_tuple()
color_srgb_t = [
    color_srgb_t.clamped_rgb_r,
    color_srgb_t.clamped_rgb_g,
    color_srgb_t.clamped_rgb_b,
]

Eg_1 = 1240/E1
Eg_2 = 1240/E2
Eg_3 = 1240/E3

A1 = np.zeros_like(wl)
A1[wl < Eg_1] = 1

A2 = np.zeros_like(wl)
A2[np.all((wl < Eg_2, wl > Eg_1), axis=0)] = 1

A3 = np.zeros_like(wl)
A3[np.all((wl < Eg_3, wl > Eg_2), axis=0)] = 1

fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(5,5))

ax_fl1 = ax2.twinx()
ax_fl2 = ax3.twinx()

ax_fl1.plot(wl, AM15G.spectrum()[1]*R/1e18, color='royalblue', alpha=1, linewidth=0.7)
ax_fl2.plot(wl, AM15G.spectrum()[1]*A1*(1-R)/1e18, color='royalblue', alpha=1, linewidth=0.7)
ax_fl2.plot(wl, AM15G.spectrum()[1]*A2*(1-R)/1e18, color='royalblue', alpha=1, linewidth=0.7)
ax_fl2.plot(wl, AM15G.spectrum()[1]*A3*(1-R)/1e18, color='royalblue', alpha=1, linewidth=0.7)

ax1.set_ylabel("Photon flux\n" + r"($\times 10^{18}$ m$^{-2}$ nm$^{-1}$ s$^{-1}$)")
ax2.set_ylabel("Reflectance (%)")
ax3.set_ylabel("Absorptance (%)")

ax1.plot(wl, AM15G.spectrum()[1]/1e18, color='royalblue', linewidth=0.7)
ax2.fill_between(wl, 100*R, color='k', alpha=0.3)
ax3.fill_between(wl, 100*A1*(1-R), color=cols[0], alpha=0.5)
ax3.fill_between(wl, 100*A2*(1-R), color=cols[1], alpha=0.5)
ax3.fill_between(wl, 100*A3*(1-R), color=cols[2], alpha=0.5)

for ax in [ax1, ax_fl1, ax_fl2]:
    ax.set_ylabel("Photon flux\n" + r"($\times 10^{18}$ m$^{-2}$ nm$^{-1}$ s$^{-1}$)")
    ax.set_ylim(0, 5)
    ax.set_yticks(np.arange(0, 5.1, 1))
    ax.set_xticks(np.arange(250, 2501, 250))

for ax in [ax1, ax2, ax3]:
    ax.set_xlim(250, max_plot_wl)
    ax.grid()

    if ax != ax1:
        ax.set_yticks(np.arange(0, 101, 20))
        ax.set_ylim(0, 100)

    if ax != ax3:
        ax.set_xticklabels([])

ax_fl2.axvline(x=Eg_1, color='k', linestyle='--')
ax_fl2.axvline(x=Eg_2, color='k', linestyle='--')
ax_fl2.axvline(x=Eg_3, color='k', linestyle='--')

ax3.set_xlabel('Wavelength (nm)')

# labels = ['AM1.5G\nphoton flux',
#           'Cell\nreflectance',
#           'Absorbed in each sub-cell',
#           ]
#
# x_pos = [1250, R_c_2 + R_w_2/2 + 20, Eg_1 + 20]
# for i1, ax in enumerate([ax1, ax_fl1, ax_fl2]):
#     ax.text(x_pos[i1], 3.6, labels[i1])

# ax3.spines['left'].set_color('indianred')
# ax3.yaxis.label.set_color('indianred')
# ax3.tick_params(axis='y', colors='indianred')

for ax in [ax_fl1, ax_fl2]:
    ax.spines['right'].set_color('royalblue')
    ax.yaxis.label.set_color('royalblue')
    ax.tick_params(axis='y', colors='royalblue')

ax_fl1.text(1000, 3.5, 'Perceived colour:')

ax_fl1.add_patch(
        Rectangle(
            xy=(1200, 2.2),
            width=250,
            height=1,
            facecolor=color_srgb_t,
            edgecolor='k',
        )
        )

plt.tight_layout()
plt.show()