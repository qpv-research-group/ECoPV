from colormath.color_objects import xyYColor, XYZColor
from ecopv.plot_utilities import *
from colour import wavelength_to_XYZ

Y = 0.5

wl_vis = np.linspace(360, 780, 500)

XYZ = wavelength_to_XYZ(wl_vis)

sumXYZ = np.sum(XYZ, axis=1)

xg = XYZ[:, 0] / sumXYZ
yg = XYZ[:, 1] / sumXYZ


color_names, color_xyY = load_colorchecker(
    output_coords="xyY"
)  # load the names and XYZ coordinates of the 24 default Babel colors

xs = np.arange(np.min(xg), np.max(xg), 0.005)
ys = np.arange(np.min(yg), np.max(yg), 0.005)

width = np.diff(xs)[0]
height = np.diff(ys)[0]

is_inside = np.full((len(xs), len(ys)), False)

peak = np.argmax(yg)

left_edge = [xg[:peak], yg[:peak]]
right_edge = [xg[peak:], yg[peak:]]

# now check if the points are inside the gamut defined by the spectral locus

for j, yc in enumerate(ys):
    left_y = np.argmin(np.abs(left_edge[1] - yc))
    right_y = np.argmin(np.abs(right_edge[1] - yc))

    left_x = left_edge[0][left_y]
    right_x = right_edge[0][right_y]
    is_inside[np.all((xs > left_x, xs < right_x), axis=0), j] = True

# eliminate everything below the line of purples:

# equation for line of purples:

slope = (yg[-1] - yg[0]) / (xg[-1] - xg[0])
c = yg[0] - slope * xg[0]

for j, yc in enumerate(ys):
    above = yc > slope * xs + c
    is_inside[:, j] = np.all((above, is_inside[:, j]), axis=0)


standard_illuminant = [0.3128, 0.3290, Y]
XYZ = convert_color(xyYColor(*standard_illuminant), XYZColor)
s_i_RGB = np.array(convert_color(XYZ, sRGBColor).get_value_tuple())
s_i_RGB[s_i_RGB > 1] = 1

label_wls = np.arange(440, 660, 20)

XYZlab = wavelength_to_XYZ(label_wls)

sumXYZlab = np.sum(XYZlab, axis=1)

xgl = XYZlab[:, 0] / sumXYZlab
ygl = XYZlab[:, 1] / sumXYZlab

tick_orig = np.zeros((len(label_wls), 2))
tick_dir = np.zeros((len(label_wls), 2))
# create ticks
for m1, lwl in enumerate(label_wls):
    p0 = wavelength_to_XYZ(lwl)
    p1 = wavelength_to_XYZ(lwl - 1)
    p2 = wavelength_to_XYZ(lwl + 1)

    p0 = p0 / np.sum(p0)
    p1 = p1 / np.sum(p1)
    p2 = p2 / np.sum(p2)

    m = np.array([p2[0] - p1[0], p2[1] - p1[1]])
    mp = np.array([-m[1], m[0]])
    mp = mp / np.linalg.norm(mp)
    # b = p1[1] + (1/m) * p1[0]

    tick_orig[m1] = p0[:2]
    tick_dir[m1] = p0[:2] + 0.02 * mp

fig, ax = plt.subplots(1, figsize=(4, 3))

# ax.set_aspect('equal')
ax.set_facecolor(s_i_RGB)
ax.plot(xg, yg, "k")
ax.plot([xg[0], xg[-1]], [yg[0], yg[-1]], "k")

for m1, lwl in enumerate(label_wls):
    ax.plot(
        [tick_orig[m1, 0], tick_dir[m1, 0]], [tick_orig[m1, 1], tick_dir[m1, 1]], "-k"
    )

    if lwl > 520:
        ax.text(*tick_dir[m1], str(lwl))

    elif lwl == 520:
        ax.text(*tick_dir[m1], str(lwl), horizontalalignment="center")

    else:
        ax.text(
            *tick_dir[m1],
            str(lwl),
            horizontalalignment="right",
            verticalalignment="center"
        )

ax.set_xlim(-0.09, 0.8)
ax.set_ylim(-0.07, 0.9)
ax.set_xlabel("x")
ax.set_ylabel("y")

for j1, x in enumerate(xs):
    for k1, y in enumerate(ys):

        if is_inside[j1, k1]:
            XYZ = convert_color(xyYColor(x, y, Y), XYZColor)
            RGB = np.array(convert_color(XYZ, sRGBColor).get_value_tuple())

            RGB[RGB > 1] = 1
            #     plt.plot(x, y, '.')
            ax.add_patch(
                Rectangle(
                    xy=(x - width / 2, y - height / 2),
                    width=width,
                    height=height,
                    facecolor=RGB,
                )
            )


for m1, c in enumerate(color_xyY[:18]):
    ax.scatter(c[0], c[1], s=4, facecolor="none", edgecolor="k", linewidth=0.5)
    ax.text(c[0], c[1], str(m1 + 1), size=8)

ax.scatter(
    color_xyY[19, 0], color_xyY[19, 1], s=8, facecolor="k", edgecolor="k", linewidth=0.5
)
ax.text(color_xyY[19, 0], color_xyY[19, 1], "19-24", size=8)

ax.yaxis.set_minor_locator(tck.AutoMinorLocator())
ax.xaxis.set_minor_locator(tck.AutoMinorLocator())
ax.grid(axis="both", color="0.4", alpha=0.5)
ax.tick_params(direction="in", which="both", top=True, right=True)
# ax.set_axisbelow(True)

fig.savefig("gamut_colorchecker.pdf", bbox_inches="tight")
plt.tight_layout()
plt.show()
