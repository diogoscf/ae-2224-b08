import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from load_data import piv_data

xcount, ycount = 395, 57
ylim = 57
minl = (57 - ylim)*395
step = 20 # Resolution
u, v = piv_data[:, 2], piv_data[:, 3]
X, Y = piv_data[minl::step, 0], piv_data[minl::step, 1]
U, V = piv_data[minl::step, 2], piv_data[minl::step, 3]

colors = np.linalg.norm(np.column_stack((U, V)), axis=1)
colors /= np.max(colors)
norm = Normalize()
norm.autoscale(colors)

fig_piv, ax_piv = plt.subplots()
ax_piv.quiver(X, Y, U, V, color=cm.jet(norm(colors)), pivot="mid", scale=100, scale_units="xy", width=0.001, headwidth=3, headlength=4, headaxislength=3)
# ax_piv.set_title("PIV data for AoA 2°, Re = 200 000")
ax_piv.set_xlabel("x/c [-]")
ax_piv.set_ylabel("y/c [-]")

absv = np.linalg.norm(np.column_stack((u, v)), axis=1)
fig_hm, ax_hm = plt.subplots()
heatmap = plt.imshow(absv.reshape((ycount,xcount))[:,::-1], extent=(piv_data[0,0], piv_data[-1,0], piv_data[0,1], piv_data[-1,1]), cmap="hot", interpolation="nearest")
plt.colorbar()

sorted_data = piv_data[np.lexsort((piv_data[:,1], piv_data[:,0]))]
X, Y = sorted_data[:, 0], sorted_data[:, 1]
U, V = sorted_data[:, 2], sorted_data[:, 3]
# regularly spaced grid spanning the domain of x and y 
xi = np.linspace(X.min(), X.max(), xcount)
yi = np.linspace(Y.min(), Y.max(), ycount)
U_grid = U.reshape((xcount, ycount))
V_grid = V.reshape((xcount, ycount))
xy_points = np.array([[x, y] for x in xi for y in yi])

# bicubic interpolation
uCi = RegularGridInterpolator((np.unique(X), np.unique(Y)), U_grid)(xy_points)
vCi = RegularGridInterpolator((np.unique(X), np.unique(Y)), V_grid)(xy_points)
speed = np.sqrt(uCi**2 + vCi**2)

fig_strm, ax_strm = plt.subplots()
ax_strm.streamplot(xi, yi, uCi.reshape((xcount, ycount)).T, vCi.reshape((xcount, ycount)).T, color=speed.reshape((xcount, ycount)).T, cmap="jet", linewidth=1, density=2, arrowstyle="->", arrowsize=1.5)

# ax_quiver.quiver(xy_points[:,0], xy_points[:,1], uCi, vCi, color=cm.jet(norm(qv_speed)), pivot="mid", scale=100, scale_units="xy", width=0.001, headwidth=3, headlength=4, headaxislength=3)
plt.show()