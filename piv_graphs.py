import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from load_data import piv_data

# Vector Field Colour Scheme
vccsm = cm.turbo 
# vccsm = cm.coolwarm

xcount, ycount = 395, 57
ylim = 57
minl = (57 - ylim)*395
step = 20 # Resolution
x, y = piv_data[:, 0], piv_data[:, 1]
xmin, xmax = x.min()/1.02, x.max()*1.02
ymin, ymax = -0.001, y.max()*1.03
u, v = piv_data[:, 2], piv_data[:, 3]
X, Y = piv_data[minl::step, 0], piv_data[minl::step, 1]
U, V = piv_data[minl::step, 2], piv_data[minl::step, 3]

colors = np.linalg.norm(np.column_stack((U, V)), axis=1)
norm = Normalize()
norm.autoscale(colors)

fig_piv, ax_piv = plt.subplots()
ax_piv.quiver(X, Y, U, V, color=vccsm(norm(colors)), pivot="mid", scale=100, scale_units="xy", width=0.001, headwidth=3, headlength=4, headaxislength=3)
# ax_piv.set_title("PIV data for AoA 2°, Re = 200 000")
ax_piv.set_xlabel("x/c [-]")
ax_piv.set_ylabel("y/c [-]")
ax_piv.set(ylim=(ymin, ymax), xlim=(xmin, xmax))
sm = cm.ScalarMappable(cmap=vccsm, norm=norm)
sm.set_array([])
plt.colorbar(sm, ax=ax_piv, label="Absolute velocity [1/U$_{inf}$]")

# absv = np.linalg.norm(np.column_stack((u, v)), axis=1)
absv = np.abs(u)
fig_hm, ax_hm = plt.subplots()
heatmap = ax_hm.imshow(absv.reshape((ycount,xcount))[:,::-1], extent=(piv_data[-1,0], piv_data[0,0], piv_data[-1,1], piv_data[0,1]), cmap=cm.coolwarm, interpolation="nearest", aspect="auto")
plt.colorbar(heatmap, label="Absolute velocity [1/U$_{inf}$]", ax=ax_hm)
ax_hm.set_xlabel("x/c [-]")
ax_hm.set_ylabel("y/c [-]")

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

norm = Normalize()
norm.autoscale(speed)

fig_strm, ax_strm = plt.subplots()
ax_strm.streamplot(xi, yi, uCi.reshape((xcount, ycount)).T, vCi.reshape((xcount, ycount)).T, color=speed.reshape((xcount, ycount)).T, cmap=vccsm, linewidth=1, density=2, arrowstyle="->", arrowsize=1.5)
ax_strm.set_xlabel("x/c [-]")
ax_strm.set_ylabel("y/c [-]")
ax_strm.set(ylim=(ymin, ymax), xlim=(xmin, xmax))
sm = cm.ScalarMappable(cmap=vccsm, norm=norm)
sm.set_array([])
plt.colorbar(sm, ax=ax_strm, label="Absolute velocity [1/U$_{inf}$]")

# ax_quiver.quiver(xy_points[:,0], xy_points[:,1], uCi, vCi, color=cm.jet(norm(qv_speed)), pivot="mid", scale=100, scale_units="xy", width=0.001, headwidth=3, headlength=4, headaxislength=3)

# Velocity plot contour
# fig_vc, ax_vc = plt.subplots()
# absv = u**2 + v**2
# ax_vc.contour(x.reshape(57, 395), y.reshape(57, 395), absv.reshape(57, 395))

plt.show()