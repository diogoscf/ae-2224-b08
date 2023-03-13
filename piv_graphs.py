import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import solve_ivp

from load_data import piv_data

# Vector Field Colour Scheme
vccsm = cm.turbo 
# vccsm = cm.coolwarm

plot_vector_field = False
plot_velocity_heatmap = True
plot_streamlines = False
plot_individual_streamline = False

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

if plot_vector_field:
    fig_piv, ax_piv = plt.subplots()
    ax_piv.quiver(X, Y, U, V, color=vccsm(norm(colors)), pivot="mid", scale=100, scale_units="xy", width=0.001, headwidth=3, headlength=4, headaxislength=3)
    # ax_piv.set_title("PIV data for AoA 2°, Re = 200 000")
    ax_piv.set_xlabel("x/c [-]")
    ax_piv.set_ylabel("y/c [-]")
    ax_piv.set(ylim=(ymin, ymax), xlim=(xmin, xmax))
    sm = cm.ScalarMappable(cmap=vccsm, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax_piv, label="Absolute velocity [1/U$_{inf}$]")

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
u_intpl = RegularGridInterpolator((np.unique(X), np.unique(Y)), U_grid)
v_intpl = RegularGridInterpolator((np.unique(X), np.unique(Y)), V_grid)
uCi, vCi = u_intpl(xy_points), v_intpl(xy_points)
speed = np.sqrt(uCi**2 + vCi**2)

norm = Normalize()
norm.autoscale(speed)

def f(t, xy):
    if xy[1] < y.min() or xy[1] > y.max() or xy[0] < x.min() or xy[0] > x.max():
        return np.array([0, 0])
    return np.squeeze([u_intpl(xy), v_intpl(xy)])

def gen_strmln(seed_point):
    sol = solve_ivp(f, [0, 10], seed_point, first_step=1e-3, max_step=1e-2, method="RK45", dense_output=True)
    positions = sol.y
    while (positions[0, -1] > x.max() or positions[0, -1] < x.min() or positions[1, -1] > y.max() or positions[1, -1] < y.min()):
        positions = positions[:,:-1]
    return positions

if plot_velocity_heatmap:
    # absv = np.linalg.norm(np.column_stack((u, v)), axis=1)
    # absv = np.abs(u)
    factor = 5 # Resolution increase
    xj = np.linspace(X.min(), X.max(), xcount*factor)
    yj = np.linspace(Y.min(), Y.max(), ycount*factor)
    xy_grid = np.array([[x, y] for x in xj for y in yj])
    absv = u_intpl(xy_grid).reshape((xcount*factor,ycount*factor)).T
    # absv = np.abs(absv)
    # absv = (absv > 0.1)
    fig_hm, ax_hm = plt.subplots()
    heatmap = ax_hm.imshow(absv[::-1,:], extent=(piv_data[-1,0], piv_data[0,0], piv_data[-1,1], piv_data[0,1]), cmap=cm.turbo, interpolation="nearest", aspect="auto")
    plt.colorbar(heatmap, label="Absolute velocity [1/U$_{inf}$]", ax=ax_hm)
    ax_hm.set_xlabel("x/c [-]")
    ax_hm.set_ylabel("y/c [-]")

    ax_hm.contour(xj,  yj, absv.reshape(ycount*factor,xcount*factor), levels=[0.45], colors="red", linewidths=1)

    # seed_points = np.array([[xk,1e-10] for xk in np.linspace(0.45, 0.6, 8)])
    seed_points = np.array([[0.48, 1e-10]])
    ax_hm.streamplot(xi, yi, uCi.reshape((xcount, ycount)).T, vCi.reshape((xcount, ycount)).T, color="black", cmap=vccsm, linewidth=1, density=2, arrowstyle="->", arrowsize=1.5, start_points=seed_points, broken_streamlines=False)
    sol = gen_strmln(seed_points[0])
    # print(sol)
    ax_hm.plot(sol[0], sol[1], "r-")

if plot_streamlines:
    fig_strm, ax_strm = plt.subplots()
    colour = speed.reshape((xcount, ycount)).T
    ax_strm.streamplot(xi, yi, uCi.reshape((xcount, ycount)).T, vCi.reshape((xcount, ycount)).T, color=colour, cmap=vccsm, linewidth=1, density=2, arrowstyle="->", arrowsize=1.5)
    ax_strm.set_xlabel("x/c [-]")
    ax_strm.set_ylabel("y/c [-]")
    ax_strm.set(ylim=(ymin, ymax), xlim=(xmin, xmax))
    sm = cm.ScalarMappable(cmap=vccsm, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax_strm, label="Absolute velocity [1/U$_{inf}$]")

if plot_individual_streamline:
    seed_point = np.array([[0.55,1e-10]])
    fig_strm, ax_strm = plt.subplots()
    # colour = speed.reshape((xcount, ycount)).T
    colour = "red"
    ax_strm.streamplot(xi, yi, uCi.reshape((xcount, ycount)).T, vCi.reshape((xcount, ycount)).T, color="black", linewidth=0.5, density=1, arrowstyle="->", arrowsize=1.5)
    ax_strm.streamplot(xi, yi, uCi.reshape((xcount, ycount)).T, vCi.reshape((xcount, ycount)).T, color="red", cmap=vccsm, linewidth=1, density=2, arrowstyle="->", arrowsize=1.5, start_points=seed_point)
    ax_strm.set_xlabel("x/c [-]")
    ax_strm.set_ylabel("y/c [-]")
    ax_strm.set(ylim=(ymin, ymax), xlim=(xmin, xmax))
    sm = cm.ScalarMappable(cmap=vccsm, norm=norm)
    sm.set_array([])
    ax_strm.plot(seed_point.T[0], seed_point.T[1], "bo")
    plt.colorbar(sm, ax=ax_strm, label="Absolute velocity [1/U$_{inf}$]")

plt.show()