import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize, LinearSegmentedColormap, BoundaryNorm
from matplotlib.widgets import Slider
import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.integrate import solve_ivp, simpson

from load_data import piv_data

# Vector Field Colour Scheme
vccsm = cm.turbo  # coolwarm, turbo, jet
hmcm = cm.jet  # coolwarm, turbo, jet

plot_vector_field = False
plot_velocity_heatmap = True
plot_streamlines = False
plot_individual_streamline = False
plot_transition = False
include_slider = False
discrete_colormap = True
include_streamplot = True
error_bar = False
heatmap_velocity = "x"  # "x" or "absolute" or "absx"
intpl_method = "linear"  # "linear", "cubic", "quadratic"

xcount, ycount = 395, 57
ylim = 57
minl = (57 - ylim) * 395
step = 20  # Resolution
x, y = piv_data[:, 0], piv_data[:, 1]
xmin, xmax = x.min() / 1.02, x.max() * 1.02
ymin, ymax = -0.001, y.max() * 1.03
u, v = piv_data[:, 2], piv_data[:, 3]
X, Y = piv_data[minl::step, 0], piv_data[minl::step, 1]
U, V = piv_data[minl::step, 2], piv_data[minl::step, 3]
cutoff = 0.007
seed_x_init = 0.5649
sep, sep_err = 0.4851, 0.0147
reatt, reatt_err = 0.7390, 0.0091

colors = np.linalg.norm(np.column_stack((U, V)), axis=1)
norm = Normalize()
norm.autoscale(colors)

if plot_vector_field:
    fig_piv, ax_piv = plt.subplots()
    ax_piv.quiver(
        X,
        Y,
        U,
        V,
        color=vccsm(norm(colors)),
        pivot="mid",
        scale=100,
        scale_units="xy",
        width=0.001,
        headwidth=3,
        headlength=4,
        headaxislength=3,
    )
    # ax_piv.set_title("PIV data for AoA 2°, Re = 200 000")
    ax_piv.set_xlabel("x/c [-]")
    ax_piv.set_ylabel("y/c [-]")
    ax_piv.set(ylim=(ymin, ymax), xlim=(xmin, xmax))
    sm = cm.ScalarMappable(cmap=vccsm, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax_piv, label="Absolute velocity [1/U$_{inf}$]")

sorted_data = piv_data[np.lexsort((piv_data[:, 1], piv_data[:, 0]))]
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


def gen_strmln(seed_point, min_y=y.min()):
    sol = solve_ivp(
        f,
        [0, 10],
        seed_point,
        first_step=1e-3,
        max_step=1e-2,
        method="RK45",
        dense_output=True,
    )
    positions = sol.y
    while (
        positions[0, -1] > x.max()
        or positions[0, -1] < x.min()
        or positions[1, -1] > y.max()
        or positions[1, -1] < min_y
    ):
        positions = positions[:, :-1]

    intpl = interp1d(
        positions[0], positions[1], kind=intpl_method, fill_value="extrapolate"
    )
    return positions, intpl


def get_strml_pts(seed_point, xspace, min_y=y.min()):
    positions, intpl = gen_strmln(seed_point, min_y)
    extrapolated = np.array([[xk, intpl(xk)] for xk in xspace if intpl(xk) > y.min()])
    separation, reattachment = extrapolated[0, :], extrapolated[-1, :]
    return positions, extrapolated, separation, reattachment


def away_from_zero(a, precision=0):
    return np.copysign(
        np.true_divide(np.ceil(np.abs(a) * 10**precision), 10**precision), a
    )


if plot_velocity_heatmap:
    # absv = np.linalg.norm(np.column_stack((u, v)), axis=1)
    # absv = np.abs(u)
    factor = 10  # Resolution increase
    xj = np.linspace(X.min(), X.max(), xcount * factor)
    yj = np.linspace(Y.min(), Y.max(), ycount * factor)
    xy_grid = np.array([[xk, yk] for xk in xj for yk in yj])
    u_spd = u_intpl(xy_grid).reshape((xcount * factor, ycount * factor)).T
    label = "x velocity"
    if heatmap_velocity == "x":
        absv = u_spd
    elif heatmap_velocity == "absx":
        absv = np.abs(u_spd)
        label = "Absolute x velocity"
    elif heatmap_velocity == "absolute":
        v_spd = v_intpl(xy_grid).reshape((xcount * factor, ycount * factor)).T
        absv = np.sqrt(u_spd**2 + v_spd**2)
        label = "Absolute velocity"
    else:
        raise ValueError("Invalid heatmap_velocity value: " + heatmap_velocity)
    # absv = (absv > 0.1)
    norm = "linear"
    if discrete_colormap:
        cmaplist = [hmcm(i) for i in range(hmcm.N)]
        hmcm = LinearSegmentedColormap.from_list(
            "Custom Discrete map", cmaplist, hmcm.N
        )
        bounds = np.linspace(absv.min(), absv.max(), 21)
        minmax = away_from_zero([absv.min(), absv.max()], 1)
        bounds = np.linspace(
            minmax[0], minmax[1], ((minmax[1] - minmax[0]) * 10 + 1).astype(int)
        )
        norm = BoundaryNorm(bounds, hmcm.N)
    fig_hm, ax_hm = plt.subplots()
    heatmap = ax_hm.imshow(
        absv[::-1, :],
        extent=(piv_data[-1, 0], piv_data[0, 0], piv_data[-1, 1], piv_data[0, 1]),
        cmap=hmcm,
        norm=norm,
        interpolation="nearest",
        aspect="auto",
    )
    plt.colorbar(heatmap, label=f"{label} [1/U$_{{inf}}$]", ax=ax_hm)
    ax_hm.set_xlabel("x/c [-]")
    ax_hm.set_ylabel("y/c [-]")

    # ax_hm.contour(xj,  yj, absv.reshape(ycount*factor,xcount*factor), levels=[0.4, 0.8], colors="purple", linewidths=1)

    # seed_points = np.array([[xk,1e-10] for xk in np.linspace(0.45, 0.6, 8)])
    seed_points = np.array([[seed_x_init, cutoff]])
    sol, extrapolated, separation, reattachment = get_strml_pts(
        seed_points[0], xj, min_y=cutoff
    )
    ax_hm.plot((x.min(), x.max()), (cutoff, cutoff), "k--", linewidth=0.5)
    if include_streamplot:
        ax_hm.streamplot(
            xi,
            yi,
            uCi.reshape((xcount, ycount)).T,
            vCi.reshape((xcount, ycount)).T,
            color="gray",
            linewidth=0.5,
            density=1,
            arrowstyle="->",
            arrowsize=1,
            broken_streamlines=False,
        )
    ax_hm.streamplot(
        xi,
        yi,
        uCi.reshape((xcount, ycount)).T,
        vCi.reshape((xcount, ycount)).T,
        color="black",
        cmap=vccsm,
        linewidth=1,
        density=2,
        arrowstyle="->",
        arrowsize=1.5,
        start_points=seed_points,
        broken_streamlines=False,
    )
    # sol, intpl = gen_strmln(seed_points[0], min_y=cutoff_turbulent)
    # print(sol)
    # print(f"Separation point: {separation[0]}")
    # print(f"Reattachment point: {reattachment[0]}")
    # ax_hm.plot(np.unique(X), [np.unique(Y)[i] for i in np.argmax(U_grid, axis=1)], "k-") # delta_max
    if error_bar:
        ax_hm.errorbar(
            sep,
            0.0003,
            xerr=sep_err,
            fmt="|",
            color="white",
            label="Separation point",
            ms=7,
            capsize=3,
        )
        ax_hm.errorbar(
            reatt,
            0.0003,
            xerr=reatt_err,
            fmt="|",
            color="white",
            label="Reattachment point",
            ms=7,
            capsize=3,
        )
    ax_hm.plot(extrapolated[:, 0], extrapolated[:, 1], "r--")
    ax_hm.plot(sol[0], sol[1], "r-")
    ax_hm.plot(seed_points[:, 0], seed_points[:, 1], "bo", markersize=3)
    ax_hm.plot((separation[0], reattachment[0]), (separation[1], reattachment[1]), "ro")

    if include_slider:
        fig_hm.subplots_adjust(left=0.25, bottom=0.25)
        ax_seed = fig_hm.add_axes([0.25, 0.1, 0.65, 0.03])
        ax_divider = fig_hm.add_axes([0.1, 0.25, 0.0225, 0.63])
        seed_slider = Slider(
            ax=ax_seed,
            label="Seed Point (x/c)",
            valmin=0.54,
            valmax=0.58,
            valinit=seed_points[0, 0],
            orientation="horizontal",
        )
        divider_slider = Slider(
            ax=ax_divider,
            label="Cutoff Line (y/c)",
            valmin=0.0065,
            valmax=0.008,
            valinit=cutoff,
            orientation="vertical",
        )
        septxt = ax_seed.text(
            0, -1, f"Separation: {separation[0]}", transform=ax_seed.transAxes
        )
        attachtxt = ax_seed.text(
            0, -2, f"Reattachment: {reattachment[0]}", transform=ax_seed.transAxes
        )

        def new_seed(strml_points, new_y):
            idxmax = np.argmax(strml_points[:, 1])
            intpl = interp1d(
                strml_points[: idxmax + 1, 1],
                strml_points[: idxmax + 1, 0],
                kind=intpl_method,
                fill_value="extrapolate",
            )
            return intpl(new_y)

        def update(val):
            global cutoff, extrapolated
            extent = ax_hm.axis()
            ax_hm.cla()
            ax_hm.imshow(
                absv[::-1, :],
                extent=(
                    piv_data[-1, 0],
                    piv_data[0, 0],
                    piv_data[-1, 1],
                    piv_data[0, 1],
                ),
                cmap=hmcm,
                norm=norm,
                interpolation="nearest",
                aspect="auto",
            )
            ax_hm.set_xlabel("x/c [-]")
            ax_hm.set_ylabel("y/c [-]")
            seed_x = seed_slider.val

            if cutoff != divider_slider.val:
                seed_x = new_seed(extrapolated, divider_slider.val)
                pts_min = get_strml_pts([seed_slider.valmin, cutoff], xj, min_y=cutoff)[
                    1
                ]
                new_min = new_seed(pts_min, divider_slider.val)
                pts_max = get_strml_pts([seed_slider.valmax, cutoff], xj, min_y=cutoff)[
                    1
                ]
                new_max = new_seed(pts_max, divider_slider.val)
                seed_slider.eventson = False
                seed_slider.set_val(seed_x)
                seed_slider.valmin = new_min
                seed_slider.valmax = new_max
                seed_slider.ax.set_xlim(seed_slider.valmin, seed_slider.valmax)
                seed_slider.eventson = True
                cutoff = divider_slider.val

            seed_points = np.array([[seed_x, cutoff]])
            sol, extrapolated, separation, reattachment = get_strml_pts(
                seed_points[0], xj, min_y=cutoff
            )
            ax_hm.plot((x.min(), x.max()), (cutoff, cutoff), "k--", linewidth=0.5)
            ax_hm.plot((x.min(), x.max()), (cutoff, cutoff), "k--", linewidth=0.5)
            if include_streamplot:
                ax_hm.streamplot(
                    xi,
                    yi,
                    uCi.reshape((xcount, ycount)).T,
                    vCi.reshape((xcount, ycount)).T,
                    color="gray",
                    linewidth=0.5,
                    density=1,
                    arrowstyle="->",
                    arrowsize=1,
                    broken_streamlines=False,
                )
            ax_hm.streamplot(
                xi,
                yi,
                uCi.reshape((xcount, ycount)).T,
                vCi.reshape((xcount, ycount)).T,
                color="black",
                cmap=vccsm,
                linewidth=1,
                density=2,
                arrowstyle="->",
                arrowsize=1.5,
                start_points=seed_points,
                broken_streamlines=False,
            )
            if error_bar:
                ax_hm.errorbar(
                    sep,
                    0.0003,
                    xerr=sep_err,
                    fmt="|",
                    color="white",
                    label="Separation point",
                    ms=7,
                    capsize=3,
                )
                ax_hm.errorbar(
                    reatt,
                    0.0003,
                    xerr=reatt_err,
                    fmt="|",
                    color="white",
                    label="Reattachment point",
                    ms=7,
                    capsize=3,
                )
            ax_hm.plot(extrapolated[:, 0], extrapolated[:, 1], "r--")
            ax_hm.plot(sol[0], sol[1], "r-")
            ax_hm.plot(seed_points[:, 0], seed_points[:, 1], "bo", markersize=3)
            ax_hm.plot(
                (separation[0], reattachment[0]), (separation[1], reattachment[1]), "ro"
            )
            septxt.set_text(f"Separation: {separation[0]}")
            attachtxt.set_text(f"Reattachment: {reattachment[0]}")
            ax_hm.axis(extent)

        seed_slider.on_changed(update)
        divider_slider.on_changed(update)


if plot_streamlines:
    fig_strm, ax_strm = plt.subplots()
    colour = speed.reshape((xcount, ycount)).T
    ax_strm.streamplot(
        xi,
        yi,
        uCi.reshape((xcount, ycount)).T,
        vCi.reshape((xcount, ycount)).T,
        color=colour,
        cmap=vccsm,
        linewidth=1,
        density=2,
        arrowstyle="->",
        arrowsize=1.5,
    )
    ax_strm.set_xlabel("x/c [-]")
    ax_strm.set_ylabel("y/c [-]")
    ax_strm.set(ylim=(ymin, ymax), xlim=(xmin, xmax))
    sm = cm.ScalarMappable(cmap=vccsm, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax_strm, label="Absolute velocity [1/U$_{inf}$]")

if plot_individual_streamline:
    seed_point = np.array([[0.55, 1e-10]])
    fig_strm, ax_strm = plt.subplots()
    # colour = speed.reshape((xcount, ycount)).T
    colour = "red"
    ax_strm.streamplot(
        xi,
        yi,
        uCi.reshape((xcount, ycount)).T,
        vCi.reshape((xcount, ycount)).T,
        color="black",
        linewidth=0.5,
        density=1,
        arrowstyle="->",
        arrowsize=1.5,
    )
    ax_strm.streamplot(
        xi,
        yi,
        uCi.reshape((xcount, ycount)).T,
        vCi.reshape((xcount, ycount)).T,
        color="red",
        cmap=vccsm,
        linewidth=1,
        density=2,
        arrowstyle="->",
        arrowsize=1.5,
        start_points=seed_point,
    )
    ax_strm.set_xlabel("x/c [-]")
    ax_strm.set_ylabel("y/c [-]")
    ax_strm.set(ylim=(ymin, ymax), xlim=(xmin, xmax))
    sm = cm.ScalarMappable(cmap=vccsm, norm=norm)
    sm.set_array([])
    ax_strm.plot(seed_point.T[0], seed_point.T[1], "bo")
    plt.colorbar(sm, ax=ax_strm, label="Absolute velocity [1/U$_{inf}$]")

if plot_transition:
    # print(xy_points.reshape((xcount, ycount, 2)))
    delta_max = np.array(
        [[np.unique(Y)[i], i] for i in np.argmax(U_grid, axis=1)]
    )  # point of maximum velocity along span
    delta_star = []
    delta_99 = []
    delta_95 = []
    theta_star = []
    for j, (dmax, i) in enumerate(delta_max):
        i = int(i)
        umax = U_grid[j, i]
        d99, d95 = 0, 0
        for k, uj in enumerate(U_grid[j, : i + 1]):
            if uj > 0.95 * umax and d95 == 0:
                d95 = np.unique(Y)[k]
            if uj > 0.99 * umax:
                d99 = np.unique(Y)[k]
                break
        dstar = simpson(
            [(1 - (uj / umax)) for uj in U_grid[j, : i + 1]], np.unique(Y)[: i + 1]
        )
        thstar = simpson(
            [(uj / umax) * (1 - (uj / umax)) for uj in U_grid[j, : i + 1]],
            np.unique(Y)[: i + 1],
        )
        # print(dstar, thstar)
        delta_95.append(d95)
        delta_99.append(d99)
        delta_star.append(dstar)
        theta_star.append(thstar)

    delta_star = np.array(delta_star)
    theta_star = np.array(theta_star)
    fig_trnst, ax_trnst = plt.subplots()
    plt1 = ax_trnst.plot(np.unique(X), delta_max[:, 0], "k-", label="$\delta_{max}$")
    plt2 = ax_trnst.plot(np.unique(X), delta_star, "r-", label="$\delta{*}$")
    plt3 = ax_trnst.plot(np.unique(X), theta_star, "b-", label="$\\theta{*}$")
    plt4 = ax_trnst.plot(np.unique(X), delta_99, "g-", label="$\delta_{99}$")
    plt5 = ax_trnst.plot(np.unique(X), delta_95, "y-", label="$\delta_{95}$")
    ax_trnst.set_xlabel("x/c [-]")
    ax_trnst.set_ylabel("y/c [-]")
    # ax_trnst.legend()
    H12 = delta_star / theta_star
    transition = np.unique(X)[np.argmax(H12)]
    print("Transition point: ", transition)
    ax_trnst_2 = ax_trnst.twinx()
    plt6 = ax_trnst_2.plot(np.unique(X), H12, "m-", label="H$_{12}$")
    ax_trnst_2.set_ylabel("H$_{12}$ [-]")

    lns = plt1 + plt2 + plt3 + plt4 + plt5 + plt6
    labs = [l.get_label() for l in lns]
    ax_trnst.legend(lns, labs, loc=0)

plt.show()
