import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
from matplotlib.colors import Normalize, LinearSegmentedColormap, BoundaryNorm
from matplotlib.widgets import Slider
import numpy as np
import argparse
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.integrate import solve_ivp, simpson
from scipy.optimize import curve_fit

from load_data import piv_data

# Vector Field Colour Scheme
vccsm = cm.jet  # coolwarm, turbo, jet
hmcm = cm.jet  # coolwarm, turbo, jet

mpl.rcParams["axes.labelsize"] = 14

thicknesses_in_heatmap = {
    "δ_max": [True, "c-"],
    "δ*": [True, "m-"],
    "θ*": [True, "w-"],
    "δ_99": [True, "g-"],
    "δ_95": [False, "y-"],
}

xcount, ycount = 395, 57
ylim = 57
minl = (57 - ylim) * 395
step = 20  # Resolution
x, y = piv_data[:, 0], piv_data[:, 1]
# print((x.max() - x.min()) / (xcount - 1))
print(x.min(), x.max(), y.min(), y.max())
xmin, xmax = x.min() / 1.02, x.max() * 1.02
ymin, ymax = -0.001, y.max() * 1.03
u, v = piv_data[:, 2], piv_data[:, 3]
X_step, Y_step = piv_data[minl::step, 0], piv_data[minl::step, 1]
U_step, V_step = piv_data[minl::step, 2], piv_data[minl::step, 3]
cutoff = 0.007
seed_x_init = 0.5515
sep, sep_err = 0.4851, 0.0147
reatt, reatt_err = 0.7390, 0.0091

colors = np.linalg.norm(np.column_stack((U_step, V_step)), axis=1)
norm = Normalize()
norm.autoscale(colors)


def plot_vector_field():
    norm_ = Normalize()
    norm_.autoscale([0, 1.4])
    fig_piv, ax_piv = plt.subplots()
    ax_piv.quiver(
        X_step,
        Y_step,
        U_step,
        V_step,
        color=vccsm(norm_(colors)),
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
    sm = cm.ScalarMappable(cmap=vccsm, norm=norm_)
    sm.set_array([])
    plt.colorbar(
        sm,
        ax=ax_piv,
        label="Absolute velocity [1/U$_{inf}$]",
        ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4],
    )


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


def gen_strmln(seed_point, min_y=y.min(), intpl_method="linear"):
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


def get_strml_pts(seed_point, xspace, min_y=y.min(), intpl_method="linear"):
    positions, intpl = gen_strmln(seed_point, min_y, intpl_method=intpl_method)
    extrapolated = np.array([[xk, intpl(xk)] for xk in xspace if intpl(xk) > y.min()])
    separation, reattachment = extrapolated[0, :], extrapolated[-1, :]
    transition = positions[:, np.argmax(positions[1, :])]
    return positions, extrapolated, separation, reattachment, transition


def away_from_zero(a, precision=0):
    return np.copysign(
        np.true_divide(np.ceil(np.abs(a) * 10**precision), 10**precision), a
    )


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
        if uj > 0.95 * 1 and d95 == 0:
            d95 = np.unique(Y)[k]
        if uj > 0.99 * 1:
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

delta_star, theta_star, delta_99, delta_95 = (
    np.array(delta_star),
    np.array(theta_star),
    np.array(delta_99),
    np.array(delta_95),
)

H12 = delta_star / theta_star
transition_h12 = np.unique(X)[np.argmax(H12)]
transition_delta_star = np.unique(X)[np.argmax(delta_star)]
transition = transition_h12

thicknesses_in_heatmap["δ_max"].append(delta_max[:, 0])
thicknesses_in_heatmap["δ_99"].append(delta_99)
thicknesses_in_heatmap["δ_95"].append(delta_95)
thicknesses_in_heatmap["δ*"].append(delta_star)
thicknesses_in_heatmap["θ*"].append(theta_star)


def plot_velocity_heatmap(
    heatmap_velocity="x",
    include_slider=False,
    discrete_colormap=False,
    include_streamplot=False,
    error_bar=False,
    intpl_method="linear",
):
    global hmcm
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
    sol, extrapolated, separation, reattachment, transition = get_strml_pts(
        seed_points[0], xj, min_y=cutoff, intpl_method=intpl_method
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
    for label, (plot, style, data) in thicknesses_in_heatmap.items():
        # print(label, plot, style, data.shape)
        if plot:
            ax_hm.plot(np.unique(X), data, style, label=label)

    ax_hm.plot(extrapolated[:, 0], extrapolated[:, 1], "r--")
    ax_hm.plot(sol[0], sol[1], "r-")
    ax_hm.plot(seed_points[:, 0], seed_points[:, 1], "bo", markersize=3)
    ax_hm.plot(
        (separation[0], reattachment[0], transition[0]),
        (separation[1], reattachment[1], transition[1]),
        "ro",
    )

    if any([x[0] for x in thicknesses_in_heatmap.values()]):
        ax_hm.legend(loc="upper right")

    if not include_slider:
        print(f"Separation point   : {np.round(separation[0], 3)} ({separation[0]})")
        print(f"Transition point   : {np.round(transition[0], 3)} ({transition[0]})")
        print(
            f"Reattachment point : {np.round(reattachment[0], 3)} ({reattachment[0]})"
        )

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
        text = [
            ["Separation point: ", f"{np.round(separation[0], 3)} ({separation[0]})"],
            ["Transition point: ", f"{np.round(transition[0], 3)} ({transition[0]})"],
            [
                "Reattachment point: ",
                f"{np.round(reattachment[0], 3)} ({reattachment[0]})",
            ],
        ]
        txt_table = ax_seed.table(
            text,
            cellLoc="left",
            colWidths=[0.5, 0.5],
            transform=ax_seed.transAxes,
            bbox=[0, -3, 1, 3],
            edges="",
        )

        # septxt = ax_seed.text(
        #     0,
        #     -1,
        #     f"Separation point   : {np.round(separation[0], 3)} ({separation[0]})",
        #     transform=ax_seed.transAxes,
        # )
        # trnsttxt = ax_seed.text(
        #     0,
        #     -2,
        #     f"Transition point   : {np.round(transition, 3)} ({transition})",
        #     transform=ax_seed.transAxes,
        # )
        # attachtxt = ax_seed.text(
        #     0,
        #     -3,
        #     f"Reattachment point : {np.round(reattachment[0], 3)} ({reattachment[0]})",
        #     transform=ax_seed.transAxes,
        # )

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
                minpt = get_strml_pts(
                    [seed_slider.valmin, cutoff],
                    xj,
                    min_y=cutoff,
                    intpl_method=intpl_method,
                )[1]
                new_min = new_seed(minpt, divider_slider.val)
                maxpt = get_strml_pts(
                    [seed_slider.valmax, cutoff],
                    xj,
                    min_y=cutoff,
                    intpl_method=intpl_method,
                )[1]
                new_max = new_seed(maxpt, divider_slider.val)
                seed_slider.eventson = False
                seed_slider.set_val(seed_x)
                seed_slider.valmin = new_min
                seed_slider.valmax = new_max
                seed_slider.ax.set_xlim(seed_slider.valmin, seed_slider.valmax)
                seed_slider.eventson = True
                cutoff = divider_slider.val

            seed_points = np.array([[seed_x, cutoff]])
            sol, extrapolated, separation, reattachment, transition = get_strml_pts(
                seed_points[0], xj, min_y=cutoff, intpl_method=intpl_method
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
            for label, (plot, style, data) in thicknesses_in_heatmap.items():
                if plot:
                    ax_hm.plot(np.unique(X), data, style, label=label)

            ax_hm.plot(extrapolated[:, 0], extrapolated[:, 1], "r--")
            ax_hm.plot(sol[0], sol[1], "r-")
            ax_hm.plot(seed_points[:, 0], seed_points[:, 1], "bo", markersize=3)
            ax_hm.plot(
                (separation[0], reattachment[0], transition[0]),
                (separation[1], reattachment[1], transition[1]),
                "ro",
            )
            txt_table.get_celld()[(0, 1)].get_text().set_text(
                f"{np.round(separation[0], 3)} ({separation[0]})"
            )
            txt_table.get_celld()[(1, 1)].get_text().set_text(
                f"{np.round(transition[0], 3)} ({transition[0]})"
            )
            txt_table.get_celld()[(2, 1)].get_text().set_text(
                f"{np.round(reattachment[0], 3)} ({reattachment[0]})"
            )
            if any([x[0] for x in thicknesses_in_heatmap.values()]):
                ax_hm.legend(loc="upper right")
            ax_hm.axis(extent)

        seed_slider.on_changed(update)
        divider_slider.on_changed(update)


def plot_streamlines():
    fig_strm, ax_strm = plt.subplots()
    colour = speed.reshape((xcount, ycount)).T
    norm_ = Normalize()
    norm_.autoscale([0, 1.4])
    ax_strm.streamplot(
        xi,
        yi,
        uCi.reshape((xcount, ycount)).T,
        vCi.reshape((xcount, ycount)).T,
        color=colour,
        cmap=vccsm,
        norm=norm_,
        linewidth=1,
        density=(0.99, 0.5),
        arrowstyle="->",
        arrowsize=1.5,
        broken_streamlines=False,
    )
    ax_strm.set_xlabel("x/c [-]")
    ax_strm.set_ylabel("y/c [-]")
    ax_strm.set(ylim=(ymin, ymax), xlim=(xmin, xmax))
    sm = cm.ScalarMappable(cmap=vccsm, norm=norm_)
    sm.set_array([])
    plt.colorbar(sm, ax=ax_strm, label="Absolute velocity [1/U$_{inf}$]")


def plot_individual_streamline():
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


tau = 1


def make_fourier(na, nb):
    def fourier(x, *a):
        ret = 0.0
        for deg in range(0, na):
            ret += a[deg] * np.cos((deg + 1) * np.pi / tau * x)
        for deg in range(na, na + nb):
            ret += a[deg] * np.sin((deg + 1) * np.pi / tau * x)
        return ret

    return fourier


def plot_transition_graph(regression_order=0):
    fig_trnst, ax_trnst = plt.subplots()

    plt1 = ax_trnst.plot(np.unique(X), delta_max[:, 0], "k-", label="$\delta_{max}$")
    plt2 = ax_trnst.plot(np.unique(X), delta_star, "r-", label="$\delta{*}$")
    plt3 = ax_trnst.plot(np.unique(X), theta_star, "b-", label="$\\theta{*}$")
    plt4 = ax_trnst.plot(np.unique(X), delta_99, "g-", label="$\delta_{99}$")
    # plt5 = ax_trnst.plot(np.unique(X), delta_95, "y-", label="$\delta_{95}$")
    ax_trnst.set_xlabel("x/c [-]")
    ax_trnst.set_ylabel("y/c [-]")
    ax_trnst.set(ylim=(0, ymax), xlim=(xmin, xmax))
    ax_trnst_2 = ax_trnst.twinx()

    H_laminar, H_turbulent = 2.59, 1.4
    ax_trnst_2.plot(
        (np.min(X), np.max(X)),
        (H_laminar, H_laminar),
        color="gray",
        linestyle="--",
        linewidth=0.5,
        label="Theoretical Laminar $H_{12}$",
    )

    ax_trnst_2.plot(
        (np.min(X), np.max(X)),
        (H_turbulent, H_turbulent),
        color="gray",
        linestyle="--",
        linewidth=0.5,
        label="$Theoretical Turbulent H_{12}$",
    )

    ax_trnst_2.text(
        0.72,
        H_laminar + 0.07,
        "Theoretical Laminar $H_{12}$",
        ha="left",
        color="gray",
    )
    ax_trnst_2.text(
        0.72,
        H_turbulent - 0.05,
        "Theoretical Turbulent $H_{12}$",
        ha="left",
        va="top",
        color="gray",
    )

    if regression_order:
        min_H = np.max(H12) * 0.95  # Only relevant above this value
        idxs = np.where(H12 > min_H)[0]
        min_x, max_x = np.unique(X)[idxs[0]], np.unique(X)[idxs[-1]]
        x_vals = np.linspace(min_x, max_x, 1000)
        coeff = np.polyfit(
            np.unique(X)[idxs[0] : idxs[-1] + 1],
            H12[idxs[0] : idxs[-1] + 1],
            regression_order,
        )
        poly = np.poly1d(coeff)
        shape_factor = poly(x_vals)
        # print(poly)
        plt6 = ax_trnst_2.plot(np.unique(X), H12, "m--", label="H$_{12}$ (exact)")
        plt7 = ax_trnst_2.plot(x_vals, shape_factor, "m-", label="H$_{12}$ (approx.)")
        ax_trnst_2.axhline(min_H, color="k", linestyle="--", linewidth=0.4)
        H_plt = plt6 + plt7
        max_val = np.max(list(shape_factor) + list(H12))
        transition_H12 = x_vals[np.argmax(shape_factor)]
    else:
        H_plt = ax_trnst_2.plot(np.unique(X), H12, "m-", label="H$_{12}$")
        max_val = np.max(H12)
        transition_H12 = np.unique(X)[np.argmax(H12)]

    ax_trnst_2.set_ylabel("H$_{12}$ [-]")
    ax_trnst_2.set(ylim=(ymin, max_val * 1.03))

    print("Transition point (Shape Factor): ", transition_H12)
    print("Transition point (δ*): ", transition_delta_star)

    # lns = plt1 + plt2 + plt3 + plt4 + plt5 + H_plt
    lns = plt1 + plt2 + plt3 + plt4 + H_plt
    labs = [l.get_label() for l in lns]
    ax_trnst.legend(lns, labs, loc=0)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


class CommandLine:
    def __init__(self):
        parser = argparse.ArgumentParser(description="Graphs for PIV Data Analysis")
        parser.add_argument(
            "--plot-heatmap",
            help="Whether to plot the velocity heatmap. Plotted by default, unless other graphs are plotted",
            required=False,
            default=False,
            const=True,
            nargs="?",
            type=str2bool,
        )

        parser.add_argument(
            "--plot-vector-field",
            help="Whether to plot the vector field (Default: False)",
            required=False,
            default=False,
            const=True,
            nargs="?",
            type=str2bool,
        )

        parser.add_argument(
            "--plot-streamlines",
            help="Whether to plot the streamlines (Default: False)",
            required=False,
            default=False,
            const=True,
            nargs="?",
            type=str2bool,
        )

        parser.add_argument(
            "--plot-individual-streamline",
            help="Whether to plot the individual seeded streamline (Default: False)",
            required=False,
            default=False,
            const=True,
            nargs="?",
            type=str2bool,
        )

        parser.add_argument(
            "--plot-transition",
            help="Whether to plot the transition graph (Default: False)",
            required=False,
            default=False,
            const=True,
            nargs="?",
            type=str2bool,
        )

        parser.add_argument(
            "--smooth-shape-factor",
            help="Perform polynomial regression on shape factor curve with order ORDER (Default: False)",
            required=False,
            default=0,
            const=2,
            nargs="?",
            type=int,
        )

        parser.add_argument(
            "--heatmap-velocity",
            help="Which velocity to use on the heatmap (x, absx or absolute). Default: x",
            required=False,
            default="x",
        )
        parser.add_argument(
            "--include-slider",
            help="Whether to include the slider in the heatmap (Default: False)",
            required=False,
            default=False,
            const=True,
            nargs="?",
            type=str2bool,
        )
        parser.add_argument(
            "--discrete-colormap",
            help="Whether to use a discrete colormap (Default: True)",
            required=False,
            default=True,
            const=True,
            nargs="?",
            type=str2bool,
        )

        parser.add_argument(
            "--include-streamplot",
            help="Whether to include a streamline plot in the heatmap (Default: True)",
            required=False,
            default=True,
            const=True,
            nargs="?",
            type=str2bool,
        )

        parser.add_argument(
            "--intpl-method",
            help="Which interpolation method to use for the streamlines (linear, quadratic, cubic). Default: linear",
            required=False,
            default="linear",
        )

        argument = parser.parse_args()
        status = False
        # print(argument)

        if argument.plot_vector_field:
            plot_vector_field()
            status = True

        if argument.plot_streamlines:
            plot_streamlines()
            status = True

        if argument.plot_individual_streamline:
            plot_individual_streamline()
            status = True

        if argument.plot_transition:
            plot_transition_graph(regression_order=argument.smooth_shape_factor)
            status = True

        if argument.plot_heatmap or not status:
            if not status:
                print("No graphs specified, plotting heatmap by default")

            plot_velocity_heatmap(
                heatmap_velocity=argument.heatmap_velocity,
                include_slider=argument.include_slider,
                discrete_colormap=argument.discrete_colormap,
                include_streamplot=argument.include_streamplot,
                intpl_method=argument.intpl_method,
            )


if __name__ == "__main__":
    app = CommandLine()
    include_slider = True
    discrete_colormap = True
    include_streamplot = True
    error_bar = False
    plt.show()
