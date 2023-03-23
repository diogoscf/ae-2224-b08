# ================== PIV data processing =================
# Date: 16 March 2023
# Author: Julia Bertsch
# Description: Process the PIV data from the tutor

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize, LinearSegmentedColormap, BoundaryNorm
from matplotlib.widgets import Slider
import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.integrate import solve_ivp
import os

# Import the tutor's data
piv_file_path = os.path.join(os.path.dirname(__file__), "./PIV.dat")
piv_data = np.genfromtxt(piv_file_path, skip_header=1, delimiter=",")

# Data settings
xcount, ycount = 395, 57
ylim = 57

# Package related settings
vccsm = cm.turbo
hmcm = cm.jet

# Change-able variables
laminar_height = 0.0005     # Initial estimated (laminar) boundary thickness
step = 20                   # 
seed = np.array([[0.45, laminar_height],[0.48, laminar_height],[0.50, laminar_height]]) # Array of seeds (base)
heatmap_resolution = 15     # Resolution of the heatmap gradient plot
xc_cutoff = 0.60            # point at which to cutoff the streamline analysis
xc_max = 0.52               # maximum xc for valid streamline (eye-balling)
cutoff_laminar = 0.007      # same as laminar height, but in diego's work (also for turbulent below) 
cutoff_turbulent = 0.007    # <<

# ------------------ pre-processing the imported data ------------------
minl = (57 - ylim)*395
x, y = piv_data[:, 0], piv_data[:, 1]
xmin, xmax = x.min()/1.02, x.max()*1.02
ymin, ymax = -0.001, y.max()*1.03
u, v = piv_data[:, 2], piv_data[:, 3]
X, Y = piv_data[minl::step, 0], piv_data[minl::step, 1]
U, V = piv_data[minl::step, 2], piv_data[minl::step, 3]

colors = np.linalg.norm(np.column_stack((U, V)), axis=1)
norm = Normalize()
norm.autoscale(colors)

# Sort the position and velocity data
sorted_data = piv_data[np.lexsort((piv_data[:,1], piv_data[:,0]))]
X, Y = sorted_data[:, 0], sorted_data[:, 1]
U, V = sorted_data[:, 2], sorted_data[:, 3]

# Create a regularly spaced position grid spanning the domain of x and y 
xi = np.linspace(X.min(), X.max(), xcount)
yi = np.linspace(Y.min(), Y.max(), ycount)
xy_points = np.array([[x, y] for x in xi for y in yi])

# Create a regularly spaced velocity grid spanning the domain of x and y 
U_grid = U.reshape((xcount, ycount))
V_grid = V.reshape((xcount, ycount))

# Bicubic interpolation
u_intpl = RegularGridInterpolator((np.unique(X), np.unique(Y)), U_grid)
v_intpl = RegularGridInterpolator((np.unique(X), np.unique(Y)), V_grid)
uCi, vCi = u_intpl(xy_points), v_intpl(xy_points)
speed = np.sqrt(uCi**2 + vCi**2)

norm = Normalize()
norm.autoscale(speed)

# Function : Plot vector field given X and Y and the corresponding velocities U and V
def PlotVectorField():
    fig_piv, ax_piv = plt.subplots()
    ax_piv.quiver(X, Y, U, V, color=vccsm(norm(colors)), pivot="mid", scale=100, scale_units="xy", width=0.001, headwidth=3, headlength=4, headaxislength=3)
    ax_piv.set_xlabel("x/c [-]")
    ax_piv.set_ylabel("y/c [-]")
    ax_piv.set(ylim=(ymin, ymax), xlim=(xmin, xmax))
    sm = cm.ScalarMappable(cmap=vccsm, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax_piv, label="Absolute velocity [1/U$_{inf}$]")
    plt.show()

# Function : Streamline function
def f(t, xy):
    if xy[1] < y.min() or xy[1] > y.max() or xy[0] < x.min() or xy[0] > x.max():
        return np.array([0, 0])
    return np.squeeze([u_intpl(xy), v_intpl(xy)])

# Function : Find the streamline given a starting(/seed) point
def GenerateStreamline(seed_point, min_y=y.min()):
    solution = solve_ivp(f, [0, 10], seed_point, first_step=1e-3, max_step=1e-2, method="RK45", dense_output=True)
    positions = solution.y
    while (positions[0, -1] > x.max() or positions[0, -1] < x.min() or positions[1, -1] > y.max() or positions[1, -1] < min_y):
        positions = positions[:,:-1]
    intpl = interp1d(positions[0], positions[1], kind="linear", fill_value="extrapolate")
    return positions, intpl

# Function : From streamline given in GenerateStreamline, obtain extrapolated, separation and reattachment points
def GenerateStreamlinePoints(seed_point, xspace, min_y=y.min()):
    positions, intpl = GenerateStreamline(seed_point, min_y)
    extrapolated = np.array([[xk, intpl(xk)] for xk in xspace if intpl(xk) > y.min()])
    separation, reattachment = extrapolated[0,:], extrapolated[-1,:]
    return positions, extrapolated, separation, reattachment
    
# Function : Plot the heatmap of the velocity, including the streamlines in the seed-list
def PlotVelocityHeatmap(seed, plot_bl=False, plot_contour=False, plot_stream = False, manual_optimization=False, contour_levels=[0.6]):
    global hmcm
    norm = "linear"

    # Create a grid for the heatmap (sizing it up on resolution)
    xh = np.linspace(X.min(), X.max(), xcount*heatmap_resolution)
    yh = np.linspace(Y.min(), Y.max(), ycount*heatmap_resolution)
    xy_grid = np.array([[x, y] for x in xh for y in yh])
    # Use the RegularGridInterpolator to find the nicely interpolated grid
    absv = u_intpl(xy_grid).reshape((xcount*heatmap_resolution,ycount*heatmap_resolution)).T

    # Create a discerete colormap
    discrete_colormap = False    
    if discrete_colormap:
        cmaplist = [hmcm(i) for i in range(hmcm.N)]
        hmcm = LinearSegmentedColormap.from_list("Custom Discrete map", cmaplist, hmcm.N)
        bounds = np.linspace(absv.min(), absv.max(), 21)
        norm = BoundaryNorm(bounds, hmcm.N)

    # Create the subplots for the heatmap and colorbar
    fig_heatmap, ax_heatmap = plt.subplots()
    heatmap = ax_heatmap.imshow(absv[::-1,:], extent=(piv_data[-1,0], piv_data[0,0], piv_data[-1,1], piv_data[0,1]), cmap=cm.turbo, interpolation="nearest", aspect="auto")
    plt.colorbar(heatmap, label="Absolute velocity [1/U$_{inf}$]", ax=ax_heatmap)
    ax_heatmap.set_xlabel("x/c [-]")
    ax_heatmap.set_ylabel("y/c [-]")

    # Plot the laminar boundary layer thickness line
    if plot_bl: ax_heatmap.plot((x.min(), x.max()), (laminar_height, laminar_height), "black", linewidth=0.5)

    # Plot the streamline contour
    if plot_contour: 
        ax_heatmap.contour(xh, yh, absv.reshape(ycount*heatmap_resolution,xcount*heatmap_resolution), levels=contour_levels, colors="black", linewidths=1)
        
    # Plot all the streamlines from the seed array (gray) and plot the seed streamlines in black
    if plot_stream: 
        ax_heatmap.streamplot(xi, yi, uCi.reshape((xcount, ycount)).T, vCi.reshape((xcount, ycount)).T, color="gray", cmap=vccsm, linewidth=0.2, density=1, arrowstyle="-", arrowsize=1.5, broken_streamlines=False)
        ax_heatmap.streamplot(xi, yi, uCi.reshape((xcount, ycount)).T, vCi.reshape((xcount, ycount)).T, color="black", cmap=vccsm, linewidth=.5, density=2, arrowstyle="->", arrowsize=1.5, start_points=seed, broken_streamlines=False)

    # Provide option to manually (live) change the streamline by adding a slider
    if manual_optimization:
        seed_points = np.array([[0.48, cutoff_laminar]])
        xj = np.linspace(X.min(), X.max(), xcount*heatmap_resolution)
        yj = np.linspace(Y.min(), Y.max(), ycount*heatmap_resolution)
        
        # Get solution and extrapolated separation and reattachment points
        sol, extrapolated, separation, reattachment = GenerateStreamlinePoints(seed_points[0], xj, min_y=cutoff_laminar)
        ax_heatmap.plot((x.min(), x.max()), (cutoff_laminar, cutoff_laminar), "k--", linewidth=0.5)
        ax_heatmap.plot((x.min(), x.max()), (cutoff_turbulent, cutoff_turbulent), "k--", linewidth=0.5)
        ax_heatmap.streamplot(xi, yi, uCi.reshape((xcount, ycount)).T, vCi.reshape((xcount, ycount)).T, color="black", cmap=vccsm, linewidth=1, density=2, arrowstyle="->", arrowsize=1.5, start_points=seed_points, broken_streamlines=False)
        
        # Print Calculated separation and reattachment points, and plot them on the heatmap
        print(f"Separation point: {separation[0]}")
        print(f"Reattachment point: {reattachment[0]}")
        ax_heatmap.plot(extrapolated[:,0], extrapolated[:,1], "r--")
        ax_heatmap.plot(sol[0], sol[1], "r-")
        ax_heatmap.plot(seed_points[:,0], seed_points[:,1], "bo", markersize=3)
        ax_heatmap.plot((separation[0], reattachment[0]), (separation[1], reattachment[1]), "ro")
        
        # Add Manually adjustable slider
        include_slider = True
        if include_slider:
            fig_heatmap.subplots_adjust(left=0.25, bottom=0.25)
            ax_slider = fig_heatmap.add_axes([0.25, 0.1, 0.65, 0.03])
            seed_slider = Slider(
                ax=ax_slider,
                label="Seed Point (x value)",
                valmin=0.54,
                valmax=0.58,
                valinit=seed_points[0,0],
                orientation="horizontal"
            )
            
            septxt = ax_slider.text(0, -1, f"Separation: {separation[0]}", transform=ax_slider.transAxes)
            attachtxt = ax_slider.text(0, -2, f"Reattachment: {reattachment[0]}", transform=ax_slider.transAxes)

            # Function : On change of slider -> update the heatmap with new calculated values
            def UpdateHeatmap(val):
                extent = ax_heatmap.axis()
                ax_heatmap.cla()
                xy_grid = np.array([[xk, yk] for xk in xj for yk in yj])
                u_spd = u_intpl(xy_grid).reshape((xcount*heatmap_resolution,ycount*heatmap_resolution)).T
                absv = np.abs(u_spd)
                ax_heatmap.imshow(absv[::-1,:], extent=(piv_data[-1,0], piv_data[0,0], piv_data[-1,1], piv_data[0,1]), cmap=hmcm, norm=norm, interpolation="nearest", aspect="auto")
                ax_heatmap.set_xlabel("x/c [-]")
                ax_heatmap.set_ylabel("y/c [-]")
                seed_points = np.array([[val, cutoff_laminar]])
                sol, extrapolated, separation, reattachment = GenerateStreamlinePoints(seed_points[0], xj, min_y=cutoff_laminar)
                ax_heatmap.plot((x.min(), x.max()), (cutoff_laminar, cutoff_laminar), "k--", linewidth=0.5)
                ax_heatmap.plot((x.min(), x.max()), (cutoff_turbulent, cutoff_turbulent), "k--", linewidth=0.5)
                ax_heatmap.streamplot(xi, yi, uCi.reshape((xcount, ycount)).T, vCi.reshape((xcount, ycount)).T, color="black", cmap=vccsm, linewidth=1, density=2, arrowstyle="->", arrowsize=0, start_points=seed_points, broken_streamlines=False)
                ax_heatmap.plot(extrapolated[:,0], extrapolated[:,1], "r--")
                ax_heatmap.plot(sol[0], sol[1], "r-")
                ax_heatmap.plot(seed_points[:,0], seed_points[:,1], "bo", markersize=3)
                ax_heatmap.plot((separation[0], reattachment[0]), (separation[1], reattachment[1]), "ro")
                septxt.set_text(f"Separation: {separation[0]}")
                attachtxt.set_text(f"Reattachment: {reattachment[0]}")
                ax_heatmap.axis(extent)
            seed_slider.on_changed(UpdateHeatmap)

    plt.show()

# Function : Fix the ouput of the GenerateStreamline function to make it easier to access in future use
def CleanStreamlinePositions(streamline_positions):
    x_points, y_points = streamline_positions[0], streamline_positions[1]
    xy_points = [[xs, ys] for xs in x_points for ys in y_points]
    return xy_points

# Function : For given streamline positions from GenerateStreamline, find the corresponding absolute velocity        
def DetermineStreamlineVelocity(streamline_positions):
    x_vel, y_vel = u_intpl(xy_points), v_intpl(xy_points)
    vel = np.sqrt(x_vel**2 + y_vel**2)
    return vel

# Function : Plot the "best" dividing streamline option from generations function(s)
def PlotHeatmapDividingStreamline(seed_ds):
    xh = np.linspace(X.min(), X.max(), xcount*heatmap_resolution)
    yh = np.linspace(Y.min(), Y.max(), ycount*heatmap_resolution)
    xy_grid = np.array([[x, y] for x in xh for y in yh])
    
    # Use the RegularGridInterpolator to find the nicely interpolated grid
    absv = u_intpl(xy_grid).reshape((xcount*heatmap_resolution,ycount*heatmap_resolution)).T

    # Create the subplots for the heatmap and colorbar
    fig_heatmap, ax_heatmap = plt.subplots()
    heatmap = ax_heatmap.imshow(absv[::-1,:], extent=(piv_data[-1,0], piv_data[0,0], piv_data[-1,1], piv_data[0,1]), cmap=cm.turbo, interpolation="nearest", aspect="auto")
    plt.colorbar(heatmap, label="Absolute velocity [1/U$_{inf}$]", ax=ax_heatmap)
    ax_heatmap.set_xlabel("x/c [-]")
    ax_heatmap.set_ylabel("y/c [-]")

    # Plot the streamline from GenerateStreamline from the seed array
    streamline_positions = GenerateStreamlinePoints(seed_ds, xh)[0]
    ax_heatmap.plot(streamline_positions[0], streamline_positions[1], "black", linewidth=0.5)
    ax_heatmap.plot((x.min(), x.max()), (cutoff_laminar, cutoff_laminar), "k--", linewidth=0.5)
    ax_heatmap.plot((x.min(), x.max()), (cutoff_turbulent, cutoff_turbulent), "k--", linewidth=0.5)
     
    # Calculate position, extrapolated positions, seperation and reattachment points
    sol, extrapolated, separation, reattachment = GenerateStreamlinePoints(seed_ds, xh, min_y=cutoff_laminar)
    print("Separation point: ", separation[0])
    print("Reattachment point: ", reattachment[0])

    # add extrapolated solution points to the graph
    ax_heatmap.plot(extrapolated[:,0], extrapolated[:,1], "r--")
    # add actual solution points to graph
    ax_heatmap.plot(sol[0], sol[1], "r-")
    # add separation and reattachment points
    ax_heatmap.plot((separation[0], reattachment[0]), (separation[1], reattachment[1]), "ro")

    # - TODO: Add transition points here
    #ax_heaatmap.plot()

    plt.show()

# Function : Find the margin of the dividing stream line
# - NOTE: function not used in this code, but could be used to pre-proces for larger data sets
def FindDividingStreamlineMeanMargin(error=0.3, n_steps=20, plot_hm=False, min_y=laminar_height):
    # => using interpolation of points to find target -> then find "closest" streamline
    umax, umin = uCi.max(), uCi.min()
    
    # Find mean of the velocity field = color of dividing streamline in gradient plot
    mean = 0.5*(umax + umin)
    # Find error + and - to find the error margin
    de_max = (1+error)*mean
    de_min = (1-error)*mean
    contour = np.array([de_min, de_max])

    # Plot heatmap with contour margin
    # PlotVelocityHeatmap(seed, plot_contour=True, contour_levels=contour)

    xh = np.linspace(X.min(), X.max(), xcount*heatmap_resolution)
    yh = np.linspace(Y.min(), Y.max(), ycount*heatmap_resolution)
    xy_grid = np.array([[x, y] for x in xh for y in yh])
    absv = u_intpl(xy_grid).reshape((xcount*heatmap_resolution,ycount*heatmap_resolution)).T
    
    fig_contour, ax_contour = plt.subplots()
    contour = ax_contour.contour(xh, yh, absv.reshape(ycount*heatmap_resolution,xcount*heatmap_resolution), levels=contour, colors="black", linewidths=1)
    p_min = contour.allsegs[0]
    p_max = contour.allsegs[1]
    plt.close(fig_contour)

    x_min = p_min[0][:, 0]
    y_min = p_min[0][:, 1]
    x_max = p_max[0][:, 0]
    y_max = p_max[0][:, 1]

    # Filter vertices
    n_min, n_max = len(x_min), len(x_max)
    points_max, points_min = [], []

    filter_step = int(n_min/n_steps)
    for i in range(0, n_min, filter_step):
        if x_min[i] < 0.7 and y_min[i] > min_y:
            coord = [x_min[i], y_min[i]]
            points_min.append(coord)

    filter_step = int(n_max/n_steps)
    for j in range(0, n_max, filter_step):
        if x_max[j] < 0.7 and y_max[i] > min_y:
            coord = [x_max[j], y_max[j]]
            points_max.append(coord)

    points_max = np.array(points_max)
    points_min = np.array(points_min)

    # Create a grid for the heatmap (sizing it up on resolution)
    xh = np.linspace(X.min(), X.max(), xcount*heatmap_resolution)
    yh = np.linspace(Y.min(), Y.max(), ycount*heatmap_resolution)
    xy_grid = np.array([[x, y] for x in xh for y in yh])
    # Use the RegularGridInterpolator to find the nicely interpolated grid
    absv = u_intpl(xy_grid).reshape((xcount*heatmap_resolution,ycount*heatmap_resolution)).T

    if plot_hm:
        fig_hm, ax_hm = plt.subplots()
        heatmap = ax_hm.imshow(absv[::-1,:], extent=(piv_data[-1,0], piv_data[0,0], piv_data[-1,1], piv_data[0,1]), cmap=cm.turbo, interpolation="nearest", aspect="auto")
        plt.colorbar(heatmap, label="Absolute velocity [1/U$_{inf}$]", ax=ax_hm)
        ax_hm.set_xlabel("x/c [-]")
        ax_hm.set_ylabel("y/c [-]")

        ax_hm.scatter(points_min[:,0], points_min[:,1], color="black", s=1)
        ax_hm.scatter(points_max[:,0], points_max[:,1], color="black", s=1)


    return points_min, points_max

# Function : Find the actual contour points of the mean estimated dividing stream line
# - NOTE: right now mean of "temperature" on heat map is chosen, however this may not be "exact"-ly valid
#         Need some more physics reasoning to find a precise determination of the contour-level
def FindDividingStreamlineContour(n_steps=20, min_y=laminar_height, plot_hm=False):
    # => using interpolation of points to find target -> then find "closest" streamline
    umax, umin = uCi.max(), uCi.min()
    # Find mean of the velocity field = color of dividing streamline in gradient plot
    mean = 0.5*(umax + umin)
    contour = [mean]
    # - TODO: Add a more specific boundary layer condition.

    # Generate grid
    xh = np.linspace(X.min(), X.max(), xcount*heatmap_resolution)
    yh = np.linspace(Y.min(), Y.max(), ycount*heatmap_resolution)
    xy_grid = np.array([[x, y] for x in xh for y in yh])
    absv = u_intpl(xy_grid).reshape((xcount*heatmap_resolution,ycount*heatmap_resolution)).T
    
    # Plot the contour corresponding to the "temperature" on the heatmap of contour=[mean]
    fig_contour, ax_contour = plt.subplots()
    contour = ax_contour.contour(xh, yh, absv.reshape(ycount*heatmap_resolution,xcount*heatmap_resolution), levels=contour, colors="black", linewidths=1)
    p = contour.allsegs[0] # extract points from contour map
    plt.close(fig_contour) #close the contour map to avoid it showing

    # Extract points from the contour map in the right format
    contour_x = p[0][:, 0]
    contour_y = p[0][:, 1]

    # Filter vertices
    contour_points = []

    # Filter out some of the points to make it a manageble size
    filter_step = int(len(contour_x)/n_steps)
    for i in range(0, len(contour_x), filter_step):
        if contour_x[i] < xc_cutoff and contour_y[i] > min_y: # check if the contour falls within region 
            coord = [contour_x[i], contour_y[i]]
            contour_points.append(coord)

    contour_points = np.array(contour_points)

    # Create a grid for the heatmap (sizing it up on resolution)
    xh = np.linspace(X.min(), X.max(), xcount*heatmap_resolution)
    yh = np.linspace(Y.min(), Y.max(), ycount*heatmap_resolution)
    xy_grid = np.array([[x, y] for x in xh for y in yh])

    # Use the RegularGridInterpolator to find the nicely interpolated grid
    absv = u_intpl(xy_grid).reshape((xcount*heatmap_resolution,ycount*heatmap_resolution)).T

    if plot_hm:
        fig_hm, ax_hm = plt.subplots()
        heatmap = ax_hm.imshow(absv[::-1,:], extent=(piv_data[-1,0], piv_data[0,0], piv_data[-1,1], piv_data[0,1]), cmap=cm.turbo, interpolation="nearest", aspect="auto")
        plt.colorbar(heatmap, label="Absolute velocity [1/U$_{inf}$]", ax=ax_hm)
        ax_hm.set_xlabel("x/c [-]")
        ax_hm.set_ylabel("y/c [-]")

        ax_hm.scatter(contour_points[:,0], contour_points[:,1], color="black", s=1)
        plt.show()
    
    return contour_points

# Function : From contour points, find a polynomial that fits the points (avoid overfitting)
def FindContinuousContour(contour_points, degree=5, plot=False):
    contour_x = contour_points[:,0]
    contour_y = contour_points[:,1]

    # Fit the contour points into a polynomial of certain degree
    coeff = np.polyfit(contour_x,contour_y, degree)
    # Use those coefficients to create a Polynomial object
    poly = np.poly1d(coeff)

    if plot:
        fig_con, ax_con = plt.subplots()
        points=[]
        ax_con.scatter(contour_x, contour_y, color="black", s=1)
        for x_point in contour_x:
            y_point = poly(x_point)
            points.append([x_point, y_point])
        points = np.array(points)
        ax_con.plot(points[:,0], points[:,1], color="red")
        plt.show()

    return poly

# Function : Estimate a laminar height using contour points before separation (approximately <0.45)
# - NOTE: Same function as FindDividingStreamlineContour, but instead of filtering within xc determination region, look before separation
def EstimateLaminarHeight(n_steps=10):
    # => using interpolation of points to find target -> then find "closest" streamline
    umax, umin = uCi.max(), uCi.min()
    # Find mean of the velocity field = color of dividing streamline in gradient plot
    mean = 0.6*(umax + umin)
    contour = [mean]

    # Generate grid
    xh = np.linspace(X.min(), X.max(), xcount*heatmap_resolution)
    yh = np.linspace(Y.min(), Y.max(), ycount*heatmap_resolution)
    xy_grid = np.array([[x, y] for x in xh for y in yh])
    absv = u_intpl(xy_grid).reshape((xcount*heatmap_resolution,ycount*heatmap_resolution)).T
    
    # Calculate contour points, without plotting the contour map
    fig_contour, ax_contour = plt.subplots()
    contour = ax_contour.contour(xh, yh, absv.reshape(ycount*heatmap_resolution,xcount*heatmap_resolution), levels=contour, colors="black", linewidths=1)
    p = contour.allsegs[0] # extract contour points
    plt.close(fig_contour)

    # Extract contour points in the right format
    contour_x = p[0][:, 0]
    contour_y = p[0][:, 1]

    # then instead of looking beyond xc_cutoff, we want to find the height of the mean around 0.0-0.1
    heights=[]
    for i in range(0, len(contour_x), int(len(contour_x)/n_steps)):
        if contour_x[i]<0.45:
            heights.append(contour_y[i])
    mean_heights = np.mean(heights)
    return mean_heights
    
# Function : Find all the streamlines for given begin and endpoints for seed generation
def FindAllStreamlines(step_size=0.1, xc_min=0.45, xc_max=0.50):
    streamlines = []
    # Generate a list of all seeds to look through for a start, stop point and a certain stepsize
    seeds = np.arange(xc_min, xc_max, step_size)
    xj = np.linspace(X.min(), X.max(), xcount*heatmap_resolution)
    # For each seed, generate streamline, find positions
    for xc in seeds:
        streamlinepos = GenerateStreamlinePoints([xc, laminar_height], xj, min_y=cutoff_laminar)[0] #just get the position vectors of each of the seed points
        position = CleanStreamlinePositions(streamline_positions=streamlinepos)
        streamlines.append(position)
    streamlines = np.array(streamlines)
    return seeds, streamlines

# Function : Given target polynomial and total of all streamlines, determine the scores and find the best fit
def FitStreamline(target_polynomial, seeds, streamlines):
    best_score = np.inf # lowest score = best :)
    best_seed = seeds[0]
    # For each streamline, find the score and compare that to the current best one
    for id, streamline in enumerate(streamlines):
        current_score = 0
        current_seed = seeds[id]
        for point in streamline:
            # To find the score, take the difference between the actual y and the contour y
            # - TODO: Could improve this by adding mean squared estimation << how does this change the scoring process
            y_contour = target_polynomial(point[0]) 
            subscore = (y_contour - point[1])**2
            current_score += subscore
        if current_score < best_score:
            best_score = current_score
            best_seed_x = current_seed
    
    return [best_seed_x, laminar_height]

# Function : Using other functions, apply generation-method to optimize performance 
def FitOverGenerations(n_generations=10,xc_min_start=0.46, xc_max_start=0.48):
    # To avoid having to plot unneccesary points, go through generations to pin point boundaries
    # and then calculate inside those boundaries on the next generation

    print("Obtaining target points")
    # Obtain the contour points for our target streamline and obtain the polynomial object for the target
    target = FindDividingStreamlineContour(n_steps=40)
    target_polynomial = FindContinuousContour(target, degree=6)
    print("Target Polynomial obtained")

    # Setup the start and stop points of the boundary
    xc_min = xc_min_start
    xc_max = xc_max_start

    # Generate steps array to work with
    steps = np.logspace(-1, -n_generations, num=n_generations) # Output: [0.1, 0.01, 0.001 ...]

    # For each gneration:
    for generation_id in range(n_generations):
        print("----- Generation: ", generation_id, " -----")
        print("Obtaining Seeds")
        # Obtain the current seed array to work with
        seeds = np.arange(xc_min, xc_max, steps[generation_id])
        print("Seeds Obtained, Generating Streamlines")
        seeds, streamlines = FindAllStreamlines(xc_min = xc_min, xc_max = xc_max, step_size=steps[generation_id])
        print("Streamlines obtained, Fitting the streamline")
        seed = FitStreamline(target_polynomial, seeds, streamlines)

        print("Setting new boundaries")
        # After each iteration, where we found the best seed, generate new boundaries within a smaller range
        xc_min = seed[0] - steps[generation_id]*seed[0]*0.5
        xc_max = seed[0] + steps[generation_id]*seed[0]*0.5
        print("Obtained boundaries: ", xc_min, xc_max)

    print("Generations completed, seed obtained: ", seed)
    return seed
        

# ------------------ Select functions to run ------------------

laminar_height = EstimateLaminarHeight(n_steps=20)

seed = FitOverGenerations(n_generations=10)
PlotHeatmapDividingStreamline(seed)



