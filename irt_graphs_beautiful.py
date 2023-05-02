import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from load_data import irt_data
import pandas as pd

def moving_average(arr, window_size):
    """
    Computes the moving average of an array using a given window size.
    
    Parameters:
        arr (numpy array): The input array to compute the moving average of.
        window_size (int): The number of neighboring elements to include in the window.
        
    Returns:
        numpy array: An array of the same length as the input array, where each element is the average of its neighboring elements in the window.
    """
    # Pad the input array with zeros at the beginning and end to handle edge cases
    padded_arr = np.pad(arr, (window_size//2, window_size//2), mode='constant', constant_values=0)
    
    # Compute the moving average using a convolution with a uniform filter
    uniform_filter = np.ones(window_size) / window_size
    moving_avg = np.convolve(padded_arr, uniform_filter, mode='valid')
    
    # Return the moving average with the same length as the input array
    return moving_avg[:len(arr)]
def moving_average2(x, w):
    """
    Calculates the moving average of a given array x using a window size w.
    Endpoints are not removed and only the existing points are used.
    
    Args:
        x: 1-D numpy array to calculate the moving average of.
        w: Integer value of the window size.
    
    Returns:
        1-D numpy array of the same size as x containing the moving average values.
    """
    # Initialize the output array with zeros
    y = np.zeros_like(x)
    
    # Calculate the moving average
    for i in range(len(x)):
        # Calculate the start and end indices of the window
        start = max(0, i - w//2)
        end = min(len(x), i + w//2 + 1)
        # Calculate the average value of the window
        y[i] = np.mean(x[start:end])
    
    return y

colors = ['#1F75CC', '#81D4FA', '#B3B3B3', '#4F4F4F']

def graph_spalart(x_s,x_t,x_r,title):
    data_st = pd.read_excel(r'./St.xlsx') # getting data
    df_st = pd.DataFrame(data_st, columns=['x', 'y'])
    data_cf = pd.read_excel(r'./Cf.xlsx')
    df_cf = pd.DataFrame(data_cf, columns=['x', 'y'])
    data_delta = pd.read_excel(r'./Delta.xlsx')
    df_delta = pd.DataFrame(data_delta, columns=['x', 'y'])
    data_theta = pd.read_excel(r'./Theta.xlsx')
    df_theta = pd.DataFrame(data_theta, columns=['x', 'y'])

    spalart_dx = 0.001
    xs = np.arange(0.2,7,spalart_dx)
    cs_st = CubicSpline(df_st["x"].to_list(),df_st["y"].to_list()) # splining data
    cs_cf = CubicSpline(df_cf["x"].to_list(),df_cf["y"].to_list())
    cs_delta = CubicSpline(df_delta["x"].to_list(),df_delta["y"].to_list())
    cs_theta = CubicSpline(df_theta["x"].to_list(),df_theta["y"].to_list())
    n = 10
    smooth_st = moving_average2(cs_st(xs), 300)
    print(len(xs))
    fig_spal, ax_spal = plt.subplots()

    shape_factor = cs_delta(xs)/cs_theta(xs)
    g = np.gradient(smooth_st,spalart_dx)
    fig_spal.add_axes([0, 0, 0, 0.01])

    ax_spal2 = ax_spal.twinx()
    ax_spal2.plot(xs,shape_factor,label = "Shape factor", color = colors[0])
    ax_spal.plot(xs,g, label = "Stanton number gradient",color = colors[1])
    ax_spal.plot(xs,cs_cf(xs),label = "Friction coefficient", color = colors[2])
    ax_spal.plot(xs,smooth_st, label = "Stanton number",color = colors[3])

    ax_spal.set_xlabel('Chord [m]', fontsize=14, fontweight='bold')
    ax_spal.set_ylabel('[-]', fontsize=14, fontweight='bold')
    ax_spal.set_title(title, fontsize=16, fontweight='bold')

    # Adding vertical lines
    ax_spal.axvline(x_s, color='black', linestyle='dotted')
    ax_spal.text(x_s-0.02, 0.0135, r'$x_s$', ha='right', va='bottom',color = "black")

    ax_spal.axvline(x_t, color='black', linestyle='dotted')
    ax_spal.text(x_t-0.03, 0.0135, r'$x_t$', ha='right', va='bottom',color = "black")

    ax_spal.axvline(x_r, color='black', linestyle='dotted')
    ax_spal.text(x_r-0.02, 0.0135, r'$x_r$', ha='right', va='bottom',color = "black")

    # Add a legend with a frame and adjust its position
    #ax_spal.legend(frameon=True, framealpha=1, fontsize=12, loc='lower right')
    lines3, labels3 = ax_spal.get_legend_handles_labels()
    lines4, labels4 = ax_spal2.get_legend_handles_labels()
    ax_spal2.legend(lines3 + lines4, labels3 + labels4, frameon=True, framealpha=1, fontsize=12, loc='lower right')
    #fig_spal.tight_layout()

    ax_spal.spines['top'].set_position('zero')
    ax_spal.spines['right'].set_color("white")
    ax_spal.spines['left'].set_position('zero')

    ax_spal2.spines['top'].set_color('white')
    #ax_spal2.spines['right'].set_color("white")
    ax_spal2.spines['left'].set_color('white')


rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


#spalart graph


graph_spalart(2.26,3.47,4.21,'Spalarts data')
graph_spalart(2.26,3.59,4,'Wynnychuk method')
graph_spalart(1.66,2.49,4,'Ricci method')




#1/T graph and gradient
xcount, zcount = 711, 343
irt_array = np.reshape(irt_data[:, 2], (zcount, xcount))
xmin, xmax = irt_data[:,0].min()/1.02, irt_data[:,0].max()*1.02
tmin, tmax = irt_data[:,2].min()/1.02, irt_data[:,2].max()*1.02

fig_inv, ax_inv1 = plt.subplots()
irt_inv = 1/np.average(irt_array, axis=0)


ax_inv1.axvline(0.47, color='black', linestyle='dotted')
ax_inv1.text(0.45 , 0.000365, r'$x_s$', ha='right', va='bottom',color = "black")

ax_inv1.axvline(0.65, color='black', linestyle='dotted')
ax_inv1.text(0.63, 0.000365, r'$x_t$', ha='right', va='bottom',color = "black")

ax_inv1.axvline(0.72, color='black', linestyle='dotted')
ax_inv1.text(0.70, 0.000365, r'$x_r$', ha='right', va='bottom',color = "black")

dx = irt_data[1, 0] - irt_data[0, 0]
g2 = np.gradient(irt_inv, dx)

ax_inv1.plot(np.unique(irt_data[0:-1, 0]), irt_inv, label = "Inverse temperature counts",color = colors[0])

ax_inv2 = ax_inv1.twinx()
ax_inv2.plot(np.unique(irt_data[0:-1, 0]),moving_average2(g2,20),label = "Gradient",color = colors[1])


ax_inv1.set_xlabel('x/c [-]', fontsize=14, fontweight='bold')
ax_inv1.set_ylabel('1/Temperature counts', fontsize=14, fontweight='bold')
ax_inv1.set_title('Inverse of average temperature counts', fontsize=16, fontweight='bold')
lines, labels = ax_inv1.get_legend_handles_labels()
lines2, labels2 = ax_inv2.get_legend_handles_labels()
ax_inv2.legend(lines + lines2, labels + labels2, frameon=True, framealpha=1, fontsize=10, loc='upper left')
#ax_inv1.legend(frameon=True, framealpha=1, fontsize=12, loc='lower right')
#fig_inv.tight_layout()


#ax_inv.set(ylim=(tmin, tmax), xlim=(xmin, xmax))

#average graph with 3 separate points
fig_sep, ax_sep = plt.subplots()
ax_sep.plot(np.unique(irt_data[0:-1, 0]),1/irt_array[0],color = colors[0], label = f"z/s = 0")
ax_sep.plot(np.unique(irt_data[0:-1, 0]),1/irt_array[172],color = colors[1], label = f"z/s = 0.5")
ax_sep.plot(np.unique(irt_data[0:-1, 0]),1/irt_array[342],color = colors[2], label = f"z/s = 1")
ax_sep.set_xlabel('x/c [-]', fontsize=14, fontweight='bold')
ax_sep.set_ylabel('1/Temperature counts', fontsize=14, fontweight='bold')
ax_sep.set_title('Inverse of temperature counts', fontsize=16, fontweight='bold')
ax_sep.legend(frameon=True, framealpha=1, fontsize=12, loc='lower right')
plt.show()