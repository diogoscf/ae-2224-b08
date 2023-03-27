import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from load_data import irt_data
import pandas as pd

fig_irt, ax_irt = plt.subplots()

xcount, zcount = 711, 343
irt_array = np.reshape(irt_data[:, 2], (zcount, xcount))
xmin, xmax = irt_data[:,0].min()/1.02, irt_data[:,0].max()*1.02
tmin, tmax = irt_data[:,2].min()/1.02, irt_data[:,2].max()*1.02

im = plt.imshow(irt_array, cmap=plt.cm.coolwarm, extent=(irt_data[0,0], irt_data[-1,0], irt_data[0,1], irt_data[-1,1]), interpolation="bilinear")
fig_irt.colorbar(im, ax=ax_irt)

fig_avg, ax_avg = plt.subplots()
irt_avg = np.average(irt_array, axis=0)
ax_avg.plot(np.unique(irt_data[0:-1, 0]), irt_avg, color="black", marker=".", linestyle="none")
ax_avg.set_xlabel("x/c [-]")
ax_avg.set_ylabel("Temperature Counts")
ax_avg.set(ylim=(tmin, tmax), xlim=(xmin, xmax))

ax_grad = ax_avg.twinx()
dx = irt_data[1, 0] - irt_data[0, 0]
g = np.gradient(irt_avg, dx)

ax_grad.plot(np.unique(irt_data[0:-1, 0]), g, color="red", marker=".", linestyle="none")


fig_inv, ax_inv = plt.subplots()
irt_inv = 1/irt_avg
ax_inv.plot(np.unique(irt_data[0:-1, 0]),irt_inv, color="green", marker=".", linestyle="none")
ax_inv.set_xlabel("x/c [-]")
ax_inv.set_ylabel("1/Temperature Counts")
#plots for three intervals
fig_sec, ax_sec1 = plt.subplots()

x = np.unique(irt_data[0:-1, 0])
for i in range(0,len(irt_array),10):

    ax_sec1.plot(x,irt_array[i])
    dx1 = irt_data[1, 0] - irt_data[0, 0]
    g1 = np.gradient(irt_array[i], dx)
    ax_sec1.plot(x,g1)  
data_st = pd.read_excel(r'./St.xlsx')
df_st = pd.DataFrame(data_st, columns=['x', 'y'])
data_cf = pd.read_excel(r'./Cf.xlsx')
df_cf = pd.DataFrame(data_cf, columns=['x', 'y'])

spalart_dx = 0.001
cs_st = CubicSpline(df_st["x"].to_list(),df_st["y"].to_list())
cs_cf = CubicSpline(df_cf["x"].to_list(),df_cf["y"].to_list())

xs = np.arange(0.2,7,spalart_dx)
fig_spal, ax_spal = plt.subplots()
ax_spal.plot(xs,cs_cf(xs))
ax_spal.plot(xs,cs_st(xs))
xlim = ax_spal.get_xlim()
ylim = ax_spal.get_ylim()
aspect = ax_spal.get_aspect()





g = np.gradient(cs_st(xs),spalart_dx)
fig_spal.add_axes([7,0.012,1,0.003])
ax_spal.plot(xs,g)
fig_spal_grad, ax_spal_grad = plt.subplots()
ax_spal_grad.plot(xs,g)

# g2 = -g
# ax_grad.plot(np.unique(irt_data[0:-1, 0]), g2, color="green", marker=".", linestyle="none")
ax_spal.set_xlim(2*np.array(xlim))
ax_spal.set_ylim(2*np.array(ylim))


#adding shape factor
data_delta = pd.read_excel(r'./Delta.xlsx')
df_delta = pd.DataFrame(data_delta, columns=['x', 'y'])
data_theta = pd.read_excel(r'./Theta.xlsx')
df_theta = pd.DataFrame(data_theta, columns=['x', 'y'])

cs_delta = CubicSpline(df_delta["x"].to_list(),df_delta["y"].to_list())
cs_theta = CubicSpline(df_theta["x"].to_list(),df_theta["y"].to_list())

shape_factor = cs_delta(xs)/cs_theta(xs)/1500
fig_shape, ax_shape = plt.subplots()
ax_spal.plot(xs,shape_factor)

plt.show()

