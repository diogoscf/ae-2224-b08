import matplotlib.pyplot as plt
import numpy as np

from load_data import irt_data

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
    
# g2 = -g
# ax_grad.plot(np.unique(irt_data[0:-1, 0]), g2, color="green", marker=".", linestyle="none")

plt.show()

