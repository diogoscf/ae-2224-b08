import matplotlib.pyplot as plt
import numpy as np

from load_data import piv_data

x = [i[0]for i in piv_data]
y = [i[1]for i in piv_data]
u = [i[2]for i in piv_data]
v = [i[3]for i in piv_data]

x_reduced = []
y_reduced =[]
u_reduced = []
v_reduced = []
for i in range(0,len(x)):
    if i %20==0:
        x_reduced.append(x[i])
        y_reduced.append(y[i])
        u_reduced.append(u[i])
        v_reduced.append(v[i])
# plt.quiver(x,y,u,v)
# plt.show()
fig_piv, ax_piv = plt.subplots()
ax_piv.quiver(x_reduced, y_reduced, u_reduced, v_reduced, pivot="mid", scale=100, scale_units="xy", width=0.002, headwidth=3, headlength=4, headaxislength=3)

plt.show()