import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import numpy as np

from load_data import piv_data

xcount, ycount = 395, 57
ylim = 57
minl = (57 - ylim)*395
step = 5 # Resolution
fig_piv, ax_piv = plt.subplots()
X, Y = piv_data[minl::step, 0], piv_data[minl::step, 1]
U, V = piv_data[minl::step, 2], piv_data[minl::step, 3]

colors = np.linalg.norm(np.column_stack((U, V)), axis=1)
norm = Normalize()
norm.autoscale(colors)

ax_piv.quiver(X, Y, U, V, color=cm.inferno(norm(colors)), pivot="mid", scale=100, scale_units="xy", width=0.001, headwidth=3, headlength=4, headaxislength=3)

plt.show()