import matplotlib.pyplot as plt
import numpy as np

from load_data import piv_data

fig_piv, ax_piv = plt.subplots()
ax_piv.quiver(piv_data[:, 0], piv_data[:, 1], piv_data[:, 2], piv_data[:, 3], pivot="mid", scale=100, scale_units="xy", width=0.002, headwidth=3, headlength=4, headaxislength=3)

plt.show()