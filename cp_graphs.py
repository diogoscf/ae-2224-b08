import matplotlib.pyplot as plt
import numpy as np

from load_data import cp_data

# Plot the data
fig_cp, ax_cp = plt.subplots()
ax_cp.plot(cp_data[:, 0], cp_data[:, 1], label="CP")
ax_cp.invert_yaxis()

plt.show()