import matplotlib.pyplot as plt
import numpy as np

from load_data import cp_data

# Plot the data
fig_cp, ax_cp = plt.subplots()
ax_cp.plot(cp_data[:, 0], cp_data[:, 1], label="CP suction side", marker="o")
ax_cp.invert_yaxis()
ax_cp.set_title("Theoretical Cp graph for AoA 2°, Re = 200 000")
ax_cp.set_xlabel("x/c [-]")
ax_cp.set_ylabel("Cp [-]")

ax_cp.grid(linewidth =0.5)
ax_cp.axvline(0, c='black', ls = '-', linewidth = 0.8)
ax_cp.axhline(0, c='black', ls = '-', linewidth = 0.8)

plt.show()
