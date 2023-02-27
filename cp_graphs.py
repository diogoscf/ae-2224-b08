import matplotlib.pyplot as plt
import numpy as np

from load_data import cp_data

x1= cp_data[17:23, 0].reshape(-1,1)
y1=cp_data[17:23, 1]

model1= LinearRegression().fit(x1, y1)
rsq1 = model1.score(x1, y1)
a1 = model1.coef_
b1 = model1.intercept_

#fit pressure recovery
x2= cp_data[22:25, 0].reshape(-1,1)
y2=cp_data[22:25, 1]

model2= LinearRegression().fit(x2, y2)
rsq2 = model2.score(x2, y2)
a2 = model2.coef_
b2 = model2.intercept_


fig_cp, ax_cp = plt.subplots()
ax_cp.plot(cp_data[:, 0], cp_data[:, 1], label="CP suction side", marker="o")
ax_cp.plot(cp_data[17:23,0], a1*cp_data[17:23,0]+b1, label=f"linear regression pressure plateau, rsq ={rsq1}", marker = "v")
ax_cp.plot(cp_data[22:25,0], a2*cp_data[22:25,0]+b2, label=f"linear regression pressure plateau, rsq ={rsq2}", marker = "s")
ax_cp.invert_yaxis()
ax_cp.set_title("Cp graph on the suction side of a NACA 643-618 for AoA 2°, Re = 200 000")
ax_cp.set_xlabel("x/c [-]")
ax_cp.set_ylabel("Cp [-]")
ax_cp.legend()
ax_cp.grid(linewidth =0.5)
ax_cp.axvline(0, c='black', ls = '-', linewidth = 0.8)
ax_cp.axhline(0, c='black', ls = '-', linewidth = 0.8)

plt.show()

