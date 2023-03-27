import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from sklearn.linear_model import LinearRegression
from scipy.interpolate import UnivariateSpline
import curlyBrace as cb

cp_data_exp = np.genfromtxt("CP_graph project N10.txt", skip_header=1, delimiter = '')
cp_data_exp2 = np.genfromtxt("CP_graph project N9812.txt", skip_header=1, delimiter = '')

#fit linear regression pressure plateau
x0= cp_data_exp[12:15, 0].reshape(-1,1)
y0=cp_data_exp[12:15, 1]

model0= LinearRegression().fit(x0, y0)
rsq0 = model0.score(x0, y0)
a0 = model0.coef_
b0 = model0.intercept_

x1= cp_data_exp[15:18, 0].reshape(-1,1)
y1=cp_data_exp[15:18, 1]

model1= LinearRegression().fit(x1, y1)
rsq1 = model1.score(x1, y1)
a1 = model1.coef_
b1 = model1.intercept_


#fit pressure recovery
x2 = cp_data_exp[18:21, 0].reshape(-1,1)
y2 = cp_data_exp[18:21, 1]

model2= LinearRegression().fit(x2, y2)
rsq2 = model2.score(x2, y2)
a2 = model2.coef_
b2 = model2.intercept_

x3= cp_data_exp[21:26, 0].reshape(-1,1)
y3=cp_data_exp[21:26, 1]

model3= LinearRegression().fit(x3, y3)
rsq3 = model3.score(x3, y3)
a3 = model3.coef_
b3 = model3.intercept_




#plots

fig_cp, ax_cp = plt.subplots()

ax_cp.plot(cp_data_exp[:26, 0], cp_data_exp[:26, 1], label="CP suction side XFoil N=10", marker="s")
#ax_cp.plot(cp_data_exp2[:26, 0], cp_data_exp2[:26, 1], label="CP suction side XFoil N=9.812", marker="v")
#ax_cp.set_xlim(0.35,0.85)
#ax_cp.set_ylim(-1.15, 0.2)


ax_cp.plot(cp_data_exp[10:17,0], a0*cp_data_exp[10:17,0]+b0, label=f"Linear regression, $R^{2} ={float(rsq0):.4f}$", marker = "s")
ax_cp.plot(cp_data_exp[12:20,0], a1*cp_data_exp[12:20,0]+b1, label=f"Linear regression pressure plateau, $a_1 = {float(a1):.4f}$, $R^{2} = {float(rsq1):.4f}$", marker = "v")
ax_cp.plot(cp_data_exp[16:22,0], a2*cp_data_exp[16:22,0]+b2, label=f"Linear regression pressure recovery, $R^{2} ={float(rsq2):.4f}$", marker = "X")
ax_cp.plot(cp_data_exp[20:26,0], a3*cp_data_exp[20:26,0]+b3, label=f"Linear regression pressure , $R^{2} ={float(rsq3):.4f}$", marker = "|")






ax_cp.invert_yaxis()
#ax_cp.set_title("Cp graph on the suction side of a NACA 643-618 for AoA 2°, Re = 200 000")
ax_cp.set_xlabel("$x/c\ [-]$")
ax_cp.set_ylabel("$C_p\ [-]$")
ax_cp.legend()
ax_cp.grid(linewidth =0.5)
ax_cp.axvline(0, c='black', ls = '-', linewidth = 0.8)
ax_cp.axhline(0, c='black', ls = '-', linewidth = 0.8)
plt.show()
