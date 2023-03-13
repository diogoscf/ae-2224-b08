
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from load_data import cp_data


#fit before plateau
x0= cp_data[14:18, 0].reshape(-1,1)
y0=cp_data[14:18, 1]

model0= LinearRegression().fit(x0, y0)
rsq0 = model0.score(x0, y0)
a0 = model0.coef_
b0 = model0.intercept_

#fit plateau
x1= cp_data[17:20, 0].reshape(-1,1)
y1=cp_data[17:20, 1]

model1= LinearRegression().fit(x1, y1)
rsq1 = model1.score(x1, y1)
a1 = model1.coef_
b1 = model1.intercept_

#fit pressure recovery
x2= cp_data[22:26, 0].reshape(-1,1)
y2=cp_data[22:26, 1]

model2= LinearRegression().fit(x2, y2)
rsq2 = model2.score(x2, y2)
a2 = model2.coef_
b2 = model2.intercept_

#fit after pressure recovery
x3= cp_data[24:27, 0].reshape(-1,1)
y3=cp_data[24:27, 1]

model3= LinearRegression().fit(x3, y3)
rsq3 = model3.score(x3, y3)
a3 = model3.coef_
b3 = model3.intercept_


fig_cp, ax_cp = plt.subplots()
ax_cp.plot(cp_data[:, 0], cp_data[:, 1], label="CP suction side", marker="o")
ax_cp.plot(cp_data[12:19,0], a0*cp_data[12:19,0]+b0, label=f"linear regression pressure plateau, rsq ={rsq0}", marker = "s")
ax_cp.plot(cp_data[15:25,0], a1*cp_data[15:25,0]+b1, label=f"linear regression pressure plateau, rsq ={rsq1}", marker = "v")
ax_cp.plot(cp_data[22:26,0], a2*cp_data[22:26,0]+b2, label=f"linear regression pressure plateau, rsq ={rsq2}", marker = "s")
ax_cp.plot(cp_data[23:27,0], a3*cp_data[23:27,0]+b3, label=f"linear regression pressure plateau, rsq ={rsq3}", marker = "v")
ax_cp.invert_yaxis()
ax_cp.set_title("Cp graph on the suction side of a NACA 643-618 for AoA 2°, Re = 200 000")
ax_cp.set_xlabel("x/c [-]")
ax_cp.set_ylabel("Cp [-]")
ax_cp.legend()
ax_cp.grid(linewidth =0.5)
ax_cp.axvline(0, c='black', ls = '-', linewidth = 0.8)
ax_cp.axhline(0, c='black', ls = '-', linewidth = 0.8)

plt.show()



def location(feature):

    if feature == "separation":

        x = (b1-b0)/(a0-a1)
        
    if feature == "transition":
        
        x = (b2-b1)/(a1-a2)

    if feature == "reattachment":

        x = (b3-b2)/(a2-a3)

    return x

print(f'Separation location (x/c): {float(location("separation")):.3f}')        
print(f'Transition location (x/c): {float(location("transition")):.3f}')
print(f'Reattachment location (x/c): {float(location("reattachment")):.3f}')
