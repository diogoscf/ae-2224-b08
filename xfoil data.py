import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from sklearn.linear_model import LinearRegression
from scipy.interpolate import UnivariateSpline




cp_data_exp = np.genfromtxt("CP_graph project N10.txt", skip_header=1, delimiter = '')[:26]
cp_data_exp2 = np.genfromtxt("CP_graph project N9812.txt", skip_header=1, delimiter = '')[:26]
cp_data_exp = cp_data_exp[::-1]


def location(feature):

    if feature == "separation":

        x = (b1-b0)/(a0-a1)
        
    if feature == "transition":
        
        x = (b2-b1)/(a1-a2)

    if feature == "reattachment":

        x = (b3-b2)/(a2-a3)

    return x


def min_der(data):
    loc_min = 0
    loc_max = 0
    for i in range(len(data)):
        if data[i] == np.min(data[10:40]):
            loc_min=cheb_g[i]

        elif data[i] == np.max(data[10:40]):
            loc_max=cheb_g[i]
            
    return loc_min, loc_max



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

#Chebychev-Gauss grid:

x22 = cp_data_exp[17:26,0]
y22 = cp_data_exp[17:26,1]

cheb_g = np.polynomial.chebyshev.chebgauss(50)[0]

for i in range(1, len(cheb_g)+1):
    cheb_g[len(cheb_g)-i] = np.mean(x22)+(x22[-1]-x22[0])*cheb_g[i-1]/2
cheb_g = np.mean(x22) + (x22[-1]-x22[0])*np.polynomial.chebyshev.chebgauss(50)[0]/2
cheb_g = cheb_g[::-1]

y_extr = UnivariateSpline(x22,y22,s=0, k=1)
y_cheb_g = y_extr(cheb_g)

#cubic spline interpolation on Chebychev

cheb_model=UnivariateSpline(cheb_g,y_cheb_g,s=0, k=3)
cheb_model_2nd = cheb_model.derivative(n=2)

#fifth order poly interpolation on Chebychev

sq_model_cheb = np.poly1d(np.polyfit(cheb_g, cheb_model(cheb_g), 5))
sq_model_cheb_sp = UnivariateSpline(cheb_g,sq_model_cheb(cheb_g),s=0, k=5)
sq_model_cheb_2nd = sq_model_cheb_sp.derivative(n=2)


#plots

fig_cp, ax_cp = plt.subplots()

ax_cp.plot(cp_data_exp[:26, 0], cp_data_exp[:26, 1], label="CP suction side XFoil N=10", marker="s")
#ax_cp.plot(cp_data_exp2[:26, 0], cp_data_exp2[:26, 1], label="CP suction side XFoil N=9.812", marker="v")
ax_cp.set_xlim(0.35,0.85)
ax_cp.set_ylim(-1.15, 0.2)


ax_cp.plot(cp_data_exp[10:17,0], a0*cp_data_exp[10:17,0]+b0, label=f"Linear regression, $R^{2} ={float(rsq0):.4f}$", marker = "s")
ax_cp.plot(cp_data_exp[12:20,0], a1*cp_data_exp[12:20,0]+b1, label=f"Linear regression pressure plateau, $a_1 = {float(a1):.4f}$, $R^{2} = {float(rsq1):.4f}$", marker = "v")
ax_cp.plot(cp_data_exp[16:22,0], a2*cp_data_exp[16:22,0]+b2, label=f"Linear regression pressure recovery, $R^{2} ={float(rsq2):.4f}$", marker = "X")
ax_cp.plot(cp_data_exp[19:26,0], a3*cp_data_exp[19:26,0]+b3, label=f"Linear regression pressure , $R^{2} ={float(rsq3):.4f}$", marker = "|")
ax_cp.plot(cheb_g, sq_model_cheb(cheb_g), label=f"Fifth order Chebychev regression", marker = "X")
#ax_cp.plot(cheb_g, sq_model_cheb_2nd(cheb_g), label=f"quadratic regression", marker = "X")

ax_cp.plot(location("separation"), a1*location("separation")+b1, markersize = 10, color='black', marker="o")
ax_cp.plot(location("transition"), a1*location("transition")+b1, markersize = 10, color='black', marker="o")
ax_cp.plot(location("reattachment"), a2*location("reattachment")+b2, markersize = 10, color='black', marker="o")
ax_cp.axvline(location("separation"), c='black', ls = '--', linewidth = 1, label="Locations determined with linear regression")
ax_cp.axvline(location("transition"), c='black', ls = '--', linewidth = 1)
ax_cp.axvline(location("reattachment"), c='black', ls = '--', linewidth = 1)

#location derivatives

ax_cp.axvline(min_der(sq_model_cheb_2nd(cheb_g))[0], c='black', ls = ':', linewidth = 1)
ax_cp.axvline(min_der(sq_model_cheb_2nd(cheb_g))[1], c='black', ls = ':', linewidth = 1, label="Locations determined with Chebychev polynomials")
ax_cp.axvline(location("separation"), c='black', ls = ':', linewidth = 1)
ax_cp.plot(location("separation"), a1*location("separation")+b1, markersize = 10, color='black', marker="o")
ax_cp.plot(min_der(sq_model_cheb_2nd(cheb_g))[0], sq_model_cheb(min_der(sq_model_cheb_2nd(cheb_g))[0]), markersize = 10, color='black', marker="o")
ax_cp.plot(min_der(sq_model_cheb_2nd(cheb_g))[1], sq_model_cheb(min_der(sq_model_cheb_2nd(cheb_g))[1]), markersize = 10, color='black', marker="o")

y_arr=np.arange(-1.5, 1, 0.01)
ax_cp.fill_betweenx(
        x1=0.637 , 
        x2 = 0.644, 
        y= y_arr,
        color= "g",
        alpha= 0.3,
        label= "Region of uncertainty for transition XFoil")
ax_cp.fill_betweenx(
        x1=0.746 , 
        x2 = 0.756, 
        y= y_arr,
        color= "y",
        alpha= 0.3,
        label= "Region of uncertainty for reattachment XFoil")



ax_cp.invert_yaxis()
#ax_cp.set_title("Cp graph on the suction side of a NACA 643-618 for AoA 2°, Re = 200 000")
ax_cp.set_xlabel("$x/c\ [-]$")
ax_cp.set_ylabel("$C_p\ [-]$")
ax_cp.legend()
ax_cp.grid(linewidth =0.5)
ax_cp.axvline(0, c='black', ls = '-', linewidth = 0.8)
ax_cp.axhline(0, c='black', ls = '-', linewidth = 0.8)
plt.show()



print(f'From intersection of linear regression:')
print(f'Separation location (x/c): {float(location("separation")):.3f}')        
print(f'Transition location (x/c): {float(location("transition")):.3f}')
print(f'Reattachment location (x/c): {float(location("reattachment")):.3f}')
print(f'From second derivatives:')
print(f'Separation location (x/c): {float(location("separation")):.3f}')        
print(f'Transition location (x/c): {float(min_der(sq_model_cheb_2nd(cheb_g))[1]):.3f}')
print(f'Reattachment location (x/c): {float(min_der(sq_model_cheb_2nd(cheb_g))[0]):.3f}')
