
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from load_data import cp_data
import scipy as sc
from scipy.interpolate import UnivariateSpline

#fit linear regression pressure plateau
x0= cp_data[14:18, 0].reshape(-1,1)
y0=cp_data[14:18, 1]

model0= LinearRegression().fit(x0, y0)
rsq0 = model0.score(x0, y0)
a0 = model0.coef_
b0 = model0.intercept_

x1= cp_data[18:23, 0].reshape(-1,1)
y1=cp_data[18:23, 1]

model1= LinearRegression().fit(x1, y1)
rsq1 = model1.score(x1, y1)
a1 = model1.coef_
b1 = model1.intercept_

#computing R squared manually

'''y1_m = np.mean(y1)
y1_ma = y1-y1_m
SS_tot = sum(y1_ma**2)
y_tr = (a1*x1.reshape(6,)+b1-y1)**2
SS_true = sum(y_tr)
print(1-SS_true/SS_tot)'''




#fit pressure recovery
x2 = cp_data[23:26, 0].reshape(-1,1)
y2 = cp_data[23:26, 1]

model2= LinearRegression().fit(x2, y2)
rsq2 = model2.score(x2, y2)
a2 = model2.coef_
b2 = model2.intercept_

x3= cp_data[24:27, 0].reshape(-1,1)
y3=cp_data[24:27, 1]

model3= LinearRegression().fit(x3, y3)
rsq3 = model3.score(x3, y3)
a3 = model3.coef_
b3 = model3.intercept_



# quadratic interpolation
x22 = cp_data[21:27, 0]
y22 = cp_data[21:27, 1]
sq_model0 = np.poly1d(np.polyfit(x0.reshape(1,-1)[0], y0, 2))



x22_ex = np.linspace(float(cp_data[21, 0]), float(cp_data[26,0]), 100)
sq_model2 = np.poly1d(np.polyfit(x22, y22, 5))
y22_ex = sq_model2(x22)
y_spl = UnivariateSpline(x22,y22,s=0)
y_spl_2d = y_spl.derivative(n=2)



#Chebychev-Gauss grid:

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

sq_model_cheb = np.poly1d(np.polyfit(cheb_g, y_cheb_g, 5))
sq_model_cheb_sp = UnivariateSpline(cheb_g,sq_model_cheb(cheb_g),s=0, k=5)
sq_model_cheb_2nd = sq_model_cheb_sp.derivative(n=2)

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


fig_cp, ax_cp = plt.subplots()
ax_cp.plot(cp_data[:, 0], cp_data[:, 1], label="CP suction side", marker="o")
ax_cp.plot(cp_data[12:19,0], a0*cp_data[12:19,0]+b0, label=f"linear regression, $R^{2} ={float(rsq0):.4f}$", marker = "s")
ax_cp.plot(cp_data[16:24,0], a1*cp_data[16:24,0]+b1, label=f"linear regression pressure plateau, $a_1 = {float(a1):.4f}$, $R^{2} = {float(rsq1):.4f}$", marker = "v")
'''ax_cp.plot(cp_data[22:26,0], a2*cp_data[22:26,0]+b2, label=f"linear regression pressure recovery, $R^{2} ={float(rsq2):.4f}$", marker = "X")
ax_cp.plot(cp_data[23:27,0], a3*cp_data[23:27,0]+b3, label=f"linear regression pressure , $R^{2} ={float(rsq3):.4f}$", marker = "|")'''

#ax_cp.plot(cheb_g, cheb_model(cheb_g), label=f"quadratic regression", marker = "o")
#ax_cp.plot(cheb_g, sq_model_cheb(cheb_g), label=f"quadratic regression", marker = "X")
#ax_cp.plot(cheb_g, cheb_model_2nd(cheb_g), label=f"quadratic regression", marker = "o")
#ax_cp.plot(cheb_g, sq_model_cheb_2nd(cheb_g), label=f"quadratic regression", marker = "X")

#ax_cp.plot(cp_data[8:23,0], sq_model0(cp_data[8:23,0]), label=f"quadratic regression", marker = "o")
#ax_cp.plot(x22_ex, sq_model2(x22_ex), label=f"quadratic regression", marker = "o")
#ax_cp.plot(x22_ex, y_spl(x22_ex), label=f"quadratic regression", marker = "o")
#ax_cp.plot(x22_ex[:75], y_spl_2d(x22_ex[:75]), label=f"quadratic regression", marker = "o")
'''ax_cp.plot(cp_data[10:24,0], sq_model1(cp_data[10:24,0]), label=f"quadratic regression", marker = "o")'''

'''ax_cp.plot(location("separation"), a1*location("separation")+b1, markersize = 10, color='black', marker="o")
ax_cp.plot(location("transition"), a1*location("transition")+b1, markersize = 10, color='black', marker="o")
ax_cp.plot(location("reattachment"), a2*location("reattachment")+b2, markersize = 10, color='black', marker="o")'''
ax_cp.axvline(location("separation"), c='black', ls = '--', linewidth = 1)
ax_cp.axvline(location("transition"), c='black', ls = '--', linewidth = 1)
ax_cp.axvline(location("reattachment"), c='black', ls = '--', linewidth = 1)

#location derivatives

ax_cp.axvline(min_der(sq_model_cheb_2nd(cheb_g))[0], c='black', ls = ':', linewidth = 0.8)
ax_cp.axvline(min_der(sq_model_cheb_2nd(cheb_g))[1], c='black', ls = ':', linewidth = 0.8)
'''ax_cp.annotate('separation', xy=(location("separation"), a1*location("separation")+b1), xytext=(0.4,-0.6), arrowprops=dict(facecolor='black', width = 1, headwidth = 5, shrink=0.05))
ax_cp.annotate('transition', xy=(location("transition"), a1*location("transition")+b1), xytext=(0.75,-1.1), arrowprops=dict(facecolor='black', width=1, headwidth = 5, shrink=0.05))
ax_cp.annotate('reattachment', xy=(location("reattachment"), a2*location("reattachment")+b2), xytext=(0.78,-0.6), arrowprops=dict(facecolor='black', width=1, headwidth = 5, shrink=0.05))'''

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
