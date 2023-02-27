import matplotlib.pyplot as plt
import numpy as np

from load_data import irt_data

fig_irt, ax_irt = plt.subplots()

xcount, zcount = 711, 343
irt_array = np.reshape(irt_data[:, 2], (zcount, xcount))
print(irt_array.shape)

im = plt.imshow(irt_array, cmap=plt.cm.RdBu, extent=(irt_data[0,0], irt_data[-1,0], irt_data[0,1], irt_data[-1,1]), interpolation="bilinear")
fig_irt.colorbar(im, ax=ax_irt)

plt.show()