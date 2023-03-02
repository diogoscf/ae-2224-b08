import numpy as np
import os

cp_file_path = os.path.join(os.path.dirname(__file__), "./Data/cp.dat")
irt_file_path = os.path.join(os.path.dirname(__file__), "./Data/IT.dat")
piv_file_path = os.path.join(os.path.dirname(__file__), "./Data/PIV.dat")

cp_data = np.genfromtxt(cp_file_path, skip_header=1, delimiter=",")
irt_data = np.genfromtxt(irt_file_path, skip_header=1, delimiter=",")
piv_data = np.genfromtxt(piv_file_path, skip_header=1, delimiter=",")

# print(irt_data)
