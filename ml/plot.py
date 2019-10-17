#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import write_output as write
import pyanitools as pya

nfile = 50
diredftbpy = '/home/gz_fan/Documents/ML/dftb/ml/dftbpy'
diredftbplus = '/home/gz_fan/Documents/ML/dftb/ml/dftbplus'
directory2 = '/home/gz_fan/Documents/ML/prog_ml/ref'
directory3 = '/home/gz_fan/Documents/ML/prog_ml/dataset1'
directory4 = '/home/gz_fan/Documents/ML/prog_ml/dftb_mio'
hirshfeld_V_C = 38.37861207
hirshfeld_V_H = 10.31539447
fpplus_energy1 = open(os.path.join(diredftbplus, 'bandenergy1.dat'), 'r')
fpplus_energy2 = open(os.path.join(diredftbplus, 'bandenergy2.dat'), 'r')
fpaims = open(os.path.join(directory2, 'volall.dat'), 'r')
fppy_energy = open(os.path.join(diredftbpy, 'bandenergy.dat'), 'r')
fpdftbmio = open(os.path.join(directory4, 'onsiteall.dat'), 'r')
fppy_energy_data = []
try:
    for line in fppy_energy:
        num_line = line.strip().split()
        fppy_energy_data.append(num_line)
finally:
    fppy_energy.close()
dftbpy_energy1 = np.zeros(50)
dftbpy_energy2 = np.zeros(50)
for ifile in range(0, nfile):
    dftbpy_energy1[ifile] = fppy_energy_data[ifile][3]
    dftbpy_energy2[ifile] = fppy_energy_data[ifile][4]
dataenergy1 = np.fromfile(fpplus_energy1, dtype=float, count=50, sep=' ')
dataenergy2 = np.fromfile(fpplus_energy2, dtype=float, count=50, sep=' ')
print(fppy_energy_data[0][3])
x = np.linspace(1, 50, 50)
plt.plot(x, dataenergy1[:], color='r')
plt.plot(x, dftbpy_energy1[:])
plt.show()
plt.plot(x, dataenergy2[:], color='r')
plt.plot(x, dftbpy_energy2[:])
plt.show()
