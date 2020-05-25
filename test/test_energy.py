#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import os
import matplotlib.pyplot as plt

DFTB_ENERGY = {"H": -0.238600544, "C": -1.398493891}
compr = np.array([2., 2.5, 3., 3.5, 4., 4.5, 5., 5.5, 6., 7., 8., 9., 10.])
ncompr = len(compr)
energyall = np.zeros((ncompr, ncompr))
dire = '/home/gz_fan/Downloads/software/dftbplus/dftbplus/work/CH4/test_skden'
fp = open(os.path.join(dire, 'energy.dat'), 'r')

for ir in range(0, ncompr):
    for jr in range(0, ncompr):
        energyall[ir, jr] = np.fromfile(fp, dtype=float, count=1, sep=' ')
        energyall[ir, jr] = energyall[ir, jr] + 1.398493891 + 0.238600544 * 4

ref = -40.49298993 - -37.77330663 - - 0.45891649 * 4  # FHI-aims
energyall = energyall - ref

X, Y = np.meshgrid(compr, compr[::-1])
extent = np.min(compr), np.max(compr), np.max(compr), np.min(compr)
# fig = plt.figure(frameon=False)
# Z1 = np.add.outer(range(8), range(8)) % 2  # chessboard
im1 = plt.imshow(energyall, interpolation='nearest',
                 extent=extent)
plt.colorbar(im1)
plt.xlabel('compression radius of C'), plt.ylabel('compression radius of H')
plt.title('DFTB+ and FHI-aims formation energy difference')
'''Z2 = func3(X, Y)

im2 = plt.imshow(Z2, cmap=plt.cm.viridis, alpha=.9, interpolation='bilinear',
                 extent=extent)'''
plt.show()


comprpot = np.array([2., 2.5, 3., 3.5, 4., 4.5, 5., 5.5, 6., 7., 8., 9., 10.])
ncomprpot = len(comprpot)
energyallpot = np.zeros((ncomprpot, ncomprpot))
dire = '/home/gz_fan/Downloads/software/dftbplus/dftbplus/work/CH4/test_skpot'
fppot = open(os.path.join(dire, 'energy.dat'), 'r')

for ir in range(0, ncomprpot):
    for jr in range(0, ncomprpot):
        energyallpot[ir, jr] = np.fromfile(fppot, dtype=float, count=1, sep=' ')
        energyallpot[ir, jr] = energyallpot[ir, jr] + 1.398493891 + 0.238600544 * 4

ref = -40.49298993 - -37.77330663 - - 0.45891649 * 4  # FHI-aims
energyallpot = energyallpot - ref

X, Y = np.meshgrid(comprpot, comprpot[::-1])
extent = np.min(comprpot), np.max(comprpot), np.max(comprpot), np.min(comprpot)
# fig = plt.figure(frameon=False)
# Z1 = np.add.outer(range(8), range(8)) % 2  # chessboard
impot = plt.imshow(energyallpot, interpolation='nearest',
                   extent=extent)
plt.colorbar(impot)
plt.xlabel('compression radius of C'), plt.ylabel('compression radius of H')
plt.title('DFTB+ and FHI-aims formation energy difference')
'''Z2 = func3(X, Y)

im2 = plt.imshow(Z2, cmap=plt.cm.viridis, alpha=.9, interpolation='bilinear',
                 extent=extent)'''
plt.show()
