#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re
import sys
import torch as t
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import utils.pyanitools as pya
from dftbtorch.geninterpskf import SkInterpolator
from dftbtorch.matht import(BicubInterp)
ATOMNUM = {'H': 1, 'C': 6, 'N': 7, 'O': 8}


def test_bicub_interp():
    """Test Bicubic interpolation."""
    bicinterp = BicubInterp()
    xmesh = t.Tensor([1.5, 1.6, 1.9, 2.4, 3.0, 3.7, 4.5])
    ymesh = t.Tensor([1.5, 1.6, 1.9, 2.4, 3.0, 3.7, 4.5])
    xmesh_ = t.Tensor([1.5, 2., 2.5, 3.0, 3.5, 4.0, 4.5])
    ymesh_ = t.Tensor([1.5, 2., 2.5, 3.0, 3.5, 4.0, 4.5])
    zmesh = t.Tensor([[.4, .45, .51, .57, .64, .72, .73],
                      [.45, .51, .58, .64, .73, .83, .85],
                      [.51, .58, .64, .73, .83, .94, .96],
                      [.57, .64, .73, .84, .97, 1.12, 1.14],
                      [.64, .72, .83, .97, 1.16, 1.38, 1.41],
                      [.72, .83, .94, 1.12, 1.38, 1.68, 1.71],
                      [.73, .85, .96, 1.14, 1.41, 1.71, 1.74]])
    xuniform = t.linspace(1.5, 4.49, 31)
    yuniform = t.linspace(1.5, 4.49, 31)
    x, y = np.meshgrid(xmesh, ymesh)
    xnew, ynew = np.meshgrid(xuniform, yuniform)
    x_, y_ = np.meshgrid(xmesh_, ymesh_)
    nx = len(xuniform)
    znew = t.empty(nx, nx)
    znew_ = t.empty(nx, nx)
    for ix in range(0, nx):
        for jy in range(0, nx):
            znew[ix, jy] = bicinterp.bicubic_2d(xmesh, ymesh, zmesh,
                                           xuniform[ix], yuniform[jy])
            znew_[ix, jy] = bicinterp.bicubic_2d(xmesh_, ymesh_, zmesh,
                                            xuniform[ix], yuniform[jy])
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.pcolormesh(x, y, zmesh)
    plt.title('original data', fontsize=15)
    plt.subplot(1, 2, 2)
    plt.pcolormesh(xnew, ynew, znew)
    plt.title('grid difference not same', fontsize=15)
    plt.show()
    plt.subplot(1, 2, 1)
    plt.pcolormesh(x_, y_, zmesh)
    plt.title('original data', fontsize=15)
    plt.subplot(1, 2, 2)
    plt.pcolormesh(xnew, ynew, znew_)
    plt.title('grid difference same', fontsize=15)
    plt.show()

def test_bicub_interp_ml():
    """Test gradients of bicubic method."""
    bicinterp = BicubInterp()
    xmesh = t.Tensor([1.9, 2., 2.3, 2.7, 3.3, 4.0, 4.1])
    ymesh = t.Tensor([1.9, 2., 2.3, 2.7, 3.3, 4.0, 4.1])
    # xin = t.Tensor([2, 3]).clone().detach().requires_grad_()
    t.enable_grad()
    xin = Variable(t.Tensor([4.05, 3]), requires_grad=True)
    zmesh = t.Tensor([[.4, .45, .51, .57, .64, .72, .73],
                      [.45, .51, .58, .64, .73, .83, .85],
                      [.51, .58, .64, .73, .83, .94, .96],
                      [.57, .64, .73, .84, .97, 1.12, 1.14],
                      [.64, .72, .83, .97, 1.16, 1.38, 1.41],
                      [.72, .83, .94, 1.12, 1.38, 1.68, 1.71],
                      [.73, .85, .96, 1.14, 1.41, 1.71, 1.74]])
    yref = t.Tensor([1.1])
    for istep in range(0, 10):
        optimizer = t.optim.SGD([xin, xmesh, ymesh], lr=1e-1)
        ypred = bicinterp.bicubic_2d(xmesh, ymesh, zmesh, xin[0], xin[1])
        criterion = t.nn.MSELoss(reduction='sum')
        loss = criterion(ypred, yref)
        optimizer.zero_grad()
        print('ypred',  ypred, 'xin', xin)
        loss.backward(retain_graph=True)
        optimizer.step()
        print('loss', loss)

def loadhdfdata(para):
    # ani_gdb_s01.h5: 1: CH4, 2: NH3, 3: H2O
    # ani_gdb_s02.h5: 1: C2H6, 2: CH5N, 3: N2 4: NH3O, 5: NHO, 6: CH4O, 7: C2H4
    #                 8: CH2O, 9: O2, 10: O2H2, 11: C2H2, 12: N2H4, 13: N2H2
    # ani_gdb_s03.h5: 1: C3H8, 2: C2NH7, 3: CH3ON, 4: CH2O2, 5: CH4N2, 6: CH3ON
    #                 7: C3H4, 8: C3H6, 9: C2H5N, 10: C2H4O, 11: NHO2,
    #                 12: O3H2, 13: C2OH6, 14: C2H7N, 15: C2H6O, 16: C3H6,
    #                 17: C2H4O, 18: CO2, 19: C2H3N, 20: CH4N2
    # ani_gdb_s04.h5: 1: C3H9N, 2: C3H8O, 3: CH2O3, 4: C4H10, 5: C3H9N
    #                 6: C3H8O, 7:C2H8N2 , 8: C2H7ON, 9: C2H6O2, 10: C3H9N
    #                 11: NHO2, 12: C4H8, 13: C3H9N, 14: C3H6O, 15: C3H5N
    #                 16: C3H7N, 17: C3H6O,18: C2H4O2, 19: C2H3NO, 20: C2H6N2
    para['hdf_num'] = 13
    hdffilelist = []
    filelist = os.listdir('data/an1')
    filelist.sort()
    [hdffilelist.append(os.path.join('data/an1', f))
        for f in filelist if re.match(r'ani+.*\.h5', f)]
    para['hdffile'] = hdffilelist

    ntype = para['hdf_num']
    hdf5filelist = para['hdffile']
    para['coorall'] = []
    nlist = 0
    for hdf5file in hdf5filelist:
        icount = 0
        nlist += 1
        adl = pya.anidataloader(hdf5file)
        print('nlist:', nlist, hdf5file)
        if nlist > 3:
            break
        for data in adl:
            icount += 1
            print(icount)
            if icount == ntype:
                for icoor in data['coordinates']:
                    row, col = np.shape(icoor)[0], np.shape(icoor)[1]
                    coor = t.zeros(row, col + 1)
                    for iat in range(0, len(data['species'])):
                        coor[iat, 0] = ATOMNUM[data['species'][iat]]
                        coor[iat, 1:] = t.from_numpy(icoor[iat, :])
                    para['coorall'].append(coor)
                symbols = data['species']
                specie = set(symbols)
                # speciedict = Counter(symbols)
                para['symbols'] = symbols
                para['specie'] = specie
                para['atomspecie'] = []
                [para['atomspecie'].append(ispecie) for ispecie in specie]
                # para['speciedict'] = speciedict
                print(icount, len(para['coorall']), data['species'])
                print(para['coorall'])


def interpskf(para):
    '''
    read .skf data from skgen with various compR
    '''
    print('** read skf file with all compR **')
    for namei in para['atomspecie']:
        for namej in para['atomspecie']:
            SkInterpolator(para, gridmesh=0.2).readskffile(
                    namei, namej, para['dire_interpSK'])


if __name__ == '__main__':
    para = {}
    para['task'] = 'bicub_interp_ml'
    if para['task'] == 'bicub_interp':
        test_bicub_interp()
    elif para['task'] == 'bicub_interp_ml':
        test_bicub_interp_ml()
    else:
        loadhdfdata(para)
