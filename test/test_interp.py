#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re
import sys
import torch as t
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import IO.pyanitools as pya
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
        pass
