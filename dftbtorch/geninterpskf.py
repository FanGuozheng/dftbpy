# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import os.path
import numpy as np
import torch as t
from torch.autograd import Variable
import scipy.interpolate
import matplotlib.pyplot as plt

intergraltyperef = {'[2, 2, 0, 0]': 0, '[2, 2, 1, 0]': 1, '[2, 2, 2, 0]': 2,
                    '[1, 2, 0, 0]': 3, '[1, 2, 1, 0]': 4, '[1, 1, 0, 0]': 5,
                    '[1, 1, 1, 0]': 6, '[0, 2, 0, 0]': 7, '[0, 1, 0, 0]': 8,
                    '[0, 0, 0, 0]': 9, '[2, 2, 0, 1]': 10, '[2, 2, 1, 1]': 11,
                    '[2, 2, 2, 1]': 12, '[1, 2, 0, 1]': 13, '[1, 2, 1, 1]': 14,
                    '[1, 1, 0, 1]': 15, '[1, 1, 1, 1]': 16, '[0, 2, 0, 1]': 17,
                    '[0, 1, 0, 1]': 18, '[0, 0, 0, 1]': 19}
ATOM_NUM = {"H": 1, "C": 6, "N": 7, "O": 8}


class BicubInterp:

    def __init__(self):
        '''
        This class aims to get interpolation with two variables
        https://en.wikipedia.org/wiki/Bicubic_interpolation
        '''
        pass

    def bicubic_2d(self, xmesh, ymesh, zmesh, xi, yi):
        '''
        famt will be the value of grid point and its derivative:
            [[f(0, 0),  f(0, 1),   f_y(0, 0),  f_y(0, 1)],
            [f(1, 0),   f(1, 1),   f_y(1, 0),  f_y(1, 1)],
            [f_x(0, 0), f_x(0, 1), f_xy(0, 0), f_xy(0, 1)],
            [f_x(1, 0), f_x(1, 1), f_xy(1, 0), f_xy(1, 1)]]
        a_mat = coeff * famt * coeff_
        therefore, this function returns:
            p(x, y) = [1, x, x**2, x**3] * a_mat * [1, y, y**2, y**3].T
        Args:
            xmesh, ymesh: x (1D) and y (1D)
            zmesh: z (2D)
            ix, iy: the interpolation point
        '''
        coeff = t.Tensor([[1, 0, 0, 0],
                          [0, 0, 1, 0],
                          [-3, 3, -2, -1],
                          [2, -2, 1, 1]])
        coeff_ = t.Tensor([[1, 0, -3, 2],
                           [0, 0, 3, -2],
                           [0, 1, -2, 1],
                           [0, 0, -1, 1]])
        fmat = t.zeros(4, 4)

        # get the indices of xi and yi u=in xmesh and ymesh
        if xi in xmesh:
            self.nxi = np.searchsorted(xmesh.detach().numpy(),
                                       xi.detach().numpy())
        else:
            self.nxi = np.searchsorted(xmesh.detach().numpy(),
                                       xi.detach().numpy()) - 1
        if yi in ymesh:
            self.nyi = np.searchsorted(ymesh.detach().numpy(),
                                       yi.detach().numpy())
        else:
            self.nyi = np.searchsorted(ymesh.detach().numpy(),
                                       yi.detach().numpy()) - 1

        # this is to transfer x or y to fraction way
        try:
            if xmesh[0] <= xi < xmesh[-1]:
                xi_ = (xi - xmesh[self.nxi]) / \
                    (xmesh[self.nxi + 1] - xmesh[self.nxi])
        except ValueError:
            print('x is out of grid point range, x0 < x < xn)')
        try:
            if ymesh[0] <= yi < ymesh[-1]:
                yi_ = (yi - ymesh[self.nyi]) / \
                    (ymesh[self.nyi + 1] - ymesh[self.nyi])
        except ValueError:
            print('y is out of grid point range, y0 < y < yn)')

        # build [1, x, x**2, x**3] and [1, y, y**2, y**3] matrices
        xmat, ymat = t.zeros(4), t.zeros(4)
        xmat[0], xmat[1], xmat[2], xmat[3] = 1, xi_, xi_ * xi_, xi_ * xi_ * xi_
        ymat[0], ymat[1], ymat[2], ymat[3] = 1, yi_, yi_ * yi_, yi_ * yi_ * yi_

        self.fmat_0(fmat, zmesh)
        self.fmat_1(fmat, zmesh, xmesh, ymesh, 'x')
        self.fmat_1(fmat, zmesh, xmesh, ymesh, 'y')
        self.fmat_xy(fmat)
        amat = t.mm(t.mm(coeff, fmat), coeff_)
        return t.matmul(t.matmul(xmat, amat), ymat)

    def fmat_0(self, fmat, zmesh):
        '''this function will construct f(0/1, 0/1) in fmat'''
        f00 = zmesh[self.nxi, self.nyi]
        f10 = zmesh[self.nxi + 1, self.nyi]
        f01 = zmesh[self.nxi, self.nyi + 1]
        f11 = zmesh[self.nxi + 1, self.nyi + 1]
        fmat[0, 0], fmat[1, 0], fmat[0, 1], fmat[1, 1] = f00, f10, f01, f11
        return fmat

    def fmat_1(self, fmat, zmesh, xmesh, ymesh, ty):
        '''this function will construct fx(0/1, 0) or fy(0, 0/1) in fmat'''
        x10 = xmesh[self.nxi + 1] - xmesh[self.nxi]
        y10 = ymesh[self.nyi + 1] - ymesh[self.nyi]

        if ty == 'x':
            if self.nxi + 1 == 1:
                x21 = xmesh[self.nxi + 2] - xmesh[self.nxi + 1]
                z1000, z2010 = \
                    self.get_diff_bound(zmesh, self.nxi, self.nyi, 'begx')
                z1101, z2111 = \
                    self.get_diff_bound(zmesh, self.nxi, self.nyi + 1, 'begx')
                fmat[2, 0], fmat[3, 0] = z1000 / x10, \
                    (z1000 + z2010) / (x10 + x21)
                fmat[2, 1], fmat[3, 1] = z1101 / x10, \
                    (z1101 + z2111) / (x10 + x21)
            elif 1 < self.nxi + 1 < len(xmesh) - 1:
                x0_1 = xmesh[self.nxi] - xmesh[self.nxi - 1]
                x21 = xmesh[self.nxi + 2] - xmesh[self.nxi + 1]
                z2010, z1000, z00_10 = \
                    self.get_diff_bound(zmesh, self.nxi, self.nyi, 'x')
                z2111, z1101, z01_11 = \
                    self.get_diff_bound(zmesh, self.nxi, self.nyi + 1, 'x')
                fmat[2, 0], fmat[3, 0] = (z1000 + z00_10) / (x10 + x0_1), \
                    (z1000 + z2010) / (x10 + x21)
                fmat[2, 1], fmat[3, 1] = (z1101 + z01_11) / (x10 + x0_1), \
                    (z1101 + z2111) / (x10 + x21)
            else:
                x0_1 = xmesh[self.nxi] - xmesh[self.nxi - 1]
                z1000, z00_10 = \
                    self.get_diff_bound(zmesh, self.nxi, self.nyi, 'endx')
                z1101, z01_11 = \
                    self.get_diff_bound(zmesh, self.nxi, self.nyi + 1, 'endx')
                fmat[2, 1], fmat[3, 1] = (z1101 + z01_11) / (x10 + x0_1), \
                    z1101 / x10
        elif ty == 'y':
            if self.nyi + 1 == 1:
                y21 = ymesh[self.nyi + 2] - ymesh[self.nyi + 1]
                z0100, z0201 = \
                    self.get_diff_bound(zmesh, self.nxi, self.nyi, 'begy')
                z1110, z1211 = \
                    self.get_diff_bound(zmesh, self.nxi + 1, self.nyi, 'begy')
                fmat[0, 2], fmat[0, 3] = z0100 / y10, \
                    (z0100 + z0201) / (y10 + y21)
                fmat[1, 2], fmat[1, 3] = z1110 / y10, \
                    (z1110 + z1211) / (y10 + y21)
            elif 1 < self.nyi + 1 < len(ymesh) - 1:
                y0_1 = xmesh[self.nyi] - xmesh[self.nyi - 1]
                y21 = xmesh[self.nyi + 2] - xmesh[self.nyi + 1]
                z0201, z0100, z000_1 = \
                    self.get_diff_bound(zmesh, self.nxi, self.nyi, 'y')
                z1211, z1110, z101_1 = \
                    self.get_diff_bound(zmesh, self.nxi + 1, self.nyi, 'y')
                fmat[0, 2], fmat[0, 3] = (z0100 + z000_1) / (y10 + y0_1), \
                    (z0100 + z0201) / (y10 + y21)
                fmat[1, 2], fmat[1, 3] = (z1110 + z101_1) / (y10 + y0_1), \
                    (z1110 + z0201) / (y10 + y21)
            else:
                y0_1 = ymesh[self.nyi] - ymesh[self.nyi - 1]
                z0100, z000_1 = \
                    self.get_diff_bound(zmesh, self.nxi, self.nyi, 'endy')
                z1110, z101_1 = \
                    self.get_diff_bound(zmesh, self.nxi + 1, self.nyi, 'endy')
                fmat[1, 2], fmat[1, 3] = (z1110 + z101_1) / (y10 + y0_1), \
                    z1110 / y10
        return fmat

    def fmat_xy(self, fmat):
        '''this function will construct f(0/1, 0/1) in fmat'''
        fmat[2, 2] = fmat[0, 2] * fmat[2, 0]
        fmat[3, 2] = fmat[3, 0] * fmat[1, 2]
        fmat[2, 3] = fmat[2, 1] * fmat[0, 3]
        fmat[3, 3] = fmat[3, 1] * fmat[1, 3]
        return fmat

    def get_diff(self, mesh, nxi=None, nyi=None, ty=None):
        '''
        this function will get derivative over x and y direction
        e.g, 10_00 means difference between (1, 0) and (0, 0)
        '''
        if nxi is not None and nyi is not None:
            z1000 = mesh[nxi + 1, nyi] - mesh[nxi, nyi]
            z00_10 = mesh[nxi, nyi] - mesh[nxi - 1, nyi]
            z0100 = mesh[nxi, nyi + 1] - mesh[nxi, nyi]
            z00_01 = mesh[nxi, nyi] - mesh[nxi, nyi - 1]
            return z1000, z00_10, z0100, z00_01
        if nxi is not None:
            x10 = mesh[nxi + 1] - mesh[nxi]
            x0_1 = mesh[nxi] - mesh[nxi - 1]
            return x10, x0_1
        if nyi is not None:
            y10 = mesh[nyi + 1] - mesh[nyi]
            y0_1 = mesh[nyi] - mesh[nyi - 1]
            return y10, y0_1

    def get_diff_bound(self, mesh, nxi, nyi, ty):
        '''
        this function will get derivative over x and y direction in boundary
        e.g, 10_00 means difference between (1, 0) and (0, 0)
        '''
        if ty == 'begx':
            z1000 = mesh[nxi + 1, nyi] - mesh[nxi, nyi]
            z2010 = mesh[nxi + 2, nyi] - mesh[nxi + 1, nyi]
            return z1000, z2010
        elif ty == 'x':
            z2010 = mesh[nxi + 2, nyi] - mesh[nxi + 1, nyi]
            z1000 = mesh[nxi + 1, nyi] - mesh[nxi, nyi]
            z00_10 = mesh[nxi, nyi] - mesh[nxi - 1, nyi]
            return z2010, z1000, z00_10
        elif ty == 'endx':
            z1000 = mesh[nxi + 1, nyi] - mesh[nxi, nyi]
            z00_10 = mesh[nxi, nyi] - mesh[nxi - 1, nyi]
            return z1000, z00_10
        elif ty == 'begy':
            z0100 = mesh[nxi, nyi + 1] - mesh[nxi, nyi]
            z0201 = mesh[nxi, nyi + 2] - mesh[nxi, nyi + 1]
            return z0100, z0201
        elif ty == 'y':
            z0201 = mesh[nxi, nyi + 2] - mesh[nxi, nyi + 1]
            z0100 = mesh[nxi, nyi + 1] - mesh[nxi, nyi]
            z000_1 = mesh[nxi, nyi] - mesh[nxi, nyi - 1]
            return z0201, z0100, z000_1
        elif ty == 'endy':
            z0100 = mesh[nxi, nyi + 1] - mesh[nxi, nyi]
            z000_1 = mesh[nxi, nyi] - mesh[nxi, nyi - 1]
            return z0100, z000_1
        elif ty == 'xy':
            z1000 = mesh[nxi + 1, nyi] - mesh[nxi, nyi]
            z00_10 = mesh[nxi, nyi] - mesh[nxi - 1, nyi]
            z0100 = mesh[nxi, nyi + 1] - mesh[nxi, nyi]
            z00_01 = mesh[nxi, nyi] - mesh[nxi, nyi - 1]
            return z1000, z00_10, z0100, z00_01

    def test(self):
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
                znew[ix, jy] = self.bicubic_2d(xmesh, ymesh, zmesh,
                                               xuniform[ix], yuniform[jy])
                znew_[ix, jy] = self.bicubic_2d(xmesh_, ymesh_, zmesh,
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

    def test_ml(self):
        '''xmesh = Variable(t.Tensor([1.9, 2., 2.3, 2.7, 3.3, 4.0, 4.1]),
                         requires_grad=True)
        ymesh = Variable(t.Tensor([1.9, 2., 2.3, 2.7, 3.3, 4.0, 4.1]),
                         requires_grad=True)'''
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
            ypred = self.bicubic_2d(xmesh, ymesh, zmesh, xin[0], xin[1])
            criterion = t.nn.MSELoss(reduction='sum')
            loss = criterion(ypred, yref)
            optimizer.zero_grad()
            print('ypred',  ypred, 'xin', xin)
            loss.backward(retain_graph=True)
            optimizer.step()
            print('loss', loss)

    def bicubic_3d(self, xmesh, ymesh, zmesh, xi, yi):
        '''
        famt will be the value of grid point and its derivative:
            [[f(0, 0),  f(0, 1),   f_y(0, 0),  f_y(0, 1)],
            [f(1, 0),   f(1, 1),   f_y(1, 0),  f_y(1, 1)],
            [f_x(0, 0), f_x(0, 1), f_xy(0, 0), f_xy(0, 1)],
            [f_x(1, 0), f_x(1, 1), f_xy(1, 0), f_xy(1, 1)]]
        a_mat = coeff * famt * coeff_
        therefore, this function returns:
            p(x, y) = [1, x, x**2, x**3] * a_mat * [1, y, y**2, y**3].T
        Args:
            xmesh, ymesh: x (1D) and y (1D)
            zmesh: z (3D)
            ix, iy: the interpolation point
        '''
        coeff11 = t.Tensor([[1, 0, 0, 0],
                            [0, 0, 1, 0],
                            [-3, 3, -2, -1],
                            [2, -2, 1, 1]])
        coeff11_ = t.Tensor([[1, 0, -3, 2],
                             [0, 0, 3, -2],
                             [0, 1, -2, 1],
                             [0, 0, -1, 1]])
        [nn1, nn2, nn3] = zmesh.shape
        coeff, coeff_ = t.zeros(4, 4, nn3), t.zeros(4, 4, nn3)
        fmat = t.zeros(4, 4, nn3)
        for ii in  range(nn3):
            coeff[:, :, ii] = coeff11[:, :]
            coeff_[:, :, ii] = coeff11_[:, :]

        # get the indices of xi and yi u=in xmesh and ymesh
        if xi in xmesh:
            self.nxi = np.searchsorted(xmesh.detach().numpy(),
                                       xi.detach().numpy())
        else:
            self.nxi = np.searchsorted(xmesh.detach().numpy(),
                                       xi.detach().numpy()) - 1
        if yi in ymesh:
            self.nyi = np.searchsorted(ymesh.detach().numpy(),
                                       yi.detach().numpy())
        else:
            self.nyi = np.searchsorted(ymesh.detach().numpy(),
                                       yi.detach().numpy()) - 1

        # this is to transfer x or y to fraction way
        try:
            if xmesh[0] <= xi < xmesh[-1]:
                xi_ = (xi - xmesh[self.nxi]) / \
                    (xmesh[self.nxi + 1] - xmesh[self.nxi])
        except ValueError:
            print('x is out of grid point range, x0 < x < xn)')
        try:
            if ymesh[0] <= yi < ymesh[-1]:
                yi_ = (yi - ymesh[self.nyi]) / \
                    (ymesh[self.nyi + 1] - ymesh[self.nyi])
        except ValueError:
            print('y is out of grid point range, y0 < y < yn)')

        # build [1, x, x**2, x**3] and [1, y, y**2, y**3] matrices
        xmat, ymat = t.zeros(4), t.zeros(4)
        xmat[0], xmat[1], xmat[2], xmat[3] = 1, xi_, xi_ * xi_, xi_ * xi_ * xi_
        ymat[0], ymat[1], ymat[2], ymat[3] = 1, yi_, yi_ * yi_, yi_ * yi_ * yi_

        self.fmat_03d(fmat, zmesh)
        self.fmat_13d(fmat, zmesh, xmesh, ymesh, 'x')
        self.fmat_13d(fmat, zmesh, xmesh, ymesh, 'y')
        self.fmat_xy3d(fmat)
        amat = t.mm(t.mm(coeff, fmat), coeff_)
        return t.matmul(t.matmul(xmat, amat), ymat)

    def fmat_03d(self, fmat, zmesh):
        '''this function will construct f(0/1, 0/1) in fmat'''
        f00 = zmesh[self.nxi, self.nyi, :]
        f10 = zmesh[self.nxi + 1, self.nyi, :]
        f01 = zmesh[self.nxi, self.nyi + 1, :]
        f11 = zmesh[self.nxi + 1, self.nyi + 1, :]
        fmat[0, 0, :], fmat[1, 0, :], fmat[0, 1, :], fmat[1, 1, :] = \
            f00, f10, f01, f11
        return fmat

    def fmat_13d(self, fmat, zmesh, xmesh, ymesh, ty):
        '''this function will construct fx(0/1, 0) or fy(0, 0/1) in fmat'''
        x10 = xmesh[self.nxi + 1] - xmesh[self.nxi]
        y10 = ymesh[self.nyi + 1] - ymesh[self.nyi]

        if ty == 'x':
            if self.nxi + 1 == 1:
                x21 = xmesh[self.nxi + 2] - xmesh[self.nxi + 1]
                z1000, z2010 = \
                    self.get_diff_bound3d(zmesh, self.nxi, self.nyi, 'begx')
                z1101, z2111 = \
                    self.get_diff_bound3d(zmesh, self.nxi, self.nyi + 1, 'begx')
                fmat[2, 0], fmat[3, 0] = z1000 / x10, \
                    (z1000 + z2010) / (x10 + x21)
                fmat[2, 1], fmat[3, 1] = z1101 / x10, \
                    (z1101 + z2111) / (x10 + x21)
            elif 1 < self.nxi + 1 < len(xmesh) - 1:
                x0_1 = xmesh[self.nxi] - xmesh[self.nxi - 1]
                x21 = xmesh[self.nxi + 2] - xmesh[self.nxi + 1]
                z2010, z1000, z00_10 = \
                    self.get_diff_bound3d(zmesh, self.nxi, self.nyi, 'x')
                z2111, z1101, z01_11 = \
                    self.get_diff_bound3d(zmesh, self.nxi, self.nyi + 1, 'x')
                fmat[2, 0], fmat[3, 0] = (z1000 + z00_10) / (x10 + x0_1), \
                    (z1000 + z2010) / (x10 + x21)
                fmat[2, 1], fmat[3, 1] = (z1101 + z01_11) / (x10 + x0_1), \
                    (z1101 + z2111) / (x10 + x21)
            else:
                x0_1 = xmesh[self.nxi] - xmesh[self.nxi - 1]
                z1000, z00_10 = \
                    self.get_diff_bound3d(zmesh, self.nxi, self.nyi, 'endx')
                z1101, z01_11 = \
                    self.get_diff_bound3d(zmesh, self.nxi, self.nyi + 1, 'endx')
                fmat[2, 1], fmat[3, 1] = (z1101 + z01_11) / (x10 + x0_1), \
                    z1101 / x10
        elif ty == 'y':
            if self.nyi + 1 == 1:
                y21 = ymesh[self.nyi + 2] - ymesh[self.nyi + 1]
                z0100, z0201 = \
                    self.get_diff_bound3d(zmesh, self.nxi, self.nyi, 'begy')
                z1110, z1211 = \
                    self.get_diff_bound3d(zmesh, self.nxi + 1, self.nyi, 'begy')
                fmat[0, 2], fmat[0, 3] = z0100 / y10, \
                    (z0100 + z0201) / (y10 + y21)
                fmat[1, 2], fmat[1, 3] = z1110 / y10, \
                    (z1110 + z1211) / (y10 + y21)
            elif 1 < self.nyi + 1 < len(ymesh) - 1:
                y0_1 = xmesh[self.nyi] - xmesh[self.nyi - 1]
                y21 = xmesh[self.nyi + 2] - xmesh[self.nyi + 1]
                z0201, z0100, z000_1 = \
                    self.get_diff_bound3d(zmesh, self.nxi, self.nyi, 'y')
                z1211, z1110, z101_1 = \
                    self.get_diff_bound3d(zmesh, self.nxi + 1, self.nyi, 'y')
                fmat[0, 2], fmat[0, 3] = (z0100 + z000_1) / (y10 + y0_1), \
                    (z0100 + z0201) / (y10 + y21)
                fmat[1, 2], fmat[1, 3] = (z1110 + z101_1) / (y10 + y0_1), \
                    (z1110 + z0201) / (y10 + y21)
            else:
                y0_1 = ymesh[self.nyi] - ymesh[self.nyi - 1]
                z0100, z000_1 = \
                    self.get_diff_bound3d(zmesh, self.nxi, self.nyi, 'endy')
                z1110, z101_1 = \
                    self.get_diff_bound3d(zmesh, self.nxi + 1, self.nyi, 'endy')
                fmat[1, 2], fmat[1, 3] = (z1110 + z101_1) / (y10 + y0_1), \
                    z1110 / y10
        return fmat

    def fmat_xy3d(self, fmat):
        '''this function will construct f(0/1, 0/1) in fmat'''
        fmat[2, 2] = fmat[0, 2] * fmat[2, 0]
        fmat[3, 2] = fmat[3, 0] * fmat[1, 2]
        fmat[2, 3] = fmat[2, 1] * fmat[0, 3]
        fmat[3, 3] = fmat[3, 1] * fmat[1, 3]
        return fmat

    def get_diff3d(self, mesh, nxi=None, nyi=None, ty=None):
        '''
        this function will get derivative over x and y direction
        e.g, 10_00 means difference between (1, 0) and (0, 0)
        '''
        if nxi is not None and nyi is not None:
            z1000 = mesh[nxi + 1, nyi] - mesh[nxi, nyi]
            z00_10 = mesh[nxi, nyi] - mesh[nxi - 1, nyi]
            z0100 = mesh[nxi, nyi + 1] - mesh[nxi, nyi]
            z00_01 = mesh[nxi, nyi] - mesh[nxi, nyi - 1]
            return z1000, z00_10, z0100, z00_01
        if nxi is not None:
            x10 = mesh[nxi + 1] - mesh[nxi]
            x0_1 = mesh[nxi] - mesh[nxi - 1]
            return x10, x0_1
        if nyi is not None:
            y10 = mesh[nyi + 1] - mesh[nyi]
            y0_1 = mesh[nyi] - mesh[nyi - 1]
            return y10, y0_1

    def get_diff_bound3d(self, mesh, nxi, nyi, ty):
        '''
        this function will get derivative over x and y direction in boundary
        e.g, 10_00 means difference between (1, 0) and (0, 0)
        '''
        if ty == 'begx':
            z1000 = mesh[nxi + 1, nyi] - mesh[nxi, nyi]
            z2010 = mesh[nxi + 2, nyi] - mesh[nxi + 1, nyi]
            return z1000, z2010
        elif ty == 'x':
            z2010 = mesh[nxi + 2, nyi] - mesh[nxi + 1, nyi]
            z1000 = mesh[nxi + 1, nyi] - mesh[nxi, nyi]
            z00_10 = mesh[nxi, nyi] - mesh[nxi - 1, nyi]
            return z2010, z1000, z00_10
        elif ty == 'endx':
            z1000 = mesh[nxi + 1, nyi] - mesh[nxi, nyi]
            z00_10 = mesh[nxi, nyi] - mesh[nxi - 1, nyi]
            return z1000, z00_10
        elif ty == 'begy':
            z0100 = mesh[nxi, nyi + 1] - mesh[nxi, nyi]
            z0201 = mesh[nxi, nyi + 2] - mesh[nxi, nyi + 1]
            return z0100, z0201
        elif ty == 'y':
            z0201 = mesh[nxi, nyi + 2] - mesh[nxi, nyi + 1]
            z0100 = mesh[nxi, nyi + 1] - mesh[nxi, nyi]
            z000_1 = mesh[nxi, nyi] - mesh[nxi, nyi - 1]
            return z0201, z0100, z000_1
        elif ty == 'endy':
            z0100 = mesh[nxi, nyi + 1] - mesh[nxi, nyi]
            z000_1 = mesh[nxi, nyi] - mesh[nxi, nyi - 1]
            return z0100, z000_1
        elif ty == 'xy':
            z1000 = mesh[nxi + 1, nyi] - mesh[nxi, nyi]
            z00_10 = mesh[nxi, nyi] - mesh[nxi - 1, nyi]
            z0100 = mesh[nxi, nyi + 1] - mesh[nxi, nyi]
            z00_01 = mesh[nxi, nyi] - mesh[nxi, nyi - 1]
            return z1000, z00_10, z0100, z00_01


class SkInterpolator:
    """This code aims to generate integrals by interpolation method.
    Therefore, the inputs include the grid points of the integrals,
    the compression radius of atom1 (r1) and atom2 (r2)
    """
    def __init__(self, para, gridmesh):
        self.para = para
        self.gridmesh = gridmesh

    def readskffile(self, namei, namej, directory):
        """
        Input:
            all .skf file, atom names, directory
        Output:
            gridmesh_points, onsite_spe_u, mass_rcut, integrals
        Namestyle:
            C-H.skf.02.77.03.34, compr of C, H are 2.77 and 3.34, respectively
        """
        nameij = namei + namej
        filenamelist = self.getfilenamelist(namei, namej, directory)
        nfile, ncompr = len(filenamelist), int(np.sqrt(len(filenamelist)))

        ngridpoint, grid_dist = t.empty(nfile), t.empty(nfile)
        onsite, spe, uhubb, occ_skf = t.empty(nfile, 3), t.empty(nfile), \
            t.empty(nfile, 3), t.empty(nfile, 3)
        mass_rcut = t.empty(nfile, 20)
        integrals, atomname_filename, self.para['rest'] = [], [], []

        icount = 0
        for filename in filenamelist:
            fp = open(os.path.join(directory, filename), 'r')
            words = fp.readline().split()
            grid_dist[icount] = float(words[0])
            ngridpoint[icount] = int(words[1])
            nitem = int(ngridpoint[icount] * 20)
            atomname_filename.append((filename.split('.')[0]).split("-"))
            split = filename.split('.')

            if [namei, split[-1], split[-2]] == [namej, split[-3], split[-4]]:
                fp_line2 = [float(ii) for ii in fp.readline().split()]
                fp_line2_ = t.from_numpy(np.asarray(fp_line2))
                onsite[icount, :] = fp_line2_[0:3]
                spe[icount] = fp_line2_[3]
                uhubb[icount, :] = fp_line2_[4:7]
                occ_skf[icount, :] = fp_line2_[7:10]
                data = np.fromfile(fp, dtype=float, count=20, sep=' ')
                mass_rcut[icount, :] = t.from_numpy(data)
                data = np.fromfile(fp, dtype=float, count=nitem, sep=' ')
                data.shape = (int(ngridpoint[icount]), 20)
                integrals.append(data)
                self.para['rest'].append(fp.read())
            else:
                data = np.fromfile(fp, dtype=float, count=20, sep=' ')
                mass_rcut[icount, :] = t.from_numpy(data)
                data = np.fromfile(fp, dtype=float, count=nitem, sep=' ')
                data.shape = (int(ngridpoint[icount]), 20)
                integrals.append(data)
                self.para['rest'].append(fp.read())
            icount += 1

        if self.para['Lrepulsive']:
            fp = open(os.path.join(directory, namei + '-' + namej + '.rep'), "r")
            first_line = fp.readline().split()
            assert 'Spline' in first_line
            nInt_cutoff = fp.readline().split()
            nint_ = int(nInt_cutoff[0])
            self.para['nint_rep' + nameij] = nint_
            self.para['cutoff_rep' + nameij] = float(nInt_cutoff[1])
            a123 = fp.readline().split()
            self.para['a1_rep' + nameij] = float(a123[0])
            self.para['a2_rep' + nameij] = float(a123[1])
            self.para['a3_rep' + nameij] = float(a123[2])
            datarep = np.fromfile(fp, dtype=float,
                                  count=(nint_-1)*6, sep=' ')
            datarep.shape = (nint_ - 1, 6)
            self.para['rep' + nameij] = t.from_numpy(datarep)
            datarepend = np.fromfile(fp, dtype=float, count=8, sep=' ')
            self.para['repend' + nameij] = t.from_numpy(datarepend)

        self.para['skf_line_tail' + nameij] = int(max(ngridpoint) + 5)
        superskf = t.zeros(ncompr, ncompr,
                           self.para['skf_line_tail' + nameij], 20)
        mass_rcut_ = t.zeros(ncompr, ncompr, 20)
        onsite_ = t.zeros(ncompr, ncompr, 3)
        spe_ = t.zeros(ncompr, ncompr)
        uhubb_ = t.zeros(ncompr, ncompr, 3)
        occ_skf_ = t.zeros(ncompr, ncompr, 3)
        ngridpoint_ = t.zeros(ncompr, ncompr)
        grid_dist_ = t.zeros(ncompr, ncompr)

        # transfer 1D [nfile, n] to 2D [ncompr, ncompr, n]
        for skfi in range(0, nfile):
            rowi = int(skfi // ncompr)
            colj = int(skfi % ncompr)
            ingridpoint = int(ngridpoint[skfi])
            superskf[rowi, colj, :ingridpoint, :] = \
                t.from_numpy(integrals[skfi])
            grid_dist_[rowi, colj] = grid_dist[skfi]
            ngridpoint_[rowi, colj] = ngridpoint[skfi]
            mass_rcut_[rowi, colj, :] = mass_rcut[skfi, :]
            onsite_[rowi, colj, :] = onsite[skfi, :]
            spe_[rowi, colj] = spe[skfi]
            uhubb_[rowi, colj, :] = uhubb[skfi, :]
            occ_skf_[rowi, colj, :] = occ_skf[skfi, :]
        self.para['massrcut_rall' + nameij] = mass_rcut_
        self.para['onsite_rall' + nameij] = onsite_
        self.para['spe_rall' + nameij] = spe_
        self.para['uhubb_rall' + nameij] = uhubb_
        self.para['occ_skf_rall' + nameij] = occ_skf_
        self.para['nfile_rall' + nameij] = nfile
        self.para['grid_dist_rall' + nameij] = grid_dist_
        self.para['ngridpoint_rall' + nameij] = ngridpoint_
        self.para['hs_all_rall' + nameij] = superskf
        self.para['atomnameInSkf' + nameij] = atomname_filename

    def getfilenamelist(self, namei, namej, directory):
        """read all the skf files and return lists of skf files according to
        the types of skf """
        filename = namei + '-' + namej + '.skf.'
        filenamelist = []
        filenames = os.listdir(directory)
        filenames.sort()
        for name in filenames:
            if name.startswith(filename):
                filenamelist.append(name)
        return filenamelist

    def getallgenintegral(self, ninterpfile, skffile, r1, r2, gridarr1,
                          gridarr2):
        """this function is to generate the whole integrals"""
        superskf = skffile["intergrals"]
        nfile = skffile["nfilenamelist"]
        row = int(np.sqrt(nfile))
        xneigh = (np.abs(gridarr1 - r1)).argmin()
        yneigh = (np.abs(gridarr2 - r2)).argmin()
        ninterp = round(xneigh*row + yneigh)
        ninterpline = int(skffile["gridmeshpoint"][ninterp, 1])
        # print("ninterpline", ninterpline)
        hs_skf = np.empty((ninterpline+5, 20))
        for lineskf in range(0, ninterpline):
            distance = lineskf*self.gridmesh + self.grid0
            counti = 0
            for intergrali in intergraltyperef:
                znew3 = SkInterpolator.getintegral(self, r1, r2, intergrali,
                                                   distance, gridarr1,
                                                   gridarr2, superskf)
                hs_skf[lineskf, counti] = znew3
                counti += 1
        return hs_skf, ninterpline

    def getintegral(self, interpr1, interpr2, integraltype, distance,
                    gridarr1, gridarr2, superskf):
        """this function is to generate interpolation at given distance and
        given compression radius"""
        numgridpoints = len(gridarr1)
        numgridpoints2 = len(gridarr2)
        if numgridpoints != numgridpoints2:
            print('Error: the dimension is not equal')
        skftable = np.empty((numgridpoints, numgridpoints))
        numline = int((distance - self.grid0)/self.gridmesh)
        numtypeline = intergraltyperef[integraltype]
        skftable = superskf[:, :, numline, numtypeline]
        # print('skftable', skftable)
        funcubic = scipy.interpolate.interp2d(gridarr2, gridarr1, skftable,
                                              kind='cubic')
        interporbital = funcubic(interpr2, interpr1)
        return interporbital

    def polytozero(self, hs_skf, ninterpline):
        """Here, we fit the tail of skf file (5lines, 5th order)"""
        ni = ninterpline
        dx = self.gridmesh * 5
        ytail = hs_skf[ni - 1, :]
        ytailp = (hs_skf[ni - 1, :] - hs_skf[ni - 2, :]) / self.gridmesh
        ytailp2 = (hs_skf[ni - 2, :]-hs_skf[ni - 3, :]) / self.gridmesh
        ytailpp = (ytailp - ytailp2) / self.gridmesh
        xx = np.array([self.gridmesh * 4, self.gridmesh * 3, self.gridmesh * 2,
                       self.gridmesh, 0.0])
        nline = ninterpline
        for xxi in xx:
            dx1 = ytailp * dx
            dx2 = ytailpp * dx * dx
            dd = 10.0 * ytail - 4.0 * dx1 + 0.5 * dx2
            ee = -15.0 * ytail + 7.0 * dx1 - 1.0 * dx2
            ff = 6.0 * ytail - 3.0 * dx1 + 0.5 * dx2
            xr = xxi / dx
            yy = ((ff * xr + ee) * xr + dd) * xr * xr * xr
            hs_skf[nline, :] = yy
            nline += 1
        return hs_skf

    def saveskffile(self, ninterpfile, atomnameall, skffile, hs_skf,
                    ninterpline):
        """this function is to save all parts in skf file"""
        atomname1 = atomnameall[0]
        atomname2 = atomnameall[1]
        nfile = skffile["nfilenamelist"]
        if ninterpfile in (0, 3):
            print('generate {}-{}.skf'.format(atomname1, atomname2))
            with open('{}-{}.skf'.format(atomname1, atomname1), 'w') as fopen:
                fopen.write(str(skffile["gridmeshpoint"][nfile-1][0])+" ")
                fopen.write(str(int(ninterpline)))
                fopen.write('\n')
                np.savetxt(fopen, skffile["onsitespeu"], fmt="%s", newline=" ")
                fopen.write('\n')
                np.savetxt(fopen, skffile["massrcut"][nfile-1], newline=" ")
                fopen.write('\n')
                np.savetxt(fopen, hs_skf)
                fopen.write('\n')
                fopen.write(skffile["rest"])
        elif ninterpfile in (1, 2):
            print('generate {}-{}.skf'.format(atomname1, atomname2))
            with open('{}-{}.skf'.format(atomname1, atomname2), 'w') as fopen:
                fopen.write(str(skffile["gridmeshpoint"][nfile-1][0])+" ")
                fopen.write(str(int(ninterpline)))
                fopen.write('\n')
                np.savetxt(fopen, skffile["massrcut"][nfile-1], newline=" ")
                fopen.write('\n')
                np.savetxt(fopen, hs_skf)
                fopen.write('\n')
                fopen.write(skffile["rest"])
