#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''this file will conclude mathematical solution needed in dfyb'''

import numpy as np
import torch
import scipy.interpolate
import bisect
import matplotlib.pyplot as plt
from torch.autograd import Variable


class DFTBmath(object):

    def __init__(self):
        '''
        this class aims to deal with interpolation of intergral tables, tail
        near the cutoff
        '''
        pass

    def polysk3thsk(self, allarr, darr, dd):
        '''
        this function is interpolation mathod with input 2D and output 1D
        '''
        row, col = np.shape(allarr)
        skftable = np.zeros(row)
        hs = np.zeros(col)
        for ii in range(0, col):
            skftable[:] = allarr[:, ii]
            fcubic = scipy.interpolate.interp1d(darr, skftable, kind='cubic')
            hs[ii] = fcubic(dd)
        return hs

    def polysk5thsk(self):
        pass


class Bspline():

    def __init__(self):
        '''
        this function is revised from:
        https://docs.scipy.org/doc/scipy-1.0.0/reference/generated/scipy.interpolate.BSpline.html
        '''
        pass

    def bspline(self, t, c, k, x):
        n = len(t) - k - 1
        assert (n >= k + 1) and (len(c) >= n)
        return sum(c[i] * self.B(t, k, x, i) for i in range(n))

    def B(self, t, k, x, i):
        if k == 0:
            return 1.0 if t[i] <= x < t[i+1] else 0.0
        if t[i+k] == t[i]:
            c1 = 0.0
        else:
            c1 = (x - t[i]) / (t[i+k] - t[i]) * self.B(t, k-1, x, i)
            if t[i + k + 1] == t[i + 1]:
                c2 = 0.0
            else:
                c2 = (t[i + k + 1] - x) / (t[i + k + 1] - t[i + 1]) * \
                    self.B(t, k - 1, x, i + 1)
        return c1 + c2

    def test(self):
        '''test function for Bspline'''
        tarr = [0, 1, 2, 3, 4, 5, 6]
        carr = [0, -2, -1.5, -1]
        fig, ax = plt.subplots()
        xx = np.linspace(1.5, 4.5, 50)
        ax.plot(xx, [self.bspline(tarr, carr, 2, ix) for ix in xx], 'r-',
                lw=3, label='naive')
        ax.grid(True)
        ax.legend(loc='best')
        plt.show()


class polySpline():
    '''
    This code revised from pycubicspline:
    https://github.com/AtsushiSakai/pycubicspline
    '''
    def __init__(self):
        pass

    def linear():
        pass

    def cubic(self, xp, yp, dd):
        self.xp = xp
        self.yp = yp
        self.dd = dd
        self.nx = self.xp.shape[0]
        self.linex = len(xp)
        self.diffx = diff(self.xp)
        if torch.is_tensor(xp):
            xnp = xp.numpy()
            self.ddind = bisect.bisect(xnp, dd) - 1
        elif type(xp) is np.ndarray:
            self.ddind = bisect.bisect(xp, dd) - 1
        if self.dd < self.xp[0] or self.dd > self.xp[-1]:
            return None
        b = Variable(torch.empty(self.nx - 1))
        d = Variable(torch.empty(self.nx - 1))
        A = self.cala()
        B = self.calb()
        c, _ = torch.lstsq(B, A)
        for i in range(self.nx - 1):
            tb = (self.yp[i + 1] - self.yp[i]) / self.diffx[i] - \
                self.diffx[i] * (c[i + 1] + 2.0 * c[i]) / 3.0
            b[i] = tb
            d[i] = (c[i + 1] - c[i]) / (3.0 * self.diffx[i])
        i = self.ddind
        dx = self.dd - self.xp[i]
        result = self.yp[i] + b[i] * dx + c[i] * dx ** 2.0 + d[i] * dx ** 3.0
        return result

    def cala(self):
        '''calculate a para in spline interpolation'''
        aarr = Variable(torch.zeros(self.nx, self.nx))
        aarr[0, 0] = 1.0
        for i in range(self.nx - 1):
            if i != (self.nx - 2):
                aarr[i + 1, i + 1] = 2.0 * (self.diffx[i] + self.diffx[i + 1])
            aarr[i + 1, i] = self.diffx[i]
            aarr[i, i + 1] = self.diffx[i]

        aarr[0, 1] = 0.0
        aarr[self.nx - 1, self.nx - 2] = 0.0
        aarr[self.nx - 1, self.nx - 1] = 1.0
        return aarr

    def calb(self):
        '''calculate b para in spline interpolation'''
        barr = Variable(torch.zeros(self.nx))
        for i in range(self.nx - 2):
            barr[i + 1] = 3.0 * (self.y[i + 2] - self.y[i + 1]) / \
                self.diffx[i + 1] - 3.0 * (self.y[i + 1] - self.y[i]) / \
                self.diffx[i]
        return barr

    def test_spline(self):
        '''test spline interpolation'''
        print("Spline test")
        xarr = np.array([-0.5, 0.0, 0.5, 1.0, 1.5])
        yarr = np.array([3.2, 2.7, 6, 5, 6.5])
        # spline = Spline(x, y)
        rx = np.arange(-1, 2, 0.01)
        ry = [polySpline().cubic(xarr, yarr, i) for i in rx]
        # ry = [spline.calc(i) for i in rx]
        plt.plot(xarr, yarr, "xb")
        plt.plot(rx, ry, "-r")
        plt.grid(True)
        plt.axis("equal")
        plt.show()


def test_polyInter():
    '''test function polyInter'''
    xarr = torch.linspace(0, 12, 61)
    yarr = torch.tensor([-3.5859E-06, -1.6676E-03, 3.3786E-01, 4.0800E-01,
                         4.3270E-01, 4.3179E-01, 4.1171E-01, 3.8387E-01,
                         3.5151E-01, 3.1994E-01, 2.8555E-01, 2.5578E-01,
                         2.2666E-01, 1.9775E-01, 1.7267E-01, 1.4858E-01,
                         1.2696E-01, 1.0885E-01, 9.1326E-02, 7.6910E-02,
                         6.1086E-02, 5.0735E-02, 4.1224E-02, 3.3132E-02,
                         2.7246E-02, 2.1254E-02, 1.5104E-02, 1.4104E-02,
                         9.5411E-03, 8.1451E-03, 4.5374E-03, 1.3409E-03,
                         3.9118E-03, 2.5759E-03, 9.6833E-04, 2.7833E-03,
                         3.5365E-04, 1.1994E-03, -8.7598E-04, -1.0084E-04,
                         5.1155E-04, 5.4544E-04, -1.1716E-04, -1.5349E-03,
                         -3.8741E-04, -4.7522E-04, -8.4352E-04, 3.2984E-04,
                         -9.7705E-04, 8.5015E-04, -7.5229E-04, -2.3253E-06,
                         -1.6848E-03, -1.1552E-03, 5.8176E-04, -4.6326E-04,
                         7.6713E-04, 2.9290E-04, 1.7461E-04, -1.2381E-03,
                         -2.9332E-04])
    rx = torch.linspace(1, 11, 101)
    ry = [polyInter(xarr, yarr, i) for i in rx]
    plt.plot(xarr, yarr, "xb")
    plt.plot(rx, ry, "-r")
    plt.grid(True)
    plt.axis("equal")
    plt.show()


def polyInter(xp, yp, rr, threshold=5E-3):
    '''
    this function is for interpolation from DFTB+ (lib_math)
    Args:
        x array, y array, and the interpolation point rr (x[0]< rr < x[-1])
    '''
    icl = 0
    nn = xp.shape[0]
    cc = Variable(torch.zeros(nn))
    dd = Variable(torch.zeros(nn))

    # if y_m-y_n is small enough, rTmp1 tends to be inf
    cc[:] = yp[:]
    dd[:] = yp[:]
    dxp = abs(rr - xp[icl])

    # this loop is to find the most close point to rr
    for ii in range(1, nn):
        dxNew = abs(rr - xp[ii])
        if dxNew < dxp:
            icl = ii
            dxp = dxNew
    yy = yp[icl]

    for mm in range(0, nn - 1):
        for ii in range(0, nn - mm - 1):
            rtmp0 = xp[ii] - xp[ii + mm + 1]
            rtmp1 = (cc[ii + 1] - dd[ii]) / rtmp0
            cc[ii] = (xp[ii] - rr) * rtmp1
            dd[ii] = (xp[ii + mm + 1] - rr) * rtmp1
        if 2 * icl < nn - mm - 1:
            dyy = cc[icl]
        else:
            dyy = dd[icl - 1]
            icl = icl - 1
        yy = yy + dyy
    return yy


def diff(mat, axis=-1):
    if torch.is_tensor(mat):
        pass
    elif type(mat) is np.ndarray:
        mat = torch.from_numpy(mat)
    else:
        raise ValueError('input matrix is not tensor or numpy')
    if len(mat.shape) == 1:
        nmat = len(mat)
        nmat_out = nmat - 1
        mat_out = torch.zeros(nmat_out)
        for imat in range(0, nmat_out):
            mat_out[imat] = mat[imat+1] - mat[imat]
    elif len(mat.shape) == 2:
        if axis < 0:
            row = mat.shape[0]
            col = mat.shape[1]-1
            mat_out = torch.zeros(row, col)
            for jmat in range(0, col):
                mat_out[jmat] = mat[:, jmat+1] - mat[:, jmat]
        elif axis == 0:
            row = mat.shape[0]-1
            col = mat.shape[1]
            mat_out = torch.zeros(row, col)
            for imat in range(0, row):
                mat_out[imat, :] = mat[imat+1, :] - mat[imat, :]
    return mat_out
