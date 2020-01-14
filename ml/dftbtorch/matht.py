#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
import scipy.interpolate
import bisect
import matplotlib.pyplot as plt
from torch.autograd import Variable


class DFTBmath(object):

    def __init__(self):
        pass

    def polysk3thsk(self, allarr, darr, dd):
        row, col = np.shape(allarr)
        skftable = np.zeros(row)
        hs = np.zeros(col)
        for ii in range(0, col):
            skftable[:] = allarr[:, ii]
            fcubic = scipy.interpolate.interp1d(darr, skftable, kind='cubic')
            hs[ii] = fcubic(dd)
        return hs

    def polysk5thsk(self, allarr, darr, dd):
        ni = ninterpline
        dx = self.gridmesh*5
        ytail = hs_skf[ni-1, :]
        ytailp = (hs_skf[ni-1, :]-hs_skf[ni-2, :])/self.gridmesh
        ytailp2 = (hs_skf[ni-2, :]-hs_skf[ni-3, :])/self.gridmesh
        ytailpp = (ytailp-ytailp2)/self.gridmesh
        xx = np.array([self.gridmesh*4, self.gridmesh*3, self.gridmesh*2,
                       self.gridmesh, 0.0])
        nline = ninterpline
        for xxi in xx:
            dx1 = ytailp * dx
            dx2 = ytailpp * dx * dx
            dd = 10.0 * ytail - 4.0 * dx1 + 0.5 * dx2
            ee = -15.0 * ytail + 7.0 * dx1 - 1.0 * dx2
            ff = 6.0 * ytail - 3.0 * dx1 + 0.5 * dx2
            xr = xxi / dx
            yy = ((ff*xr + ee)*xr + dd)*xr*xr*xr
            hs_skf[nline, :] = yy
            nline += 1
        return hs_skf


class Bspline():

    def __init__(self):
        pass

    def bspline(self, x, t, c, k):
        n = len(t) - k - 1
        assert (n >= k+1) and (len(c) >= n)
        return sum(c[i] * self.B(x, k, i, t) for i in range(n))

    def B(self, x, k, i, t):
        if k == 0:
            return 1.0 if t[i] <= x < t[i+1] else 0.0
        if t[i+k] == t[i]:
            c1 = 0.0
        else:
            c1 = (x - t[i])/(t[i+k] - t[i]) * self.B(x, k-1, i, t)
            if t[i+k+1] == t[i+1]:
                c2 = 0.0
            else:
                c2 = (t[i+k+1] - x)/(t[i+k+1] - t[i+1]) * self.B(
                        x, k-1, i+1, t)
        return c1 + c2


class polySpline():
    '''
    This code revised from pycubicspline:
    https://github.com/AtsushiSakai/pycubicspline
    '''
    def __init__(self, x, y, dd):
        self.x = x
        self.y = y
        self.dd = dd
        self.nx = self.x.shape[0]
        self.linex = len(x)
        self.diffx = diff(self.x)
        if torch.is_tensor(x):
            xnp = x.numpy()
            self.ddind = bisect.bisect(xnp, dd) - 1
        elif type(x) is np.ndarray:
            self.ddind = bisect.bisect(x, dd) - 1

    def linear():
        pass

    def cubic(self):
        if self.dd < self.x[0] or self.dd > self.x[-1]:
            return None
        b = Variable(torch.empty(self.nx - 1))
        d = Variable(torch.empty(self.nx - 1))
        A = self.cala()
        B = self.calb()
        c, _ = torch.lstsq(B, A)
        for i in range(self.nx - 1):
            tb = (self.y[i + 1] - self.y[i]) / self.diffx[i] - self.diffx[i] \
                * (c[i + 1] + 2.0 * c[i]) / 3.0
            b[i] = tb
            d[i] = (c[i + 1] - c[i]) / (3.0 * self.diffx[i])
        i = self.ddind
        dx = self.dd - self.x[i]
        result = self.y[i] + b[i] * dx + c[i] * dx ** 2.0 + d[i] * dx ** 3.0
        return result

    def cala(self):
        A = Variable(torch.zeros(self.nx, self.nx))
        A[0, 0] = 1.0
        for i in range(self.nx - 1):
            if i != (self.nx - 2):
                A[i + 1, i + 1] = 2.0 * (self.diffx[i] + self.diffx[i + 1])
            A[i + 1, i] = self.diffx[i]
            A[i, i + 1] = self.diffx[i]

        A[0, 1] = 0.0
        A[self.nx - 1, self.nx - 2] = 0.0
        A[self.nx - 1, self.nx - 1] = 1.0
        return A

    def calb(self):
        B = Variable(torch.zeros(self.nx))
        for i in range(self.nx - 2):
            B[i + 1] = 3.0 * (self.y[i + 2] - self.y[i + 1]) / \
                self.diffx[i + 1] - 3.0 * (self.y[i + 1] - self.y[i]) / \
                self.diffx[i]
        return B

    def test_spline(self):
        print("Spline test")
        x = np.array([-0.5, 0.0, 0.5, 1.0, 1.5])
        y = np.array([3.2, 2.7, 6, 5, 6.5])
        # spline = Spline(x, y)
        rx = np.arange(-1, 2, 0.01)
        ry = [polySpline(x, y, i).cubic() for i in rx]
        # ry = [spline.calc(i) for i in rx]
        plt.plot(x, y, "xb")
        plt.plot(rx, ry, "-r")
        plt.grid(True)
        plt.axis("equal")
        plt.show()


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




