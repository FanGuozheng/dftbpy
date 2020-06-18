"""Mathematical libraries for DFTB.

including:
    Interpolation for SKF
    Bicubic
    Spline interpolation
    Linear Algebra
    General eigenvalue problem

"""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch as t
import scipy.interpolate
import bisect
import matplotlib.pyplot as plt
from torch.autograd import Variable


class DFTBmath:
    """Interpolation method for SKF in DFTB.

    see interpolation.F90 in DFTB+
    """

    def __init__(self, para):
        """Deal with interpolation of intergral tables.

        Args:
            skf tables
            geometry information

        """
        self.para = para

    def sk_interp(self, rr, nameij):
        """Interpolation SKF according to distance from intergral tables.

        Args:
            grid distance
            number of grid points
            distance between atoms

        """
        datalist = self.para['hs_all' + nameij]
        incr = self.para['grid_dist' + nameij]
        ngridpoint = self.para['ngridpoint' + nameij]
        if type(datalist) is np.ndarray:
            datalist = t.from_numpy(np.asarray(datalist))

        # cutoff = self.para['cutoffsk' + nameij]
        distFudge = 5 * incr  # tail = 5 * incr
        ninterp = 8  # number of interplation
        rMax = ngridpoint * incr + distFudge
        ind = int(rr / incr)
        leng = ngridpoint
        xa = t.zeros((ninterp), dtype=t.float64)
        yb = t.zeros((ninterp, 20), dtype=t.float64)
        dd = t.zeros((20), dtype=t.float64)
        if leng < ninterp + 1:
            print("Warning: not enough points for interpolation!")
        if rr >= rMax:
            pass
        elif ind < leng:  # => polynomial fit
            # iLast = min(leng, ind + nRightInterNew_)
            ilast = min(leng, int(ind + ninterp / 2 + 1))
            ilast = max(ninterp, ilast)
            for ii in range(0, ninterp):
                xa[ii] = (ilast - ninterp + ii) * incr
            yb[:, :] = datalist[ilast - ninterp - 1:ilast - 1]
            # dd = self.polysk3thsk(yb, xa, rr)  # method 1
            dd = self.polyInter_2d(xa, yb, rr)  # method 2
            # for ii in range(0, 20):  # method 3
            #    dd[ii] = self.polyInter(xa, yb[:, ii], rr)

        else:  # Beyond the grid => extrapolation with polynomial of 5th order
            dr = rr - rMax
            ilast = leng
            pass
            '''for ii in range(0, ninterp):
                xa[ii] = (ilast - ninterp + ii) * incr
            yb = datalist[ilast - ninterp - 1:ilast - 1]
            y0 = self.polyInter_u(xa, yb, xa(ninterp) - deltaR_)
            y2 = self.polyInter_u(xa, yb, xa(ninterp) + deltaR_)
            for ii in range(0, ):
                ya[:] = datalist[iLast-ninterp+1:iLast, ii]
                y1 = ya(ninterp)
                y1p = (y2[ii] - y0[ii]) / (2.0 * deltaR_)
                y1pp = (y2[ii] + y0[ii] - 2.0 * y1) / (deltaR_ * deltaR_)
                dd[ii] = poly5ToZero(y1, y1p, y1pp, dr, -1.0 * distFudge)'''
        return dd

    def sk_interp_4d(self, rr, nameij, ncompr):
        """Interpolation SKF (4D) according to distance from intergral tables.

        the normal SKF interpolation is 2D, here this function is for DFTB
        with given compression radius, extra dimensions will be for
        compression radius for atom specie one and atom specie two
        """
        datalist = self.para['hs_all' + nameij]
        incr = self.para['grid_dist' + nameij]
        ngridpoint = self.para['ngridpoint' + nameij]
        if type(datalist) is np.array:
            datalist = t.from_numpy(datalist)

        distFudge = 5 * incr  # tail = 5 * incr
        ninterp = 8  # number of interplation
        rMax = ngridpoint * incr + distFudge
        ind = int(rr / incr)
        leng = ngridpoint
        xa = t.zeros(ninterp)
        yb = t.zeros(ncompr, ncompr, ninterp, 20)
        dd = t.zeros(ncompr, ncompr, 20)
        if leng < ninterp + 1:
            print("Warning: not enough points for interpolation!")
        if rr >= rMax:
            pass
        elif ind < leng:  # => polynomial fit
            # iLast = min(leng, ind + nRightInterNew_)
            ilast = min(leng, int(ind + ninterp / 2 + 1))
            ilast = max(ninterp, ilast)
            for ii in range(0, ninterp):
                xa[ii] = (ilast - ninterp + ii) * incr
            yb = datalist[:, :, ilast - ninterp - 1:ilast - 1]
            # dd = self.polysk3thsk(yb, xa, rr)  # method 1
            dd = self.polyInter_4d(xa, yb, rr)  # method 2
            # for ii in range(0, 20):  # method 3
            #    dd[ii] = self.polyInter(xa, yb[:, ii], rr)

        else:  # Beyond the grid => extrapolation with polynomial of 5th order
            dr = rr - rMax
            ilast = leng
            pass
        return dd

    def polysk3thsk(self, allarr, darr, dd):
        """Interpolation SKF for certain orbitals with given distance.

        e.g: interpolation ss0 orbital with given distance from various
        grid points
        """
        row, col = np.shape(allarr)
        if type(allarr) is t.Tensor:
            allarr.numpy(), darr.numpy()
            Ltensor = True
        else:
            Ltensor = False
        skftable = np.zeros(row)
        hs = np.zeros(col)
        for ii in range(0, col):
            skftable[:] = allarr[:, ii]
            fcubic = scipy.interpolate.interp1d(darr, skftable, kind='cubic')
            hs[ii] = fcubic(dd)
        if Ltensor:
            hs = t.from_numpy(hs)
        return hs

    def polysk5thsk(self):
        pass

    def polyInter_4d(self, xp, yp, rr):
        """Interpolation from DFTB+ (lib_math) with uniform grid.

        Args:
            x array, y array
            the interpolation point rr (x[0]< rr < x[-1])

        """
        icl = 0
        nn = xp.shape[0]
        nn1, nn2, row, col = yp.shape[0], yp.shape[1], yp.shape[2], yp.shape[3]
        assert row == nn
        cc = t.zeros(nn1, nn2, row, col)
        dd = t.zeros(nn1, nn2, row, col)

        # if y_m-y_n is small enough, rTmp1 tends to be inf
        cc[:, :, :, :] = yp[:, :, :, :]
        dd[:, :, :, :] = yp[:, :, :, :]
        dxp = abs(rr - xp[icl])

        # this loop is to find the most close point to rr
        for ii in range(0, nn - 1):
            dxNew = abs(rr - xp[ii])
            if dxNew < dxp:
                icl = ii
                dxp = dxNew
        yy = yp[:, :, icl, :]

        for mm in range(0, nn - 1):
            for ii in range(0, nn - mm - 1):
                rtmp0 = xp[ii] - xp[ii + mm + 1]
                rtmp1 = (cc[:, :, ii + 1, :] - dd[:, :, ii, :]) / rtmp0
                cc[:, :, ii, :] = (xp[ii] - rr) * rtmp1
                dd[:, :, ii, :] = (xp[ii + mm + 1] - rr) * rtmp1
            if 2 * icl < nn - mm - 1:
                dyy = cc[:, :, icl, :]
            else:
                dyy = dd[:, :, icl - 1, :]
                icl = icl - 1
            yy = yy + dyy
        return yy

    def polyInter_2d(self, xp, yp, rr):
        """Interpolation from DFTB+ (lib_math) with uniform grid.

        Args:
            x array, y array
            the interpolation point rr (x[0]< rr < x[-1])

        """
        '''nn = len(xp)
        delta = t.zeros(nn)
        for ii in range(0, len(xp) - 1):
            delta[ii] = 1.0 / (xp[ii + 1] - xp[0])
        cc = yp
        dd = yp
        iCl = math.ceil(((xx - xp[0]) * delta[0]).numpy())
        yy = yp[iCl, :]
        iCl = iCl - 1
        for mm in range(0, nn - 1):
            for ii in range(0, nn - mm - 1):
                r2Tmp = (dd[ii, :] - cc[ii + 1, :]) * delta[mm]
                cc[ii, :] = (xp[ii] - xx) * r2Tmp
                dd[ii, :] = (xp[ii + mm] - xx) * r2Tmp
            if 2 * iCl < nn - mm - 1:
                dyy = cc[iCl, :]
            else:
                dyy = dd[iCl - 1, :]
                iCl = iCl - 1
            yy = yy + dyy
        return yy'''
        icl = 0
        nn = xp.shape[0]
        row, col = yp.shape[0], yp.shape[1]
        assert row == nn
        cc = t.zeros(row, col)
        dd = t.zeros(row, col)

        # if y_m-y_n is small enough, rTmp1 tends to be inf
        cc[:, :] = yp[:, :]
        dd[:, :] = yp[:, :]
        dxp = abs(rr - xp[icl])

        # this loop is to find the most close point to rr
        for ii in range(0, nn - 1):
            dxNew = abs(rr - xp[ii])
            if dxNew < dxp:
                icl = ii
                dxp = dxNew
        yy = yp[icl, :]

        for mm in range(0, nn - 1):
            for ii in range(0, nn - mm - 1):
                rtmp0 = xp[ii] - xp[ii + mm + 1]
                rtmp1 = (cc[ii + 1, :] - dd[ii, :]) / rtmp0
                cc[ii, :] = (xp[ii] - rr) * rtmp1
                dd[ii, :] = (xp[ii + mm + 1] - rr) * rtmp1
            if 2 * icl < nn - mm - 1:
                dyy = cc[icl, :]
            else:
                dyy = dd[icl - 1, :]
                icl = icl - 1
            yy = yy + dyy
        return yy

    def polyInter(self, xp, yp, rr, threshold=5E-3):
        """Interpolation from DFTB+ (lib_math).

        Args:
            x array, y array
            the interpolation point rr (x[0]< rr < x[-1])

        """
        icl = 0
        nn = xp.shape[0]
        cc = t.zeros(nn)
        dd = t.zeros(nn)

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


class Bspline():
    """Bspline interpolation for DFTB.

    originate from
    """

    def __init__(self):
        """Revised from the following.

        https://docs.scipy.org/doc/scipy-1.0.0/reference/generated/
        scipy.interpolate.BSpline.html
        """
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
        """Test function for Bspline."""
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
    """Revised from pycubicspline.

    https://github.com/AtsushiSakai/pycubicspline
    """

    def __init__(self):
        pass

    def linear():
        """Calculate linear interpolation."""
        pass

    def cubic(self, xp, yp, dd):
        """Calculate a para in spline interpolation."""
        self.xp = xp
        self.yp = yp
        self.dd = dd
        self.nx = self.xp.shape[0]
        self.linex = len(xp)
        self.diffx = diff(self.xp)
        if t.is_tensor(xp):
            xnp = xp.numpy()
            self.ddind = bisect.bisect(xnp, dd) - 1
        elif type(xp) is np.ndarray:
            self.ddind = bisect.bisect(xp, dd) - 1
        if self.dd < self.xp[0] or self.dd > self.xp[-1]:
            return None
        b = Variable(t.empty(self.nx - 1))
        d = Variable(t.empty(self.nx - 1))
        A = self.cala()
        B = self.calb()
        c, _ = t.lstsq(B, A)
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
        """Calculate a para in spline interpolation."""
        aarr = Variable(t.zeros(self.nx, self.nx))
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
        """Calculate b para in spline interpolation."""
        barr = Variable(t.zeros(self.nx))
        for i in range(self.nx - 2):
            barr[i + 1] = 3.0 * (self.y[i + 2] - self.y[i + 1]) / \
                self.diffx[i + 1] - 3.0 * (self.y[i + 1] - self.y[i]) / \
                self.diffx[i]
        return barr

    def test_spline(self):
        """Test spline interpolation."""
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
    """Test function polyInter."""
    xarr = t.linspace(0, 12, 61)
    yarr = t.tensor([-3.5859E-06, -1.6676E-03, 3.3786E-01, 4.0800E-01,
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
    rx = t.linspace(1, 11, 101)
    ry = [polyInter(xarr, yarr, i) for i in rx]
    plt.plot(xarr, yarr, "xb")
    plt.plot(rx, ry, "-r")
    plt.grid(True)
    plt.axis("equal")
    plt.show()


def polyInter(xp, yp, rr, threshold=5E-3):
    """Interpolation from DFTB+ (lib_math).

    Args:
        x array, y array, and the interpolation point rr (x[0]< rr < x[-1])

    """
    icl = 0
    nn = xp.shape[0]
    cc = Variable(t.zeros(nn))
    dd = Variable(t.zeros(nn))

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
    if t.is_tensor(mat):
        pass
    elif type(mat) is np.ndarray:
        mat = t.from_numpy(mat)
    else:
        raise ValueError('input matrix is not tensor or numpy')
    if len(mat.shape) == 1:
        nmat = len(mat)
        nmat_out = nmat - 1
        mat_out = t.zeros(nmat_out)
        for imat in range(0, nmat_out):
            mat_out[imat] = mat[imat+1] - mat[imat]
    elif len(mat.shape) == 2:
        if axis < 0:
            row = mat.shape[0]
            col = mat.shape[1]-1
            mat_out = t.zeros(row, col)
            for jmat in range(0, col):
                mat_out[jmat] = mat[:, jmat+1] - mat[:, jmat]
        elif axis == 0:
            row = mat.shape[0]-1
            col = mat.shape[1]
            mat_out = t.zeros(row, col)
            for imat in range(0, row):
                mat_out[imat, :] = mat[imat+1, :] - mat[imat, :]
    return mat_out


class BicubInterp:
    """Bicubic interpolation method for DFTB.

    reference: https://en.wikipedia.org/wiki/Bicubic_interpolation
    """

    def __init__(self):
        """Get interpolation with two variables."""
        pass

    def bicubic_2d(self, xmesh, ymesh, zmesh, xi, yi):
        """Build fmat.

        [[f(0, 0),  f(0, 1),   f_y(0, 0),  f_y(0, 1)],
         [f(1, 0),   f(1, 1),   f_y(1, 0),  f_y(1, 1)],
         [f_x(0, 0), f_x(0, 1), f_xy(0, 0), f_xy(0, 1)],
         [f_x(1, 0), f_x(1, 1), f_xy(1, 0), f_xy(1, 1)]]
        a_mat = coeff * famt * coeff_
        Then returns:
            p(x, y) = [1, x, x**2, x**3] * a_mat * [1, y, y**2, y**3].T
        Args:
            xmesh, ymesh: x (1D) and y (1D)
            zmesh: z (2D)
            ix, iy: the interpolation point
        """
        # check if xi, yi is out of range of xmesh, ymesh, maybe not good!!!!!
        if xi < xmesh[0]:
            xi = xmesh[0]
        if yi < ymesh[0]:
            yi = ymesh[0]

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
        """Construct f(0/1, 0/1) in fmat."""
        f00 = zmesh[self.nxi, self.nyi]
        f10 = zmesh[self.nxi + 1, self.nyi]
        f01 = zmesh[self.nxi, self.nyi + 1]
        f11 = zmesh[self.nxi + 1, self.nyi + 1]
        fmat[0, 0], fmat[1, 0], fmat[0, 1], fmat[1, 1] = f00, f10, f01, f11
        return fmat

    def fmat_1(self, fmat, zmesh, xmesh, ymesh, ty):
        """Construct fx(0/1, 0) or fy(0, 0/1) in fmat."""
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
        """Construct f(0/1, 0/1) in fmat."""
        fmat[2, 2] = fmat[0, 2] * fmat[2, 0]
        fmat[3, 2] = fmat[3, 0] * fmat[1, 2]
        fmat[2, 3] = fmat[2, 1] * fmat[0, 3]
        fmat[3, 3] = fmat[3, 1] * fmat[1, 3]
        return fmat

    def get_diff(self, mesh, nxi=None, nyi=None, ty=None):
        """Get derivative over x and y direction."""
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
        """Get derivative over x and y direction in boundary."""
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
        """Test Bicubic method."""
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
        """Test gradients of bicubic method."""
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
        """Build the following fmat.

        [[f(0, 0),  f(0, 1),   f_y(0, 0),  f_y(0, 1)],
         [f(1, 0),   f(1, 1),   f_y(1, 0),  f_y(1, 1)],
         [f_x(0, 0), f_x(0, 1), f_xy(0, 0), f_xy(0, 1)],
         [f_x(1, 0), f_x(1, 1), f_xy(1, 0), f_xy(1, 1)]]
        a_mat = coeff * famt * coeff_
        then returns:
            p(x, y) = [1, x, x**2, x**3] * a_mat * [1, y, y**2, y**3].T
        Args:
            xmesh, ymesh: x (1D) and y (1D)
            zmesh: z (3D)
            ix, iy: the interpolation point

        """
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
        """Construct f(0/1, 0/1) in fmat."""
        f00 = zmesh[self.nxi, self.nyi, :]
        f10 = zmesh[self.nxi + 1, self.nyi, :]
        f01 = zmesh[self.nxi, self.nyi + 1, :]
        f11 = zmesh[self.nxi + 1, self.nyi + 1, :]
        fmat[0, 0, :], fmat[1, 0, :], fmat[0, 1, :], fmat[1, 1, :] = \
            f00, f10, f01, f11
        return fmat

    def fmat_13d(self, fmat, zmesh, xmesh, ymesh, ty):
        """Construct fx(0/1, 0) or fy(0, 0/1) in fmat."""
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
        """Construct f(0/1, 0/1) in fmat."""
        fmat[2, 2] = fmat[0, 2] * fmat[2, 0]
        fmat[3, 2] = fmat[3, 0] * fmat[1, 2]
        fmat[2, 3] = fmat[2, 1] * fmat[0, 3]
        fmat[3, 3] = fmat[3, 1] * fmat[1, 3]
        return fmat

    def get_diff3d(self, mesh, nxi=None, nyi=None, ty=None):
        """Get derivative over x and y direction."""
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
        """Get derivative over x and y direction in boundary."""
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


class LinAl:
    """Linear Algebra."""

    def __init__(self, para):
        """Initialize parameters."""
        self.para = para

    def inv33_mat(self, in_):
        """Return inverse of 3 * 3 matrix, out_ = 1 / det(in_) adj(in_)."""
        out_ = t.zeros((3, 3), dtype=t.float64)
        det = self.det33_mat(in_)

        out_[0, 0] = (in_[1, 1] * in_[2, 2] - in_[1, 2] * in_[2, 1]) / det
        out_[0, 1] = -(in_[0, 1] * in_[2, 2] - in_[0, 2] * in_[2, 1]) / det
        out_[0, 2] = (in_[0, 1] * in_[1, 2] - in_[0, 2] * in_[1, 1]) / det
        out_[1, 0] = -(in_[1, 0] * in_[2, 2] - in_[1, 2] * in_[2, 0]) / det
        out_[1, 1] = (in_[0, 0] * in_[2, 2] - in_[0, 2] * in_[2, 0]) / det
        out_[1, 2] = -(in_[0, 0] * in_[1, 2] - in_[0, 2] * in_[1, 0]) / det
        out_[2, 0] = (in_[1, 0] * in_[2, 1] - in_[1, 1] * in_[2, 0]) / det
        out_[2, 1] = -(in_[0, 0] * in_[2, 1] - in_[0, 1] * in_[2, 0]) / det
        out_[2, 2] = (in_[0, 0] * in_[1, 1] - in_[0, 1] * in_[1, 0]) / det
        return out_

    def det33_mat(self, in_):
        """Return 3*3 determinant."""
        det = in_[0, 0] * in_[1, 1] * in_[2, 2] - \
            in_[0, 0] * in_[1, 2] * in_[2, 1] - \
            in_[0, 1] * in_[1, 0] * in_[2, 2] + \
            in_[0, 1] * in_[1, 2] * in_[2, 0] + \
            in_[0, 2] * in_[1, 0] * in_[2, 1] - \
            in_[0, 2] * in_[1, 1] * in_[2, 0]
        return det
