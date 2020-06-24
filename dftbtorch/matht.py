"""Mathematical libraries for DFTB-ML.

including:
    Interpolation for SKF
    Bicubic
    Spline interpolation
    Linear Algebra: eigenvalue problem, linear equation...

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
        """Initialize parameters."""
        self.para = para

    def sk_interp(self, rr, nameij):
        """Interpolation SKF according to distance from intergral tables.

        Args:
            incr: grid distance
            ngridpoint: number of grid points
            distance between atoms
            ninterp: interpolate from up and lower SKF grid points number

        """
        datalist = self.para['hs_all' + nameij]
        incr = self.para['grid_dist' + nameij]
        ngridpoint = self.para['ngridpoint' + nameij]
        ninterp = self.para['ninterp']
        delta_r = self.para['delta_r_skf']

        xa = t.zeros((ninterp), dtype=t.float64)
        yb = t.zeros((ninterp, 20), dtype=t.float64)
        dd = t.zeros((20), dtype=t.float64)

        if type(datalist) is np.ndarray:
            datalist = t.from_numpy(np.asarray(datalist))

        # cutoff = self.para['cutoffsk' + nameij]
        tail = 5 * incr
        rmax = (ngridpoint - 1) * incr + tail
        ind = int(rr / incr)
        leng = ngridpoint
        if leng < ninterp + 1:
            print("Warning: not enough points for interpolation!")
        if rr >= rmax:
            dd[:] = 0.0
        elif ind < leng:  # => polynomial fit
            ilast = min(leng, int(ind + ninterp / 2 + 1))
            ilast = max(ninterp, ilast)
            for ii in range(0, ninterp):
                xa[ii] = (ilast - ninterp + ii) * incr
            yb[:, :] = datalist[ilast - ninterp - 1: ilast - 1]
            # dd = self.polysk3thsk(yb, xa, rr)  # method 1
            dd = self.poly_interp_2d(xa, yb, rr)  # method 2
        else:  # Beyond the grid => extrapolation with polynomial of 5th order
            dr = rr - rmax
            ilast = leng
            for ii in range(0, ninterp):
                xa[ii] = (ilast - ninterp + ii) * incr
            yb = datalist[ilast - ninterp - 1: ilast - 1]
            y0 = self.poly_interp_2d(xa, yb, xa[ninterp - 1] - delta_r)
            y2 = self.poly_interp_2d(xa, yb, xa[ninterp - 1] + delta_r)
            ya = datalist[ilast - ninterp - 1: ilast - 1]
            y1 = ya[ninterp - 1]
            y1p = (y2 - y0) / (2.0 * delta_r)
            y1pp = (y2 + y0 - 2.0 * y1) / (delta_r * delta_r)
            dd = self.poly5_zero(y1, y1p, y1pp, dr, -1.0 * tail)
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
        ninterp = self.para['ninterp']
        delta_r = self.para['delta_r_skf']
        xa = t.zeros(ninterp)
        yb = t.zeros(ncompr, ncompr, ninterp, 20)
        dd = t.zeros(ncompr, ncompr, 20)

        if type(datalist) is np.array:
            datalist = t.from_numpy(datalist)

        tail = 5 * incr
        rmax = (ngridpoint - 1) * incr + tail
        ind = int(rr / incr)
        leng = ngridpoint
        if leng < ninterp + 1:
            print("Warning: not enough points for interpolation!")
        if rr >= rmax:
            dd[:] = 0.0
        elif ind < leng:  # => polynomial fit
            ilast = min(leng, int(ind + ninterp / 2 + 1))
            ilast = max(ninterp, ilast)
            for ii in range(0, ninterp):
                xa[ii] = (ilast - ninterp + ii) * incr
            yb = datalist[:, :, ilast - ninterp - 1:ilast - 1]
            # dd = self.polysk3thsk(yb, xa, rr)  # method 1
            dd = self.poly_interp_4d(xa, yb, rr)  # method 2
        else:  # Beyond the grid => extrapolation with polynomial of 5th order
            dr = rr - rmax
            ilast = leng
            for ii in range(0, ninterp):
                xa[ii] = (ilast - ninterp + ii) * incr
            print(xa)
            yb = datalist[ilast - ninterp - 1: ilast - 1]
            y0 = self.poly_interp_4d(xa, yb, xa[ninterp - 1] - delta_r)
            y2 = self.poly_interp_4d(xa, yb, xa[ninterp - 1] + delta_r)
            ya = datalist[ilast - ninterp - 1: ilast - 1]
            y1 = ya[ninterp - 1]
            y1p = (y2 - y0) / (2.0 * delta_r)
            y1pp = (y2 + y0 - 2.0 * y1) / (delta_r * delta_r)
            dd = self.poly5_zero(y1, y1p, y1pp, dr, -1.0 * tail)
        return dd

    def poly5_zero(self, y0, y0p, y0pp, xx, dx):
        """Get integrals if beyond the grid range with 5th polynomial."""
        dx1 = y0p * dx
        dx2 = y0pp * dx * dx
        dd = 10.0 * y0 - 4.0 * dx1 + 0.5 * dx2
        ee = -15.0 * y0 + 7.0 * dx1 - 1.0 * dx2
        ff = 6.0 * y0 - 3.0 * dx1 + 0.5 * dx2
        xr = xx / dx
        yy = ((ff * xr + ee) * xr + dd) * xr * xr * xr
        return yy

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

    def polytozero(self, hs_skf, ninterpline, gridmesh):
        """Fit the tail of skf file (5lines, 5th order)."""
        ni = ninterpline
        dx = gridmesh * 5
        ytail = hs_skf[ni - 1, :]
        ytailp = (hs_skf[ni - 1, :] - hs_skf[ni - 2, :]) / gridmesh
        ytailp2 = (hs_skf[ni - 2, :]-hs_skf[ni - 3, :]) / gridmesh
        ytailpp = (ytailp - ytailp2) / gridmesh
        xx = np.array([gridmesh * 4, gridmesh * 3, gridmesh * 2,
                       gridmesh, 0.0])
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

    def poly_interp_4d(self, xp, yp, rr):
        """Interpolation from DFTB+ (lib_math) with uniform grid.

        Args:
            x array, y array
            the interpolation point rr (x[0]< rr < x[-1])

        """
        icl = 0
        nn = xp.shape[0]
        nn1, nn2, row, col = yp.shape[0], yp.shape[1], yp.shape[2], yp.shape[3]
        assert row == nn
        cc = t.zeros((nn1, nn2, row, col), dtype=t.float64)
        dd = t.zeros((nn1, nn2, row, col), dtype=t.float64)

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

    def poly_interp_2d(self, xp, yp, rr):
        """Interpolation from DFTB+ (lib_math) with uniform grid.

        Args:
            x array, y array
            the interpolation point rr (x[0]< rr < x[-1])

        """
        icl = 0
        nn = xp.shape[0]
        row, col = yp.shape[0], yp.shape[1]
        assert row == nn
        cc = t.zeros((row, col), dtype=t.float64)
        dd = t.zeros((row, col), dtype=t.float64)

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


class EigenSolver:
    """Eigen solver for general eigenvalue problem."""

    def __init__(self, para):
        """Initialize parameters."""
        self.para = para

    def cholesky(self, matrixa, matrixb):
        """Cholesky decomposition.

        Cholesky decomposition of B: B = LL^{T}
        transfer general eigenvalue problem AX = (lambda)BX ==>
            (L^{-1}AL^{-T})(L^{T}X) = (lambda)(L^{T}X)
        matrix_a: here is Fock operator
        matrix_b: here is overlap
        """
        chol_l = t.cholesky(matrixb)
        # self.para['eigval'] = chol_l
        linv_a = t.mm(t.inverse(chol_l), matrixa)
        l_invtran = t.inverse(chol_l.t())
        linv_a_linvtran = t.mm(linv_a, l_invtran)
        eigval, eigm = t.symeig(linv_a_linvtran, eigenvectors=True)
        eigm_ab = t.mm(l_invtran, eigm)
        return eigval, eigm_ab

    def cholesky_new(self, matrixa, matrixb):
        """Cholesky decomposition.

        difference from _cholesky is avoiding the use of inverse matrix
        """
        chol_l = t.cholesky(matrixb)
        row = matrixa.shape[1]
        A1, LU_A = t.solve(matrixa, chol_l)
        A2, LU_A1 = t.solve(A1.t(), chol_l)
        A3 = A2.t()
        eigval, eigm = t.symeig(A3, eigenvectors=True)
        l_inv, _ = t.solve(t.eye((row), dtype=t.float64), chol_l.t())
        eigm_ab = t.mm(l_inv, eigm)
        return eigval, eigm_ab

    def lowdin_symeig(self, matrixa, matrixb):
        """Use t.symeig to decompose B to realize Löwdin orthonormalization.

        BX = (lambda)X, then omega = lambda.diag()
        S_{-1/2} = Xomega_{-1/2}X_{T}
        AX' = (lambda)BX' ==>
        (S_{-1/2}AS_{-1/2})(S_{1/2}X') = (lambda)(S_{1/2}X')
        matrix_a: here is Fock operator
        matrix_b: here is overlap
        """
        lam_b, l_b = t.symeig(matrixb, eigenvectors=True)
        lam_sqrt_inv = t.sqrt(1 / lam_b)
        S_sym = t.mm(l_b, t.mm(lam_sqrt_inv.diag(), l_b.t()))
        SHS = t.mm(S_sym, t.mm(matrixa, S_sym))
        eigval, eigvec_ = t.symeig(SHS, eigenvectors=True)
        eigvec = t.mm(S_sym, eigvec_)
        # eigval3, eigvec_ = t.symeig(lam_b_2d, eigenvectors=True)
        return eigval, eigvec

    def lowdin_svd_sym(self, matrixa, matrixb):
        """Use SVD and sym to decompose B to realize Löwdin orthonormalization.

        B: B = USV_{T}
        S_{-1/2} = US_{-1/2}V_{T}
        AX = (lambda)BX ==>
        (S_{-1/2}AS_{-1/2})(S_{1/2}X) = (lambda)(S_{1/2}X)
        matrix_a: Fock operator
        matrix_b: overlap matrix
        """
        ub, sb, vb = t.svd(matrixb)
        sb_sqrt_inv = t.sqrt(1 / sb)
        S_sym = t.mm(ub, t.mm(sb_sqrt_inv.diag(), vb.t()))
        SHS = t.mm(S_sym, t.mm(matrixa, S_sym))
        eigval, eigvec_ = t.symeig(SHS, eigenvectors=True)
        eigvec = t.mm(S_sym, eigvec_)
        return eigval, eigvec

    def lowdin_svd(self, matrixa, matrixb):
        """Only SVD to decompose B to realize Löwdin orthonormalization.

        SVD decomposition of B: B = USV_{T}
            S_{-1/2} = US_{-1/2}V_{T}
            AX = (lambda)BX ==>
            (S_{-1/2}AS_{-1/2})(S_{1/2}X) = (lambda)(S_{1/2}X)
        matrix_a: Fock operator
        matrix_b: overlap matrix
        """
        ub, sb, vb = t.svd(matrixb)
        sb_sqrt_inv = t.sqrt(1 / sb)
        S_sym = t.mm(ub, t.mm(sb_sqrt_inv.diag(), vb.t()))
        SHS = t.mm(S_sym, t.mm(matrixa, S_sym))

        ub2, sb2, vb2 = t.svd(SHS)
        eigvec = t.mm(S_sym, ub2)
        return sb2, eigvec

    def lowdin_qr_eig(self, matrixa, matrixb):
        """Use QR to decompose B to realize Löwdin orthonormalization.

        QR decomposition of B: B = USV_{T}
        S_{-1/2} = US_{-1/2}V_{T}
        AX = (lambda)BX ==>
        (S_{-1/2}AS_{-1/2})(S_{1/2}X) = (lambda)(S_{1/2}X)
        matrix_a: Fock operator
        matrix_b: overlap matrix
        """
        Bval = []
        rowa = matrixb.shape[0]
        eigvec_b = t.eye(rowa)
        Bval.append(matrixb)
        icount = 0
        while True:
            Q_, R_ = t.qr(Bval[-1])
            eigvec_b = eigvec_b @ Q_
            Bval.append(R_ @ Q_)
            icount += 1
            '''if abs(Bval[-1].sum() - Bval[-2].sum()) < rowa ** 2 * 1e-6:
                break'''
            if abs((Q_ - Q_.diag().diag()).sum()) < rowa ** 2 * 1e-6:
                break
            if icount > 60:
                print('Warning: QR decomposition do not reach convergence')
                break
        eigval_b = Bval[-1]
        diagb_sqrt_inv = t.sqrt(1 / eigval_b.diag()).diag()
        S_sym = t.mm(eigvec_b, t.mm(diagb_sqrt_inv, eigvec_b.t()))
        SHS = t.mm(S_sym, t.mm(matrixa, S_sym))
        eigval, eigvec_ = t.symeig(SHS, eigenvectors=True)
        eigvec = t.mm(S_sym, eigvec_)
        return eigval, eigvec

    def lowdin_qr(self, matrixa, matrixb):
        """Use QR to decompose B to realize Löwdin orthonormalization.

        QR decomposition of B: B = USV_{T}
        S_{-1/2} = US_{-1/2}V_{T}
        AX = (lambda)BX ==>
        (S_{-1/2}AS_{-1/2})(S_{1/2}X) = (lambda)(S_{1/2}X)
        matrix_a: Fock operator
        matrix_b: overlap matrix
        """
        Bval, ABval = [], []
        rowb, colb = matrixb.shape[0], matrixb.shape[1]
        rowa, cola = matrixa.shape[0], matrixa.shape[1]
        assert rowa == rowb == cola == colb
        eigvec_b, eigvec_ab = t.eye(rowa), t.eye(rowa)
        eigval = t.zeros(rowb)
        eigvec = t.zeros(rowa, rowb)
        Bval.append(matrixb)
        icount = 0
        while True:
            Q_, R_ = t.qr(Bval[-1])
            eigvec_b = eigvec_b @ Q_
            Bval.append(R_ @ Q_)
            icount += 1
            '''if abs(Bval[-1].sum() - Bval[-2].sum()) < rowa ** 2 * 1e-6:
                break'''
            if abs((Q_ - Q_.diag().diag()).sum()) < rowa ** 2 * 1e-6:
                break
            if icount > 60:
                print('Warning: QR decomposition do not reach convergence')
                break
        eigval_b = Bval[-1]
        diagb_sqrt_inv = t.sqrt(1 / eigval_b.diag()).diag()
        S_sym = t.mm(eigvec_b, t.mm(diagb_sqrt_inv, eigvec_b.t()))
        SHS = t.mm(S_sym, t.mm(matrixa, S_sym))

        ABval.append(SHS)
        icount = 0
        while True:
            Q2_, R2_ = t.qr(ABval[-1])
            eigvec_ab = eigvec_ab @ Q2_
            ABval.append(R2_ @ Q2_)
            icount += 1
            if abs((Q2_ - Q2_.diag().diag()).sum()) < rowa ** 2 * 1e-6:
                break
            if icount > 60:
                print('Warning: QR decomposition do not reach convergence')
                break
        eigval_ab = ABval[-1].diag()
        eigvec_ = t.mm(S_sym, eigvec_ab)
        sort = eigval_ab.sort()
        for ii in range(0, matrixb.shape[0]):
            eigval[ii] = eigval_ab[int(t.tensor(sort[1])[ii])]
            eigvec[:, ii] = eigvec_[:, int(t.tensor(sort[1])[ii])]
        return eigval, eigvec


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
        self.diffx = self.diff(self.xp)
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

    def diff(self, mat, axis=-1):
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
        for ii in range(nn3):
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
