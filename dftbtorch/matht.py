"""Mathematical libraries for DFTB-ML.

including:
    Interpolation for a list of SK files with:
        Bicubic interpolation
    Spline interpolation
    Linear Algebra:
        eigenvalue problem, linear equation...

"""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch as t
import scipy.interpolate
import bisect
import matplotlib.pyplot as plt
from torch.autograd import Variable
import DFTBMaLT.dftbmalt.utils.batch as utilsbatch
from torch.nn.utils.rnn import pad_sequence


class DFTBmath:
    """Interpolation method for SKF in DFTB.

    see interpolation.F90 in DFTB+.
    """

    def __init__(self, para, skf):
        """Initialize parameters."""
        self.para = para
        self.skf = skf

    def sk_interp(self, rr, nameij):
        """Interpolation SKF according to distance from intergral tables.

        Args:
            incr: grid distance
            ngridpoint: number of grid points
            distance between atoms
            ninterp: interpolate from up and lower SKF grid points number

        """
        datalist = self.skf['hs_all' + nameij]
        incr = self.skf['grid_dist' + nameij]
        ngridpoint = self.skf['ngridpoint' + nameij]
        ninterp = self.skf['ninterp']
        delta_r = self.skf['deltaRskf']

        xa = t.zeros((ninterp), dtype=t.float64)
        yb = t.zeros((ninterp, 20), dtype=t.float64)
        dd = t.zeros((20), dtype=t.float64)

        if type(datalist) is np.ndarray:
            datalist = t.from_numpy(np.asarray(datalist))

        # cutoff = self.para['cutoffsk' + nameij]
        tail = 5 * incr
        rmax = (ngridpoint - 1) * incr + tail
        ind = int(rr / incr)
        # ind = t.round(rr / incr)
        leng = ngridpoint
        if leng < ninterp + 1:
            print("Warning: not enough points for interpolation!")
        if rr >= rmax:
            dd[:] = 0.0
        elif ind < leng:  # => polynomial fit
            ilast = min(leng, int(ind + ninterp / 2 + 1))
            ilast = max(ninterp, ilast)
            for ii in range(ninterp):
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
        compression radius for atom specie one and atom specie two.
        """
        datalist = self.para['hs_all' + nameij]
        incr = self.para['grid_dist' + nameij]
        ngridpoint = self.para['ngridpoint_rall' + nameij]
        ngridmin = int(ngridpoint.min())
        ngridmax = int(ngridpoint.max())
        ninterp = self.para['ninterp']
        delta_r = self.para['delta_r_skf']
        xa = t.zeros(ninterp)
        yb = t.zeros(ncompr, ncompr, ninterp, 20)
        dd = t.zeros(ncompr, ncompr, 20)

        if type(datalist) is np.array:
            datalist = t.from_numpy(datalist)

        tail = 5 * incr
        rmaxmin = (ngridmin - 1) * incr + tail
        rmaxmax = (ngridmax - 1) * incr + tail
        ind = int(rr / incr)
        leng = ngridmin  # the smallest grid point among all compression r
        if leng < ninterp + 1:
            print("Warning: not enough points for interpolation!")
        if rr >= rmaxmax:
            dd[:] = 0.0
        elif ind < ngridmin:  # => polynomial fit
            ilast = min(leng, int(ind + ninterp / 2 + 1))
            ilast = max(ninterp, ilast)
            xii = t.linspace(0, ninterp - 1, ninterp)
            # for ii in range(ninterp):
            xa = (ilast - ninterp + xii) * incr
            yb = datalist[:, :, ilast - ninterp - 1:ilast - 1]
            # dd = self.polysk3thsk(yb, xa, rr)  # method 1
            dd = self.poly_interp_4d(xa, yb, rr)  # method 2
        else:
            # Beyond the grid => extrapolation with polynomial of 5th order
            # here, it may appears some larger, some smaller because of compR
            for icompr in range(ngridpoint.shape[0]):
                for jcompr in range(ngridpoint.shape[0]):
                    ngrid_ = int(ngridpoint[icompr, jcompr])
                    rmax_ = (ngrid_ - 1) * incr + tail
                    if rr > rmax_:
                        dd[icompr, jcompr] = 0.0
                    elif ind < ngrid_:
                        ilast = min(ngrid_, int(ind + ninterp / 2 + 1))
                        ilast = max(ninterp, ilast)
                        for ii in range(ninterp):
                            xa[ii] = (ilast - ninterp + ii) * incr
                        yb = datalist[icompr, jcompr,
                                      ilast - ninterp - 1:ilast - 1]
                        dd[icompr, jcompr] = self.poly_interp_2d(xa, yb, rr)
                    else:
                        dr = rr - rmax_
                        ilast = ngrid_
                        xii = t.linspace(0, ninterp - 1, ninterp)
                        xa = (ilast - ninterp + xii) * incr
                        yb = datalist[icompr, jcompr,
                                      ilast - ninterp - 1: ilast - 1]
                        y0 = self.poly_interp_2d(
                            xa, yb, xa[ninterp - 1] - delta_r)
                        y2 = self.poly_interp_2d(
                            xa, yb, xa[ninterp - 1] + delta_r)
                        ya = datalist[icompr, jcompr,
                                      ilast - ninterp - 1: ilast - 1]
                        y1 = ya[ninterp - 1]
                        y1p = (y2 - y0) / (2.0 * delta_r)
                        y1pp = (y2 + y0 - 2.0 * y1) / (delta_r * delta_r)
                        dd[icompr, jcompr] = self.poly5_zero(
                            y1, y1p, y1pp, dr, -1.0 * tail)
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

    def poly_check(self, xp, yp, rr, issameatom):
        if rr < 1e-1 and not issameatom:
            raise ValueError("distance between different atoms < ", 1e-1)
        elif rr < 1e-1 and issameatom:
            return t.zeros(yp.shape[0], yp.shape[1], yp.shape[3])
        elif rr > 12.:  # temporal code, revise !!!
            return t.zeros(yp.shape[0], yp.shape[1], yp.shape[3])
        else:
            return self.poly_interp_4d(xp, yp, rr)

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
    """Eigenvalue solver for (general) eigenvalue problem.

    Notes
    -----
    In current DFTB-ML framework, general eigenvalue problems can not be
    solved directly with pytorch, therefore we have to transfer to normal
    eigenvalue problem.

    """

    def __init__(self, eigenmethod='cholesky'):
        """Initialize parameters.

        Parameters
        ----------
        self.eigenmethod: [`string`], optional
            The method for general eigenvalue problem.
        Notes
        -----
        default is cholesky method for general eigenvalue problem, other
        options include Löwdin orthonormalization.
        """
        self.eigenmethod = eigenmethod

    def eigen(self, A, B=None, Lbatch=False, atomindex=None, ibatch=None, **kwargs):
        """Choose different mothod for (general) eigenvalue problem.

        Parameters
        ----------
        A : `torch.tensor` [`float`]
            Real symmetric matrix for which the eigenvalues & eigenvectors
            are to be computed. This is typically the Fock matrix.
        B : `torch.tensor` [`float`], optional
            Complementary positive definite real symmetric matrix for the
            generalised eigenvalue problem. Typically the overlap matrix.
        """
        # get the atom orbital index of single/batch systems
        self.atomindex = atomindex

        # simple eigenvalue problem
        if B is None:
            eigval, eigvec = t.symeig(A, eigenvectors=True)

        # general eigenvalue problem
        else:
            # test the shape size
            assert A.shape == B.shape

            # select decomposition method cholesky
            if self.eigenmethod == 'cholesky':
                eigval, eigvec = self.cholesky(A, B, Lbatch, ibatch)

            # select decomposition method lowdin
            elif self.eigenmethod == 'lowdin':
                eigval, eigvec = self.lowdin_qr(A, B, Lbatch)

        # return eigenvalue and eigenvector
        return eigval, eigvec

    def cholesky(self, A, B, Lbatch, ibatch, direct_inv=False):
        """Cholesky decomposition.

        Parameters
        ----------
        A : `torch.tensor` [`float`]
            Real symmetric matrix for which the eigenvalues & eigenvectors
            are to be computed. This is typically the Fock matrix.
        B : `torch.tensor` [`float`], optional
            Complementary positive definite real symmetric matrix for the
            generalised eigenvalue problem. Typically the overlap matrix.

        Notes
        -----
        Cholesky decomposition of B: B = LL^{T}
        transfer general eigenvalue problem:
            AX = λBX ==> (L^{-1}AL^{-T})(L^{T}X) = λ(L^{T}X)

        """
        # get how many systems in batch and the largest atom index
        nbatch = len(self.atomindex)
        maxind = max([iindex[-1] for iindex in self.atomindex])
        chol_l = t.empty(nbatch, maxind, maxind)

        # get the decomposition L, B = LL^{T} and padding zero
        chol_l = utilsbatch.pack([t.cholesky(
            iB[:self.atomindex[ii][-1], :self.atomindex[ii][-1]])
            for iB, ii in zip(B, ibatch)])

        # directly use inverse matrix
        if direct_inv:

            # L^{-1}
            linv = t.inverse(chol_l)

            # (L^{-1} @ A) @ L^{-T}
            linv_a_linvt = linv @ A @ linv.T

        else:

            # get the t.eye which share the last dimension of A
            eye_ = t.eye(A.shape[-1], dtype=A.dtype)

            # get L^{-1}, the 1st method only work if all sizes are the same
            # linv, _ = t.solve(eye_.unsqueeze(0).expand(A.shape), chol_l)
            linv = utilsbatch.pack([t.solve(t.eye(
                self.atomindex[ii][-1]), il[:self.atomindex[ii][-1], :self.atomindex[ii][-1]])[0]
                for il, ii in zip(chol_l, ibatch)])

            # get L^{-T}
            linvt = t.stack([il.T for il in linv])

            # (L^{-1} @ A) @ L^{-T}
            linv_a_linvt = linv @ A @ linvt

        # get eigenvalue of (L^{-1} @ A) @ L^{-T}
        # RuntimeError: Function 'SymeigBackward' returned nan values in its 0th output.
        # eigval, eigvec_ = t.symeig(linv_a_linvt, eigenvectors=True)
        eigval_eigvec = [
            t.symeig(il[:self.atomindex[ii][-1], :self.atomindex[ii][-1]],eigenvectors=True)
            for il, ii in zip(linv_a_linvt, ibatch)]
        eigval = pad_sequence([i[0] for i in eigval_eigvec]).T
        eigvec_ = utilsbatch.pack([i[1] for i in eigval_eigvec])

        # transfer eigenvector from (L^{-1} @ A) @ L^{-T} to AX = λBX
        eigvec = linvt @ eigvec_

        # return eigenvalue and eigen vector
        return eigval, eigvec

    def lowdin_symeig(self, A, B):
        """Löwdin orthonormalization.

        Parameters
        ----------
        A : `torch.tensor` [`float`]
            Real symmetric matrix for which the eigenvalues & eigenvectors
            are to be computed. This is typically the Fock matrix.
        B : `torch.tensor` [`float`], optional
            Complementary positive definite real symmetric matrix for the
            generalised eigenvalue problem. Typically the overlap matrix.

        Notes
        -----
        BX = λX, if λ' = λ.diag(), then get S^{-1/2} = X λ'^{-1/2} X^{T},
        AX' = λBX' ==> (S^{-1/2}AS^{-1/2})(S^{1/2}X') = λ(S_{1/2}X')
        The sqrt and inverse is not friendly to gradient!!

        """
        # construct S^{-1/2} = X λ^{-1/2} X^{T}, where SX = λX
        lam_, l_vec = t.symeig(B, eigenvectors=True)
        lam_sqrt_inv = t.sqrt(lam_.inverse())
        S_sym = t.mm(l_vec, t.mm(lam_sqrt_inv.diag(), l_vec.t()))

        # build H' (SHS), where H' = S^{-1/2}AS^{-1/2})
        SHS = t.mm(S_sym, t.mm(A, S_sym))

        # get eigenvalue and eigen vector of H'
        eigval, eigvec_ = t.symeig(SHS, eigenvectors=True)

        # get eigenvector of H from H'
        eigvec = t.mm(S_sym, eigvec_)
        return eigval, eigvec

    def lowdin_svd(self, A, B):
        """Löwdin orthonormalization with SVD decomposition.

        Parameters
        ----------
        A : `torch.tensor` [`float`]
            Real symmetric matrix for which the eigenvalues & eigenvectors
            are to be computed. This is typically the Fock matrix.
        B : `torch.tensor` [`float`], optional
            Complementary positive definite real symmetric matrix for the
            generalised eigenvalue problem. Typically the overlap matrix.

        """
        ub, sb, vb = t.svd(B)
        sb_sqrt_inv = t.sqrt(1 / sb)
        S_sym = ub @ sb_sqrt_inv.diag() @ vb.t()
        SHS = S_sym @ A @ S_sym

        ub2, sb2, vb2 = t.svd(SHS)
        eigvec = S_sym @ ub2
        return sb2, eigvec

    def lowdin_qr_eig(self, A, B, Lbatch):
        """Löwdin orthonormalization.

        Parameters
        ----------
        A : `torch.tensor` [`float`]
            Real symmetric matrix for which the eigenvalues & eigenvectors
            are to be computed. This is typically the Fock matrix.
        B : `torch.tensor` [`float`], optional
            Complementary positive definite real symmetric matrix for the
            generalised eigenvalue problem. Typically the overlap matrix.

        Notes
        -----
        B = USV_{T}, B^{-1/2} = US^{-1/2}V^{T},
        AX' = λBX' ==> (S^{-1/2}AS^{-1/2})(S^{1/2}X') = λ(S_{1/2}X')
        The sqrt and inverse is not friendly to gradient!!

        """

        assert A.shape == B.shape
        assert A.dtype == B.dtype

        # the max iteration of QR decomposition
        niter = 20

        # define matrices used during QR
        size = (niter + 1, *tuple(A.shape))
        BQ = t.zeros(size, dtype=A.dtype)

        # for single system
        if not Lbatch:
            beye = t.eye(A.shape[-1], dtype=A.dtype)

        # for multi system
        elif Lbatch:
            beye = t.eye(A.shape[-1],
                         dtype=A.dtype).unsqueeze(0).expand(A.shape)

        # define temporal BQ matrix initial value
        BQ[0] = B

        # QR decomposition of B and get eigenvalue of B
        icount = 0
        while True:
            Q_, R_ = t.qr(BQ[0])
            BQ = t.roll(BQ, 1, 0)

            # QR loop to get eigenvalue
            beye = beye @ Q_
            BQ[0] = R_ @ Q_

            # convergence condition
            icount += 1
            if max(abs(Q_.masked_select(~t.eye(Q_.shape[-1], dtype=bool)))) < 1e-4:
                break
            if icount > niter:
                print('Warning: QR decomposition do not reach convergence')
                break

        # extract the  diag of last BQ as eigenvalue
        eigval_b = BQ[0]
        diagb_sqrt_inv = t.sqrt(1 / eigval_b.diag()).diag()

        # the eigenvalue of H'(H'=S'AS', S' = XS^{-1/2}X.T) is from symeig
        S_sym = beye @ diagb_sqrt_inv @ beye.T
        SHS = S_sym @ A @ S_sym
        eigval, eigvec_ = t.symeig(SHS, eigenvectors=True)

        # get eigenvector of original H from H'
        eigvec = S_sym @ eigvec_
        return eigval, eigvec

    def lowdin_qr(self, A, B, Lbatch):
        """Löwdin orthonormalization with QR decomposition.

        Parameters
        ----------
        A : `torch.tensor` [`float`]
            Real symmetric matrix for which the eigenvalues & eigenvectors
            are to be computed. This is typically the Fock matrix.
        B : `torch.tensor` [`float`], optional
            Complementary positive definite real symmetric matrix for the
            generalised eigenvalue problem. Typically the overlap matrix.

        Notes
        -----
        """
        assert A.shape == B.shape
        assert A.dtype == B.dtype

        # the max iteration of QR decomposition
        niter = 20

        # define matrices used during QR
        size = (niter + 1, *tuple(A.shape))
        BQ = t.zeros(size, dtype=A.dtype)
        ABQ = t.zeros(size, dtype=A.dtype)

        # for single system
        if not self.batch:
            beye = t.eye(A.shape[-1], dtype=A.dtype)
            abeye = t.eye(A.shape[-1], dtype=A.dtype)

        # for multi system
        elif Lbatch:
            beye = t.eye(
                A.shape[-1], dtype=A.dtype).unsqueeze(0).expand(A.shape)
            abeye = t.eye(
                A.shape[-1], dtype=A.dtype).unsqueeze(0).expand(A.shape)

        # define temporal BQ matrix initial value
        BQ[0] = B

        # QR decomposition of B and get eigenvalue of B
        icount = 0
        while True:
            Q_, R_ = t.qr(BQ[0])
            BQ = t.roll(BQ, 1, 0)

            # QR loop to get eigenvalue
            beye = beye @ Q_
            BQ[0] = R_ @ Q_

            # convergence condition
            icount += 1
            if max(abs(Q_.masked_select(~t.eye(Q_.shape[-1], dtype=bool)))) < 1e-4:
                break
            if icount > niter:
                print('Warning: QR decomposition do not reach convergence')
                break

        # extract the  diag of last BQ as eigenvalue
        eigval_b = BQ[0]
        if not Lbatch:
            diagb_sqrt_inv = t.sqrt(1 / eigval_b.diag()).diag()
            S_sym = beye @ diagb_sqrt_inv @ beye.T
        elif Lbatch:
            diagb_sqrt_inv = t.stack([t.sqrt(1 / eigval_b[i].diag()).diag()
                                      for i in range(self.nb)])
            S_sym = beye @ diagb_sqrt_inv @ t.stack([beye[i].T for i in range(self.nb)])

        # construct H' = S^{-1/2}AS^{-1/2}
        SHS = S_sym @ A @ S_sym
        ABQ[0] = SHS

        # QR decomposition of H' and get eigenvalue of H'
        icount = 0
        while True:
            Q2_, R2_ = t.qr(ABQ[0])
            ABQ = t.roll(ABQ, 1, 0)

            # QR loop to get eigenvalue
            abeye = abeye @ Q2_
            ABQ[0] = R2_ @ Q2_

            # convergence condition
            icount += 1
            if max(abs(Q2_.masked_select(~t.eye(Q2_.shape[-1], dtype=bool)))) < 1e-4:
                break
            if icount > niter:
                print('Warning: QR decomposition do not reach convergence')
                break

        # get eigenvalue of general AX = lambdaBX
        if not Lbatch:
            eigval_ab = ABQ[0].diag()
            eigvec_ = S_sym @ abeye

            # the order is not chaotic, sort from samll to large
            eigenval, indices = t.sort(eigval_ab, -1)
            eigvec = t.stack([eigvec_.T[i] for i in indices]).T

        elif Lbatch:
            eigval_ab = t.stack([ABQ[0][i].diag() for i in range(self.nb)])
            eigvec_ = S_sym @ abeye

            # the order is not chaotic, sort from samll to large
            eigenval, indices = t.sort(eigval_ab, -1)
            eigvec = t.stack([t.stack([eigvec_.T[i] for i in indices[j]]).T
                              for j in range(self.nb)])

        '''
        sort = eigval_ab.sort()
        for ii in range(0, B.shape[0]):
            eigval[ii] = eigval_ab[int(t.tensor(sort[1])[ii])]
            eigvec[:, ii] = eigvec_[:, int(t.tensor(sort[1])[ii])]
        print(eigvec_, 'ss \n', eigvec, 'val \n', eigval)'''
        return eigenval, eigvec


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


class polySpline:
    """Polynomial spline.

    See: https://en.wikipedia.org/wiki/Spline_(mathematics)

    Args:
        xp (tensor): one dimension grid points
        yp (tensor): one or two dimension grid points
        dd (torch.float64): the point to return spline values
        parameter (list, optional): a list of parameters get from grid points,
            e.g., for cubic, parameter=[a, b, c, d]

    Returns:
        result (torch.float64): spline interpolation value at dd
    """

    def __init__(self, x=None, y=None, d=None, abcd=None, kind='cubic'):
        self.xp = x
        self.yp = y
        self.dd = d
        self.abcd = abcd

        # boundary condition
        if self.dd is not None:
            if not self.xp[0] < self.dd < self.xp[-1]:
                raise ValueError("%s is out of boundary" % self.dd)

            # get the nearest grid point index
            if t.is_tensor(self.xp):
                self.ddind = bisect.bisect(self.xp.numpy(), self.dd) - 1
            elif type(self.xp) is np.ndarray:
                self.ddind = bisect.bisect(self.xp, self.dd) - 1

            # according to the order to choose spline method
            if kind =='linear':
                self.ynew = self.linear()
            elif kind == 'cubic':
                self.ynew = self.cubic()
            else:
                raise NotImplementedError("%s is unsupported" % kind)

    def linear(self):
        """Calculate linear interpolation."""
        pass

    def cubic(self):
        """Calculate cubic in spline interpolation."""
        if self.abcd is None:
            a, b, c, d = self.get_abcd()
        else:
            a, b, c, d = self.abcd

        dx = self.dd - self.xp[self.ddind]
        return a[self.ddind] + b[self.ddind] * dx + c[self.ddind] * dx ** 2.0 + d[self.ddind] * dx ** 3.0

    def get_abcd(self):
        """Get parameter a, b, c, d for cubic spline interpolation."""
        self.nx = self.xp.shape[0]

        # a is grid point values, step 1
        a = self.yp

        # get the differnce between grid points
        self.diff_xp = self.xp[1:] - self.xp[:-1]

        # get b, c, d from reference website: step 3~9
        if self.yp.dim() == 1:
            b = t.zeros(self.nx - 1)
            d = t.zeros(self.nx - 1)
            A = self.cala()
        else:
            b = t.zeros(self.nx - 1, self.yp.shape[1])
            d = t.zeros(self.nx - 1, self.yp.shape[1])

        A = self.cala()
        B = self.calb()

        # least squares and least norm problems
        c, _ = t.lstsq(B, A)
        for i in range(self.nx - 1):
            b[i] = (self.yp[i + 1] - self.yp[i]) / self.diff_xp[i] - \
                self.diff_xp[i] * (c[i + 1] + 2.0 * c[i]) / 3.0
            d[i] = (c[i + 1] - c[i]) / (3.0 * self.diff_xp[i])
        return a, b, c, d

        '''A = self.get_A()
        B = self.get_B()

        # least squares and least norm problems
        M, _ = t.lstsq(B, A)
        for i in range(self.nx - 1):
            a[i] = (M[i + 1]- M[i]) / self.xp[i] / 6
            b[i] = (self.xp[i + 1] * M[i]- self.xp[i] * M[i + 1]) / self.diff_xp[i] / 2
            c[i] = (self.yp[i + 1] - self.yp[i]) / self.diff_xp[i] - self.diff_xp[i] / 6 * (M[i+1] - M[i])
            d[i] = (self.xp[i + 1] * self.yp[i] - self.xp[i] * self.yp[i + 1]) / self.diff_xp[i] - \
                self.diff_xp[i] / 6 * (self.xp[i + 1] * M[i] - self.xp[i] * M[i + 1])
        return d, c, b, a'''

    def get_B(self):
        # natural boundary condition, the first and last are zero
        B = t.zeros(self.nx, dtype=t.float64)
        for i in range(self.nx - 2):
            B[i + 1] = 6.0 * ((self.yp[i + 2] - self.yp[i + 1]) /
                        self.diff_xp[i + 1] -
                        (self.yp[i + 1] - self.yp[i]) /
                        self.diff_xp[i]) / (self.diff_xp[i + 1] + self.diff_xp[i])
        return B

    def get_A(self):
        """Calculate a para in spline interpolation."""
        A = t.zeros((self.nx, self.nx), dtype=t.float64)
        A[0, 0] = 1.0
        for i in range(self.nx - 2):
            A[i + 1, i + 1] = 2.0
            A[i + 1, i] = self.diff_xp[i] / (self.diff_xp[i] + self.diff_xp[i + 1])
            A[i + 1, i + 2] = 1 - A[i + 1, i]

        A[self.nx - 1, self.nx - 1] = 1.0
        return A

    def cala(self):
        """Calculate a para in spline interpolation."""
        aarr = t.zeros(self.nx, self.nx)
        aarr[0, 0] = 1.0
        for i in range(self.nx - 1):
            if i != (self.nx - 2):
                aarr[i + 1, i + 1] = 2.0 * (self.diff_xp[i] + self.diff_xp[i + 1])
            aarr[i + 1, i] = self.diff_xp[i]
            aarr[i, i + 1] = self.diff_xp[i]

        aarr[0, 1] = 0.0
        aarr[self.nx - 1, self.nx - 2] = 0.0
        aarr[self.nx - 1, self.nx - 1] = 1.0
        return aarr

    def calb(self):
        """Calculate b para in spline interpolation."""
        barr = t.zeros(*self.yp.shape, dtype=self.diff_xp.dtype)
        for i in range(self.nx - 2):
            barr[i + 1] = 3.0 * (self.yp[i + 2] - self.yp[i + 1]) / \
                self.diff_xp[i + 1] - 3.0 * (self.yp[i + 1] - self.yp[i]) / \
                self.diff_xp[i]
        return barr

    def diff(self, mat, axis=-1):
        if len(mat.shape) == 2:
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


"""Test spline interpolation.
print("Spline test")
xarr = t.tensor([-0.5, 0.0, 0.5, 1.0, 1.5])
yarr = t.tensor([3.2, 2.7, 6, 5, 6.5])
# spline = Spline(x, y)
rx = t.linspace(-0.2, 1.3, 100)
ry = [polySpline(xarr, yarr, i).cubic() for i in rx]
# ry = [spline.calc(i) for i in rx]
plt.plot(xarr, yarr, "xb")
plt.plot(rx, ry, "-r")
plt.grid(True)
plt.axis("equal")
plt.show()"""


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

        coeff = t.tensor([[1., 0., 0., 0.], [0., 0., 1., 0.],
                          [-3., 3., -2., -1.], [2., -2., 1., 1.]])
        coeff_ = t.tensor([[1., 0., -3., 2.], [0., 0., 3., -2.],
                           [0., 1., -2., 1.], [0., 0., -1., 1.]])
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


class BicubInterpVec:
    """Vectorized bicubic interpolation method for DFTB.

    reference: https://en.wikipedia.org/wiki/Bicubic_interpolation
    """

    def __init__(self, para, ml):
        """Get interpolation with two variables."""
        self.para = para
        self.ml = ml

    def bicubic_2d(self, xmesh, zmesh, xi, yi):
        """Build fmat.

        [[f(0, 0),  f(0, 1),   f_y(0, 0),  f_y(0, 1)],
         [f(1, 0),   f(1, 1),   f_y(1, 0),  f_y(1, 1)],
         [f_x(0, 0), f_x(0, 1), f_xy(0, 0), f_xy(0, 1)],
         [f_x(1, 0), f_x(1, 1), f_xy(1, 0), f_xy(1, 1)]]
        a_mat = coeff * famt * coeff_

        Args:
            xmesh: x (2D), [natom, ngrid_point]
            zmesh: z (5D), [natom, natom, ngrid_point, ngrid_point, 20]
            ix, iy: the interpolation point
        Returns:
            p(x, y) = [1, x, x**2, x**3] * a_mat * [1, y, y**2, y**3].T
        """
        # check if xi, yi is out of range of xmesh, ymesh
        xmin = t.ge(xi, self.ml['compr_min'])
        xmax = t.le(xi, self.ml['compr_max'])
        ymin = t.ge(yi, self.ml['compr_min'])
        ymax = t.le(yi, self.ml['compr_max'])
        assert False not in xmin
        assert False not in xmax
        assert False not in ymin
        assert False not in ymax

        # directly give the coefficients matrices
        coeff = t.tensor([[1., 0., 0., 0.], [0., 0., 1., 0.],
                          [-3., 3., -2., -1.], [2., -2., 1., 1.]])
        coeff_ = t.tensor([[1., 0., -3., 2.], [0., 0., 3., -2.],
                           [0., 1., -2., 1.], [0., 0., -1., 1.]])

        # get the nearest grid points indices of xi and yi
        self.nx0 = []
        for iat in range(len(xi)):

            # xi[iat] is in one of the grid point
            if xi[iat] in xmesh[iat]:
                self.nx0.append(np.searchsorted(
                    xmesh[iat].detach().numpy(), xi[iat].detach().numpy()))

            # xi[iat] is between two grid points
            else:
                self.nx0.append(np.searchsorted(
                    xmesh[iat].detach().numpy(), xi[iat].detach().numpy()) - 1)

        # build surrounding grid points indices, where _1 means -1, the
        # previous grid points, so is the 0, 1 ... along x, and y axes
        self.nx_1, self.nx1, self.nx2, self.nind = [], [], [], []
        [self.nind.append(i) for i in range(len(xi))]
        [self.nx1.append(i + 1) for i in self.nx0]

        # get the grid indices of the surrounding two points
        # xmesh, and ymesh is the same, we can write index togetehr
        for i in self.nx0:
            if i >= self.nind[1]:
                self.nx_1.append(i - 1)

            # if x, y is in the first grid mesh, self.nx_1 and self.nx0 will be
            # the same, there will be no gradient (the derivative is zero then)
            else:
                self.nx_1.append(i)
            if i <= self.nind[-3]:
                self.nx2.append(i + 2)

            # if x, y is in the last grid mesh, self.nx2 and self.nx1 will be
            # the same, there will be no gradient (the derivative is zero then)
            else:
                self.nx2.append(i + 1)

        # this is to transfer x or y to fraction, with natom element
        x_ = (xi - xmesh.T[self.nx0, self.nind]) / (xmesh.T[
            self.nx1, self.nind] - xmesh.T[self.nx0, self.nind])

        # build [1, x, x**2, x**3] matrices of all atoms, dimension: [4, natom]
        xmat = t.stack([x_ ** 0, x_ ** 1, x_ ** 2, x_ ** 3])

        # get four nearest grid points values, each will be: [natom, natom, 20]
        f00, f10, f01, f11 = self.fmat0th(zmesh)

        # get four nearest grid points derivative over x, y, xy
        f02, f03, f12, f13, f20, f21, f30, f31, f22, f23, f32, f33 = \
            self.fmat1th(xmesh, zmesh, f00, f10, f01, f11)
        fmat = t.stack([t.stack([f00, f01, f02, f03]),
                        t.stack([f10, f11, f12, f13]),
                        t.stack([f20, f21, f22, f23]),
                        t.stack([f30, f31, f32, f33])])
        a_mat = t.einsum('ii,ijlmn,jj->ijlmn', coeff, fmat, coeff_)
        return t.einsum('ij,iijkn,ik->jkn', xmat, a_mat, xmat)
        # return t.stack([t.matmul(t.matmul(xmat[i], a_mat[i]), xmat[i]) for i in range(len(xi))])

    def fmat0th(self, zmesh):
        """Construct f(0/1, 0/1) in fmat."""
        f00 = t.stack([t.stack([zmesh[i, j, self.nx0[i], self.nx0[j]]
                                for j in self.nind]) for i in self.nind])
        f10 = t.stack([t.stack([zmesh[i, j, self.nx1[i], self.nx0[j]]
                                for j in self.nind]) for i in self.nind])
        f01 = t.stack([t.stack([zmesh[i, j, self.nx0[i], self.nx1[j]]
                                for j in self.nind]) for i in self.nind])
        f11 = t.stack([t.stack([zmesh[i, j, self.nx1[i], self.nx1[j]]
                                for j in self.nind]) for i in self.nind])
        return f00, f10, f01, f11

    def fmat1th(self, xmesh, zmesh, f00, f10, f01, f11):
        """Get the 1st derivative of four grid points over x, y and xy."""
        f_10 = t.stack([t.stack([zmesh[i, j, self.nx_1[i], self.nx0[j]]
                                 for j in self.nind]) for i in self.nind])
        f_11 = t.stack([t.stack([zmesh[i, j, self.nx_1[i], self.nx1[j]]
                                 for j in self.nind]) for i in self.nind])
        f0_1 = t.stack([t.stack([zmesh[i, j, self.nx0[i], self.nx_1[j]]
                                 for j in self.nind]) for i in self.nind])
        f02 = t.stack([t.stack([zmesh[i, j, self.nx0[i], self.nx2[j]]
                                for j in self.nind]) for i in self.nind])
        f1_1 = t.stack([t.stack([zmesh[i, j, self.nx1[i], self.nx_1[j]]
                                 for j in self.nind]) for i in self.nind])
        f12 = t.stack([t.stack([zmesh[i, j, self.nx1[i], self.nx2[j]]
                                for j in self.nind]) for i in self.nind])
        f20 = t.stack([t.stack([zmesh[i, j, self.nx2[i], self.nx0[j]]
                                for j in self.nind]) for i in self.nind])
        f21 = t.stack([t.stack([zmesh[i, j, self.nx2[i], self.nx1[j]]
                                for j in self.nind]) for i in self.nind])

        # calculate the derivative: (F(1) - F(-1) / (2 * grid)
        # if there is no previous or next grdi point, it will be:
        # (F(1) - F(0) / grid or (F(0) - F(-1) / grid
        fy00 = t.stack([t.stack([(f01[i, j] - f0_1[i, j]) /
                                 (self.nx1[j] - self.nx_1[i])
                                 for j in self.nind]) for i in self.nind])
        fy01 = t.stack([t.stack([(f02[i, j] - f00[i, j]) /
                                 (self.nx1[j] - self.nx_1[i])
                                 for j in self.nind]) for i in self.nind])
        fy10 = t.stack([t.stack([(f11[i, j] - f1_1[i, j]) /
                                 (self.nx1[j] - self.nx_1[i])
                                 for j in self.nind]) for i in self.nind])
        fy11 = t.stack([t.stack([(f12[i, j] - f10[i, j]) /
                                 (self.nx1[j] - self.nx_1[i])
                                 for j in self.nind]) for i in self.nind])
        fx00 = t.stack([t.stack([(f10[i, j] - f_10[i, j]) /
                                 (self.nx1[j] - self.nx_1[i])
                                 for j in self.nind]) for i in self.nind])
        fx01 = t.stack([t.stack([(f20[i, j] - f00[i, j]) /
                                 (self.nx1[j] - self.nx_1[i])
                                 for j in self.nind]) for i in self.nind])
        fx10 = t.stack([t.stack([(f11[i, j] - f_11[i, j]) /
                                 (self.nx1[j] - self.nx_1[i])
                                 for j in self.nind]) for i in self.nind])
        fx11 = t.stack([t.stack([(f21[i, j] - f01[i, j]) /
                                 (self.nx1[j] - self.nx_1[i])
                                 for j in self.nind]) for i in self.nind])
        fxy00, fxy11 = fy00 * fx00, fx11 * fy11
        fxy01, fxy10 = fx01 * fy01, fx10 * fy10
        return fy00, fy01, fy10, fy11, fx00, fx01, fx10, fx11, fxy00, fxy01, \
            fxy10, fxy11


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
