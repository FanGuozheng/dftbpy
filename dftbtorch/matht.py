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
from torch.autograd import Variable
from ml.padding import pad1d, pad2d
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
        ninterp = self.skf['sizeInterpolationPoints']
        delta_r = self.skf['deltaSK']

        if type(datalist) is np.ndarray:
            datalist = t.from_numpy(np.asarray(datalist))

        tail = 5 * incr
        rmax = (ngridpoint - 1) * incr + tail
        ind = int(rr / incr)
        leng = ngridpoint

        # thye input SKF must have more than 8 grid points
        if leng < ninterp + 1:
            raise ValueError("Warning: not enough points for interpolation!")

        # distance beyond grid points in SKF
        if rr >= rmax:
            dd = t.zeros(20)
        # => polynomial fit
        elif ind < leng:
            ilast = min(leng, int(ind + ninterp / 2 + 1))
            ilast = max(ninterp, ilast)
            xa = (ilast - ninterp) * incr + t.arange(ninterp) * incr
            yb = datalist[ilast - ninterp - 1: ilast - 1]

            # two interpolation methods
            # dd = self.polysk3thsk(yb, xa, rr)  # method 1
            dd = self.poly_interp_2d(xa, yb, rr)  # method 2
        # Beyond the grid => extrapolation with polynomial of 5th order
        else:
            dr = rr - rmax
            ilast = leng
            xa = (ilast - ninterp) * incr + t.arange(ninterp) * incr
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


class EigenSolver(t.autograd.Function):
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

    # Note that 'none' is included only for testing purposes
    KNOWN_METHODS = ['cond', 'lorn', 'none']

    @staticmethod
    def forward(ctx, a, method='cond', factor=1E-12):
        """Calculate the eigenvalues and eigenvectors of a symmetric matrix.
        Finds the eigenvalues and eigenvectors of a real symmetric
        matrix using the torch.symeig function.
        Arguments:
            a: A real symmetric matrix whose eigenvalues & eigenvectors will
                be computed.
            method: Broadening method to used, available options are:
                    - "cond" for conditional broadening.
                    - "lorn" for Lorentzian broadening.
                See class doc-string for more info on these methods.
                [DEFAULT='cond']
            factor: Degree of broadening (broadening factor). [Default=1E-12]
        Returns:
            w: The eigenvalues, in ascending order.
            v: The eigenvectors.
        Notes:
            The ctx argument is auto-parsed by PyTorch & is used to pass data
            from the .forward() method to the .backward() method. This is not
            normally described in the docstring but has been done here to act
            as an example.
        Warnings:
            Under no circumstances should `factor` be a torch.tensor entity.
            The `method` and `factor` parameters MUST be passed as positional
            arguments and NOT keyword arguments.
        """
        # Check that the method is of a known type
        if method not in EigenSolver.KNOWN_METHODS:
            raise ValueError('Unknown broadening method selected.')

        # Compute eigen-values & vectors using torch.symeig.
        w, v = t.symeig(a, eigenvectors=True)

        # Save tensors that will be needed in the backward pass
        ctx.save_for_backward(w, v)

        # Save the broadening factor and the selecte broadening method.
        ctx.bf, ctx.bm = factor, method

        # Store dtype to prevent dtype mixing (don't mix dtypes)
        ctx.dtype = a.dtype

        # Return the eigenvalues and eigenvectors
        return (w, v)

    @staticmethod
    def backward(ctx, w_bar, v_bar):
        """Evaluates gradients of the eigen decomposition operation.
        Evaluates gradients of the matrix from which the eigenvalues
        and eigenvectors were taken.
        Arguments:
            w_bar: Gradients associated with the the eigenvalues.
            v_bar: Gradients associated with the eigenvectors.
        Returns:
            a_bar: Gradients associated with the `a` tensor.
        Notes:
            See class doc-string for a more detailed description of this
            method.
        """
        # Equation to variable legend
        #   w <- λ
        #   v <- U

        # __Preamble__
        # Retrieve eigenvalues (w) and eigenvectors (v) from ctx
        w, v = ctx.saved_tensors

        # Retrieve, the broadening factor and convert to a tensor entity
        bf = t.tensor(ctx.bf, dtype=ctx.dtype)

        # Retrieve the broadening method
        bm = ctx.bm

        # Form the eigenvalue gradients into diagonal matrix
        lambda_bar = w_bar.diag_embed()

        # Identify the indices of the upper triangle of the F matrix
        tri_u = t.triu_indices(*v.shape[-2:], 1)

        # Construct the deltas
        deltas = w[..., tri_u[1]] - w[..., tri_u[0]]

        # Apply broadening
        if ctx.bm == 'cond':  # <- Conditional broadening
            deltas = 1 / t.where(t.abs(deltas) > bf,
                                 deltas, bf) * t.sign(deltas)
        elif ctx.bm == 'lorn':  # <- Lorentzian broadening
            deltas = deltas / (deltas**2 + bf)
        elif ctx.bm == 'none':  # <- Debugging only
            deltas = 1 / deltas
        else:  # <- Should be impossible to get here
            raise ValueError(f'Unknown broadening method {ctx.bm}')

        # Construct F matrix where F_ij = v_bar_j - v_bar_i; construction is
        # done in this manner to avoid 1/0 which can cause intermittent and
        # hard-to-diagnose issues.
        F = t.zeros(*w.shape, w.shape[-1], dtype=ctx.dtype)
        F[..., tri_u[0], tri_u[1]] = deltas  # <- Upper triangle
        F[..., tri_u[1], tri_u[0]] -= F[..., tri_u[0], tri_u[1]]  # <- lower triangle

        # Construct the gradient following the equation in the doc-string.
        a_bar = v @ (lambda_bar + sym(F * (v.transpose(-2, -1) @ v_bar))) @ v.transpose(-2, -1)

        # Return the gradient. PyTorch expects a gradient for each parameter
        # (method, bf) hence two extra Nones are returned
        return a_bar, None, None


    def eigen(self, A, B=None, Lbatch=False, norbital=None, ibatch=None, **kwargs):
        """Choose different mothod for (general) eigenvalue problem.

        Parameters
        ----------
        A :
            Real symmetric matrix for which the eigenvalues & eigenvectors
            are to be computed. This is typically the Fock matrix.
        B :
            Complementary positive definite real symmetric matrix for the
            generalised eigenvalue problem. Typically the overlap matrix.
        """
        self.inverse = kwargs['inverse'] if 'inverse' in kwargs else False
        self.norb = [sum(iorb) for iorb in norbital]

        # simple eigenvalue problem
        if B is None:
            eigval, eigvec = t.symeig(A, eigenvectors=True)
        else:
            # test the shape size
            assert A.shape == B.shape

            # select decomposition method cholesky
            if self.eigenmethod == 'cholesky':
                eigval, eigvec = self.cholesky(A, B, Lbatch, ibatch, self.inverse)
            # select decomposition method lowdin
            elif self.eigenmethod == 'lowdin':
                eigval, eigvec = self.lowdin_qr(A, B, Lbatch, self.inverse)

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
        # get how many systems in batch and the largest atom index, old method
        # nbatch = A.shape[0]
        # maxind = max([ia.shape[-1] for ia in A])
        # chol_l = t.zeros(nbatch, maxind, maxind, dtype=A.dtype)

        # get the decomposition L, B = LL^{T} and padding zero
        # chol_l = pad2d([t.cholesky(
        #     iB[:self.norb[ii], :self.norb[ii]]) for ii, iB in enumerate(B)])

        # Create a mask which is True wherever a column/row pair is 0-valued
        _is_zero = t.eq(B, 0)
        mask = t.all(_is_zero, dim=-1) & t.all(_is_zero, dim=-2)
        # Set the diagonals at these locations to 1
        B = B + t.diag_embed(mask.type(B.dtype))
        A = A + t.diag_embed(mask.type(A.dtype))
        chol_l = t.cholesky(B)

        # directly use inverse matrix
        if direct_inv:

            # L^{-1}
            linv = t.inverse(chol_l)
            linvt = t.transpose(linv, -1, -2)

            # (L^{-1} @ A) @ L^{-T}
            linv_a_linvt = linv @ A @ linvt

        else:

            # get the t.eye which share the last dimension of A
            # eye_ = t.eye(A.shape[-1], dtype=A.dtype)

            # get L^{-1}, the 1st method only work if all sizes are the same
            # linv, _ = t.solve(eye_.unsqueeze(0).expand(A.shape), chol_l)
            # linv = pad2d([t.solve(t.eye(
            #     self.norb[ii]), il[:self.norb[ii], :self.norb[ii]])[0]
            #     for ii, il in enumerate(chol_l)])

            # get L^{-T}
            # linvt = t.stack([il.T for il in linv])
            linv = t.solve(t.eye(A.shape[-1], dtype=A.dtype), chol_l)[0]
            linvt = t.transpose(linv, -1, -2)

            # (L^{-1} @ A) @ L^{-T}
            linv_a_linvt = linv @ A @ linvt

        # get eigenvalue of (L^{-1} @ A) @ L^{-T}
        # RuntimeError: Function 'SymeigBackward' returned nan values in its 0th output.
        # eigval_eigvec = [
        #      t.symeig(il[:self.norb[ii], :self.norb[ii]], eigenvectors=True)
        #      for ii, il in enumerate(linv_a_linvt)]
        # eigval = pad_sequence([i[0] for i in eigval_eigvec]).T
        # eigvec_ = pad2d([ieigv[1] for ieigv in eigval_eigvec])
        
        # eigval, eigvec_ = t.symeig(linv_a_linvt, eigenvectors=True)
        eigval, eigvec_ = EigenSolver.apply(linv_a_linvt)

        # transfer eigenvector from (L^{-1} @ A) @ L^{-T} to AX = λBX
        eigvec = linvt @ eigvec_
        eigval, eigvec = self._eig_sort_out(eigval, eigvec, False)
        # return eigenvalue and eigen vector
        return eigval, eigvec

    def _eig_sort_out(self, w, v, ghost=True):
        """Move ghost eigen values/vectors to the end of the array.
        Discuss the difference between ghosts (w=0) and auxiliaries (w=1)
        Performing and eigen-decomposition operation on a zero-padded packed
        tensor results in the emergence of ghost eigen-values/vectors. This can
        cause issues downstream, thus they are moved to the end here which means
        they can be easily clipped off should the user wish to do so.
        Arguments:
            w: The eigen-values.
            v: The eigen-vectors.
            ghost: Ghost-eigen-vlaues are assumed to be 0 if True, else assumed to
                be 1. If zero padded then this should be True, if zero padding is
                turned into identity padding then False should be used. This will
                also change the ghost eigenvalues from 1 to zero when appropriate.
                [DEFAULT=True]
        Returns:
            w: The eigen-values, with ghosts moved to the end.
            v: The eigen-vectors, with ghosts moved to the end.
        """
        eval = 0 if ghost else 1

        # Create a mask that is True when an eigen value is zero
        mask = w == eval
        # and its associated eigen vector is a column of a identity matrix:
        # i.e. all values are 1 or 0 and there is only a single 1.
        _is_one = t.eq(v, 1)  # <- precompute
        mask &= t.all(t.eq(v, 0) | _is_one, dim=1)
        mask &= t.sum(_is_one, dim=1) == 1  # <- Only a single "1"

        # Convert any auxiliary eigenvalues into ghosts
        if not ghost:
            w = w - mask.type(w.dtype)

        # Pull out the indices of the true & ghost entries and cat them together
        # so that the ghost entries are at the end.
        # noinspection PyTypeChecker
        indices = t.cat((t.stack(t.where(~mask)), t.stack(t.where(mask))), dim=-1)

        # argsort fixes the batch order and stops eigen-values accidentally being
        # mixed between different systems. As PyTorch's argsort is not stable, i.e.
        # it dose not respect any order already present in the data, numpy's argsort
        # must be used for now.
        if indices.device.type == 'cuda':
            sorter = np.argsort(indices.cpu()[0], kind='stable')
        else:
            sorter = np.argsort(indices[0], kind='stable')

        # Apply sorter to indices; use a tuple to make 1D & 2D cases compatible
        sorted_indices = tuple(indices[..., sorter])

        # Fix the order of the eigen values and eigen vectors.
        w = w[sorted_indices].reshape(w.shape)
        # Reshaping is needed to allow sorted_indices to be used for 2D & 3D
        v = v.transpose(-1, -2)[sorted_indices].reshape(v.shape).transpose(-1, -2)

        # Return the eigenvalues and eigenvectors
        return w, v

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

def sym(x, dim0=-1, dim1=-2):
    """Symmetries the specified tensor.
    Arguments:
        x: The tensor to be symmetrised.
        dim0: First dimension to be transposed. [DEFAULT=-1]
        dim1: Second dimension to be transposed [DEFAULT=-2]
    Returns:
        x_sym: The symmetrised tensor.
    """
    return (x + x.transpose(dim0, dim1)) / 2

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


class LinAl:
    """Linear Algebra."""

    def __init__(self, para):
        """Initialize parameters."""
        self.para = para

    def inv33_mat(self, in_):
        """Return inverse of 3 * 3 matrix, out_ = 1 / det(in_) adj(in_)."""
        out_ = t.zeros(3, 3)
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
