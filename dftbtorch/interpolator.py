import bisect
import numpy as np
import torch as t
import matplotlib.pyplot as plt
import json
import logging
import argparse
from cvxopt import matrix, solvers
from scipy.linalg import block_diag


class _Interpolate(object):
    """Common features for interpolation.

    Deal with the input data dtype, shape, dimension problems.

    Methods:
        __call__
        set_dtype
        set_shape
    """

    def __init__(self, x=None, y=None, d=None):
        self.set_dtype(x, y)
        self.set_shape(x, y)

    def __call__(self, x):
        pass

    def set_dtype(self, x, y):
        """Check the input dtype."""
        if type(x) is np.ndarray:
            x = t.from_numpy(x)
        if type(y) is np.ndarray:
            x = t.from_numpy(x)

        # return torch type
        x_dtype = x.dtype
        y_dtype = x.dtype

        # x and y dtype should be the same
        if x_dtype != y_dtype:
            raise ValueError("input x, y dtype not the same")
        self.dtype = x_dtype

    def set_shape(self, x, y):
        """Set the shape of x, y."""
        x_shape = x.shape
        if y is not None:
            y_shape = y.shape
            if x_shape == y_shape:
                self.is_same_shape = True
            else:
                self.is_same_shape = False

    def bound(self):
        pass


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
        # directly give the coefficients matrices
        coeff = t.tensor([[1., 0., 0., 0.], [0., 0., 1., 0.],
                          [-3., 3., -2., -1.], [2., -2., 1., 1.]])
        coeff_ = t.tensor([[1., 0., -3., 2.], [0., 0., 3., -2.],
                           [0., 1., -2., 1.], [0., 0., -1., 1.]])

        # get the nearest grid points indices of xi and yi
        xmesh_ = xmesh.cpu() if xmesh.device.type == 'cuda' else xmesh
        xi_ = xi.cpu() if xi.device.type == 'cuda' else xi
        self.nx0 = [bisect.bisect(xmesh_[ii].detach().numpy(),
                                  xi_[ii].detach().numpy()) - 1
                    for ii in range(len(xi))]
        # get all surrounding 4 grid points indices, _1 means previous grid point index
        self.nind = [ii for ii in range(len(xi))]
        self.nx1 = [ii + 1 for ii in self.nx0]
        self.nx_1 = [ii - 1 if ii >= 1 else ii for ii in self.nx0]
        self.nx2 = [ii + 2 if ii <= len(xmesh) - 3 else ii + 1 for ii in self.nx0]

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

        # method 1 to calculate a_mat, not stable
        # a_mat = t.einsum('ii,ijlmn,jj->ijlmn', coeff, fmat, coeff_)
        # return t.einsum('ij,iijkn,ik->jkn', xmat, a_mat, xmat)
        a_mat = t.matmul(t.matmul(coeff, fmat.permute(2, 3, 4, 0, 1)), coeff_)
        return t.stack([t.stack(
            [t.matmul(t.matmul(xmat[:, i], a_mat[i, j]), xmat[:, j])
             for j in range(len(xi))]) for i in range(len(xi))])

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
                                 (self.nx1[j] - self.nx_1[j])
                                 for j in self.nind]) for i in self.nind])
        fy01 = t.stack([t.stack([(f02[i, j] - f00[i, j]) /
                                 (self.nx1[j] - self.nx_1[j])
                                 for j in self.nind]) for i in self.nind])
        fy10 = t.stack([t.stack([(f11[i, j] - f1_1[i, j]) /
                                 (self.nx1[j] - self.nx_1[j])
                                 for j in self.nind]) for i in self.nind])
        fy11 = t.stack([t.stack([(f12[i, j] - f10[i, j]) /
                                 (self.nx1[j] - self.nx_1[j])
                                 for j in self.nind]) for i in self.nind])
        fx00 = t.stack([t.stack([(f10[i, j] - f_10[i, j]) /
                                 (self.nx1[i] - self.nx_1[i])
                                 for j in self.nind]) for i in self.nind])
        fx01 = t.stack([t.stack([(f20[i, j] - f00[i, j]) /
                                 (self.nx1[i] - self.nx_1[i])
                                 for j in self.nind]) for i in self.nind])
        fx10 = t.stack([t.stack([(f11[i, j] - f_11[i, j]) /
                                 (self.nx1[i] - self.nx_1[i])
                                 for j in self.nind]) for i in self.nind])
        fx11 = t.stack([t.stack([(f21[i, j] - f01[i, j]) /
                                 (self.nx1[i] - self.nx_1[i])
                                 for j in self.nind]) for i in self.nind])
        fxy00, fxy11 = fy00 * fx00, fx11 * fy11
        fxy01, fxy10 = fx01 * fy01, fx10 * fy10
        return fy00, fy01, fy10, fy11, fx00, fx01, fx10, fx11, fxy00, fxy01, \
            fxy10, fxy11


class PolySpline(_Interpolate):
    """Polynomial natural (linear, cubic) spline.

    See: https://en.wikipedia.org/wiki/Spline_(mathematics)
    You can either generate spline parameters(abcd) or offer spline
    parameters, and get spline interpolation results.

    Args:
        xp (tensor, optional): one dimension grid points
        yp (tensor, optional): one or two dimension grid points
        parameter (list, optional): a list of parameters get from grid points,
            e.g., for cubic, parameter=[a, b, c, d]

    Returns:
        result (tensor): spline interpolation value at dd
    """

    def __init__(self, x=None, y=None, abcd=None, kind='cubic'):
        """Initialize the interpolation class."""
        _Interpolate.__init__(self, x, y)
        self.xp = x
        self.yp = y
        self.abcd = abcd
        self.kind = kind

        # delete original input
        del x, y, abcd, kind

    def __call__(self, dd=None):
        """Evaluate the polynomial spline.

        Args:
        dd : torch.tensor
            Points to evaluate the interpolant at.

        Returns:
        ynew : torch.tensor
            Interpolated values. Shape is determined by replacing
            the interpolation axis in the original array with the shape of x.
        """
        # according to the order to choose spline method
        self.dd = dd

        # if d is not None, it will return spline interpolation values
        if self.dd is not None:

            # boundary condition of d
            if not self.xp[0] <= self.dd <= self.xp[-1]:
                raise ValueError("%s is out of boundary" % self.dd)

            # get the nearest grid point index of d in x
            if t.is_tensor(self.xp):
                self.dind = bisect.bisect(self.xp.numpy(), self.dd) - 1
            elif type(self.xp) is np.ndarray:
                self.dind = bisect.bisect(self.xp, self.dd) - 1

        if self.kind == 'linear':
            self.ynew = self.linear()
        elif self.kind == 'cubic':
            self.ynew = self.cubic()
        else:
            raise NotImplementedError("%s is unsupported" % self.kind)
        return self.ynew

    def linear(self):
        """Calculate linear interpolation."""
        pass

    def cubic(self):
        """Calculate cubic spline interpolation."""
        # calculate a, b, c, d parameters, need input x and y
        if self.abcd is None:
            a, b, c, d = self.get_abcd()
        else:
            a, b, c, d = self.abcd

        dx = self.dd - self.xp[self.dind]
        return a[self.dind] + b[self.dind] * dx + c[self.dind] * dx ** 2.0 + d[self.dind] * dx ** 3.0

    def get_abcd(self):
        """Get parameter a, b, c, d for cubic spline interpolation."""
        assert self.xp is not None and self.yp is not None

        # get the first dim of x
        self.nx = self.xp.shape[0]

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

        # a is grid point values
        a = self.yp

        # return c (Ac=B) with least squares and least norm problems
        c, _ = t.lstsq(B, A)
        for i in range(self.nx - 1):
            b[i] = (a[i + 1] - a[i]) / self.diff_xp[i] - \
                self.diff_xp[i] * (c[i + 1] + 2.0 * c[i]) / 3.0
            d[i] = (c[i + 1] - c[i]) / (3.0 * self.diff_xp[i])
        return a, b, c.squeeze(), d

    def _get_abcd(self):
        """Get parameter a, b, c, d for cubic spline interpolation."""
        # get the first dim of x
        self.nx = self.xp.shape[0]

        # get the differnce between grid points
        self.h = self.xp[1:] - self.xp[:-1]

        # get the differnce between grid points
        self.ydiff = self.yp[1:] - self.yp[:-1]

        # setp 6, define l, mu, z
        ll = t.zeros(self.nx, dtype=self.dtype)
        mu = t.zeros(self.nx, dtype=self.dtype)
        zz = t.zeros(self.nx, dtype=self.dtype)
        alpha = t.zeros(self.nx, dtype=self.dtype)
        ll[0] = ll[-1] = 1.

        # step 7, calculate alpha, l, mu, z
        for i in range(1, self.nx - 1):
            alpha[i] = 3. * self.ydiff[i] / self.h[i] - \
                3. * self.ydiff[i - 1] / self.h[i - 1]
            ll[i] = 2 * (self.xp[i + 1] - self.xp[i - 1]) - \
                self.h[i - 1] * mu[i - 1]
            mu[i] = self.h[i] / ll[i]
            zz[i] = (alpha[i] - self.h[i - 1] * zz[i - 1]) / ll[i]

        # step 8, define b, c, d
        b = t.zeros(self.nx, dtype=self.dtype)
        c = t.zeros(self.nx, dtype=self.dtype)
        d = t.zeros(self.nx, dtype=self.dtype)

        # step 9, get b, c, d
        for i in range(self.nx - 2, -1, -1):
            c[i] = zz[i] - mu[i] * c[i + 1]
            b[i] = self.ydiff[i] / self.h[i] - \
                self.h[i] * (c[i + 1] + 2 * c[i]) / 3
            d[i] = (c[i + 1] - c[i]) / 3 / self.h[i]

        return self.yp, b, c, d

    def cala(self):
        """Calculate a para in spline interpolation."""
        aa = t.zeros(self.nx, self.nx)
        aa[0, 0] = 1.0
        for i in range(self.nx - 1):
            if i != (self.nx - 2):
                aa[i + 1, i + 1] = 2.0 * (self.diff_xp[i] + self.diff_xp[i + 1])
            aa[i + 1, i] = self.diff_xp[i]
            aa[i, i + 1] = self.diff_xp[i]

        aa[0, 1] = 0.0
        aa[self.nx - 1, self.nx - 2] = 0.0
        aa[self.nx - 1, self.nx - 1] = 1.0
        return aa

    def calb(self):
        """Calculate b para in spline interpolation."""
        bb = t.zeros(*self.yp.shape, dtype=self.diff_xp.dtype)
        for i in range(self.nx - 2):
            bb[i + 1] = 3.0 * (self.yp[i + 2] - self.yp[i + 1]) / \
                self.diff_xp[i + 1] - 3.0 * (self.yp[i + 1] - self.yp[i]) / \
                self.diff_xp[i]
        return bb

logger = logging.getLogger(__name__)

class CCS:

    def __init__(self):
        pass

    def twp_fit(self, filename):
        """ The function parses the input files and fits the reference data.

        Args:
            filename (str): The input file (input.json).
        """
        from collections import OrderedDict
        import pandas as pd

        try:
            with open(filename) as json_file:
                data = json.load(json_file, object_pairs_hook=OrderedDict)
        except FileNotFoundError:
            logger.critical(" input.json file missing")
            raise
        try:
            with open(data['Reference']) as json_file:
                struct_data = json.load(json_file, object_pairs_hook=OrderedDict)
        except FileNotFoundError:
            logger.critical(" Reference file with paiwise distances missing")
            raise
        except ValueError:
            logger.critical("Reference file not in json format")
            raise

        # Reading the input.json file and structure file to see the keys are matching
        atom_pairs = []
        ref_energies = []
        # Loop over different species
        for atmpair, values in data['Twobody'].items():
            logger.debug("\n The atom pair is : %s" % (atmpair))
            list_dist = []
            for snum, v in struct_data.items():  # loop over structures
                try:
                    list_dist.append(v[atmpair])
                except KeyError:
                    logger.critical(
                        " Name mismatch in input.json and structures.json")
                    raise
                try:
                    ref_energies.append(v['Energy'])
                except KeyError:
                    logger.critical(" Check Energy key in structure file")
                    raise

            dist_mat = pd.DataFrame(list_dist)
            dist_mat = dist_mat.values
            logger.debug(" Distance matrix for %s is \n %s " % (atmpair, dist_mat))
            atom_pairs.append(
                Twobody(atmpair, dist_mat, len(struct_data), **values))

        sto = t.zeros(len(struct_data), len(data['Onebody']))
        for i, key in enumerate(data['Onebody']):
            count = 0
            for k, v in struct_data.items():
                sto[count][i] = v['Atoms'][key]
                count = count+1
        n = Objective(atom_pairs, sto, ref_energies)
        n.solution()


def write_splinecoeffs(twb, coeffs, fname='splines.out', exp_head=False):
    """This function writes the spline output

    Args:
        twb (Twobody): Twobody class object.
        coeffs (ndarray): Array containing spline coefficients.
        fname (str, optional): Filename to output the spline coefficients. Defaults to 'splines.out'.
        exp_head (bool, optional): To fit an exponential function at shorter atomic distances. Defaults to False.
    """
    coeffs_format = ' '.join(['{:6.3f}'] * 2 + ['{:15.8E}'] * 4) + '\n'
    with open(fname, 'w') as fout:
        fout.write('Spline table\n')
        for index in range(len(twb.interval)-1):
            r_start = twb.interval[index]
            r_stop = twb.interval[index+1]
            fout.write(coeffs_format.format(r_start, r_stop, *coeffs[index]))


def write_error(mdl_eng, ref_eng, mse, fname='error.out'):
    """ Prints the errors in a file

    Args:
        mdl_eng (ndarray): Energy prediction values from splines.
        ref_eng (ndarray): Reference energy values.
        mse (float): Mean square error.
        fname (str, optional): Output filename.. Defaults to 'error.out'.
    """
    header = "{:<15}{:<15}{:<15}".format("Reference", "Predicted", "Error")
    error = abs(ref_eng - mdl_eng)
    maxerror = max(abs(error))
    footer = "MSE = {:2.5E}\nMaxerror = {:2.5E}".format(mse, maxerror)
    np.savetxt(fname,
               np.transpose([ref_eng, mdl_eng, error]),
               header=header,
               footer=footer,
               fmt="%-15.5f")

class Objective():
    """  Objective function for ccs method """

    def __init__(self, l_twb, sto, ref_E, c='C', RT=None, RF=1e-6, switch=False, ST=None):
        """ Generates Objective class object

        Args:
            l_twb (list): list of Twobody class objects.
            sto (ndarray): An array containing number of atoms of each type.
            ref_E (ndarray): Reference energies.
            c (str, optional): Type of solver. Defaults to 'C'.
            RT ([type], optional): Regularization Type. Defaults to None.
            RF ([type], optional): Regularization factor. Defaults to 1e-6.
            switch (bool, optional): switch condition. Defaults to False.
            ST ([type], optional): switch search where there is data. Defaults to None.
        """
        self.l_twb = l_twb
        self.sto = sto
        self.ref_E = np.asarray(ref_E)
        self.c = c
        self.RT = RT
        self.RF = RF
        self.switch = switch
        self.cols_sto = sto.shape[1]
        self.NP = len(l_twb)
        self.cparams = self.l_twb[0].cols
        self.ns = len(ref_E)
        logger.debug(" The reference energy : \n %s", self.ref_E)

    @staticmethod
    def solver(P, q, G, h, MAXITER=300, tol=(1e-10, 1e-10, 1e-10)):
        """ The solver for the objective

        Args:
            P (matrix): P matrix as per standard Quadratic Programming(QP) notation.
            q (matrix): q matrix as per standard QP notation.
            G (matrix): G matrix as per standard QP notation.
            h (matrix): h matrix as per standard QP notation
            MAXITER (int, optional): Maximum iteration steps. Defaults to 300.
            tol (tuple, optional): Tolerance value of the solution. Defaults to (1e-10, 1e-10, 1e-10).

        Returns:
            dictionary: The solution details are present in this dictionary
        """

        solvers.options['maxiters'] = MAXITER
        solvers.options['feastol'] = tol[0]
        solvers.options['abstol'] = tol[1]
        solvers.options['reltol'] = tol[2]
        sol = solvers.qp(P, q, G, h)
        return sol

    def eval_obj(self, x):
        """ mean square error function

        Args:
            x (ndarray): The solution for the objective.

        Returns:
            float: mean square error.
        """
        return np.format_float_scientific(np.sum((self.ref_E - (np.ravel(self.M.dot(x))))**2)/self.ns, precision=4)

    def plot(self, E_model, s_interval, s_a, x):
        """ function to plot the results

        Args:
            E_model (ndarray): Predicted energies via spline.
            s_interval (list): Spline interval.
            s_a (ndarray): Spline a coeffcients.
            x (ndarrray): The solution array.
        """

        fig = plt.figure()

        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(E_model, self.ref_E, 'bo')
        ax1.set_xlabel('Predicted energies')
        ax1.set_ylabel('Ref. energies')
        z = np.polyfit(E_model, self.ref_E, 1)
        p = np.poly1d(z)
        ax1.plot(E_model, p(E_model), 'r--')

        ax2 = fig.add_subplot(2, 2, 2)
        ax2.scatter(s_interval[1:], s_a, c=[i < 0 for i in s_a])
        ax2.set_xlabel('Distance')
        ax2.set_ylabel('a coefficients')

        ax3 = fig.add_subplot(2, 2, 3)
        c = [i < 0 for i in x]
        ax3.scatter(s_interval[1:], x, c=c)
        ax3.set_ylabel('c coefficients')
        ax3.set_xlabel('Distance')

        ax4 = fig.add_subplot(2, 2, 4)
        n, bins, patches = plt.hist(x=np.ravel(
            self.l_twb[0].Dismat), bins=self.l_twb[0].interval, color='g', rwidth=0.85)
        ax4.set_ylabel('Frequency of a distance')
        ax4.set_xlabel('Spline interval')
        plt.tight_layout()
        plt.savefig('summary.png')

    def solution(self):
        """ Function to solve the objective with constraints
        """
        self.M = self.get_M()
        P = matrix(np.transpose(self.M).dot(self.M))
        q = -1*matrix(np.transpose(self.M).dot(self.ref_E))
        N_switch_id = 0
        obj = np.zeros(self.l_twb[0].cols)
        sol_list = []
        if self.l_twb[0].Nswitch is None:
            for count, N_switch_id in enumerate(range(self.l_twb[0].Nknots+1)):
                G = self.get_G(N_switch_id)
                logger.debug(
                    "\n Nswitch_id : %d and G matrix:\n %s", N_switch_id, G)
                h = np.zeros(G.shape[1])
                sol = self.solver(P, q, matrix(G), matrix(h))
                obj[count] = self.eval_obj(sol['x'])
                sol_list.append(sol)

            mse = np.min(obj)
            opt_sol_index = np.ravel(np.argwhere(obj == mse))
            logger.info("\n The best switch is : %d", opt_sol_index)
            opt_sol = sol_list[opt_sol_index[0]]

        else:
            N_switch_id = self.l_twb[0].Nswitch
            G = self.get_G(N_switch_id)
            h = np.zeros(G.shape[1])
            opt_sol = self.solver(P, q, matrix(G), matrix(h))
            mse = float(self.eval_obj(opt_sol['x']))

        x = np.array(opt_sol['x'])
        model_eng = np.ravel(self.M.dot(x))
        curvatures = x[0:self.cparams]
        epsilon = x[-self.cols_sto:]
        logger.info("\n The optimal solution is : \n %s", x)
        logger.info("\n The optimal curvatures are:\n%s\nepsilon:%s",
                    curvatures, epsilon)

        s_a = np.dot(self.l_twb[0].A, curvatures)
        s_b = np.dot(self.l_twb[0].B, curvatures)
        s_c = np.dot(self.l_twb[0].C, curvatures)
        s_d = np.dot(self.l_twb[0].D, curvatures)

        write_error(model_eng, self.ref_E, mse)
        splcoeffs = np.hstack((s_a, s_b, s_c, s_d))
        write_splinecoeffs(self.l_twb[0], splcoeffs)
        self.plot(model_eng, self.l_twb[0].interval, s_a, s_c)

    def get_M(self):
        """ Returns the M matrix

        Returns:
            ndarray: The M matrix
        """
        v = self.l_twb[0].v
        logger.debug("\n The first v matrix is:\n %s", v)
        logger.debug("\n Shape of the first v matrix is:\t%s", v.shape)
        logger.debug("\n The stochiometry matrix is:\n%s", self.sto)
        if self.NP == 1:
            m = np.hstack((v, self.sto))
            logger.debug("\n The m  matrix is:\n %s \n shape:%s", m, m.shape)
            return m
        else:
            for i in range(1, self.NP):
                logger.debug("\n The %d pair v matrix is :\n %s",
                             i+1, self.l_twb[i].v)
                v = np.hstack((v, self.l_twb[i].v))
                logger.debug(
                    "\n The v  matrix shape after stacking :\t %s", v.shape)
            m = np.hstack((v, self.sto))
            return m

    def get_G(self, n_switch):
        """ returns constraints matrix

        Args:
            n_switch (int): switching point to cahnge signs of curvatures.

        Returns:
            ndarray: returns G matrix
        """
        g = block_diag(-1*np.identity(n_switch),
                       np.identity(self.l_twb[0].cols-n_switch))
        logger.debug("\n g matrix:\n%s", g)
        if self.NP == 1:
            G = block_diag(g, np.identity(self.cols_sto))
            return G
        else:
            for elem in range(1, self.NP):
                tmp_G = block_diag(g, np.identity(self.l_twb[elem].cols))
                g = tmp_G
        G = block_diag(g, self.cols_sto)

class Twobody():
    """ Twobody class describes properties of an Atom pair"""

    def __init__(self,
                 name,
                 Dismat,
                 Nconfigs,
                 Rcut,
                 Nknots,
                 Rmin=None,
                 Nswitch=None):
        """ Constructs an Two body object

        Args:
            name (str): Name of the atom pair.
            Dismat (dataframe): Pairwise  distance matrix.
            Nconfigs (int): Number of configurations
            Rcut (float): Maximum cut off value for spline interval
            Nknots (int): Number of knots in the spline interval
            Rmin (float, optional): Minimum value of the spline interval. Defaults to None.
            Nswitch (int, optional): The switching point for the spline. Defaults to None.
        """
        self.name = name
        self.Rcut = Rcut
        self.Rmin = Rmin
        self.Nknots = Nknots
        self.Nswitch = Nswitch
        self.Dismat = Dismat
        self.Nconfigs = Nconfigs
        self.dx = (self.Rcut - self.Rmin) / self.Nknots
        self.cols = self.Nknots + 1
        self.interval = np.linspace(self.Rmin,
                                    self.Rcut,
                                    self.cols,
                                    dtype=float)
        self.C, self.D, self.B, self.A = spline_construction(
            self.cols - 1, self.cols, self.dx)
        self.v = self.get_v()

    def get_v(self):
        """ Function for spline matrix

        Returns:
            ndarray: v matrix
        """
        return spline_energy_model(self.Rcut, self.Rmin, self.Dismat,
                                   self.cols, self.dx, self.Nconfigs,
                                   self.interval)


def spline_energy_model(Rcut, Rmin, df, cols, dx, size, x):
    """ Constructs the v matrix

    Args:
        Rcut (float): The max value cut off for spline interval.
        Rmin (float): The min value cut off for spline interval.
        df (ndarray): The paiwise distance matrix.
        cols (int):  Number of unknown parameters.
        dx (float): Grid size.
        size (int): Number of configuration.
        x (list): Spline interval.

    Returns:
        ndarray: The v matrix for a pair.
    """
    C, D, B, A = spline_construction(cols - 1, cols, dx)
    logger.debug(" Number of configuration for v matrix: %s", size)
    logger.debug("\n A matrix is: \n %s \n Spline interval = %s", A, x)
    v = np.zeros((size, cols))
    indices = []
    for config in range(size):
        distances = [i for i in df[config, :] if i <= Rcut and i >= Rmin]
        u = 0
        for r in distances:
            index = int(np.ceil(np.around(((r - Rmin) / dx), decimals=5)))
            indices.append(index)
            delta = r - x[index]
            logger.debug("\n In config %s\t distance r = %s\tindex=%s\tbin=%s",
                         config, r, index, x[index])
            a = A[index - 1]
            b = B[index - 1] * delta
            d = D[index - 1] * np.power(delta, 3) / 6.0
            c_d = C[index - 1] * np.power(delta, 2) / 2.0
            u = u + a + b + c_d + d

        v[config, :] = u
    logger.debug("\n V matrix :%s", v)
    return v

def spline_construction(rows, cols, dx):
    """ This function constructs the matrices A, B, C, D.
    Args:
        rows (int): The row dimension for matrix
        cols (int): The column dimension of the matrix
        dx (list): grid space((Rcut-Rmin)/N)

    Returns:
        A,B,C,D matrices
    """

    C = np.zeros((rows, cols), dtype=float)
    np.fill_diagonal(C, 1, wrap=True)
    C = np.roll(C, 1, axis=1)

    D = np.zeros((rows, cols), dtype=float)
    i, j = np.indices(D.shape)
    D[i == j] = -1
    D[i == j - 1] = 1
    D = D / dx

    B = np.zeros((rows, cols), dtype=float)
    i, j = np.indices(B.shape)
    B[i == j] = -0.5
    B[i < j] = -1
    B[j == cols - 1] = -0.5
    B = np.delete(B, 0, 0)
    B = np.vstack((B, np.zeros(B.shape[1])))
    B = B * dx

    A = np.zeros((rows, cols), dtype=float)
    tmp = 1 / 3.0
    for row in range(rows - 1, -1, -1):
        A[row][cols - 1] = tmp
        tmp = tmp + 0.5
        for col in range(cols - 2, -1, -1):
            if row == col:
                A[row][col] = 1 / 6.0
            if col > row:
                A[row][col] = col - row

    A = np.delete(A, 0, 0)
    A = np.vstack((A, np.zeros(A.shape[1])))
    A = A * dx * dx

    return C, D, B, A

def ccs_fit():
    """ parser for ccs"""
    FILENAME = 'input.json'
    parser = argparse.ArgumentParser(
        description=" A tool to fit two body potentials using constrained cubic splines")
    parser.add_argument(
        'input',
        nargs='?',
        default=FILENAME,
        help=' Json file containing pairwise distances and energies')
    parser.add_argument("-d",
                        "--debug",
                        dest="loglvl",
                        default=logging.INFO,
                        const=logging.DEBUG,
                        help="Set log level to debug",
                        action='store_const')
    args = parser.parse_args()

    logging.basicConfig(
        filename='ccs.log',
        format='%(asctime)s - %(name)s - %(levelname)s -       %(message)s',
        level=args.loglvl)
    logging.info('Started')
    CCS().twp_fit(args.input)
    logging.info('ended')


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


class Spline1:
    def __init__(self, x, y, d):
        self.xp = x
        self.yp = y
        self.dd = d
        self.nx = self.xp.shape[0]
        self.h = self.xp[1:] - self.xp[:-1]
        self.diffy = self.yp[1:] - self.yp[:-1]
        self.dind = bisect.bisect(self.xp.numpy(), self.dd) - 1
        self.ynew = self.cubic()

    def cubic(self):
        A = self.get_A()
        B = self.get_B()
        a = t.zeros(self.nx)
        c = t.zeros(self.nx)
        d = t.zeros(self.nx)

        # least squares and least norm problems
        M, _ = t.lstsq(B, A)
        for i in range(self.nx - 2):
            # b[i] = (self.xp[i + 1] * M[i]- self.xp[i] * M[i + 1]) / self.diff_xp[i] / 2
            c[i + 1] = (self.diffy[i]) / self.h[i + 1] - self.h[i + 1] / 6 * (M[i + 1] - M[i])
            d[i + 1] = (self.xp[i + 1] * self.yp[i] - self.xp[i] * self.yp[i + 1]) / self.h[i + 1] - \
            self.h[i + 1] / 6 * (self.xp[i + 1] * M[i] - self.xp[i] * M[i + 1])

        return (M[self.dind] * (self.xp[self.dind + 1] - self.dd) ** 3 + \
                M[self.dind + 1] * (self.dd - self.xp[self.dind]) ** 3) / self.xp[self.dind] / 6 + \
            c[self.dind + 1] * self.dd + d[self.dind + 1]

    def get_B(self):
        # natural boundary condition, the first and last are zero
        B = t.zeros(self.nx)
        for i in range(self.nx - 2):
            B[i + 1] = 6.0 * ((self.diffy[i + 1]) / self.h[i + 1] -
                              (self.diffy[i]) / self.h[i]) / \
            (self.h[i + 1] + self.h[i])
        return B

    def get_A(self):
        """Calculate a para in spline interpolation."""
        A = t.zeros(self.nx, self.nx)
        A[0, 0] = 1.
        # A[0, 1] = 1.
        for i in range(self.nx - 2):
            A[i + 1, i + 1] = 2.
            A[i + 1, i] = self.h[i] / (self.h[i + 1] + self.h[i])
            A[i + 1, i + 2] = 1 - A[i + 1, i]
        A[self.nx - 1, self.nx - 1] = 1.
        # A[self.nx - 1, self.nx - 2] = 1.
        return A


