"""Created on Wed Jun 10 11:01:38 2020.

The code are revised based on MBD-DFTB, the name style... will not follow PEP8
@author: gz_fan
"""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch as t
import dftbtorch.parameters as parameters
from dftbtorch.matht import LinAl
from ml.padding import pad1d, pad2d

class MBD:
    """Realize MBD-DFTB, revised based on MBD-DFTB branch."""

    def __init__(self, para, dataset):
        """Accrdong to args, run different functions.

        Initialize parameters;
        Get onsite population with charge population analysis;
        Calculate TS, SCS@MBD polarizability while NO MBD energy!!!
        """
        self.para = para
        self.dataset = dataset
        self.nb = self.dataset['nbatch']
        self.nat = self.dataset['natomAll']
        self.natommax = max(self.nat)
        self.mbd_init()
        self.onsite_population()
        self.get_cpa()
        self.get_mbdenergy()

    def mbd_init(self):
        """Initialize parameters for MBD-DFTB.

        Initialize parameters;
        """
        parameters.mbd_parameter(self.para)
        self.para['num_pairs'] = [int((iat ** 2 - iat) / 2 + iat) for iat in self.nat]
        self.para['alpha_tsall'] = []
        self.para['R_TS_VdW'] = t.zeros((self.nb, self.natommax), dtype=t.float64)
        self.para['sigma'] = t.zeros((self.nb, self.natommax), dtype=t.float64)

        if not self.para['Lperiodic']:
            latvec = t.ones((3), dtype=t.float64) * 1000000000
            self.para['latvec'] = latvec.diag()

        # get the return values of alpha_free, C6 and R
        alpha_c_r = pad2d([t.tensor([list(parameters.mbd_vdw_para(
            self.dataset['atomNumber'][ibatch][iat]))
            for iat in range(self.nat[ibatch])])
            for ibatch in range(self.dataset['nbatch'])])

        # write the return values to dict
        self.para['alpha_free'], self.para['C6_free'], self.para['R_vdw_free'] = \
            alpha_c_r[:, :, 0], alpha_c_r[:, :, 1], alpha_c_r[:, :, 2]

        # get the indice of each matrix (each matrixsieze: natom, natom)
        pairs_pq = pad2d([t.triu_indices(self.nat[ib], self.nat[ib], 0)
                          for ib in range(self.dataset['nbatch'])])
        self.para['pairs_p'] = pairs_pq[:, 0]
        self.para['pairs_q'] = pairs_pq[:, 1]

    def onsite_population(self):
        """Get onsite population for CPA DFTB.

        sum density matrix diagnal value for each atom
        """
        self.para['OnsitePopulation'] = t.zeros((self.nat), dtype=t.float64)
        atomind = self.dataset['atomind']

        # get the diagnal of density matrice
        denmat = [idensity.diag() for idensity in self.para['denmat']]

        # get onsite population
        self.para['OnsitePopulation'] = \
            [[denmat[ib][atomind[ib][iat]: atomind[ib][iat + 1]].sum()
              for iat in range(self.nat[ib])] for ib in range(self.dataset['nbatch'])]

    def get_cpa(self):
        """Get onsite population for CPA DFTB.

        J. Chem. Phys. 144, 151101 (2016)
        """
        onsite = self.para['OnsitePopulation']
        qzero = self.para['qzero']
        atomnumber = self.dataset['atomNumber']
        self.para['cpa'] = \
            [[1.0 + (onsite[ib][iat] - qzero[ib][iat]) / atomnumber[ib][iat]
              for iat in range(self.nat[ib])] for ib in range(self.dataset['nbatch'])]
        self.para['vefftsvdw'] = \
            [[atomnumber[ib][iat] + onsite[ib][iat] - qzero[ib][iat]
              for iat in range(self.nat[ib])] for ib in range(self.dataset['nbatch'])]

    def get_mbdenergy(self):
        """Get MBD energy for DFTB.

        J. Chem. Phys. 140, 18A508 (2014)
        Phys. Rev. Lett. 108, 236402 (2012)
        """
        omega = self.para['omega']
        '''orig_idx = t.zeros((self.nat), dtype=t.float64)
        orig_idx =
        for ip in range(self.nat):
            orig_idx[ip] = self.para['pairs_p'][ip]'''

        self.para['ainv_'] = LinAl(self.para).inv33_mat(self.para['latvec'])
        h_, ainv_ = self.para['latvec'], self.para['ainv_']
        self.mbdvdw_pbc(self.dataset['positions'], h_, ainv_, self.nat)
        for ieff in range(self.para['nOmegaGrid'] + 1):
            self.mbdvdw_effqts(ieff, omega[ieff])
            self.mbdvdw_SCS(ieff)
            if ieff == 0:
                alpha_isotropic = self.mbdvdw_screened_pol()
                self.para['alpha_ts'] = self.para['alpha_tsall'][0]
                self.para['alpha_mbd'] = alpha_isotropic[:]

    def mbdvdw_pbc(self, coor, h_, ainv_, nat):
        pass

    def mbdvdw_effqts(self, ieff, omega):
        """Calculate TS polarizability, R_vdw.

        J. Chem. Phys. 140, 18A508 (2014)
        Phys. Rev. Lett. 108, 236402 (2012)
        """
        alpha_free = self.para['alpha_free']
        C6_free = self.para['C6_free']
        R_vdw_free = self.para['R_vdw_free']
        vfree = self.dataset['atomNumber']
        VefftsvdW = self.para['vefftsvdw']
        alpha_ts_ = t.zeros((self.nb, self.natommax), dtype=t.float64)
        if self.para['vdwConsistent']:
            dsigmadV = t.zeros((self.nat, self.nat), dtype=t.float64)
            dR_TS_VdWdV = t.zeros((self.nat, self.nat), dtype=t.float64)
            dalpha_tsdV = t.zeros((self.nat, self.nat), dtype=t.float64)
        for ib in range(self.dataset['nbatch']):
            for iat in range(self.nat[ib]):
                omega_free = ((4.0 / 3.0) * C6_free[ib][iat] / (alpha_free[ib][iat] ** 2))

                # Padé Approx: Tang, K.T, M. Karplus. Phys Rev 171.1 (1968): 70
                pade_approx = 1.0 / (1.0 + (omega / omega_free) ** 2)

                # Computes sigma
                gamma = (1.0 / 3.0) * np.sqrt(2.0 / np.pi) * pade_approx * \
                    (alpha_free[ib][iat] / vfree[ib][iat])
                gamma = gamma ** (1.0 / 3.0)
                self.para['sigma'][ib, iat] = gamma * VefftsvdW[ib][iat] ** (1.0 / 3.0)

                # Computes R_TS: equation 11 in [PRL 102, 073005 (2009)]
                xi = R_vdw_free[ib][iat] / (vfree[ib][iat]) ** (1.0 / 3.0)
                self.para['R_TS_VdW'][ib, iat] = xi * VefftsvdW[ib][iat] ** (1.0 / 3.0)

                # Computes alpha_ts: equation 1 in [PRL 108 236402 (2012)]
                lambda_ = pade_approx * alpha_free[ib][iat] / vfree[ib][iat]
                # self.para['alpha_tsall'][ieff, iat] = lambda_ * VefftsvdW[iat]
                alpha_ts_[ib, iat] = lambda_ * VefftsvdW[ib][iat]

                if self.para['vdwConsistent']:
                    for jat in range(self.nat):
                        if iat == jat:
                            dsigmadV[iat, jat] = gamma / \
                                (3.0 * VefftsvdW[iat] ** (2.0 / 3.0))
                            dR_TS_VdWdV[iat, jat] = xi / \
                                (3.0 * VefftsvdW[iat] ** (2.0 / 3.0))
                            dalpha_tsdV[iat, jat] = lambda_
                self.para['alpha_tsall'].append(alpha_ts_)

    def mbdvdw_SCS(self, ieff):
        r"""Calculate SCS@MBD.

        $\bar{A}(i\omega) = \left(  A^{-1} + TSR \right)^{-1}$
        """
        num_pairs = self.para['num_pairs']
        pairs_p = self.para['pairs_p']
        pairs_q = self.para['pairs_q']
        alpha_ts = self.para['alpha_tsall'][ieff]
        mytsr = t.zeros((self.nb, max(num_pairs), 3, 3), dtype=t.float64)
        collected_tsr = t.zeros((self.nb, max(num_pairs), 3, 3), dtype=t.float64)
        A_matrix = t.zeros((self.nb, 3 * self.natommax, 3 * self.natommax), dtype=t.float64)

        for ib in range(self.nb):
            for ii in range(num_pairs[ib]):
                p, q = int(pairs_p[ib, ii]), int(pairs_q[ib, ii])

                sl_mult = 1.0
                h_, ainv_ = self.para['latvec'], self.para['ainv_']
                tsr = self.mbdvdw_TGG(p, q, self.nat[ib], h_, ainv_, ib)
                mytsr[ib, ii, :, :] = tsr
                # print('mytsr', tsr)

            # only loops over the upper triangle
            counter = 0
            for p in range(self.nat[ib]):
                for q in range(p, self.nat[ib]):
                    tsr = mytsr[ib, counter, :, :]
                    counter = counter + 1
                    for i_idx in range(3):
                        for j_idx in range(i_idx, 3):
                            ind = (3 * p + i_idx)
                            jnd = (3 * q + j_idx)
                            ind_T = (3 * p + j_idx)
                            jnd_T = (3 * q + i_idx)
                            if p == q:
                                if i_idx == j_idx:
                                    A_matrix[ib, ind, jnd] = 1.0 / alpha_ts[ib, p]
                                    divisor = 1.0 / alpha_ts[ib, p] ** 2.0
                            A_matrix[ib, ind, jnd] = A_matrix[ib, ind, jnd] + tsr[i_idx, j_idx]
                            A_matrix[ib, jnd, ind] = A_matrix[ib, ind, jnd]
                            A_matrix[ib, ind_T, jnd_T] = A_matrix[ib, ind, jnd]
                            A_matrix[ib, jnd_T, ind_T] = A_matrix[ib, jnd, ind]

        A_LU, pivots = t.lu(A_matrix)
        self.para['A_matrix'] = A_matrix.inverse()

    def mbdvdw_screened_pol(self):
        """Calculate polarizability."""
        alpha_isotropic = t.zeros((self.nb, self.natommax), dtype=t.float64)
        A_matrix = self.para['A_matrix']
        # vdw_self_consistent = False
        # do_forces = False
        # mbd_vdw_isolated = True
        for ib in range(self.nb):
            for pat in range(self.nat[ib]):
                for qat in range(self.nat[ib]):
                    for i_idx in range(3):
                        # cnt_v = 1
                        # cnt_h = 1
                        # cnt_f = 1
                        ind = 3 * pat
                        jnd = 3 * qat
                        for i_idx in range(3):
                            ind = (3 * pat)
                            jnd = (3 * qat)
                            alpha_isotropic[ib, pat] = alpha_isotropic[ib, pat] + \
                                A_matrix[ib, ind + i_idx, jnd + i_idx]
        alpha_isotropic = alpha_isotropic / 3.0 ** 2
        # self.para['alpha_mbd'] = alpha_isotropic
        return alpha_isotropic

    def mbdvdw_TGG(self, p, q, n, h_in, ainv_in, ib):
        """TGG = TSR + TLR.

        h_in: normally the latice parameters
        ainv_in: inverse matrix of of 3*3 h_in
        """
        coor = self.dataset['positions']
        spq = t.zeros((3), dtype=t.float64)
        spq_lat = t.zeros((3), dtype=t.float64)
        rpq_lat = t.zeros((3), dtype=t.float64)
        tsr = t.zeros((3, 3), dtype=t.float64)
        Rc = 20.0

        rpq = coor[ib, p, :] - coor[ib, q, :]
        spq[:] = ainv_in[0, :] * rpq[0] + ainv_in[1, :] * rpq[1] + \
            ainv_in[2, :] * rpq[2]
        sc = self.mbdvdw_circumscribe(h_in, Rc)
        if sc[0] == sc[1] == sc[2] == 0 and p != q:
            n1, n2, n3 = 0, 0, 0
            # if((n1.eq.0).and.(n2.eq.0).and.(n3.eq.0)) then
            #    if(p.eq.q) cycle
            spq_lat[:] = spq[:] + t.tensor([n1, n2, n3])
            rpq_lat[:] = h_in[0, :] * spq_lat[0] + \
                h_in[1, :] * spq_lat[1] + h_in[2, :] * spq_lat[2]
            tsr = self.mbdvdw_compute_TSR(p, q, rpq_lat, spq_lat, ib)
        return tsr

    def mbdvdw_circumscribe(self, uc, radius):
        nn = len(uc)
        layer_sep = t.zeros((nn), dtype=t.float64)
        sc = t.zeros((nn), dtype=t.float64)
        ruc = 2 * np.pi * LinAl(self.para).inv33_mat(uc).t()
        for inn in range(nn):
            layer_sep[inn] = sum(uc[inn, :] * ruc[inn, :] /
                                 (sum(ruc[inn, :] ** 2)).sqrt())
        sc = t.ceil((radius / layer_sep + 0.5))
        try:
            vacuum = self.para['VacuumAxis']
        except:
            vacuum = [True, True, True]
        if vacuum[0]:
            sc[0] = 0
        if vacuum[1]:
            sc[1] = 0
        if vacuum[2]:
            sc[2] = 0
        return sc

    def mbdvdw_compute_TSR(self, p, q, rpq, spq_lat, ib):
        sigma = self.para['sigma']
        R_TS_VdW = self.para['R_TS_VdW']
        beta = self.para['beta']
        Rmat = t.zeros((3, 3), dtype=t.float64)
        Tdip = t.zeros((3, 3), dtype=t.float64)
        sqrtpi = np.pi ** 0.5
        rpq_norm = (rpq[:] ** 2.0).sum().sqrt()
        # Computes the effective correlation length of the interaction potential
        # defined from the widths of the QHO Gaussians
        Sigma_pq = (sigma[ib, p].clone() ** 2.0 + sigma[ib, q].clone() ** 2.0).sqrt()
        # sigma_p, sigma_q = sigma[p].clone(), sigma[q].clone()
        # Sigma_pq = (sigma_p ** 2.0 + sigma_q ** 2.0) ** (0.5)
        # Computes the damping radius
        R_VdW_pq = R_TS_VdW[ib, p] + R_TS_VdW[ib, q]
        Spq = beta * R_VdW_pq
        Z = 6.0 * (rpq_norm / Spq - 1.0)
        fermi_fn = 1.0
        dfn_pre = 0.0


        # zeta = rpq_norm / Sigma_pq
        # computes the fermi damping function. The latex for this is
        # f_{damp}(R_{pq}) = \frac{1}{ 1 + exp( - Z(R_{pq}) ) }
        # where Z = 6 \left( \frac{R_{pq} }{ S_{pq}} - 1 \right)
        # and S_{pq} = \beta \left(  R_{p, VdW} + R_{q, VdW} \right)
        if Z <= 35.0:
            fermi_fn = 1.0 / (1.0 + t.exp(-Z))
            dfn_pre = t.exp(-Z) / (1.0 + t.exp(-Z)) ** 2.0
            # Computes the factors for U
            # U = {\rm erf}\left[\zeta\right] -  \frac{2}{\sqrt{\pi}}\zeta \exp\left[-\zeta^2\right]
            zeta = rpq_norm / Sigma_pq
        else:
            zeta = rpq_norm / Sigma_pq

        if zeta >= 6.0:
            U = 1.0
            W = 0.0
            gaussian = 0.0
        else:
            gaussian = t.exp(-zeta * zeta)
            U = t.erf(zeta) - (2.0 * zeta) / sqrtpi * gaussian
            # Computes the first half of the factor for W before we multiply by the R^i R^j tensor
            # \mathbf{W}^{ij} &\equiv&   \left(\frac{R^i R^j}{R^5}\right) \, \frac{4}{\sqrt{\pi}}  \zeta^3  \exp\left[-\zeta^2\right]
            W = 4.0 * zeta ** 3.0 / sqrtpi * gaussian

        # Loops over the cartesian coordinates to compute the R^i*R^j
        # matrix that goes in to constructing the dipole matrix
        for i in range(3):
            for j in range(3):
                Rmat[i, j] = rpq[i] * rpq[j] / (rpq_norm**5.0)
                Tdip[i, j] = -3.0 * Rmat[i, j]

            # This just applies the kronheker delta center for the dipole tensor. Recall
            # that T^{ij}_{dip} = \frac{-3 R^i R^j + R^2 \delta_{ij}}{ R^5 }
            Tdip[i, i] = Tdip[i, i] + 1.0 / (rpq_norm ** 3.0)

        # Computes the short range dipole coupling quantity using the fact that
        # T = T_{dip}\left[ U \right] + W
        TGG = (Tdip * U + Rmat * W)
        TSR = (1.0 - fermi_fn) * TGG
        return TSR
