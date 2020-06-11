#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 11:01:38 2020

@author: gz_fan
"""
import numpy as np
import torch as t
import parameters


class MBD:

    def __init__(self, para):
        self.para = para
        self.nat = para['natom']
        self.mbd_init()
        self.qatom_population()
        self.get_cpa()
        self.get_mbdenergy()

    def mbd_init(self):
        parameters.mbd_parameter(self.para)
        self.para['alpha_free'] = t.zeros((self.nat), dtype=t.float64)
        self.para['C6_free'] = t.zeros((self.nat), dtype=t.float64)
        self.para['R_vdw_free'] = t.zeros((self.nat), dtype=t.float64)
        self.para['alpha_ts'] = t.zeros((self.nat), dtype=t.float64)
        self.para['R_TS_VdW'] = t.zeros((self.nat), dtype=t.float64)
        self.para['sigma'] = t.zeros((self.nat), dtype=t.float64)
        latvec = t.ones((3), dtype=t.float64) * 1000000000
        if not self.para['Ldipole']:
            self.para['latvec'] = latvec.diag()

        for iat in range(self.nat):
            parameters.mbd_vdw_para(self.para, iat)

    def mbdvdw_para_init(self):
        self.para['num_pairs'] = int((self.nat ** 2 - self.nat) / 2 + self.nat)
        pairs_scs_p = t.zeros((self.para['num_pairs']), dtype=t.float64)
        pairs_scs_q = t.zeros((self.para['num_pairs']), dtype=t.float64)
        counter = 0
        for p in range(self.nat):
            for q in range(p, self.nat):
                pairs_scs_p[counter], pairs_scs_q[counter] = p, q
                counter += 1
        self.para['pairs_scs_p'] = pairs_scs_p
        self.para['pairs_scs_q'] = pairs_scs_q

    def qatom_population(self):
        '''sum density matrix diagnal value for each atom'''
        self.para['qatompopulation'] = t.zeros((self.nat), dtype=t.float64)
        atomind = self.para['atomind']
        denmat = self.para['denmat'][-1].diag()
        for iatom in range(self.nat):
            ii1 = atomind[iatom]
            ii2 = atomind[iatom + 1]
            self.para['qatompopulation'][iatom] = denmat[ii1: ii2].sum()

    def get_cpa(self):
        '''
        this is from MBD-DFTB for charge population analysis
        J. Chem. Phys. 144, 151101 (2016)
        '''
        cpa = t.zeros((self.nat), dtype=t.float64)
        vefftsvdw = t.zeros((self.nat), dtype=t.float64)
        onsite = self.para['qatompopulation']
        qzero = self.para['qzero']
        coor = self.para['coor']
        for iatom in range(self.nat):
            cpa[iatom] = 1.0 + (onsite[iatom] - qzero[iatom]) / coor[iatom][0]
            vefftsvdw[iatom] = coor[iatom][0] + onsite[iatom] - qzero[iatom]
        # sedc_ts_veff_div_vfree = scaling_ratio
        self.para['cpa'] = cpa
        self.para['vefftsvdw'] = vefftsvdw

    def get_mbdenergy(self):
        omega = self.para['omega']
        for ieff in range(self.para['n_omega_grid'] + 1):
            self.mbdvdw_effqts(omega[ieff])
            self.mbdvdw_SCS()

    def mbdvdw_effqts(self, omega):
        alpha_free = self.para['alpha_free']
        C6_free = self.para['C6_free']
        R_vdw_free = self.para['R_vdw_free']
        vfree = self.para['atomNumber']
        VefftsvdW = self.para['vefftsvdw']
        if self.para['vdw_self_consistent']:
            dsigmadV = t.zeros((self.nat, self.nat), dtype=t.float64)
            dR_TS_VdWdV = t.zeros((self.nat, self.nat), dtype=t.float64)
            dalpha_tsdV = t.zeros((self.nat, self.nat), dtype=t.float64)
        for iat in range(self.nat):
            omega_free = ((4.0 / 3.0) * C6_free[iat] / (alpha_free[iat] ** 2))

            # Padé Approx: Tang, K.T, M. Karplus. Phys Rev 171.1 (1968): 70
            pade_approx = 1.0 / (1.0 + (omega / omega_free) ** 2)

            # Computes sigma
            gamma = (1.0 / 3.0) * np.sqrt(2.0 / np.pi) * pade_approx * \
                (alpha_free[iat] / vfree[iat])
            gamma = gamma ** (1.0 / 3.0)
            self.para['sigma'][iat] = gamma * VefftsvdW[iat] ** (1.0 / 3.0)

            # Computes R_TS: equation 11 in [PRL 102, 073005 (2009)]
            xi = R_vdw_free[iat] / (vfree[iat]) ** (1.0 / 3.0)
            self.para['R_TS_VdW'][iat] = xi * VefftsvdW[iat] ** (1.0 / 3.0)

            # Computes alpha_ts: equation 1 in [PRL 108 236402 (2012)]
            lambda_ = pade_approx * alpha_free[iat] / vfree[iat]
            self.para['alpha_ts'][iat] = lambda_ * VefftsvdW[iat]

            if self.para['vdw_self_consistent']:
                for jat in range(self.nat):
                    if iat == jat:
                        dsigmadV[iat, jat] = gamma / \
                            (3.0 * VefftsvdW[iat] ** (2.0 / 3.0))
                        dR_TS_VdWdV[iat, jat] = xi / \
                            (3.0 * VefftsvdW[iat] ** (2.0 / 3.0))
                        dalpha_tsdV[iat, jat] = lambda_

    def mbdvdw_SCS(self):
        '''
        calculate:
            $\bar{A}(i\omega) = \left(  A^{-1} + TSR \right)^{-1}$
        '''
        self.mbdvdw_para_init()
        num_pairs = self.para['num_pairs']
        pairs_scs_p = self.para['pairs_scs_p']
        pairs_scs_q = self.para['pairs_scs_p']
        for ii in range(num_pairs):
            p, q = pairs_scs_p[ii], pairs_scs_q[ii]
            # cpuid = pairs_scs(i)%cpu

            if cpuid + 1 == me:
                sl_mult = 1.0
                self.mbdvdw_TGG(1, p, q, nat, h_, ainv_, tau, Tsr, dTsrdR, dTsrdh, dTsrdV)

                my_tsr[counter,: ,:] = tsr

                if self.para['vdw_self_consistent']:
                    my_dtsrdv[counter,:,:,:] = dtsrdv

                if self.para['do_forces']:
                    for s in range(self.nat):
                        for i_f in range(3):
                            my_dtsrdr[counter, :, :, 3 * (s - 1) + i_f] = \
                                dtsrdr[:, :, s, i_f]
                if not self.para['mbd_vdw_isolated']:
                    for s in range(3):
                        for i_f in range(3):
                            my_dtsrdh[counter, :, :, 3 * (s - 1) + i_f] = \
                                dtsrdh[:, :, s, i_f]
                            counter = counter + 1
    def mbdvdw_TGG(self):
        pass

    def mbdvdw_calculate_screened_pol(self):
        pass

