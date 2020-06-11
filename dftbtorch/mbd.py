#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 11:01:38 2020

@author: gz_fan
"""
import math
import numpy as np
import torch as t
import parameters
from matht import LinAl


class MBD:

    def __init__(self, para):
        self.para = para
        self.nat = para['natom']
        self.mbd_init()
        self.mbdvdw_para_init()
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
        if not self.para['Lperiodic']:
            latvec = t.ones((3), dtype=t.float64) * 1000000000
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
        orig_idx = t.zeros((self.nat), dtype=t.float64)
        for ip in range(self.nat):
            orig_idx[ip] = self.para['pairs_scs_p'][ip]

        self.para['ainv_'] = LinAl(self.para).inv33_mat(self.para['latvec'])
        h_, ainv_ = self.para['latvec'], self.para['ainv_']
        self.mbdvdw_pbc(self.para['coor'], h_, ainv_, self.nat)
        for ieff in range(self.para['n_omega_grid'] + 1):
            self.mbdvdw_effqts(omega[ieff])
            self.mbdvdw_SCS()
            if ieff == 0:
                alpha_isotropic = self.mbdvdw_calculate_screened_pol()
                print(alpha_isotropic)

    def mbdvdw_pbc(self, coor, h_, ainv_, nat):
        pass

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
        num_pairs = self.para['num_pairs']
        pairs_scs_p = self.para['pairs_scs_p']
        pairs_scs_q = self.para['pairs_scs_q']
        alpha_ts = self.para['alpha_ts']
        mytsr = t.zeros((num_pairs, 3, 3), dtype=t.float64)
        collected_tsr = t.zeros((num_pairs, 3, 3), dtype=t.float64)
        A_matrix = t.zeros((3 * self.nat, 3 * self.nat), dtype=t.float64)

        for ii in range(num_pairs):
            p, q = int(pairs_scs_p[ii]), int(pairs_scs_q[ii])

            sl_mult = 1.0
            h_, ainv_ = self.para['latvec'], self.para['ainv_']
            tsr = self.mbdvdw_TGG(p, q, self.nat, h_, ainv_)
            mytsr[ii, :, :] = tsr
            
            # only loops over the upper triangle
            counter = 0
            for p in range(self.nat):
                for q in range(p, self.nat):
                    tsr = collected_tsr[counter, :, :]
                    #if(vdw_self_consistent) TSRdv(:,:,:) = collected_dtsrdv(counter,:,:,:)
                    #if(do_forces) then
                    #    TSRdr(:,:,:) = collected_dtsrdr(counter, :, :, :)
                    #if(.not.mbd_vdw_isolated) TSRdh(:,:,:) = collected_dtsrdh(counter, :, :, :)
                    counter = counter + 1
                    for i_idx in range(3):
                        for j_idx in range(i_idx, 3):
                            i_index = (3 * p + i_idx)
                            j_index = (3 * q + j_idx)
                            i_index_T = (3 * p + j_idx)
                            j_index_T = (3 * q + i_idx)
                            if p == q:
                                if i_idx == j_idx:
                                    A_matrix[i_index, j_index] = 1.0 / alpha_ts[p]
                                    divisor = 1.0 / alpha_ts[p] ** 2.0
                                    '''if vdw_self_consistent:
                                        cnt_v = 1
                                        for i_f in range(self.nat):
                                            if v_cpu_id[i_f] == me_image:
                                                s = i_f
                                                dA_matrixdV[i_index, j_index, cnt_v] = -dalpha_tsdV[p, i_f] * divisor
                                                cnt_v = cnt_v + 1
                                    if do_forces:
                                        cnt_f = 1
                                        for i_f in range(3 * self.nat):
                                            if f_cpu_id[i_f] == me_image:
                                                self.mbdvdw_get_is(i_f, s, i)
                                                dA_matrixdR[i_index, j_index, cnt_f] = -dalpha_tsdR(p, s, i)*divisor
                                                cnt_f = cnt_f + 1
                                    if not mbd_vdw_isolated:
                                        cnt_h = 1
                                        for i_f in range(9):
                                            if h_cpu_id[i_f] == me_image:
                                                self.mbdvdw_get_is(i_f, s, i)
                                                dA_matrixdh(i_index, j_index, cnt_h) = -dalpha_tsdh(p, s, i) * divisor
                                                cnt_h = cnt_h + 1'''
                            A_matrix[i_index, j_index] = A_matrix[i_index, j_index] + tsr[i_idx, j_idx]
                            A_matrix[j_index, i_index] = A_matrix[i_index, j_index]
                            A_matrix[i_index_T, j_index_T] = A_matrix[i_index, j_index]
                            A_matrix[j_index_T, i_index_T] = A_matrix[j_index, i_index]
                            '''if vdw_self_consistent:
                                cnt_v = 1
                                for i_f in range(self.nat):
                                    if v_cpu_id[i_f] == me_image:
                                        dA_matrixdV(i_index,j_index,cnt_v)=dA_matrixdV(i_index,j_index,cnt_v)+TSRdV(i_idx,j_idx,cnt_v)
                                        dA_matrixdV(j_index,i_index,cnt_v)=dA_matrixdV(i_index,j_index,cnt_v)
                                        dA_matrixdV(i_index_T,j_index_T,cnt_v)=dA_matrixdV(i_index,j_index,cnt_v)
                                        dA_matrixdV(j_index_T,i_index_T,cnt_v)=dA_matrixdV(j_index,i_index,cnt_v)

                            if do_forces:
                                cnt_f = 1
                                for i_f in range(3 * self.nat):
                                    if f_cpu_id[i_f] == me_image:
                                        self.mbdvdw_get_is(i_f, s, i)
                                        dA_matrixdR(i_index,j_index,cnt_f)=dA_matrixdR(i_index,j_index,cnt_f)+TSRdR(i_idx,j_idx,cnt_f)
                                        dA_matrixdR(j_index,i_index,cnt_f)=dA_matrixdR(i_index, j_index, cnt_f)
                                        dA_matrixdR(i_index_T,j_index_T,cnt_f)=dA_matrixdR(i_index,j_index,cnt_f)
                                        dA_matrixdR(j_index_T,i_index_T,cnt_f)=dA_matrixdR(j_index,i_index,cnt_f)
                                        cnt_f = cnt_f + 1
                            if not mbd_vdw_isolated:
                                cnt_h = 1
                                for i_f in range(9):
                                    if h_cpu_id[i_f] == me_image:
                                        self.mbdvdw_get_is(i_f, s, i)
                                        dA_matrixdh(i_index,j_index,cnt_h)=dA_matrixdh(i_index, j_index,cnt_h)+TSRdh(i_idx,j_idx,cnt_h)
                                        dA_matrixdh(j_index,i_index,cnt_h)=dA_matrixdh(i_index, j_index,cnt_h) # fill in symmetric elements
                                        dA_matrixdh(i_index_T,j_index_T,cnt_h)=dA_matrixdh(i_index,j_index,cnt_h)
                                        dA_matrixdh(j_index_T,i_index_T,cnt_h)=dA_matrixdh(j_index,i_index,cnt_h)
                                        cnt_h = cnt_h + 1'''

            '''my_tsr[counter,: ,:] = tsr
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
                        counter = counter + 1'''
        A_LU, pivots = t.lu(A_matrix)
        self.para['A_matrix'] = A_matrix.inverse()

    def mbdvdw_calculate_screened_pol(self):
        alpha_isotropic = t.zeros((self.nat), dtype=t.float64)
        A_matrix = self.para['A_matrix']
        vdw_self_consistent = False
        do_forces = False
        mbd_vdw_isolated = True
        for pat in range(self.nat):
            for qat in range(self.nat):
                for i_idx in range(3):
                    cnt_v = 1
                    cnt_h = 1
                    cnt_f = 1
                    i_index = (3 * pat)
                    j_index = (3 * qat)

                    '''if vdw_self_consistent:
                        for i_f in range(self.nat):
                            dalpha_isotropicdV[pat, i_f] = dalpha_isotropicdV[pat, i_f] + dA_matrixdV[i_index+i_idx, j_index+i_idx, cnt_v]
                            cnt_v = cnt_v + 1

                    if do_forces:
                        for i_f in range(3 * self.nat):
                            self.mbdvdw_get_is(i_f, s, i)
                            dalpha_isotropicdR[pat, s, i] = dalpha_isotropicdR[pat, s, i] + dA_matrixdR[i_index+i_idx, j_index+i_idx, cnt_f]
                            cnt_f = cnt_f + 1
                    if not mbd_vdw_isolated:
                        for i_f in range(9):
                            self.mbdvdw_get_is(i_f, s, i)
                            dalpha_isotropicdh[pat, s, i] = dalpha_isotropicdh[pat, s, i] + dA_matrixdh[i_index+i_idx, j_index+i_idx, cnt_h]
                            cnt_h = cnt_h + 1'''
                    for i_idx in range(3):
                        i_index = (3 * pat)
                        j_index = (3 * qat)
                        alpha_isotropic[pat] = alpha_isotropic[pat] + A_matrix[i_index+i_idx, j_index+i_idx]
        alpha_isotropic = alpha_isotropic / 3.0
        alpha_isotropic = alpha_isotropic / 3.0
        '''if do_forces:
            dalpha_isotropicdR = dalpha_isotropicdR/3.0
        if do_forces:
            dalpha_isotropicdh = dalpha_isotropicdh/3.0
        if vdw_self_consistent:
            dalpha_isotropicdV = dalpha_isotropicdV/3.0'''
        print('alpha_mbd', alpha_isotropic)
        self.para['alpha_mbd'] = alpha_isotropic


    def mbdvdw_TGG(self, p, q, n, h_in, ainv_in):
        '''
        TGG = TSR + TLR
        h_in: normally the latice parameters
        ainv_in: inverse matrix of of 3*3 h_in
        '''
        coor = self.para['coorbohr']
        spq = t.zeros((3), dtype=t.float64)
        spq_lat = t.zeros((3), dtype=t.float64)
        rpq_lat = t.zeros((3), dtype=t.float64)
        tsr = t.zeros((3, 3), dtype=t.float64)
        Rc = 20.0

        rpq = coor[p, :] - coor[q, :]
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
            rnorm = (rpq_lat[:] ** 2.0).sum().sqrt()
            tsr = self.mbdvdw_compute_TSR(p, q, rpq_lat, spq_lat)
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

    def mbdvdw_compute_TSR(self, p, q, rpq, spq_lat):
        sigma = self.para['sigma']
        R_TS_VdW = self.para['R_TS_VdW']
        beta = self.para['beta']
        Rmat = t.zeros((3, 3), dtype=t.float64)
        Tdip = t.zeros((3, 3), dtype=t.float64)
        sqrtpi = np.pi ** 0.5
        rpq_norm = (rpq[:] ** 2.0).sum().sqrt()
        # Computes the effective correlation length of the interaction potential
        # defined from the widths of the QHO Gaussians
        Sigma_pq = (sigma[p] ** 2.0 + sigma[q] ** 2.0).sqrt()
        # Computes the damping radius
        R_VdW_pq = R_TS_VdW[p] + R_TS_VdW[q]
        Spq = beta * R_VdW_pq
        Z = 6.0 * (rpq_norm / Spq - 1.0)
        zeta = rpq_norm / Sigma_pq
        fermi_fn = 1.0
        dfn_pre = 0.0

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

        if zeta >= 6.0:
            U = 1.0
            W = 0.0
            gaussian = 0.0
        else:
            gaussian = t.exp(-zeta * zeta)
            U = t.erf(zeta)-(2.0 * zeta) / sqrtpi * gaussian
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

        '''if vdw_self_consistent:
            for s in range(self.nat):
        dsigma_pqdV = (sigma(p)*dsigmadV(p, s) + sigma(q)*dsigmadV(q, s))/Sigma_pq
        dSpqdV = beta*(dR_TS_VdWdV(p, s) + dR_TS_VdWdV(q, s))
        dZetadV = -rpq_norm*dsigma_pqdV/Sigma_pq**2.0_DP
        dZdV = 6.0_DP*( -rpq_norm*dSpqdV/Spq**2.0_DP)
        dFermi_fndV = dfn_pre*dZdV

        if(zeta.ge.6.0_DP) then
          dUdV = 0.0_DP
          dWdV = 0.0_DP
        else
          dUdV = dZetadV*zeta*zeta*4.0_DP*gaussian/SQRTPI
          dWdV = 4.0_DP/SQRTPI*zeta*zeta*gaussian*dzetadV*(3.0_DP - 2.0_DP*zeta*zeta)
        end if
        dTSRdV(:, :, s) = -TGG*dFermi_fndV + (1.0_DP - fermi_fn)*(Tdip*dUdV+Rmat*dWdV)
      end do ! Loop over all possible components
    end if ! if to check if self consistent derivatives need to be computed

    if(do_forces) then
      ! dR
      do s = 1, nat, 1
        do i = 1, 3, 1
          call mbdvdw_compute_dRdR(s, i, p, q, rpq, rpq_norm, drpqdR, drpq_normdR, dRmatdR, dTdipdR)
          dsigma_pqdR = (sigma(p)*dsigmadR(p, s, i) + sigma(q)*dsigmadR(q, s, i))/Sigma_pq
          dZetadR = drpq_normdR/Sigma_pq - rpq_norm*dsigma_pqdR/Sigma_pq**2.0_DP
          dSpqdR = beta*(dR_TS_VdWdR(p, s, i) + dR_TS_VdWdR(q, s, i))
          dZdR = 6.0_DP*( drpq_normdR/Spq - rpq_norm*dSpqdR/Spq**2.0_DP)
          dFermi_fndR = dfn_pre*dZdR
          if(zeta.ge.6.0_DP) then
            dUdR = 0.0_DP
            dWdR = 0.0_DP
          else
            dUdR = dZetadR*zeta*zeta*4.0_DP*gaussian/SQRTPI
            dWdR = 4.0_DP/SQRTPI*zeta*zeta*gaussian*dzetadR*(3.0_DP - 2.0_DP*zeta*zeta)
          end if
          dTSRdr(:, :, s, i) = -TGG*dFermi_fndR + (1.0_DP - fermi_fn)*(dTdipdR*U + Tdip*dUdR + W*dRmatdR + Rmat*dWdR)
        end do ! loop over cartesian components
      end do ! Loop over all atoms
      if(.not.mbd_vdw_isolated) then
        do s = 1, 3, 1
          do i = 1, 3, 1
            call mbdvdw_compute_dRdh(s, i, rpq, rpq_norm, Spq_lat, drpqdh, drpq_normdh, dRmatdh, dTdipdh)
            dsigma_pqdH = (sigma(p)*dsigmadH(p, s, i) + sigma(q)*dsigmadH(q, s, i))/Sigma_pq
            dZetadH = drpq_normdH/Sigma_pq - rpq_norm*dsigma_pqdH/Sigma_pq**2.0_DP
            dSpqdH = beta*(dR_ts_VdWdH(p, s, i) + dR_ts_VdWdH(q, s, i))
            dZdH = 6.0_DP*( drpq_normdH/Spq - rpq_norm*dSpqdH/Spq**2.0_DP)
            dFermi_fndH = dfn_pre*dZdH
            if(zeta.ge.6.0_DP) then
              dUdH = 0.0_DP
              dWdH = 0.0_DP
            else
              dUdH = dZetadH*zeta*zeta*4.0_DP*gaussian/SQRTPI
              dWdH = 4.0_DP/SQRTPI*zeta*zeta*gaussian*dzetadH*(3.0_DP - 2.0_DP*zeta*zeta)
            end if
            dTSRdH(:, :, s, i) = -TGG*dFermi_fndH + (1.0_DP - fermi_fn)*(dTdipdH*U + Tdip*dUdH + W*dRmatdH + Rmat*dWdH)
          end do ! loop over all cell vector components
        end do ! Loop over all cell vectors
      end if
    end if ! if to check if forces and stresses need to be computed
    write(6, *) 'if to check if forces and stresses need to be computed' !rev'''
