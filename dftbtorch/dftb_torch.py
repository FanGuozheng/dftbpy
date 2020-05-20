#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
main DFTB code
required:
    numpy
    pytorch
'''

import argparse
import os
import numpy as np
import torch as t
import bisect
from slakot import ReadSlaKo, SlaKo, SKTran
from electront import DFTBelect
from readt import ReadInt, ReadSKt
from periodic import Periodic
GEN_PARA = {"inputfile_name": 'in.ground'}
VAL_ELEC = {"H": 1, "C": 4, "N": 5, "O": 6, "Ti": 4}
PUBPARA = {"LDIM": 9, "AUEV": 27.2113845, "BOHR": 0.529177210903, "tol": 1E-4}


def main(para):
    '''
    We have implemented pytorch into this code with different interface,
    we will_ use different ty values for different interface
    '''
    # read input construct these data for next DFTB calculations
    Initialization(para)

    # with all necessary data, run dftb calculation
    Rundftbpy(para)


class Initialization:
    '''
    this class aims to read input coor, calculation parameters and SK tables;
    Then with SK transformation, construct 1D Hamiltonian and overlap matrice
    for the following DFTB calculations
    '''
    def __init__(self, para):
        '''
        here several steps will_ be easy to have more interface in the future
        Input:
            LCmdArgs (optional): True or False
            LReadInput (optional): True or False
        '''
        self.para = para
        self.slako, self.readsk = SlaKo(self.para), ReadSlaKo(self.para)
        self.readin = ReadInt(para)

        # step 1: whether get parameters from command line
        if 'LCmdArgs' in self.para.keys():
            if self.para['LCmdArgs']:  # define 'LCmdArgs' True
                self.parser_cmd_args()
        elif 'LCmdArgs' not in self.para.keys():  # not define 'LCmdArgs'
            self.parser_cmd_args()

        # step 2: if read para from dftb_in, or define yourself
        if 'LReadInput' in self.para.keys():
            if self.para['LReadInput']:
                self.readin.get_task(para)
                self.readin.get_coor(para)
        elif 'LReadInput' not in self.para.keys():
            self.readin.get_task(para)
            self.readin.get_coor(para)

        # step 3: generate vector, distance ... from given geometry
        self.readin.cal_coor()

        if not self.para['Lml']:
            self.normal_sk()

    def parser_cmd_args(self):
        '''
        raed some input information, including path, names of files, etc.
        either from command line or define yourself somewhere
            default path of input: current path
            default path of .skf file: ./slko
            default inout name: dftb_in (json type)
        '''
        _description = 'Test script demonstrating argparse'
        parser = argparse.ArgumentParser(description=_description)
        msg = 'Directory (default: .)'
        parser.add_argument('-d', '--directory', default='.', help=msg)
        msg = 'Directory_SK (default: .)'
        parser.add_argument('-sk', '--directorySK', default='slko', help=msg)
        msg = 'input filename'
        parser.add_argument('-fn', '--filename', type=str, default='dftb_in',
                            metavar='NAME', help=msg)
        args = parser.parse_args()
        path = os.getcwd()
        if 'filename' not in self.para:
            self.para['filename'] = args.filename
        if 'direInput' not in self.para:
            self.para['direInput'] = os.path.join(path, args.directory)
        if 'direSK' not in self.para:
            self.para['direSK'] = os.path.join(path, args.directorySK)

    def normal_sk(self):
        '''
        This is for normal DFTB calculations, read integrals from .skf
        '''
        self.readsk.read_sk_specie()
        SKTran(self.para)

    def form_sk_spline(self):
        '''use SK table data to build spline interpolation'''
        self.readsk.read_sk_specie()
        self.slako.get_sk_spldata()

    def gen_sk_matrix(self, para):
        '''SK transformations'''
        SKTran(para)


class Rundftbpy:
    '''
    According to the input parameters, call different calculation tasks
    input parameters:
        Lperiodic: False or True
        scc: scc, nonscc, xlbomd
    '''
    def __init__(self, para):
        '''run dftb with multi interface'''
        self.para = para
        self.cal_repulsive()
        self.rundftbplus()
        # self.sum_dftb()

    def rundftbplus(self):
        '''run dftb-torch
        if solid / molecule, scc / nonscc'''
        scf = SCF(self.para)
        Print(self.para).print_scf_title()

        if not self.para['Lperiodic'] and self.para['scc'] == 'nonscc':
            scf.scf_npe_nscc()
        elif not self.para['Lperiodic'] and self.para['scc'] == 'scc':
            if self.para['HSsym'] == 'symhalf':
                scf.scf_npe_scc()
            elif self.para['HSsym'] in ['symall', 'symall_chol']:
                scf.scf_npe_scc_symall()
        elif not self.para['Lperiodic'] and self.para['scc'] == 'xlbomd':
            scf.scf_npe_xlbomd()
        elif self.para['Lperiodic'] and self.para['scc'] is 'scc':
            scf.scf_pe_scc()

    def case_mathod():
        '''case method'''
        pass

    def cal_repulsive(self):
        if self.para['Lrepulsive']:
            Repulsive(self.para)

    def sum_dftb(self):
        Analysis(self.para).dftb_energy()


class SCF:
    '''
    This class is for self-consistent field method
    input parameters:
        Lperiodic: Ture or False
        scc: scc, nonscc or xlbomd
        hammat: this is H0 after SK transformations
        overm: this is overlap after SK transformations
        natom: number of atoms
        atomind: lmax of atoms (valence electrons)
        atomind2: sum of all lamx od atoms (the length of Hamiltonian matrix)
        HSsym: if write all / half of the matrices (due to symmetry)
    '''
    def __init__(self, para):
        '''different options (scc, periodic) for scf calculations'''
        self.para = para
        self.nat = para['natom']
        self.atind = para['atomind']
        self.atind2 = para['atomind2']
        self.hmat = para['hammat']
        self.smat = para['overmat']

    def scf_npe_nscc(self):
        '''
        atomind is the number of atom, for C, lmax is 2, therefore
        we need 2**2 orbitals (s, px, py, pz), then define atomind2
        '''
        get_qatom(self.para)
        elect = DFTBelect(self.para)
        analysis = Analysis(self.para)
        print_ = Print(self.para)
        self.para['qzero'], ind_nat = self.para['qatom'], self.atind[self.nat]

        icount = 0
        if self.para['HSsym'] in ['symall', 'symall_chol']:
            eigm, overm = self.hmat, self.smat
        elif self.para['HSsym'] == 'symhalf':
            eigm, overm = t.zeros(ind_nat, ind_nat), t.zeros(ind_nat, ind_nat)
            for iind in range(0, ind_nat):
                for jind in range(0, iind + 1):
                    eigm[jind, iind] = self.hmat[icount]
                    overm[jind, iind] = self.smat[icount]
                    eigm[iind, jind] = self.hmat[icount]
                    overm[iind, jind] = self.smat[icount]
                    icount += 1

        # get eigenvector and eigenvalue (and cholesky decomposition)
        eigval_ch, eigm_ch = self._cholesky(eigm, overm)

        # calculate the occupation of electrons
        energy = 0
        occ = elect.fermi(eigval_ch)
        for iind in range(0, int(self.atind[self.nat])):
            if occ[iind] > PUBPARA['tol']:
                energy = energy + occ[iind] * eigval_ch[iind]

        # density matrix, work controls the unoccupied eigm as 0!!
        work = t.sqrt(occ)
        for jind in range(0, ind_nat):  # n = no. of occupied orbitals
            for iind in range(0, ind_nat):
                eigm_ch[iind, jind] = eigm_ch[iind, jind] * work[jind]
        denmat = t.mm(eigm_ch, eigm_ch.t())

        # calculate mulliken charges
        if self.para['HSsym'] == 'symhalf':
            denmat_ = t.zeros(self.atind2)
            for iind in range(0, ind_nat):
                for j_i in range(0, iind + 1):
                    inum = int(iind * (iind + 1) / 2 + j_i)
                    denmat_[inum] = denmat[j_i, iind]
            qatom = elect.mulliken(self.para['HSsym'], self.smat, denmat_)
        elif self.para['HSsym'] in ['symall', 'symall_chol']:
            qatom = elect.mulliken(self.para['HSsym'], self.smat, denmat)

        # print and write non-SCC DFTB results
        self.para['eigenvalue'], self.para['qatomall'] = eigval_ch, qatom
        analysis.dftb_energy()
        analysis.sum_property(), print_.print_dftb_caltail()

    def scf_npe_scc(self):
        '''
        scf for non-periodic-ML system with scc
        atomind is the number of atom, for C, lmax is 2, therefore
        we need 2**2 orbitals (s, px, py, pz), then define atomind2
        '''
        print('*' * 35, 'Non-periodic SCC DFTB', '*' * 35)
        elect = DFTBelect(self.para)
        mix = Mixing(self.para)
        elect = DFTBelect(self.para)
        analysis = Analysis(self.para)
        print_ = Print(self.para)
        maxiter = self.para['maxIter']
        get_qatom(self.para)
        gmat = elect.gmatrix()

        energy = t.zeros(maxiter)
        self.para['qzero'] = qzero = self.para['qatom']
        eigm, eigval, qatom, qmix, qdiff = [], [], [], [], []
        denmat, denmat_2d = [], []
        ind_nat = self.atind[self.nat]

        for iiter in range(0, maxiter):
            # calculate the sum of gamma * delta_q, the 1st cycle is zero
            eigm_ = t.zeros(ind_nat, ind_nat)
            oldsmat_ = t.zeros(ind_nat, ind_nat)
            denmat_, qatom_ = t.zeros(self.atind2), t.zeros(self.nat)
            fockmat_ = t.zeros(self.atind2)
            shift_, shiftorb_ = t.zeros(self.nat), t.zeros(ind_nat)
            occ_, work_ = t.zeros(ind_nat), t.zeros(ind_nat)

            if iiter > 0:
                shift_ = elect.shifthamgam(self.para, qmix[-1], qzero, gmat)
            for iat in range(0, self.nat):
                for jind in range(self.atind[iat], self.atind[iat + 1]):
                    shiftorb_[jind] = shift_[iat]

            # Hamiltonian = H0 + H2, where
            # H2 = 0.5 * sum(overlap * (gamma_IK + gamma_JK))
            icount = 0
            for iind in range(0, int(self.atind[self.nat])):
                for j_i in range(0, iind + 1):
                    fockmat_[icount] = self.hmat[icount] + 0.5 * \
                        self.smat[icount] * (shiftorb_[iind] + shiftorb_[j_i])
                    icount += 1

            # transfer 1D to 2D H, S matrice
            icount = 0
            for iind in range(0, int(self.atind[self.nat])):
                for j_i in range(0, iind + 1):
                    eigm_[j_i, iind] = fockmat_[icount]
                    oldsmat_[j_i, iind] = self.smat[icount]
                    eigm_[iind, j_i] = fockmat_[icount]
                    oldsmat_[iind, j_i] = self.smat[icount]
                    icount += 1

            # get eigenvector and eigenvalue (and cholesky decomposition)
            eigval_, eigm_ch = self._cholesky2(eigm_, oldsmat_)
            eigval.append(eigval_), eigm.append(eigm_ch)

            # calculate the occupation of electrons
            occ_ = elect.fermi(eigval_)
            for iind in range(0, int(self.atind[self.nat])):
                if occ_[iind] > PUBPARA['tol']:
                    energy[iiter] = energy[iiter] + occ_[iind] * eigval_[iind]

            # density matrix, work_ controls the unoccupied eigm as 0!!
            work_ = t.sqrt(occ_)
            for j in range(0, ind_nat):  # n = no. of occupied orbitals
                for i in range(0, self.atind[self.nat]):
                    eigm_ch[i, j] = eigm_ch[i, j].clone() * work_[j]
            denmat_2d_ = t.mm(eigm_ch, eigm_ch.t())
            denmat_2d.append(denmat_2d_)
            for iind in range(0, int(self.atind[self.nat])):
                for j_i in range(0, iind + 1):
                    inum = int(iind * (iind + 1) / 2 + j_i)
                    denmat_[inum] = denmat_2d_[j_i, iind]
            denmat.append(denmat_)

            # calculate mulliken charges
            qatom_ = elect.mulliken(self.para['HSsym'], self.smat[:], denmat_)
            qatom.append(qatom_)
            ecoul = 0.0
            for i in range(0, self.nat):
                ecoul = ecoul + shift_[i] * (qatom_[i] + qzero[i])
            energy[iiter] = energy[iiter] - 0.5 * ecoul
            mix.mix(iiter, qzero, qatom, qmix, qdiff)
 
           # if reached convergence
            self.print_energy(iiter, energy)
            reach_convergence = self.convergence(iiter, maxiter, qdiff)
            if reach_convergence:
                break

        # print and write non-SCC DFTB results
        self.para['eigenvalue'], self.para['qatomall'] = eigval_, qatom[-1]
        analysis.sum_property(), print_.print_dftb_caltail()

    def scf_npe_scc_symall(self):
        '''
        scf for non-periodic-ML system with scc
        atomind is the number of atom, for C, lmax is 2, therefore
        we need 2**2 orbitals (s, px, py, pz), then define atomind2
        '''
        elect = DFTBelect(self.para)
        mix = Mixing(self.para)
        elect = DFTBelect(self.para)
        analysis = Analysis(self.para)
        print_ = Print(self.para)
        maxiter = self.para['maxIter']
        get_qatom(self.para)
        gmat = elect.gmatrix()

        energy = t.zeros(maxiter)
        self.para['qzero'] = qzero = self.para['qatom']
        eigm, eigval, qatom, qmix, qdiff, denmat = [], [], [], [], [], []
        ind_nat = self.atind[self.nat]
        # print('hamt:', self.hmat)

        for iiter in range(0, maxiter):
            # calculate the sum of gamma * delta_q, the 1st cycle is zero
            qatom_, fockmat_ = t.zeros(self.nat), t.zeros(ind_nat, ind_nat)
            shift_, shiftorb_ = t.zeros(self.nat), t.zeros(ind_nat)
            occ_, work_ = t.zeros(ind_nat), t.zeros(ind_nat)

            if iiter > 0:
                shift_ = elect.shifthamgam(self.para, qmix[-1], qzero, gmat)
            for iat in range(0, self.nat):
                for jind in range(self.atind[iat], self.atind[iat + 1]):
                    shiftorb_[jind] = shift_[iat]

            # Hamiltonian = H0 + H2, where
            # H2 = 0.5 * sum(overlap * (gamma_IK + gamma_JK))
            icount = 0
            for iind in range(0, ind_nat):
                for j_i in range(0, ind_nat):
                    fockmat_[iind, j_i] = self.hmat[iind, j_i] + 0.5 * \
                        self.smat[iind, j_i] * (shiftorb_[iind] + shiftorb_[j_i])
                    icount += 1

            # get eigenvector and eigenvalue (and cholesky decomposition)
            eigval_, eigm_ch = self._cholesky2(fockmat_, self.smat)
            eigval.append(eigval_), eigm.append(eigm_ch)

            # calculate the occupation of electrons
            occ_ = elect.fermi(eigval_)
            self.para['eigenvalue'], self.para['shift_'] = eigval_, shift_
            '''for iind in range(0, int(self.atind[self.nat])):
                if occ_[iind] > PUBPARA['tol']:
                    energy[iiter] = energy[iiter] + occ_[iind] * eigval_[iind]
            '''
            # density matrix, work_ controls the unoccupied eigm as 0!!
            work_ = t.sqrt(occ_)
            for j in range(0, ind_nat):  # n = no. of occupied orbitals
                for i in range(0, self.atind[self.nat]):
                    eigm_ch[i, j] = eigm_ch[i, j].clone() * work_[j]
            denmat_ = t.mm(eigm_ch, eigm_ch.t())
            denmat.append(denmat_)

            # calculate mulliken charges
            qatom_ = elect.mulliken(self.para['HSsym'], self.smat[:], denmat_)
            qatom.append(qatom_)
            mix.mix(iiter, qzero, qatom, qmix, qdiff)
            self.para['qatom_'] = qatom_

            # if reached convergence
            analysis.dftb_energy()
            energy[iiter] = self.para['energy']
            self.print_energy(iiter, energy)
            if self.convergence(iiter, maxiter, qdiff):
                break

        # print and write non-SCC DFTB results
        self.para['eigenvalue'], self.para['qatomall'] = eigval_, qatom[-1]
        analysis.sum_property(), print_.print_dftb_caltail()

    def scf_pe_scc(self):
        '''scf for periodic with scc'''
        pass

    def scf_npe_xlbomd(self):
        '''
        scf for non-periodic-ML system with scc
        atomind is the number of atom, for C, lmax is 2, therefore
        we need 2**2 orbitals (s, px, py, pz), then define atomind2
        '''
        elect = DFTBelect(self.para)
        get_qatom(self.para)
        gmat = elect.gmatrix()
        mix = Mixing(self.para)
        elect = DFTBelect(self.para)
        analysis = Analysis(self.para)
        print_ = Print(self.para)

        energy = 0
        self.para['qzero'] = qzero = self.para['qatom']
        qatom_xlbomd = self.para['qatom_xlbomd']
        ind_nat = self.atind[self.nat]

        # calculate the sum of gamma * delta_q, the 1st cycle is zero
        denmat_, qatom_ = t.zeros(self.atind2), t.zeros(self.nat)
        shift_, shiftorb_ = t.zeros(self.nat), t.zeros(ind_nat)
        occ_, work_ = t.zeros(ind_nat), t.zeros(ind_nat)

        shift_ = elect.shifthamgam(self.para, qatom_xlbomd, qzero, gmat)
        for iat in range(0, self.nat):
            for jind in range(self.atind[iat], self.atind[iat + 1]):
                shiftorb_[jind] = shift_[iat]

        # Hamiltonian = H0 + H2
        '''icount = 0
        if self.para['HSsym'] == 'symall':
            eigm = self.hmat
            overm = self.smat
        else:
            eigm = t.zeros(self.atind[self.nat], self.atind[self.nat])
            overm = t.zeros(self.atind[self.nat], self.atind[self.nat])
            for iind in range(0, self.atind[self.nat]):
                for jind in range(0, iind + 1):
                    eigm[jind, iind] = self.hmat[icount]
                    overm[jind, iind] = self.smat[icount]
                    eigm[iind, jind] = self.hmat[icount]
                    overm[iind, jind] = self.smat[icount]
                    icount += 1'''

        icount = 0
        if self.para['HSsym'] == 'symall':
            eigm_ = t.zeros(ind_nat, ind_nat)
            for iind in range(0, ind_nat):
                for j_i in range(0, ind_nat):
                    eigm_[iind, j_i] = self.hmat[iind, j_i] + 0.5 * \
                        self.smat[iind, j_i] * (shiftorb_[iind] + shiftorb_[j_i])
                    icount += 1
            oldsmat_ = self.hmat
        elif self.para['HSsym'] == 'symhalf':
            fockmat_ = t.zeros(self.atind2)
            eigm_ = t.zeros(ind_nat, ind_nat)
            oldsmat_ = t.zeros(ind_nat, ind_nat)
            for iind in range(0, int(self.atind[self.nat])):
                for j_i in range(0, iind + 1):
                    fockmat_[icount] = self.hmat[icount] + 0.5 * \
                        self.smat[icount] * (shiftorb_[iind] + shiftorb_[j_i])
                    icount += 1
            icount = 0
            for iind in range(0, ind_nat):
                for j_i in range(0, iind + 1):
                    eigm_[j_i, iind] = fockmat_[icount]
                    oldsmat_[j_i, iind] = self.smat[icount]
                    eigm_[iind, j_i] = fockmat_[icount]
                    oldsmat_[iind, j_i] = self.smat[icount]
                    icount += 1

        # get eigenvector and eigenvalue (and cholesky decomposition)
        eigval_, eigm_ch = self._cholesky(eigm_, oldsmat_)

        # calculate the occupation of electrons
        occ_ = elect.fermi(eigval_)
        for iind in range(0, int(self.atind[self.nat])):
            if occ_[iind] > PUBPARA['tol']:
                energy = energy + occ_[iind] * eigval_[iind]

        # density matrix, work_ controls the unoccupied eigm as 0!!
        work_ = t.sqrt(occ_)
        for j in range(0, ind_nat):  # n = no. of occupied orbitals
            for i in range(0, self.atind[self.nat]):
                eigm_ch[i, j] = eigm_ch[i, j].clone() * work_[j]
        denmat_2d_ = t.mm(eigm_ch, eigm_ch.t())
        for iind in range(0, int(self.atind[self.nat])):
            for j_i in range(0, iind + 1):
                inum = int(iind * (iind + 1) / 2 + j_i)
                denmat_[inum] = denmat_2d_[j_i, iind]

        # calculate mulliken charges
        qatom_ = elect.mulliken(self.para['HSsym'], self.smat, denmat_)
        ecoul = 0.0
        for i in range(0, self.nat):
            ecoul = ecoul + shift_[i] * (qatom_[i] + qzero[i])
        energy = energy - 0.5 * ecoul

        # print and write non-SCC DFTB results
        self.para['eigenvalue'], self.para['qatomall'] = eigval_, qatom_
        analysis.sum_property(), print_.print_dftb_caltail()

    def _cholesky(self, matrixa, matrixb):
        '''
        cholesky decomposition of B: B = LL^{T}
            AX = (lambda)BX ==> (L^{-1}AL^{-T})(L^{T}X) = (lambda)(L^{T}X)
        matrix_a: Fock operator
        matrix_b: overm'''

        chol_l = t.cholesky(matrixb)
        # self.para['eigval'] = chol_l
        linv_a = t.mm(t.inverse(chol_l), matrixa)
        l_invtran = t.inverse(chol_l.t())
        linv_a_linvtran = t.mm(linv_a, l_invtran)
        eigval, eigm = t.symeig(linv_a_linvtran, eigenvectors=True)
        eigm_ab = t.mm(l_invtran, eigm)
        return eigval, eigm_ab

    def _cholesky2(self, matrixa, matrixb):
        '''
        cholesky decomposition of B: B = LL^{T}
            AX = (lambda)BX ==> (L^{-1}AL^{-T})(L^{T}X) = (lambda)(L^{T}X)
        matrix_a: Fock operator
        matrix_b: overm'''

        chol_l = t.cholesky(matrixb)
        row = matrixa.shape[1]
        A1, LU_A = t.solve(matrixa, chol_l)
        A2, LU_A1 = t.solve(A1.t(), chol_l)
        A3 = A2.t()
        eigval, eigm = t.symeig(A3, eigenvectors=True)
        l_inv, _ = t.solve(t.eye(row), chol_l.t())
        eigm_ab = t.mm(l_inv, eigm)
        return eigval, eigm_ab

    def lowdin_symeig(self, matrixa, matrixb):
        '''
        use t.symeig to decompose B: BX = (lambda)X, then omega = lambda.diag()
            S_{-1/2} = Xomega_{-1/2}X_{T}
            AX' = (lambda)BX' ==>
            (S_{-1/2}AS_{-1/2})(S_{1/2}X') = (lambda)(S_{1/2}X')
        matrix_a: Fock operator
        matrix_b: overlap matrix
        '''
        lam_b, l_b = t.symeig(matrixb, eigenvectors=True)
        lam_sqrt_inv = t.sqrt(1 / lam_b)
        S_sym = t.mm(l_b, t.mm(lam_sqrt_inv.diag(), l_b.t()))
        SHS = t.mm(S_sym, t.mm(matrixa, S_sym))
        eigval, eigvec_ = t.symeig(SHS, eigenvectors=True)
        eigvec = t.mm(S_sym, eigvec_)
        # eigval3, eigvec_ = t.symeig(lam_b_2d, eigenvectors=True)
        return eigval, eigvec

    def lowdin_svd_sym(self, matrixa, matrixb):
        '''
        SVD decomposition of B: B = USV_{T}
            S_{-1/2} = US_{-1/2}V_{T}
            AX = (lambda)BX ==>
            (S_{-1/2}AS_{-1/2})(S_{1/2}X) = (lambda)(S_{1/2}X)
        matrix_a: Fock operator
        matrix_b: overlap matrix
        '''
        ub, sb, vb = t.svd(matrixb)
        sb_sqrt_inv = t.sqrt(1 / sb)
        S_sym = t.mm(ub, t.mm(sb_sqrt_inv.diag(), vb.t()))
        SHS = t.mm(S_sym, t.mm(matrixa, S_sym))
        eigval, eigvec_ = t.symeig(SHS, eigenvectors=True)
        eigvec = t.mm(S_sym, eigvec_)
        return eigval, eigvec

    def lowdin_svd(self, matrixa, matrixb):
        '''
        SVD decomposition of B: B = USV_{T}
            S_{-1/2} = US_{-1/2}V_{T}
            AX = (lambda)BX ==>
            (S_{-1/2}AS_{-1/2})(S_{1/2}X) = (lambda)(S_{1/2}X)
        matrix_a: Fock operator
        matrix_b: overlap matrix
        '''
        ub, sb, vb = t.svd(matrixb)
        sb_sqrt_inv = t.sqrt(1 / sb)
        S_sym = t.mm(ub, t.mm(sb_sqrt_inv.diag(), vb.t()))
        SHS = t.mm(S_sym, t.mm(matrixa, S_sym))

        ub2, sb2, vb2 = t.svd(SHS)
        eigvec = t.mm(S_sym, ub2)
        return sb2, eigvec

    def lowdin_qr_eig(self, matrixa, matrixb):
        '''
        SVD decomposition of B: B = USV_{T}
            S_{-1/2} = US_{-1/2}V_{T}
            AX = (lambda)BX ==>
            (S_{-1/2}AS_{-1/2})(S_{1/2}X) = (lambda)(S_{1/2}X)
        matrix_a: Fock operator
        matrix_b: overlap matrix
        '''
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
        '''
        SVD decomposition of B: B = USV_{T}
            S_{-1/2} = US_{-1/2}V_{T}
            AX = (lambda)BX ==>
            (S_{-1/2}AS_{-1/2})(S_{1/2}X) = (lambda)(S_{1/2}X)
        matrix_a: Fock operator
        matrix_b: overlap matrix
        '''
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

    def print_energy(self, iiter, energy):
        if iiter == 0:
            self.dE = energy[iiter].detach()
            print('iteration', ' '*8, 'energy', ' '*20, 'dE')
            print(f'{iiter:5} {energy[iiter].detach():25}', f'{self.dE:25}')
        elif iiter >= 1:
            self.dE = energy[iiter].detach() - energy[iiter - 1].detach()
            print(f'{iiter:5} {energy[iiter].detach():25}', f'{self.dE:25}')

    def convergence(self, iiter, maxiter, qdiff):
        if self.para['convergenceType'] == 'energy':
            if abs(self.dE) < self.para['energy_tol']:
                reach_convergence = True
            elif iiter + 1 >= maxiter and abs(self.dE) > PUBPARA['tol']:
                print('Warning: SCF donot reach required convergence')
                reach_convergence = True
            else:
                reach_convergence = False
        elif self.para['convergenceType'] == 'charge':
            qdiff_ = t.sum(qdiff[-1]) / len(qdiff[-1])
            if abs(qdiff_) < self.para['energy_tol']:
                reach_convergence = True
            elif iiter + 1 >= maxiter and abs(qdiff_) > PUBPARA['tol']:
                print('Warning: SCF donot reach required convergence')
                reach_convergence = True
            else:
                reach_convergence = False
        return reach_convergence


class Repulsive():

    def __init__(self, para):
        self.para = para
        self.nat = self.para['natom']
        self.get_rep_para()
        self.cal_rep_energy()

    def get_rep_para(self):
        Periodic(self.para).get_neighbour(cutoff='repulsive')

    def cal_rep_energy(self):
        self.rep_energy = t.zeros(self.nat)
        atomnameall = self.para['atomnameall']
        for iat in range(0, self.nat):
            for jat in range(iat + 1, self.nat):
                nameij = atomnameall[iat] + atomnameall[jat]
                cutoff_ = self.para['cutoff_rep' + nameij]
                distanceij = self.para['distance'][iat, jat]
                if distanceij < cutoff_:
                    ienergy = self.cal_erep_atomij(distanceij, nameij)
                    self.rep_energy[iat] = self.rep_energy[iat] + ienergy
        sum_energy = t.sum(self.rep_energy[:])
        self.para['rep_energy'] = sum_energy

    def cal_erep_atomij(self, distanceij, nameij):
        nint = self.para['nint_rep' + nameij]
        alldist = t.zeros(nint + 1)
        a1 = self.para['a1_rep' + nameij]
        a2 = self.para['a2_rep' + nameij]
        a3 = self.para['a3_rep' + nameij]
        alldist[:-2] = self.para['rep' + nameij][:, 0]
        alldist[nint-1:] = self.para['repend' + nameij][:2]
        if distanceij < alldist[0]:
            energy = t.exp(-a1 * distanceij + a2) + a3
        elif distanceij < alldist[-1]:
            ddind = bisect.bisect(alldist.numpy(), distanceij) - 1
            if ddind <= nint - 1:
                para = self.para['rep' + nameij][ddind]
                deltar = distanceij - para[0]
                assert deltar > 0
                energy = para[2] + para[3] * deltar + para[4] * deltar ** 2 \
                    + para[5] * deltar ** 3
            elif ddind == nint:
                para = self.para['repend' + nameij][ddind]
                deltar = distanceij - para[0]
                assert deltar > 0
                energy = para[2] + para[3] * deltar + para[4] * deltar ** 2 \
                    + para[5] * deltar ** 3 + para[6] * deltar ** 4 + \
                    para[7] * deltar ** 5
        else:
            print('Error: {} distance > cutoff'.format(nameij))
        return energy


class Mixing:

    def __init__(self, para):
        '''class for different mixing method'''
        self.para = para
        if self.para['mixMethod'] == 'broyden':
            self.df, self.uu = [], []
            self.ww = t.zeros(self.para['maxIter'])

    def mix(self, iiter, qzero, qatom, qmix, qdiff):
        '''calling different mixing methods'''
        if iiter == 0:
            qmix.append(qzero)
            if self.para['mixMethod'] == 'broyden':
                self.df.append(t.zeros(self.para['natom']))
                self.uu.append(t.zeros(self.para['natom']))
            qmix_ = self.simple_mix(qzero, qatom[-1], qdiff)
            qmix.append(qmix_)
        else:
            if self.para['mixMethod'] == 'simple':
                qmix_ = self.simple_mix(qmix[-1], qatom[-1], qdiff)
            elif self.para['mixMethod'] == 'broyden':
                qmix_ = self.broyden_mix(iiter, qmix, qatom[-1], qdiff)
            elif self.para['mixMethod'] == 'anderson':
                qmix_ = self.anderson_mix(iiter, qmix, qatom, qdiff)
            qmix.append(qmix_)
        self.para['qatomall'] = qatom

    def simple_mix(self, oldqatom, qatom, qdiff):
        '''this is for simple mixing method'''
        mixf = self.para['mixFactor']
        qdiff.append(qatom - oldqatom)
        qmix_ = oldqatom + mixf * qdiff[-1]
        return qmix_

    def anderson_mix(self, iiter, qmix, qatom, qdiff):
        '''this is for anderson mixing method'''
        mixf = self.para['mixFactor']
        qdiff.append(qatom[-1] - qmix[-1])
        df_iiter, df_prev = qdiff[-1], qdiff[-2]
        temp1 = t.dot(df_iiter, df_iiter - df_prev)
        temp2 = t.dot(df_iiter - df_prev, df_iiter - df_prev)
        beta = temp1 / temp2
        average_qin = (1.0 - beta) * qmix[-1] + beta * qmix[-2]
        average_qout = (1.0 - beta) * qatom[-1] + beta * qatom[-2]
        qmix_ = (1 - mixf) * average_qin + mixf * average_qout
        return qmix_

    def broyden_mix(self, iiter, qmix, qatom_, qdiff):
        '''this is for broyden mixing method'''
        aa = t.zeros(iiter, iiter)
        cc = t.zeros(iiter, iiter)
        beta = t.zeros(iiter, iiter)
        weight = 1e-2
        omega0 = 1e-2
        alpha = self.para['mixFactor']

        qdiff.append(qatom_ - qmix[-1])
        df_uu = qdiff[-1] - qdiff[-2]
        self.ww[iiter - 1] = weight / (t.sqrt(t.dot(qdiff[-1], qdiff[-1])))
        inv_norm = 1 / t.sqrt(t.dot(df_uu, df_uu))
        df_uu = inv_norm * df_uu

        for ii in range(0, iiter - 1):
            aa[ii, iiter - 1] = t.dot(self.df[ii], df_uu)
            aa[iiter - 1, ii] = aa[ii, iiter - 1]
            cc[0, ii] = self.ww[ii] * t.dot(self.df[ii], qdiff[-1])
        aa[iiter - 1, iiter - 1] = 1.0
        cc[0, iiter - 1] = self.ww[iiter - 1] * t.dot(df_uu, qdiff[-1])

        for ii in range(0, iiter):
            beta[:iiter - 1, ii] = self.ww[:iiter - 1] * self.ww[ii] * \
                aa[:iiter - 1, ii]
            beta[ii, ii] = beta[ii, ii] + omega0 ** 2
        beta = t.inverse(beta)
        gamma = t.mm(cc, beta)
        self.df.append(df_uu)
        df_uu = alpha * df_uu + inv_norm * (qmix[-1] - qmix[-2])

        qmix_ = qmix[-1] + alpha * qdiff[-1]
        print('qmix_1', qmix_, qmix[-1])
        for ii in range(0, iiter - 1):
            qmix_ = qmix_ - self.ww[ii] * gamma[0, ii] * self.uu[ii]
        print('qmix_2', qmix_, self.ww[:], gamma[0, :], self.uu[ii])
        qmix_ = qmix_ - self.ww[iiter - 1] * gamma[0, iiter - 1] * df_uu
        print('qmix_3', qmix_, self.ww[iiter - 1], gamma[0, iiter - 1])
        self.uu.append(df_uu)
        return qmix_


class Write:

    def __init__(self, para):
        self.para = para

    def write(self):
        pass


class Print:

    def __init__(self, para):
        self.para = para

    def print_scf_title(self):
        if not self.para['Lperiodic'] and self.para['scc'] == 'nonscc':
            print('*' * 35, 'Non-periodic Non-SCC-DFTB', '*' * 35)
        elif not self.para['Lperiodic'] and self.para['scc'] == 'scc':
            print('*' * 35, 'Non-periodic SCC-DFTB', '*' * 35)
        elif not self.para['Lperiodic'] and self.para['scc'] == 'xlbomd':
            print('*' * 35, 'Non-periodic xlbomd-DFTB', '*' * 35)
        elif self.para['Lperiodic'] and self.para['scc'] is 'scc':
            print('*' * 35, 'Periodic SCC-DFTB', '*' * 35)

    def print_dftb_caltail(self):
        print('charge: \n', self.para['qatomall'].detach())
        print('dipole: \n', self.para['dipole'].detach())
        print('energy: \n', self.para['energy'].detach())
        print('TS energy: \n', self.para['H0_energy'].detach())
        if self.para['scc'] == 'scc':
            print('Coulomb energy: \n', -self.para['coul_energy'].detach() / 2)
        print('repulsive energy: \n', self.para['rep_energy'].detach())


class Analysis:

    def __init__(self, para):
        self.para = para
        self.nat = self.para['natom']

    def dftb_energy(self):
        eigval = self.para['eigenvalue']
        occ = self.para['occ']
        if self.para['scc'] == 'nonscc':
            self.para['H0_energy'] = t.dot(eigval, occ)
            if self.para['Lrepulsive']:
                self.para['energy'] = self.para['H0_energy'] + \
                    self.para['rep_energy']
            else:
                self.para['energy'] = self.para['H0_energy']
        if self.para['scc'] == 'scc':
            qzero = self.para['qzero']
            shift_ = self.para['shift_']
            qatom_ = self.para['qatom_']
            self.para['H0_energy'] = t.dot(eigval, occ)
            ecoul = 0.0
            for i in range(0, self.nat):
                ecoul = ecoul + shift_[i] * (qatom_[i] + qzero[i])
            # energy[iiter] = energy[iiter] - 0.5 * ecoul
            self.para['coul_energy'] = ecoul
            if self.para['Lrepulsive']:
                self.para['energy'] = self.para['H0_energy'] + \
                    self.para['rep_energy'] - 0.5 * self.para['coul_energy']
            else:
                self.para['energy'] = self.para['H0_energy'] + \
                    self.para['coul_energy']

    def sum_property(self):
        nocc = self.para['nocc']
        eigval = self.para['eigenvalue']
        qzero, qatom = self.para['qzero'], self.para['qatomall']
        self.para['homo_lumo'] = eigval[int(nocc) - 1:int(nocc) + 1] * \
            PUBPARA['AUEV']
        self.para['dipole'] = get_dipole(self.para, qzero, qatom)


def read_sk(para):
    '''
    generate the electrons, the onsite only includes s, p and d oribitals
    '''
    atomname = para['atomnameall']
    onsite = np.zeros((len(atomname), 3))
    spe = np.zeros(len(atomname))
    uhubb = np.zeros((len(atomname), 3))
    occ_atom = np.zeros((len(atomname), 3))
    atomname_set = list(set(atomname))
    icount = 0
    for namei in atomname:
        for namej in atomname:
            ReadSKt(para, namei, namej)
        onsite[icount, :] = para['espd_uspd'+namei+namei][0:3]
        spe[icount] = para['espd_uspd'+namei+namei][3]
        uhubb[icount, :] = para['espd_uspd'+namei+namei][4:7]
        occ_atom[icount, :] = para['espd_uspd'+namei+namei][7:10]
        icount += 1
    para['atomname_set'] = atomname_set
    para['onsite'] = onsite
    para['spe'] = spe
    para['uhubb'] = uhubb
    para['occ_atom'] = occ_atom
    return para


def get_qatom(para):
    '''get the basic electronic info of each atom'''
    natom = para['natom']
    atomname = para['atomnameall']
    num_electrons = 0
    qatom = t.empty(natom)
    for i in range(0, natom):
        qatom[i] = VAL_ELEC[atomname[i]]
        num_electrons += qatom[i]
    para['qatom'] = qatom
    para['nelectrons'] = num_electrons
    return para


def get_dipole(para, qzero, qatom):
    '''read and process dipole data'''
    coor = para['coor']
    natom = para['natom']
    dipole = t.zeros(3)
    for iatom in range(0, natom):
        if type(coor[iatom][:]) is list:
            coor_t = t.from_numpy(np.asarray(coor[iatom][1:]))
            dipole[:] = dipole[:] + (qzero[iatom] - qatom[iatom]) * coor_t
        else:
            dipole[:] = dipole[:] + (qzero[iatom] - qatom[iatom]) * \
                coor[iatom][1:]
    return dipole
