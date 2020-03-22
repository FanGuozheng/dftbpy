#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''implement of pytorch into DFTB code'''

import argparse
import os
import numpy as np
import torch as t
from torch.autograd import Variable
from slakot import SlaKo
from electront import DFTBelect
from readt import ReadInt, ReadSKt
GEN_PARA = {"inputfile_name": 'in.ground'}
VAL_ELEC = {"H": 1, "C": 4, "N": 5, "O": 6, "Ti": 4}
PUBPARA = {"LDIM": 9, "AUEV": 27.2113845, "BOHR": 0.529177210903, "tol": 1E-4}


def main(para):
    '''
    We have implemented pytorch into this code with different interface,
    we will_ use different ty values for different interface
    '''
    # the output parameters
    parser_cmd_args(para)

    # read input construct these data for next DFTB calculations
    Initialization(para)

    # with all necessary data, run dftb calculation
    Rundftbpy(para)


def parser_cmd_args(para):
    '''
    raed some input information, including path, names of files, etc.
    default path of input: current path
    default path of .skf file: ./slko
    default inout name: dftb_in (json)
    '''
    _description = 'Test script demonstrating argparse'
    parser = argparse.ArgumentParser(description=_description)
    msg = 'Directory (default: .)'
    parser.add_argument('-d', '--directory', default='.', help=msg)
    msg = 'Directory_SK (default: .)'
    parser.add_argument('-sk', '--directorySK', default='slko', help=msg)
    msg = 'input filename'
    parser.add_argument('-f', '--filename', type=str, default='dftb_in',
                        metavar='NAME', help=msg)
    args = parser.parse_args()
    path = os.getcwd()
    para['filename'] = args.filename
    para['direInput'] = os.path.join(path, args.directory)
    para['direSK'] = os.path.join(path, args.directorySK)
    return para


class Initialization:
    '''
    this class aims to read input coor, calculation parameters and
    SK tables;
    Then with SK transformation, construct 1D Hamiltonian and
    overlap matrix for next DFTB calculations
    '''
    def __init__(self, para):
        '''
        here several steps will_ be easy to have more interface in the future
        '''
        self.para = para
        self.slako = SlaKo(para)
        self.read = ReadInt(para)
        # step 1: if read input para from dftb_in file or define para as input
        if self.para['readInput']:
            self.read.get_task(para)

        # step 2: read geo (coor) info
        self.read.get_coor(para)

        # step 3: read SK table and operate SK data for next step
        if not para['ml']:
            self.slako.read_skdata(para)
            self.slako.sk_tranold(para)

    def form_sk_spline(self, para):
        '''use SK table data to build spline interpolation'''
        self.slako.get_sk_spldata(para)

    def gen_sk_matrix(self, para):
        '''SK transformations'''
        self.slako.sk_tranold(para)


class Rundftbpy:
    '''
    According to the task of input parameters, this code will call different
    calculation tasks
    '''
    def __init__(self, para):
        '''run dftb with multi interface'''
        self.para = para
        self.rundftbplus()

    def rundftbplus(self):
        '''run dftb-torch'''
        # if solid / molecule, scc / nonscc
        if not self.para['periodic'] and not self.para['scc']:
            SCF(self.para).scf_npe_nscc()
        elif not self.para['periodic'] and self.para['scc']:
            SCF(self.para).scf_npe_scc()
        elif self.para['periodic'] and self.para['scc']:
            SCF(self.para).scf_pe()

    def case_mathod():
        '''case method'''
        pass


class SCF:
    '''
    This class is for self-consistent field method, you need the
    following parameters:
        periodic: if Ture, solid system, elif False, molecule system
        scc: if True, scc-DFTB, elif False, non-scc-DFTB
        hammat: this is H0 after SK transformations
        overmat: this is overlap after SK transformations
    atomic coordination and electron information:
        natom: number of atoms
        atomind: lmax of atoms (valence electrons)
        atomind2: sum of all lamx od atoms (the length of Hamiltonian matrix)
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
        eigvec = t.zeros(self.atind[self.nat], self.atind[self.nat])
        overmat = t.zeros(self.atind[self.nat], self.atind[self.nat])
        print(' ' * 35, 'start for NON-SCC')
        icount = 0
        for iind in range(0, self.atind[self.nat]):
            for jind in range(0, iind + 1):
                eigvec[jind, iind] = self.hmat[icount]
                overmat[jind, iind] = self.smat[icount]
                eigvec[iind, jind] = self.hmat[icount]
                overmat[iind, jind] = self.smat[icount]
                icount += 1

        # get eigenvector and eigenvalue (and cholesky decomposition)
        eigvec_, eigval = self._cholesky(eigvec, overmat)

        # print and write non-SCC DFTB results
        print('nonscc eigen value (eV): \n', eigval * PUBPARA['AUEV'])
        nelect = self.para['nelectrons'] / 2
        self.para['homo_lumo'] = eigval[int(nelect) - 1:int(nelect) + 1] * \
            PUBPARA['AUEV']

    def scf_npe_scc(self):
        '''
        scf for non-periodic-ML system with scc
        atomind is the number of atom, for C, lmax is 2, therefore
        we need 2**2 orbitals (s, px, py, pz), then define atomind2
        '''
        elect, mix = DFTBelect(self.para), Mixing(self.para)
        maxiter = self.para['maxIter']
        get_qatom(self.para)
        gmat = elect.gmatrix()

        energy = t.zeros(maxiter)
        qzero = self.para['qatom']
        eigvec, eigval, qatom, qmix, qdiff = [], [], [], [], []
        denmat, denmat_2d = [], []
        ind_nat = self.atind[self.nat]

        # starting SCC loop
        for iiter in range(0, maxiter):

            # calculate the sum of gamma * delta_q, the 1st cycle is zero
            eigvec_ = t.zeros(ind_nat, ind_nat)
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
                    eigvec_[j_i, iind] = fockmat_[icount]
                    oldsmat_[j_i, iind] = self.smat[icount]
                    eigvec_[iind, j_i] = fockmat_[icount]
                    oldsmat_[iind, j_i] = self.smat[icount]
                    icount += 1

            # get eigenvector and eigenvalue (and cholesky decomposition)
            eigvec_ch, eigval_ = self._cholesky(eigvec_, oldsmat_)
            eigval.append(eigval_), eigvec.append(eigvec_ch)

            # calculate the occupation of electrons
            occ_ = elect.fermi(eigval_)
            nocc = self.para['nocc']

            for iind in range(0, int(self.atind[self.nat])):
                if occ_[iind] > PUBPARA['tol']:
                    energy[iiter] = energy[iiter] + occ_[iind] * eigval_[iind]

            # density matrix, work_ controls the unoccupied eigvec as 0!!
            work_ = t.sqrt(occ_)
            for j in range(0, ind_nat):  # n = no. of occupied orbitals
                for i in range(0, self.atind[self.nat]):
                    eigvec_ch[i, j] = eigvec_ch[i, j].clone() * work_[j]
            denmat_2d_ = t.mm(eigvec_ch, eigvec_ch.t())
            denmat_2d.append(denmat_2d_)
            for iind in range(0, int(self.atind[self.nat])):
                for j_i in range(0, iind + 1):
                    inum = int(iind * (iind + 1) / 2 + j_i)
                    denmat_[inum] = denmat_2d_[j_i, iind]
            denmat.append(denmat_)

            # calculate mulliken charges
            qatom_ = elect.mulliken(self.smat[:], denmat_)
            qatom.append(qatom_)
            ecoul = 0.0
            for i in range(0, self.nat):
                ecoul = ecoul + shift_[i] * (qatom_[i] + qzero[i])
            energy[iiter] = energy[iiter] - 0.5 * ecoul
            mix.mix(iiter, qzero, qatom, qmix, qdiff)
            self.print_energy(iiter, energy)

            # if reached convergence
            if iiter > 0 and abs(self.dE) < PUBPARA['tol']:
                break
            if iiter + 1 >= maxiter and abs(self.dE) > PUBPARA['tol']:
                print('Warning: SCF donot reach required convergence')

        self.para['dipole'] = get_dipole(self.para, qzero, qmix[-1])
        self.para['homo_lumo'] = eigval_[nocc - 1:nocc + 1] * PUBPARA['AUEV']
        print("self.para['homo_lumo']", self.para['homo_lumo'], qmix[-1])

    def scf_pe(self):
        '''scf for periodic with scc'''
        pass

    def _cholesky(self, matrixa, matrixb):
        '''
        cholesky decomposition of B: B = LL^{T}
            AX = (lambda)BX ==> (L^{-1}AL^{-T})(L^{T}X) = (lambda)(L^{T}X)
        matrix_a: Fock operator
        matrix_b: overmat
        '''
        chol_l = t.cholesky(matrixb)
        linv_a = t.mm(t.inverse(chol_l), matrixa)
        l_invtran = t.inverse(chol_l.t())
        linv_a_linvtran = t.mm(linv_a, l_invtran)
        eigval, eigvec = t.symeig(linv_a_linvtran, eigenvectors=True)
        eigvec_ab = t.mm(l_invtran, eigvec)
        return eigvec_ab, eigval

    def print_energy(self, iiter, energy):
        if iiter == 0:
            print('iteration', ' '*8, 'energy', ' '*20, 'dE')
            self.dE = energy[iiter].detach()
        elif iiter >= 1:
            self.dE = energy[iiter].detach() - energy[iiter - 1].detach()
            print(f'{iiter:5} {energy[iiter].detach():25}', f'{self.dE:25}')


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
        print('df_iiter, df_prev', df_iiter, df_prev)
        print(qmix[-1], qmix[-2], qatom[-1], qatom[-2])
        print(beta, temp1, temp2, average_qin, average_qout)
        return qmix_

    def broyden_mix(self, iiter, qmix, qatom_, qdiff):
        '''this is for broyden mixing method'''
        pass


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
            dipole[:] = dipole[:] + (
                    qzero[iatom] - qatom[iatom]) * coor[iatom][:]
    return dipole
