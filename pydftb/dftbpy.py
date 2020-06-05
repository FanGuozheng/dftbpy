#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
main structure of DFTBpy
'''
import os
import argparse
import numpy as np
import scipy.linalg as linalg
import slakotran
from electron import DFTB_elect
from read import ReadInput, ReadSK
GEN_PARA = {"inputfile_name": 'in.ground'}
VAL_ELEC = {"H": 1, "C": 4, "N": 5, "O": 6, "Ti": 4}
PUBPARA = {"LDIM": 9, "AUEV": 27.2113845, "BOHR": 0.529177249, "tol": 1E-4}


def main(para):
    '''
    This is the main function, with different interface, we will use
    different ty values.
    dftb:
        read from defined doucument
    dftbpy:
        all defined by python input
    dftbml:
        dftb code for machine learning
    '''
    parser_cmd_args(para)

    initialization(para)

    DFTBpy(para)


def initialization(para):
    '''
    reading and processing geometry data
    reading and processing SKF data
    '''
    ReadInput(para)

    if para['ty'] in ['dftbml']:
        read_sk_interp(para)
    elif para['ty'] in ['dftb', 'dftbpy']:
        read_sk(para)


def parser_cmd_args(para):
    '''
    read from commond line
    '''
    _DESCRIPTION = 'Test script demonstrating argparse'
    parser = argparse.ArgumentParser(description=_DESCRIPTION)
    msg = 'Directory (default: .)'
    parser.add_argument('-d', '--directory', default=os.getcwd(), help=msg)
    msg = 'Directory_SK (default: slko; relative path)'
    parser.add_argument('-s', '--directorySK', default='./slko', help=msg)
    msg = 'input filename'
    parser.add_argument('-f', '--filename', type=str, default='dftb_in',
                        metavar='NAME', help=msg)
    parser.add_argument('-t', '--MainType', type=str, default='dftb',
                        metavar='NAME', help=msg)
    args = parser.parse_args()
    para['filename'] = args.filename
    para['dire'] = args.directory
    para['direSK'] = os.path.join(args.directory, args.directorySK)
    para['ty'] = args.MainType


class DFTBpy:
    """
    this is dftb python code module
    """

    def __init__(self, para):
        '''
        calculate solid, molecule with (non-)SCC-DFTB
        '''
        self.para = para
        self.elect = DFTB_elect(self.para)
        if not para['periodic']:
            self.scf_nonpe()
        else:
            self.scf_pe()

    def scf_nonpe(self):
        '''self consistent field calculation'''
        natom = self.para['natom']
        oldqatom = np.zeros(natom)
        qzero = np.zeros(natom)
        qatom = np.zeros(natom)
        qzero[:] = self.para['qatom']
        qatom[:] = self.para['qatom']
        atomind = self.para['atomind']
        nind = atomind[natom]
        atomind2 = int(nind * (nind + 1) / 2)
        eigvec = np.zeros((nind, nind), dtype=float)
        eigval = np.zeros((nind), dtype=float)
        fockmat = np.zeros((atomind2), dtype=float)
        occ = np.zeros((nind), dtype=float)
        oldovermat = np.zeros((nind, nind), dtype=float)
        qatom, qmix, qdiff = [], [], []

        hammat, overmat = slakotran.sk_tran(self.para)  # SK transfer
        gmat = self.elect.gmatrix()  # construct gamma in second-order term

        if self.para['scc']:
            niter = self.para['maxIter']
        else:
            niter = 1
        energy = np.zeros((niter), dtype=float)

        for iiter in range(niter):
            shift_ = np.zeros((natom), dtype=float)
            shiftorb_ = np.zeros((atomind2), dtype=float)
            if iiter > 0:
                shift_ = self.elect.shifthamgam(natom, qmix[-1], qzero, gmat)
            for iat in range(natom):
                for jind in range(atomind[iat], atomind[iat + 1]):
                    shiftorb_[jind] = shift_[iat]

            icount = 0
            for iind in range(0, nind):  # construct the whole Hamiltonian
                for jind in range(0, iind + 1):
                    fockmat[icount] = hammat[icount] + 0.5 * \
                        overmat[icount] * (shiftorb_[iind] + shiftorb_[jind])
                    icount += 1

            icount = 0
            for iind in range(0, nind):
                for jind in range(iind + 1):
                    eigvec[jind, iind] = fockmat[icount]
                    oldovermat[jind, iind] = overmat[icount]
                    icount += 1

            # get eigenvector and eigenvalue
            eigval, eigvec = linalg.eigh(eigvec, oldovermat, lower=False,
                                         overwrite_a=True, overwrite_b=True)

            # calculate the occupation of electrons
            occ = self.elect.fermi(eigval, occ)[0]
            for i in range(nind):
                if occ[i] > PUBPARA['tol']:
                    energy[iiter] += occ[i] * eigval[i]
                else:
                    break
            for nocc in range(nind):
                if occ[nocc] > PUBPARA['tol']:
                    nocc += 1
                else:
                    break
            # construct density matrix
            denmat = np.zeros((atomind2))
            work = []
            for iocc in range(nocc + 1):
                work.append(np.sqrt(occ[iocc]))
            for jocc in range(nocc + 1):
                for iind in range(0, nind):
                    eigvec[iind, jocc] = eigvec[iind, jocc] * work[jocc]
            eigvec[:, nocc + 1: nind] = 0.0

            oldovermat = linalg.blas.dgemm(alpha=1.0, a=eigvec, b=eigvec,
                                           beta=0.0, trans_b=1)
            for iind in range(0, nind):
                for jind in range(iind + 1):
                    mm_ = int(iind * (iind + 1) / 2 + jind)
                    denmat[mm_] = denmat[mm_] + oldovermat[jind, iind]

            qatom.append(self.elect.mulliken(overmat, denmat))

            ecoul = 0.0
            for iat in range(natom):
                ecoul = ecoul + shift_[iat] * (qatom[-1][iat] + qzero[iat])
            energy[iiter] -= 0.5 * ecoul

            Mix(self.para, iiter, qzero, qatom, qmix, qdiff)
            self.print_energy(iiter, energy)
            if self.convergence(iiter, niter, qdiff):
                break

        if self.para['dipole']:
            get_dipole(self.para, qzero, self.para['qmix'][-1])
        self.para['humo_lumo'] = eigval[:] * PUBPARA['AUEV']

    def scf_pe(self):
        '''
        periodic calculation
        '''
        pass

    def print_energy(self, iiter, energy):
        if iiter == 0:
            self.dE = energy[iiter]
            print('iteration', ' '*8, 'energy', ' '*20, 'dE')
            print(f'{iiter:5} {energy[iiter]:25}', f'{self.dE:25}')
        elif iiter >= 1:
            self.dE = energy[iiter] - energy[iiter - 1]
            print(f'{iiter:5} {energy[iiter]:25}', f'{self.dE:25}')

    def convergence(self, iiter, maxiter, qdiff):
        if self.para['convergenceType'] == 'energy':
            if abs(self.dE) < 1E-6:
                reach_convergence = True
            elif iiter + 1 >= maxiter and abs(self.dE) > PUBPARA['tol']:
                print('Warning: SCF donot reach required convergence')
                reach_convergence = True
            else:
                reach_convergence = False
        elif self.para['convergenceType'] == 'charge':
            qdiff_ = np.sum(qdiff[-1]) / len(qdiff[-1])
            if abs(qdiff_) < self.para['energy_tol']:
                reach_convergence = True
            elif iiter + 1 >= maxiter and abs(qdiff_) > PUBPARA['tol']:
                print('Warning: SCF donot reach required convergence')
                reach_convergence = True
            else:
                reach_convergence = False
        return reach_convergence


class Mix:
    '''
    mixing method
    '''

    def __init__(self, para, iiter, qzero, qatom, qmix, qdiff):
        '''
        call different mixing method
        '''
        self.para = para
        self.iiter = iiter
        self.qzero = qzero
        self.qatom = qatom
        self.qmix = qmix
        self.qdiff = qdiff
        self.natom = self.para['natom']
        self.df, self.uu = [], []
        self.mix()

    def mix(self):
        '''calling different mixing methods'''
        if self.iiter == 0:
            self.qmix.append(self.qzero)
            if self.para['mixMethod'] == 'broyden':
                self.df.append(np.zeros((self.natom), dtype=float))
                self.uu.append(np.zeros((self.natom), dtype=float))
            qmix_ = self.simple_mix()
            self.qmix.append(qmix_)
        else:
            if self.para['mixMethod'] == 'simple':
                qmix_ = self.simple_mix()
            elif self.para['mixMethod'] == 'broyden':
                qmix_ = self.broyden_mix()
            elif self.para['mixMethod'] == 'anderson':
                qmix_ = self.anderson_mix()
            self.qmix.append(qmix_)
        # self.para['qatomall'] = qatom
        self.para['qmix'] = self.qmix

    def simple_mix(self):
        '''this is for simple mixing method'''
        mixf = self.para['mixFactor']
        self.qdiff.append(self.qatom[-1] - self.qmix[-1])
        qmix_ = self.qmix[-1] + mixf * self.qdiff[-1]
        return qmix_

    def anderson_mix(self):
        '''this is for anderson mixing method'''
        mixf = self.para['mixFactor']
        self.qdiff.append(self.qatom[-1] - self.qmix[-1])
        df_iiter, df_prev = self.qdiff[-1], self.qdiff[-2]
        temp1 = np.dot(df_iiter, df_iiter - df_prev)
        temp2 = np.dot(df_iiter - df_prev, df_iiter - df_prev)
        beta = temp1 / temp2
        average_qin = (1.0 - beta) * self.qmix[-1] + beta * self.qmix[-2]
        average_qout = (1.0 - beta) * self.qatom[-1] + beta * self.qatom[-2]
        qmix_ = (1 - mixf) * average_qin + mixf * average_qout
        return qmix_

    def broyden_mix(self):
        pass


def read_sk(para):
    '''
    generate the electrons, the onsite only includes s, p and d oribitals
    '''
    natom = para['natom']
    atomname = para['atomnameall']
    qatom = np.zeros(natom)
    num_electrons = 0
    onsite = np.zeros((len(atomname), 3))
    spe = np.zeros(len(atomname))
    uhubb = np.zeros((len(atomname), 3))
    occ_atom = np.zeros((len(atomname), 3))
    atomname_set = list(set(atomname))
    icount = 0
    for namei in atomname:
        for namej in atomname:
            ReadSK(para, namei, namej)
        onsite[icount, :] = para['Espd_Uspd' + namei + namei][0:3]
        spe[icount] = para['Espd_Uspd' + namei + namei][3]
        uhubb[icount, :] = para['Espd_Uspd' + namei + namei][4:7]
        occ_atom[icount, :] = para['Espd_Uspd' + namei + namei][7:10]
        icount += 1
    for i in range(natom):
        qatom[i] = VAL_ELEC[atomname[i]]
        num_electrons += qatom[i]
    para['atomname_set'] = atomname_set
    para['onsite'] = onsite
    para['spe'] = spe
    para['uhubb'] = uhubb
    para['occ_atom'] = occ_atom
    para['qatom'] = qatom
    para['nelectrons'] = num_electrons


def read_sk_interp(para):
    '''
    read from interpolation
    '''
    natom = para['natom']
    atomname = para['atomnameall']
    atomname_set = list(set(atomname))
    qatom = np.zeros(natom)
    num_electrons = 0
    for i in range(0, natom):
        qatom[i] = VAL_ELEC[atomname[i]]
        num_electrons += qatom[i]
    onsite = np.zeros((len(atomname), 3))
    spe = np.zeros(len(atomname))
    uhubb = np.zeros((len(atomname), 3))
    occ_atom = np.zeros((len(atomname), 3))
    icount = 0
    for namei in atomname:
        for namej in atomname:
            ReadSK(para, namei, namej)
        onsite[icount, :] = para['Espd_Uspd' + namei + namei][: 3]
        spe[icount] = para['Espd_Uspd' + namei + namei][3]
        uhubb[icount, :] = para['Espd_Uspd' + namei + namei][4: 7]
        occ_atom[icount, :] = para['Espd_Uspd' + namei + namei][7: 10]
        icount += 1
    para['atomname_set'] = atomname_set
    para['onsite'] = onsite
    para['spe'] = spe
    para['uhubb'] = uhubb
    para['occ_atom'] = occ_atom
    para['qatom'] = qatom
    para['nelectrons'] = num_electrons


def get_dipole(para, qzero, qatom):
    '''
    calculate dipole moment
    '''
    coor = para['coor']
    natom = para['natom']
    dipole = np.zeros((3), dtype=float)
    for iat in range(0, natom):
        if para['ty'] in ['ml']:
            dipole[:] += (qzero[iat] - qatom[iat]) * coor[iat, 1:]
        else:
            dipole[:] += (qzero[iat] - qatom[iat]) * np.array(coor[iat][1:])
    para['dipole'] = dipole


if __name__ == '__main__':
    '''
    call main function
    '''
    para = {}
    main(para)
