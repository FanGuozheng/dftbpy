#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import argparse
import os
import slakot
from electront import DFTB_elect
from readt import ReadInt, ReadSKt
import torch as t
from torch.autograd import Variable
GEN_PARA = {"inputfile_name": 'in.ground'}
VAL_ELEC = {"H": 1, "C": 4, "N": 5, "O": 6, "Ti": 4}
PUBPARA = {"LDIM": 9, "AUEV": 27.2113845, "BOHR": 0.529177210903, "tol": 1E-4}


def main(generalpara):
    '''This is the main function, with different interface, we will use
    different ty values. For example,
    For use the main function in this file and read input from test, ty is 1;
    With input coor and .skf file, ty s 2;
    For hamiltonian from interpolation, ty is 5;
    For generating hamiltonian using dftbtorch code for ML, ty is 6;
    '''
    parser_cmd_args(generalpara)
    Initialization(generalpara)
    if generalpara['scf']:
        SCF(generalpara)
    WriteOut(generalpara)


class SCF(object):
    '''this is dftb python code module'''
    def __init__(self, generalpara):
        self.generalpara = generalpara
        if not generalpara['periodic']:
            self.scf_nonpe(generalpara)
        else:
            self.scf_pe(generalpara)

    def scf_nonpe(self, generalpara):
        # if generalpara['ty'] == 6:
        #    slkoML(generalpara)
        natom = generalpara['natom']
        getqAtom0(generalpara)
        oldqatom = Variable(t.empty(natom))
        qzero = Variable(t.empty(natom))
        qatom = Variable(t.empty(natom))
        qzero[:] = generalpara['qatom']
        qatom[:] = generalpara['qatom']
        # ----atomind is the number of atom, for C, lmax is 2, therefore --- #
        # ----we need 2**2 orbitals (s, px, py, pz), then define atomind2 ---#
        atomind = generalpara['atomind']
        atomind2 = generalpara['atomind2']
        eigvec = Variable(t.empty(int(atomind[natom]), int(atomind[natom])))
        eigval = Variable(t.empty(int(atomind[natom])))
        fockmat = Variable(t.empty(atomind2))
        occ = Variable(t.empty(int(atomind[natom])))
        oldovermat = Variable(t.empty(int(atomind[natom]),
                                      int(atomind[natom])))
        hammat = generalpara['hammat']
        overmat = generalpara['overmat']
        # -------------construct gamma in second-order term----------------- #
        if not generalpara['scc']:
            # ************************************************************ #
            # *********************start for NON-SCC********************** #
            # ************************************************************ #
            # return torch.tensor([1., 2.], requires_grad=True)
            oldqatom[:] = qatom
            fockmat[:] = hammat
            k = 0
            for i in range(0, int(atomind[natom])):
                for j in range(0, i+1):
                    eigvec[j, i] = hammat[k]
                    oldovermat[j, i] = overmat[k]
                    # To make symmetric positive-definite, the following
                    # two lines are added compare with linalg.eigh
                    eigvec[i, j] = fockmat[k]
                    oldovermat[i, j] = overmat[k]
                    k += 1
            # ---------------get eigenvector and eigenvalue---------------- #
            ll = t.cholesky(oldovermat)
            C1 = t.mm(t.inverse(ll), eigvec)
            lltinv = t.inverse(ll.t())
            C2 = t.mm(C1, lltinv)
            eigval, YY = t.symeig(C2, eigenvectors=True, upper=True)
            eigvec = t.mm(t.inverse(ll.t()), YY)
            # eigval, eigvec = linalg.eigh(eigvec, oldovermat,
            # lower=False, overwrite_a=True, overwrite_b=True)
            print('nonscc eigen value: ', eigval*PUBPARA['AUEV'])
            # ************************************************************ #
            # *********************end of (NON) SCC*********************** #
        else:
            gmat = DFTB_elect().gmatrix(generalpara)
            maxIter = generalpara['maxIter']
            oldener = 0.0
            # ************************************************************ #
            # ***********************start SCC loop*********************** #
            # ************************************************************ #
            for niter in range(0, maxIter):
                oldqatom[:] = qatom[:]
                fockmat[:] = hammat[:]
                # --------------(gamma_ik+gamma_jk)*delta_qk-------------- #
                shift = DFTB_elect().shifthamgam(natom, qatom, qzero, gmat)
                for i in range(0, natom):
                    for ii in range(int(atomind[i]), int(atomind[i+1])):
                        occ[ii] = shift[i]
                k = 0
                # -------------construct the whole Hamiltonian------------ #
                for i in range(0, int(atomind[natom])):
                    for j in range(0, i+1):
                        fockmat[k] = fockmat[k]+0.5*overmat[k]*(occ[i]+occ[j])
                        k += 1
                k = 0
                for i in range(0, int(atomind[natom])):
                    for j in range(0, i+1):
                        eigvec[j, i] = fockmat[k]
                        oldovermat[j, i] = overmat[k]
                        # To make symmetric positive-definite, the following
                        # two lines are added compare with linalg.eigh
                        eigvec[i, j] = fockmat[k]
                        oldovermat[i, j] = overmat[k]
                        k += 1
                # -------------get eigenvector and eigenvalue-------------- #
                ll = t.cholesky(oldovermat)
                C1 = t.mm(t.inverse(ll), eigvec)
                lltinv = t.inverse(ll.t())
                C2 = t.mm(C1, lltinv)
                eigval, YY = t.symeig(C2, eigenvectors=True, upper=True)
                eigvec = t.mm(t.inverse(ll.t()), YY)
                '''eigval, eigvec = linalg.eigh(eigvec, oldovermat,
                lower=False, overwrite_a=True, overwrite_b=True)'''
                # ---------calculate the occupation of electrons---------- #
                occ = DFTB_elect().fermi(generalpara, eigval, occ)[0]
                energy = 0.0
                for i in range(0, int(atomind[natom])):
                    if occ[i] > PUBPARA['tol']:
                        energy = energy + occ[i]*eigval[i]
                    else:
                        break
                for nocc in range(0, int(atomind[natom])):
                    if occ[nocc] > PUBPARA['tol']:
                        nocc += 1
                    else:
                        break
                # ---------------construct density matrix----------------- #
                denmat = np.zeros((atomind2))
                work = []
                for i in range(0, nocc+1):
                    work.append(np.sqrt(occ[i]))
                for j in range(0, nocc+1):
                    for i in range(0, int(atomind[natom])):
                        eigvec[i, j] = eigvec[i, j] * work[j]
                for j in range(nocc+1, int(atomind[natom])):
                    eigvec[:, j] = 0.0
                oldovermat = t.mm(eigvec, t.t(eigvec))
                # oldovermat = linalg.blas.dgemm(alpha=1.0, a=eigvec, b=eigvec,
                #                               beta=0.0, trans_b=1)
                for i in range(0, int(atomind[natom])):
                    for j in range(0, i+1):
                        m = int(i*(i+1)/2+j)
                        denmat[m] = denmat[m]+oldovermat[j, i]
                # ---------------calculate mulliken charges---------------- #
                DFTB_elect().mulliken(generalpara, overmat, denmat, qatom)
                # ---------------calculate electronic energy--------------- #
                ecoul = 0.0
                for i in range(0, natom):
                    ecoul = ecoul + shift[i] * (qatom[i] + qzero[i])
                energy = energy - 0.5 * ecoul
                # -----------------if reached convergence------------------ #
                if niter == 0:
                    print('niter     energy       dE')
                print(niter, float(energy), float(abs(oldener-energy)))
                if abs(oldener-energy) < PUBPARA['tol']:
                    print('\n')
                    print('No Occ       au               eV')
                    for ii in range(0, int(atomind[natom])):
                        print(ii, int(occ[ii]), eigval[ii],
                              eigval[ii]*PUBPARA['AUEV'])
                    break
                if niter+1 >= maxIter and abs(oldener-energy) > PUBPARA['tol']:
                    print('Warning: SCF donot reach required convergence')
                oldqatom = Mixing(generalpara).simplemix(oldqatom, qatom)
                qatom[:] = oldqatom[:]
                oldener = energy
            # ************************************************************ #
            # *********************** end of SCC ************************* #
        if generalpara['dipole'] and generalpara['scc']:
            dipolemoment = getDipole(generalpara, qzero, qatom)
            print('dipolemoment', dipolemoment)
        elif generalpara['dipole'] and not generalpara['scc']:
            print('For dipolemoment calculations, plesese set "scc": True')
        nelect = generalpara['nelectrons']/2
        return eigval[int(nelect)-1:int(nelect)+1]*PUBPARA['AUEV']

    def scf_pe():
        pass


def parser_cmd_args(generalpara):
    '''raed some input information, including path, name of file, etc.
       default path of input: current path
       default path od .skf: ./slko
       default inout name: dftb_in'''
    _DESCRIPTION = 'Test script demonstrating argparse'
    parser = argparse.ArgumentParser(description=_DESCRIPTION)
    msg = 'Directory (default: .)'
    parser.add_argument('-d', '--directory', default='.', help=msg)
    msg = 'Directory_SK (default: .)'
    parser.add_argument('-sk', '--directorySK', default='slko', help=msg)
    msg = 'input filename'
    parser.add_argument('-f', '--filename', type=str, default='dftb_in',
                        metavar='NAME', help=msg)
    args = parser.parse_args()
    Path0 = os.getcwd()
    generalpara['filename'] = args.filename
    generalpara['direInput'] = os.path.join(Path0, args.directory)
    generalpara['direSK'] = os.path.join(Path0, args.directorySK)
    return generalpara


class Initialization(object):

    def __init__(self, generalpara):
        self.generalpara = generalpara
        self.ReadInput(generalpara)
        self.ReadGeo(generalpara)
        self.ReadSK(generalpara)
        self.getSKTable(generalpara)
        self.GenSKMatrix(generalpara)

    def ReadInput(self, generalpara):
        if self.generalpara['ty'] == 0:
            ReadInt().get_task(generalpara)

    def ReadGeo(self, generalpara):
        if self.generalpara['ty'] == 0:
            ReadInt().get_coor(generalpara)
        elif self.generalpara['ty'] == 5:
            ReadInt().get_coor5(generalpara)

    def ReadSK(self, generalpara):
        read_sk(generalpara)

    def getSKTable(self, generalpara):
        if generalpara['ty'] == 6:
            slakot.getSKTable(generalpara)

    def GenSKMatrix(self, generalpara):
        if generalpara['ty'] == 6:
            slakot.sk_tranml(generalpara)
        else:
            slakot.sk_tranold(generalpara)
        return generalpara

    def SlKotrans(self, generalpara):
        pass


class Mixing():

    def __init__(self, generalpara):
        self.generalpara = generalpara

    def simplemix(self, oldqatom, qatom):
        mixf = self.generalpara['mixFactor']
        natom = self.generalpara['natom']
        qdiff = torch.zeros(natom)
        qmix = torch.zeros(natom)
        qdiff[:] = qatom[:]-oldqatom[:]
        qmix[:] = oldqatom[:]+mixf*qdiff[:]
        return qmix

    def broyden():
        pass


class WriteOut(object):

    def __init__(self, generalpara):
        self.generalpara = generalpara

    def writeElectron(self):
        pass


def slkoML(generalpara):
    pass


def read_sk(generalpara):
    '''generate the electrons, the onsite only includes s, p and d oribitals'''
    atomname = generalpara['atomnameall']
    onsite = np.zeros((len(atomname), 3))
    spe = np.zeros(len(atomname))
    uhubb = np.zeros((len(atomname), 3))
    occ_atom = np.zeros((len(atomname), 3))
    atomname_set = list(set(atomname))
    icount = 0
    for namei in atomname:
        for namej in atomname:
            ReadSKt(generalpara, namei, namej)
        onsite[icount, :] = generalpara['Espd_Uspd'+namei+namei][0:3]
        spe[icount] = generalpara['Espd_Uspd'+namei+namei][3]
        uhubb[icount, :] = generalpara['Espd_Uspd'+namei+namei][4:7]
        occ_atom[icount, :] = generalpara['Espd_Uspd'+namei+namei][7:10]
        icount += 1
    generalpara['atomname_set'] = atomname_set
    generalpara['onsite'] = onsite
    generalpara['spe'] = spe
    generalpara['uhubb'] = uhubb
    generalpara['occ_atom'] = occ_atom
    return generalpara


def getqAtom0(generalpara):
    natom = generalpara['natom']
    atomname = generalpara['atomnameall']
    num_electrons = 0
    qatom = t.empty(natom, requires_grad=True)
    for i in range(0, natom):
        qatom[i] = VAL_ELEC[atomname[i]]
        num_electrons += qatom[i]
    generalpara['qatom'] = qatom
    generalpara['nelectrons'] = num_electrons
    return generalpara


def getDipole(generalpara, qzero, qatom):
    coor = generalpara['coor']
    natom = generalpara['natom']
    dipolemoment = t.zeros(3)
    for ii in range(0, natom):
        if generalpara['ty'] == 5:
            dipolemoment[:] += (qzero[ii]-qatom[ii])*coor[ii, 1:]
        else:
            dipolemoment[:] += (qzero[ii]-qatom[ii])*t.from_numpy(
                    np.array(coor[ii][1:]))
    return dipolemoment


if __name__ == '__main__':
    generalpara = {}
    generalpara['ty'] = 0
    main(generalpara)
