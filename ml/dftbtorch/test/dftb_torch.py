#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import numpy as np
import torch as t
from torch.autograd import Variable
import slakot
from electront import DFTB_elect
from readt import ReadInt, ReadSKt
GEN_PARA = {"inputfile_name": 'in.ground'}
VAL_ELEC = {"H": 1, "C": 4, "N": 5, "O": 6, "Ti": 4}
PUBPARA = {"LDIM": 9, "AUEV": 27.2113845, "BOHR": 0.529177210903, "tol": 1E-4}


def main(outpara):
    '''
    We have implemented pytorch into this code with different interface,
    we will_ use different ty values for different interface. For example,
    ty = 0: read dftb_in (.json, coor and calculation parameters) and skf
            files (.skf, for SK tables), you have to define the directory;
    ty = 1: read dftb_in (.json, coor and calculation parameters) and skf
            files (.skf, for SK tables) for test (path: ./test);
    ty = 5: input (coor, calculation parameters), SK tables from interpolation
            with different compression radius;
    ty = 6: for ML, input (coor, calculation parameters and skf data);
    ty = 7: for ML, input (calculation parameters) SK tables from spline
            interpolation, coor from dataset;
    '''
    parser_cmd_args(outpara)
    Initialization(outpara)
    Rundftbpy(outpara)
    WriteOut(outpara)


def parser_cmd_args(outpara):
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
    Path0 = os.getcwd()
    outpara['filename'] = args.filename
    outpara['direInput'] = os.path.join(Path0, args.directory)
    outpara['direSK'] = os.path.join(Path0, args.directorySK)
    return outpara


class Initialization():
    '''
    this class aims to read input coor, calculation parameters and
    SK tables;
    Then with SK transformation, construct 1D Hamiltonian and
    overlap matrix for next DFTB calculations
    '''
    def __init__(self, outpara):
        '''
        here several steps will_ be easy to have more interface in the future
        '''
        self.outpara = outpara
        # step 1: whether read input para from dftb_in file
        if self.outpara['ty'] == 0 or self.outpara['ty'] == 1:
            self.readinput_(outpara)
        # step 2: how to read geo info
        if self.outpara['ty'] == 5:
            self.readgeo_all_(outpara)
        else:
            self.readgeo_(outpara)
        # step 3: read SK table and operate SK data for next step
        if self.outpara['ty'] == 6:
            self.get_sktable(outpara)
            slakot.sk_tranml(outpara)
        elif self.outpara['ty'] == 0 or self.outpara['ty'] == 1:
            self.read_skdata(outpara)
            slakot.SlaKo(outpara).sk_tranold(outpara)

    def readinput_(self, outpara):
        if self.outpara['ty'] == 0:
            ReadInt(outpara).get_task(outpara)

    def readgeo_(self, outpara):
        ReadInt(outpara).get_coor(outpara)

    def readgeo_all_(self, outpara):
        ReadInt(outpara).get_coorall_(outpara)

    def read_skdata(self, outpara):
        slakot.SlaKo(outpara).read_skdata(outpara)

    def get_sktable(self, outpara):
        slakot.getSKTable(outpara)

    def formSKDataSpline(self, outpara):
        slakot.getSKSplPara(outpara)

    def GenSKMatrix(self, outpara):
        slakot.SlaKo(outpara).sk_tranold(outpara)


class Rundftbpy():
    '''
    According to the task of input parameters, this code will_ call_ different
    calculation tasks
    '''
    def __init__(self, outpara):
        self.outpara = outpara
        self.rundftbplus(outpara)

    def rundftbplus(self, outpara):
        if outpara['scf']:
            SCF(outpara)


class SCF():
    '''
    this is for self-consistent field method
    '''
    def __init__(self, outpara):
        self.outpara = outpara
        if not outpara['periodic'] and not outpara['scc']:
            self.scf_npe_nscc(outpara)
        elif not outpara['periodic'] and outpara['scc']:
            self.scf_npe(outpara)
        else:
            self.scf_pe(outpara)

    def scf_npe_nscc(self, outpara):
        natom = outpara['natom']
        get_qatom(outpara)
        oldqatom = Variable(t.empty(natom))
        qzero = Variable(t.empty(natom))
        qatom = Variable(t.empty(natom))
        qzero[:] = outpara['qatom']
        qatom[:] = outpara['qatom']
        # ----atomind is the number of atom, for C, lmax is 2, therefore --- #
        # ----we need 2**2 orbitals (s, px, py, pz), then define atomind2 ---#
        atomind = outpara['atomind']
        atomind2 = outpara['atomind2']
        eigvec = Variable(t.empty(int(atomind[natom]), int(atomind[natom])))
        eigval = Variable(t.empty(int(atomind[natom])))
        fockmat = Variable(t.empty(atomind2))
        oldovermat = Variable(t.empty(int(atomind[natom]),
                                      int(atomind[natom])))
        hammat = outpara['hammat']
        overmat = outpara['overmat']
        # ************************************************************ #
        # *********************start for NON-SCC********************** #
        # ************************************************************ #
        oldqatom[:] = qatom
        fockmat[:] = hammat
        k = 0
        for i in range(0, int(atomind[natom])):
            for j in range(0, i + 1):
                eigvec[j, i] = hammat[k]
                oldovermat[j, i] = overmat[k]
                # To make symmetric positive-definite, the foll_owing
                # two lines are added compare with linalg.eigh
                eigvec[i, j] = fockmat[k]
                oldovermat[i, j] = overmat[k]
                k += 1
        # ---------------get eigenvector and eigenvalue---------------- #
        ll_ = t.cholesky(oldovermat)
        mm1_ = t.mm(t.inverse(ll_), eigvec)
        ll_tinv = t.inverse(ll_.t())
        mm2_ = t.mm(mm1_, ll_tinv)
        eigval, eigvec_ = t.symeig(mm2_, eigenvectors=True, upper=True)
        eigvec = t.mm(t.inverse(ll_.t()), eigvec_)
        # eigval, eigvec = linalg.eigh(eigvec, oldovermat,
        # lower=False, overwrite_a=True, overwrite_b=True)
        print('nonscc eigen value: ', eigval*PUBPARA['AUEV'])
        # ************************************************************ #
        # *********************end of (NON) SCC*********************** #
        nelect = outpara['nelectrons'] / 2
        return eigval[int(nelect) - 1:int(nelect) + 1] * PUBPARA['AUEV']

    def scf_npe(self, outpara):
        natom = outpara['natom']
        get_qatom(outpara)
        oldqatom = Variable(t.empty(natom))
        qzero = Variable(t.empty(natom))
        qatom = Variable(t.empty(natom))
        qzero[:] = outpara['qatom']
        qatom[:] = outpara['qatom']
        # ----atomind is the number of atom, for C, lmax is 2, therefore --- #
        # ----we need 2**2 orbitals (s, px, py, pz), then define atomind2 ---#
        atomind = outpara['atomind']
        atomind2 = outpara['atomind2']
        eigvec = Variable(t.empty(int(atomind[natom]), int(atomind[natom])))
        eigval = Variable(t.empty(int(atomind[natom])))
        fockmat = Variable(t.empty(atomind2))
        occ = Variable(t.empty(int(atomind[natom])))
        oldovermat = Variable(t.empty(int(atomind[natom]),
                                      int(atomind[natom])))
        hammat = outpara['hammat']
        overmat = outpara['overmat']
        gmat = DFTB_elect().gmatrix(outpara)
        maxIter = outpara['maxIter']
        oldener = 0.0
        # ************************************************************ #
        # ***********************start SCC loop*********************** #
        # ************************************************************ #
        for niter in range(0, maxIter):
            oldqatom[:] = qatom[:]
            fockmat[:] = hammat[:]
            # --------------(gamma_ik+gamma_jk)*delta_qk-------------- #
            shift = DFTB_elect().shifthamgam(natom, qatom, qzero, gmat)
            for i_ in range(0, natom):
                for j_ in range(int(atomind[i_]), int(atomind[i_ + 1])):
                    occ[j_] = shift[i_]
            k = 0
            # -------------construct the whole Hamiltonian------------ #
            for i in range(0, int(atomind[natom])):
                for j in range(0, i+1):
                    fockmat[k] += 0.5 * overmat[k] * (occ[i] + occ[j])
                    k += 1
            k = 0
            for i in range(0, int(atomind[natom])):
                for j in range(0, i+1):
                    eigvec[j, i] = fockmat[k]
                    oldovermat[j, i] = overmat[k]
                    eigvec[i, j] = fockmat[k]
                    oldovermat[i, j] = overmat[k]
                    k += 1
            # -------------get eigenvector and eigenvalue-------------- #
            ll_ = t.cholesky(oldovermat)
            mm1_ = t.mm(t.inverse(ll_), eigvec)
            ll_tinv = t.inverse(ll_.t())
            mm2_ = t.mm(mm1_, ll_tinv)
            eigval, eigvec_ = t.symeig(mm2_, eigenvectors=True, upper=True)
            eigvec = t.mm(t.inverse(ll_.t()), eigvec_)
            # ---------calculate the occupation of electrons---------- #
            occ = DFTB_elect().fermi(outpara, eigval, occ)[0]
            energy = 0.0
            for i in range(0, int(atomind[natom])):
                if occ[i] > PUBPARA['tol']:
                    energy = energy + occ[i] * eigval[i]
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
            for i in range(0, nocc + 1):
                work.append(np.sqrt(occ[i]))
            for j in range(0, nocc+1):
                for i in range(0, int(atomind[natom])):
                    eigvec[i, j] = eigvec[i, j] * work[j]
            for j in range(nocc+1, int(atomind[natom])):
                eigvec[:, j] = 0.0
            oldovermat = t.mm(eigvec, t.t(eigvec))
            for i in range(0, int(atomind[natom])):
                for j in range(0, i+1):
                    m = int(i * (i + 1) / 2 + j)
                    denmat[m] = denmat[m] + oldovermat[j, i]
            # ---------------calculate mull_iken charges---------------- #
            DFTB_elect().mull_iken(outpara, overmat, denmat, qatom)
            # ---------------calculate electronic energy--------------- #
            ecoul = 0.0
            for i in range(0, natom):
                ecoul = ecoul + shift[i] * (qatom[i] + qzero[i])
            energy = energy - 0.5 * ecoul
            # -----------------if reached convergence------------------ #
            if niter == 0:
                print('niter     energy       dE')
            print(niter, float(energy), float(abs(oldener - energy)))
            if abs(oldener - energy) < PUBPARA['tol']:
                print('\n')
                print('No Occ       au               eV')
                for i_ in range(0, int(atomind[natom])):
                    print(i_, int(occ[i_]), eigval[i_],
                          eigval[i_] * PUBPARA['AUEV'])
                break
            if niter + 1 >= maxIter and abs(oldener - energy) > PUBPARA['tol']:
                print('Warning: SCF donot reach required convergence')
            oldqatom = Mixing(outpara).simple_mix(oldqatom, qatom)
            qatom[:] = oldqatom[:]
            oldener = energy
        # ************************************************************ #
        # *********************** end of SCC ************************* #
        if outpara['dipole'] and outpara['scc']:
            dipolemoment = get_dipole(outpara, qzero, qatom)
            outpara['dipolemoment'] = dipolemoment

    def scf_pe():
        pass


class Mixing():

    def __init__(self, outpara):
        self.outpara = outpara

    def simple_mix(self, oldqatom, qatom):
        mixf = self.outpara['mixFactor']
        natom = self.outpara['natom']
        qdiff = t.zeros(natom)
        qmix = t.zeros(natom)
        qdiff[:] = qatom[:]-oldqatom[:]
        qmix[:] = oldqatom[:]+mixf*qdiff[:]
        return qmix

    def broyden_mix(self):
        pass


class WriteOut(object):

    def __init__(self, outpara):
        self.outpara = outpara

    def w_electron(self):
        pass


def read_sk(outpara):
    '''
    generate the electrons, the onsite only includes s, p and d oribitals
    '''
    atomname = outpara['atomnameall_']
    onsite = np.zeros((len(atomname), 3))
    spe = np.zeros(len(atomname))
    uhubb = np.zeros((len(atomname), 3))
    occ_atom = np.zeros((len(atomname), 3))
    atomname_set = list(set(atomname))
    icount = 0
    for namei in atomname:
        for namej in atomname:
            ReadSKt(outpara, namei, namej)
        onsite[icount, :] = outpara['Espd_Uspd'+namei+namei][0:3]
        spe[icount] = outpara['Espd_Uspd'+namei+namei][3]
        uhubb[icount, :] = outpara['Espd_Uspd'+namei+namei][4:7]
        occ_atom[icount, :] = outpara['Espd_Uspd'+namei+namei][7:10]
        icount += 1
    outpara['atomname_set'] = atomname_set
    outpara['onsite'] = onsite
    outpara['spe'] = spe
    outpara['uhubb'] = uhubb
    outpara['occ_atom'] = occ_atom
    return outpara


def get_qatom(outpara):
    '''
    '''
    natom = outpara['natom']
    atomname = outpara['atomnameall']
    num_electrons = 0
    qatom = t.empty(natom, requires_grad=True)
    for i in range(0, natom):
        qatom[i] = VAL_ELEC[atomname[i]]
        num_electrons += qatom[i]
    outpara['qatom'] = qatom
    outpara['nelectrons'] = num_electrons
    return outpara


def get_dipole(outpara, qzero, qatom):
    coor = outpara['coor']
    natom = outpara['natom']
    dipolemoment = t.zeros(3)
    for i_ in range(0, natom):
        if outpara['ty'] == 5:
            dipolemoment[:] += (qzero[i_]-qatom[i_])*coor[i_, 1:]
        else:
            dipolemoment[:] += (qzero[i_]-qatom[i_])*t.from_numpy(
                np.array(coor[i_][1:]))
    return dipolemoment


if __name__ == '__main__':
    outpara = {}
    outpara['ty'] = 0
    main(outpara)
