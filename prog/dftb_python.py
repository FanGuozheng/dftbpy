#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import argparse
import scipy.linalg as linalg
import slakotran
from electron import DFTB_elect
from read import ReadInput, ReadSK
GEN_PARA = {"inputfile_name": 'in.ground'}
VAL_ELEC = {"H": 1, "C": 4, "N": 5, "O": 6, "Ti": 4}
PUBPARA = {"LDIM": 9, "AUEV": 27.2113845, "BOHR": 0.529177210903, "tol": 1E-4}


def main(outpara):
    '''This is the main function, with different interface, we will use
    different ty values. For example, for machine learning, ty is 5'''
    generalpara = {}
    generalpara['ty'] = outpara['ty']
    if outpara['ty'] == 0:
        parser_cmd_args(generalpara)
        generalpara['dire'] = '../test'
        ReadInput(generalpara)
        if not generalpara['periodic']:
            dipolem, eigval = DFTBpy().scf_nonpe(generalpara, outpara)
        else:
            DFTBpy().scf_pe(generalpara)
    elif outpara['ty'] == 1:
        generalpara['coor'] = outpara['coor']
        generalpara['symbols'] = outpara['symbols']
        parser_cmd_args(generalpara)
        generalpara['dire'] = '../test'
        ReadInput(generalpara)
        if not generalpara['periodic']:
            dipolem, eigval = DFTBpy().scf_nonpe(generalpara, outpara)
        else:
            DFTBpy().scf_pe(generalpara)
    elif outpara['ty'] == 5:
        generalpara['coor'] = outpara['coor']
        generalpara['hs'] = outpara['h_s_all']
        generalpara['symbols'] = outpara['symbols']
        parser_cmd_args(generalpara)
        ReadInput(generalpara)
        if not generalpara['periodic']:
            dipolem, eigval = DFTBpy().scf_nonpe(generalpara, outpara)
        else:
            DFTBpy().scf_pe(generalpara)
    return dipolem, eigval


class DFTBpy(object):
    """this is dftb python code module"""
    def __init__(self):
        pass

    def scf_nonpe(self, generalpara, outpara):
        '''self consistent field calculation'''
        if generalpara['ty'] == 5:
            read_sk5(generalpara, outpara)
        elif generalpara['ty'] == 0:
            read_sk(generalpara, outpara)
        elif generalpara['ty'] == 1:
            read_sk(generalpara, outpara)
        natom = generalpara['natom']
        oldqatom = np.zeros(natom)
        qzero = np.zeros(natom)
        qatom = np.zeros(natom)
        qzero[:] = generalpara['qatom']
        qatom[:] = generalpara['qatom']
        atomind = generalpara['atomind']
        # ----atomind is the number of atom, for C, lmax is 2, therefore --- #
        # ----we need 2**2 orbitals (s, px, py, pz), then define atomind2 ---#
        atomind2 = int(atomind[natom]*(atomind[natom]+1)/2)
        eigvec = np.zeros((int(atomind[natom]), int(atomind[natom])))
        eigval = np.zeros((int(atomind[natom])))
        fockmat = np.zeros(atomind2)
        occ = np.zeros(int(atomind[natom]))
        oldovermat = np.zeros((int(atomind[natom]), int(atomind[natom])))
        # -------------------Slater-Koster transfer------------------------- #
        hammat, overmat = slakotran.sk_tran(generalpara)
        print('hammat', hammat)
        # -------------construct gamma in second-order term----------------- #
        gmat = DFTB_elect().gmatrix(generalpara)
        # **************************start SCF loop************************** #
        # ****************************************************************** #
        oldener = 0.0
        maxIter = generalpara['maxIter']
        for niter in range(0, maxIter):
            print('\n \n niter:', niter)
            oldqatom[:] = qatom[:]
            fockmat[:] = hammat[:]
            # ----------------(gamma_ik+gamma_jk)*delta_qk------------------ #
            shift = DFTB_elect().shifthamgam(natom, qatom, qzero, gmat)
            for i in range(0, natom):
                for ii in range(int(atomind[i]), int(atomind[i+1])):
                    occ[ii] = shift[i]
            k = 0
            # ----------------construct the whole Hamiltonian--------------- #
            for i in range(0, int(atomind[natom])):
                for j in range(0, i+1):
                    fockmat[k] = fockmat[k]+0.5*overmat[k]*(occ[i]+occ[j])
                    k += 1
            k = 0
            for i in range(0, int(atomind[natom])):
                for j in range(0, i+1):
                    eigvec[j, i] = fockmat[k]
                    oldovermat[j, i] = overmat[k]
                    k += 1
            # --------------get eigenvector and eigenvalue------------------ #
            eigval, eigvec = linalg.eigh(eigvec, oldovermat, lower=False,
                                         overwrite_a=True, overwrite_b=True)
            # -----------calculate the occupation of electrons-------------- #
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
            # ------------------construct density matrix-------------------- #
            denmat = np.zeros((atomind2))
            work = []
            for i in range(0, nocc+1):
                work.append(np.sqrt(occ[i]))
            for j in range(0, nocc+1):
                for i in range(0, int(atomind[natom])):
                    eigvec[i, j] = eigvec[i, j] * work[j]
            for j in range(nocc+1, int(atomind[natom])):
                eigvec[:, j] = 0.0
            oldovermat = linalg.blas.dgemm(alpha=1.0, a=eigvec, b=eigvec,
                                           beta=0.0, trans_b=1)
            for i in range(0, int(atomind[natom])):
                for j in range(0, i+1):
                    m = int(i*(i+1)/2+j)
                    denmat[m] = denmat[m]+oldovermat[j, i]
            # -----------------calculate mulliken charges------------------- #
            DFTB_elect().mulliken(generalpara, overmat, denmat, qatom)
            # -----------------calculate electronic energy------------------ #
            ecoul = 0.0
            for i in range(0, natom):
                ecoul = ecoul + shift[i] * (qatom[i] + qzero[i])
            energy = energy - 0.5 * ecoul
            print('energy=energy-0.5ecoul', energy, '\n', 'ecoul', ecoul)
            # -------------------if reached convergence--------------------- #
            if abs(oldener-energy) < PUBPARA['tol']:
                print('\n')
                print('No Occ       au               eV')
                for ii in range(0, int(atomind[natom])):
                    print(ii, int(occ[ii]), eigval[ii],
                          eigval[ii]*PUBPARA['AUEV'])
                break
            if niter+1 >= maxIter and abs(oldener-energy) > PUBPARA['tol']:
                print('Warning: SCF donot reach required convergence')
            oldqatom = simplemix(generalpara, oldqatom, qatom)
            print('charge after mixing:', oldqatom)
            qatom[:] = oldqatom[:]
            oldener = energy
            '''broyden1 = np.zeros((natom, 2))
            broyden2 = np.zeros((natom, 2, generalpara['maxIter']))
            print("oldqatom", oldqatom, "qatom", qatom)
            if niter == 0:
                oldqatom = self.broyden0(niter, broyden1, broyden2,
                                         oldqatom, qatom, generalpara)[0]
                f0 = self.broyden0(niter, broyden1, broyden2,
                                   oldqatom, qatom, generalpara)[1]
            else:
                f0 = self.broyden1(niter, broyden1, broyden2,
                                   oldqatom, qatom, f0, generalpara)[1]
            print("oldqatom", oldqatom, "qatom", qatom)'''
        # ***************************end of SCF***************************** #
        if generalpara['dipole']:
            dipolemoment = getDipole(generalpara, qzero, qatom)
            print('dipolemoment', dipolemoment)
        print('charge of atom:', qatom)
        return dipolemoment, eigval[:]*PUBPARA['AUEV']

    def broyden():
        pass

    def scf_pe():
        pass


def parser_cmd_args(generalpara):
    _DESCRIPTION = 'Test script demonstrating argparse'
    parser = argparse.ArgumentParser(description=_DESCRIPTION)
    msg = 'Directory (default: .)'
    parser.add_argument('-d', '--directory', default='.', help=msg)
    msg = 'Directory_SK (default: .)'
    parser.add_argument('-s', '--directorySK', default='../test', help=msg)
    msg = 'input filename'
    parser.add_argument('-f', '--filename', type=str, default='dftb_in',
                        metavar='NAME', help=msg)
    args = parser.parse_args()
    generalpara['filename'] = args.filename
    generalpara['dire'] = args.directory
    generalpara['direSK'] = args.directorySK
    return generalpara


def read_sk(generalpara, outpara):
    '''generate the electrons, the onsite only includes s, p and d oribitals'''
    natom = generalpara['natom']
    atomname = generalpara['atomnameall']
    qatom = np.zeros(natom)
    num_electrons = 0
    '''for i in range(0, natom):
        qatom[i] = VAL_ELEC[atomname[i]]
        num_electrons += qatom[i]
        # read information from Slater-Koster file
        atomname_set = list(set(atomname))
        onsite = np.zeros((len(atomname_set), 3))
        spe = np.zeros(len(atomname_set))
        uhubb = np.zeros((len(atomname_set), 3))
        occ_atom = np.zeros((len(atomname_set), 3))
        icount = 0
        for namei in atomname_set:
            for namej in atomname_set:
                ReadSK(generalpara, outpara, namei, namej)
            onsite[icount, :] = generalpara['Espd_Uspd'+namei+namei][0:3]
            spe[icount] = generalpara['Espd_Uspd'+namei+namei][3]
            uhubb[icount, :] = generalpara['Espd_Uspd'+namei+namei][4:7]
            occ_atom[icount, :] = generalpara['Espd_Uspd'+namei+namei][7:10]
            icount += 1'''
    onsite = np.zeros((len(atomname), 3))
    spe = np.zeros(len(atomname))
    uhubb = np.zeros((len(atomname), 3))
    occ_atom = np.zeros((len(atomname), 3))
    atomname_set = list(set(atomname))
    icount = 0
    for namei in atomname:
        for namej in atomname:
            ReadSK(generalpara, outpara, namei, namej)
        onsite[icount, :] = generalpara['Espd_Uspd'+namei+namei][0:3]
        spe[icount] = generalpara['Espd_Uspd'+namei+namei][3]
        uhubb[icount, :] = generalpara['Espd_Uspd'+namei+namei][4:7]
        occ_atom[icount, :] = generalpara['Espd_Uspd'+namei+namei][7:10]
        icount += 1
    for i in range(0, natom):
        qatom[i] = VAL_ELEC[atomname[i]]
        num_electrons += qatom[i]
        # read information from Slater-Koster file
    generalpara['atomname_set'] = atomname_set
    generalpara['onsite'] = onsite
    generalpara['spe'] = spe
    generalpara['uhubb'] = uhubb
    generalpara['occ_atom'] = occ_atom
    generalpara['qatom'] = qatom
    generalpara['nelectrons'] = num_electrons
    return generalpara


def read_sk5(generalpara, outpara):
    '''read from interpolation'''
    natom = generalpara['natom']
    atomname = generalpara['atomnameall']
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
            ReadSK(generalpara, outpara, namei, namej)
        onsite[icount, :] = outpara['Espd_Uspd'+namei+namei][0:3]
        spe[icount] = outpara['Espd_Uspd'+namei+namei][3]
        uhubb[icount, :] = outpara['Espd_Uspd'+namei+namei][4:7]
        occ_atom[icount, :] = outpara['Espd_Uspd'+namei+namei][7:10]
        icount += 1
    generalpara['atomname_set'] = atomname_set
    generalpara['onsite'] = onsite
    generalpara['spe'] = spe
    generalpara['uhubb'] = uhubb
    generalpara['occ_atom'] = occ_atom
    generalpara['qatom'] = qatom
    generalpara['nelectrons'] = num_electrons
    return generalpara


def simplemix(generalpara, oldqatom, qatom):
    mixf = generalpara['mixFactor']
    natom = generalpara['natom']
    qdiff = np.zeros(natom)
    qmix = np.zeros(natom)
    qdiff[:] = qatom[:]-oldqatom[:]
    qmix[:] = oldqatom[:]+mixf*qdiff[:]
    return qmix


def getDipole(generalpara, qzero, qatom):
    coor = generalpara['coor']
    natom = generalpara['natom']
    dipolemoment = np.zeros(3)
    for ii in range(0, natom):
        if generalpara['ty'] == 5:
            print(coor[ii, 1:])
            dipolemoment[:] += (qzero[ii]-qatom[ii])*coor[ii, 1:]
        else:
            dipolemoment[:] += (qzero[ii]-qatom[ii])*np.array(coor[ii][1:])
    return dipolemoment


if __name__ == '__main__':
    outpara = {}
    outpara['ty'] = 0
    main(outpara)
