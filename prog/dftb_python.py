#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import linecache
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

    def scf_pe(self, generalpara):
        norbs = int(generalpara["norbs"])
        natom = generalpara["natom"]
        atomind = generalpara["atomind"]
        atomnamlist = list(set(generalpara["atomnameall"]))
        generalpara["skself"] = self.get_skself(generalpara)
        if len(atomnamlist) == 1:
            skf_name = atomnamlist+"-"+atomnamlist+".skf"
            generalpara["datalist"+atomnamlist[0]] = self.get_hs_skf(skf_name)
            generalpara["linespline"+atomnamlist[0]] = self.line_spline(skf_name)
            generalpara["griddist"+atomnamlist[0]] = self.get_grid(skf_name)[0]
            generalpara["ngrid"+atomnamlist[0]] = self.get_grid(skf_name)[1]
            # generalpara["skself"+atomnamlist[0]] = self.get_grid(skf_name)[2]
        else:
            skfnamelist = []
            skfnamelist.append(atomnamlist[0]+"-"+atomnamlist[0]+".skf")
            skfnamelist.append(atomnamlist[0]+"-"+atomnamlist[1]+".skf")
            skfnamelist.append(atomnamlist[1]+"-"+atomnamlist[0]+".skf")
            skfnamelist.append(atomnamlist[1]+"-"+atomnamlist[1]+".skf")
            for skfnami in skfnamelist:
                print("datalist"+skfnami)
                generalpara["datalist"+skfnami] = self.get_hs_skf(skfnami)
                generalpara["linespline"+skfnami] = self.line_spline(skfnami)
                generalpara["griddist"+skfnami] = self.get_grid(skfnami)[0]
                generalpara["ngrid"+skfnami] = self.get_grid(skfnami)[1]
                # generalpara["skself"+skfnami] = self.get_grid(skfnami)[2]
            generalpara["ngridmax"] = max(generalpara["ngrid"+skfnamelist[0]],
                                          generalpara["ngrid"+skfnamelist[1]],
                                          generalpara["ngrid"+skfnamelist[2]],
                                          generalpara["ngrid"+skfnamelist[3]])
            generalpara["gridall"] = [generalpara["griddist"+skfnamelist[0]],
                                      generalpara["griddist"+skfnamelist[1]],
                                      generalpara["griddist"+skfnamelist[2]],
                                      generalpara["griddist"+skfnamelist[3]]]
        print('generalpara["gridall"]', generalpara["gridall"])
        print('generalpara["ngridmax"]', generalpara["ngridmax"])
        generalpara = self.genmpgrids(generalpara)
        nkpoint = np.shape(generalpara["k_mesh"])[1]
        eigval_k = np.zeros((norbs, nkpoint))
        eigvec_k = np.zeros((norbs, norbs, nkpoint), dtype=complex)
        generalpara = self.get_electron(generalpara)
        gmat = self.gmatrixpe(generalpara)
        # print('gmat', gmat)
        generalpara = self.k_weight2int(generalpara)
        # print('generalpara["ik_weight"]', generalpara["ik_weight"])
        k_weight = generalpara["k_kweight"]
        oldqatom = np.empty(natom)
        nelect = np.empty(natom)
        qatom = generalpara["qatom"]
        qzero = generalpara["qatom"]
        # shift = self.shifthamgam(generalpara)
        eigvec = np.zeros((int(atomind[natom]), int(atomind[natom])))
        overmat = np.zeros((int(atomind[natom]), int(atomind[natom])))
        occ = np.zeros((int(atomind[natom])))
        shiftorb = np.zeros((int(atomind[natom])))
        nelect[:] = qzero[:]
        occ_k = np.zeros((norbs, nkpoint))
        oldenergy = 0.0
        for niter in range(0, generalpara["maxi"]):
            print("\n \n niter", niter, "\n")
            if generalpara["scf"] == "T":
                oldqatom[0: natom] = qatom[0: natom]
            generalpara["qatomi"] = qatom
            # shift = np.zeros(natom)
            shift = self.shifthamgam(natom, qatom, qzero, gmat)
            # print('shift', shift)
            for i in range(0, natom):
                for ii in range(int(atomind[i]), int(atomind[i+1])):
                    shiftorb[ii] = shift[i]
            for nk in range(0, nkpoint):
                eigvec, overmat = self.formhspe(nk, generalpara, shift,
                                                eigvec, overmat)
                # print('    nk', nk)
                eigval, eigvec = linalg.eigh(eigvec, overmat, lower=False)
                # print(np.shape(eigval_k), np.shape(eigval))
                # print('    eigvec', eigvec, 'overmat', overmat)
                # print('    eigval', eigval, norbs)
                # print(np.shape(eigval_k))
                eigval_k[:, nk] = eigval[:]
                eigvec_k[:, :, nk] = eigvec[:, :]
                # self.hermitianize(eigvec, norbs)
            # fockmat[0: atomind2] = hammat[0: atomind2]
            # print("fockmat[i] = hammat[i]", fockmat)
            # print("sys%shift", shift)
                    # print("occ[ii]", occ[ii])
            """k = 0
            for i in range(0, int(atomind[natom])):
                # print(int(atomind[natom]))
                for j in range(0, i+1):
                    fockmat[k] = fockmat[k] + 0.5 * overmat[k] * \
                                 (occ[i] + occ[j])
                    # here, using shifthamgam to reduce N*N*N dimension
                    k += 1
            # print("fockmat\n",fockmat,"occ\n",occ,"overmat\n",overmat)
            k = 0
            for i in range(0, int(atomind[natom])):
                for j in range(0, i+1):
                    eigvec[j, i] = fockmat[k]
                    oldovermat[j, i] = overmat[k]
                    # print("eigvec", eigvec[j,i], "k", k, "i", i, "j", j)
                    # print("oldovermat", oldOverMat[j, i])
                    k += 1
            eigval, eigvec = linalg.eigh(eigvec, oldovermat, lower=False,
                                         overwrite_a=True, overwrite_b=True)
            print("eigvec after eigh\n", eigvec, '\n eigveal', eigval)
            # print("oldovermat after eigh", oldovermat)"""
            occ, efermi = self.fermipe(generalpara, nkpoint, eigval_k, occ_k)
            # occ = self.fermi(int(atomind[natom]), eigval, occ)[0]
            # print("occ after fermi", occ)
            # sum of occupied eigenvalues
            energy = 0.0
            for nk in range(0, nkpoint):
                eigvec, overmat = self.formhspe(nk, generalpara, shift,
                                                eigvec, overmat)
                for i in range(0, norbs):
                    if occ[i, nk] < 8-16:
                        pass
                    else:
                        energy += occ[i, nk]*eigval_k[i, nk]*k_weight[nk]
                        print('energy', energy)
                for nocc in range(0, norbs):
                    if occ[nocc, nk] > 8E-16:
                        # print(occ[i],"n",n,int(atomind[natom]))
                        nocc += 1
                        # if only occ[0]!=0, n should be 0+1
                    else:
                        break
                denmat = np.zeros((norbs, norbs))
                work = []
                eigvec[:, :] = eigvec_k[:, :, nk]
                for i in range(0, nocc):  # n = no. of occupied orbitals
                    work.append(np.sqrt(occ[i, nk]))
                for j in range(0, nocc):  # n = no. of occupied orbitals
                    for i in range(0, norbs):
                        # print("eigvec[i,j]", eigVec[i,j], "i,j", i, j)
                        eigvec[i, j] = eigvec[i, j] * work[j]
                # print('nocc, norbs', nocc, norbs)
                for j in range(nocc, norbs):
                    eigvec[:, j] = 0.0
                    # oldovermat = np.dot(eigvec, eigvec.T)
                    oldovermat = linalg.blas.dgemm(alpha=1.0, a=eigvec,
                                                   b=eigvec, beta=0.0,
                                                   trans_b=1)
                # print("oldovermat after dgemm\n", oldovermat)
                """for i in range(0, norbs):
                    for j in range(0, i+1):
                        m = int(i*(i+1)/2+j)
                        # print('oldovermat[j, i]', oldovermat)
                        denmat[m] = denmat[m]+oldovermat[j, i]
                        # print("denmat[m]",denmat[m],i,j,"m",m)
                # print("before mulliken, oldqatom", oldqatom, qatom)"""
                # calculate mulliken charges
                for ii in range(0, natom):
                    qatom[ii] = 0.0
                    for i in range(int(atomind[ii]), int(atomind[ii+1])):
                        for j in range(0, norbs):
                            qatom[ii] += denmat[i, j]*overmat[j, i]
                print("qatom after mulliken", qatom)
                # print("after mulliken, oldqatom", oldqatom)
                # calculate electronic energy
            ecoul = 0.0
            for i in range(0, natom):
                ecoul = ecoul + shift[i] * (qatom[i] + qzero[i])
            energy = energy - 0.5 * ecoul
            print("energy=energy-0.5ecoul", energy, "\n ecoul", ecoul)
            # check convergence, broyden mixing
            # write restart data
            # a,b=np.linalg.eig(oldovermat)

            if abs(oldenergy-energy) < 1E-4:
                print("convergence energy:", energy, "niter", niter)
                break
            oldenergy = energy
            # broyden1 = np.zeros((natom, 2))
            # broyden2 = np.zeros((natom, 2, maxinter))
            print("oldqatom", oldqatom, "qatom", qatom)
            """if niter == 0:
                oldqatom = self.broyden0(niter, broyden1, broyden2,
                                         oldqatom, qatom, generalpara)[0]
                f0 = self.broyden0(niter, broyden1, broyden2, oldqatom,
                                   qatom, generalpara)[1]
            else:
                f0 = self.broyden1(niter, broyden1, broyden2, oldqatom, qatom,
                                   f0, generalpara)[1]"""
            print("oldqatom", oldqatom, "qatom", qatom)
            for i in range(0, natom):
                qatom[i] = oldqatom[i]

    def genmpgrids(self, generalpara):
        """the coordination is atomic unit, """
        # periodic, unitc, reallat, ksample, natom
        reclat = np.empty((3, 3))
        unitc = generalpara["latticepara"]
        natom = generalpara["natom"]
        reallat = generalpara["coor"]
        ksample = generalpara["ksampling"]
        vec = np.empty(3)
        skcut = generalpara["gridall"][0]*generalpara["ngridmax"]+0.3
        mcell = np.empty(3)
        shifts = np.empty(3)
        coeffs = np.empty((3, 3))
        k_vec = np.zeros((3, 3))
        # mp_grid = np.zeros((3, 3))
        xyz_shift = np.empty((natom, 3))
        reclat[0, 0] = unitc[1, 1]*unitc[2, 2]-unitc[2, 1]*unitc[1, 2]
        reclat[0, 1] = unitc[1, 2]*unitc[2, 0]-unitc[2, 2]*unitc[1, 0]
        reclat[0, 2] = unitc[1, 0]*unitc[2, 1]-unitc[2, 0]*unitc[1, 1]
        reclat[1, 0] = unitc[2, 1]*unitc[0, 2]-unitc[0, 1]*unitc[2, 2]
        reclat[1, 1] = unitc[2, 2]*unitc[0, 0]-unitc[0, 2]*unitc[2, 0]
        reclat[1, 2] = unitc[2, 0]*unitc[0, 1]-unitc[0, 0]*unitc[2, 1]
        reclat[2, 0] = unitc[0, 1]*unitc[1, 2]-unitc[1, 1]*unitc[0, 2]
        reclat[2, 1] = unitc[0, 2]*unitc[1, 0]-unitc[1, 2]*unitc[0, 0]
        reclat[2, 2] = unitc[0, 0]*unitc[1, 1]-unitc[1, 0]*unitc[0, 1]
        volume = unitc[0, 0]*reclat[0, 0] + unitc[0, 1]*reclat[0, 1] + \
                 unitc[0, 2]*reclat[0, 2]
        reclat = 2*np.pi*reclat/volume
        # print("unitc", unitc, "\n volume", volume, "\n reclat", reclat)
        for ri in range(0, 3):
            vec[ri] = np.sqrt(np.dot(unitc[ri, :], unitc[ri, :]))
            # print('unitc[ri, :]', unitc[ri, :])
            # print('vec[ri], skcut', vec[ri], skcut)
            if vec[ri] <= 1E-6:
                print("Error, lattice vector is zero!")
            elif vec[ri] <= skcut:
                mcell[ri] = int(skcut/vec[ri]) + 1
                # print('mcell[ri] = int(skcut/vec[ri])', mcell[ri])
            else:
                mcell[ri] = 1
            for j in range(0, natom):
                xyz_shift[j, :] = reallat[j, 1:4] + unitc[ri, :]
                # print('xyz_shift[j, :]', xyz_shift[j, :])
                # print('reallat[j, 1:4], unitc[ri, :]', reallat[j, 1:4],
                #      unitc[ri, :])
            mindist = 1E6
            for j in range(0, natom):
                for k in range(0, natom):
                    vec[:] = reallat[j, 1:4] - xyz_shift[k, :]
                    dist = np.sqrt(np.dot(vec[:], vec[:]))
                    mindist = min(mindist, dist)
                    # print("mindist", mindist, 'vec', vec)
            if mindist > skcut:
                mcell[ri] = 0
            # print('mcell[ri], ri', mcell, ri)
            # print('skcut, vec[ri]', skcut, vec[ri])
        # mindist
        ncell = int((2*mcell[0]+1) * (2*mcell[1]+1) * (2*mcell[2]+1))
        s_irvec = np.empty((3, ncell))
        icell = 0
        for i in range(-int(mcell[0]), int(mcell[0])+1):
            for j in range(-int(mcell[1]), int(mcell[1])+1):
                for k in range(-int(mcell[2]), int(mcell[2])+1):
                    s_irvec[0, icell] = i
                    s_irvec[1, icell] = j
                    s_irvec[2, icell] = k
                    # print('i, j, k', i, j, k, s_irvec[:, icell])
                    icell += 1
        # print('s_irvec', s_irvec)
        coeffs[:, :] = 0.0
        for i in range(0, 3):
            # print('coeffs[i, i]', coeffs[i, i], 'ksample[i]', ksample[i])
            coeffs[i, i] = ksample[i]
            if np.mod(ksample[i], 2) == 0:
                # if divisble by 2, shift k by 0.5
                shifts[i] = 0.5
            else:
                shifts[i] = 0.0
        nkmax = int(ksample[0] * ksample[1] * ksample[2])
        allkpoints = np.empty((3, nkmax))
        allkweights = np.empty(nkmax)
        allkpoints, allkweights = self.getksample(coeffs, shifts, unitc, k_vec,
                                                  allkpoints, allkweights,
                                                  nkmax, False)
        # print('allkpoints', allkpoints, 'allkweights', allkweights, nkmax)
        """
        k_mesh = np.empty((3, nkmax))
        k_weight = np.empty(nkmax)
        k_mesh[:, :nkmax] = allkpoints[:, :nkmax]
        k_weight[:nkmax] = allkweights[:nkmax]"""
        # print("reclat", reclat, "unitcell", unitc)
        generalpara["reclat"] = reclat
        generalpara["k_mesh"] = allkpoints
        generalpara["k_kweight"] = allkweights
        generalpara["numkpoints"] = nkmax
        generalpara["volume0"] = volume
        generalpara["ncell"] = ncell
        generalpara["s_irvec"] = s_irvec
        # return reclat, allkpoints, allkweights, volume, ncell, s_irvec
        return generalpara

    def getksample(self, mp_grid, shifts, latvecs, recvecs2p, kpoints,
                   kweights, nkpoint, treduce):
        tol = 1E-4
        minlim = -tol
        maxlim = 1.0 - tol
        coeffs = mp_grid*1.0
        superlatvecs = np.matmul(latvecs, coeffs)
        superrecvecs = superlatvecs
        # self.matinv(superrecvecs, 3)
        out = np.zeros_like(superrecvecs)
        superrecvecs = np.reciprocal(superrecvecs, where=superrecvecs > 0,
                                     out=out)
        superrecvecs = superrecvecs.reshape(3, 3)
        invcoeffs = coeffs
        # self.matinv(invcoeffs, 3)
        out = np.zeros_like(invcoeffs)
        invcoeffs = np.reciprocal(invcoeffs, where=invcoeffs > 0, out=out)
        invcoeffs = invcoeffs.reshape(3, 3)
        nkmax = mp_grid[0, 0]*mp_grid[1, 1]*mp_grid[2, 2]
        allkpoints = np.empty((3, int(nkmax)))
        k = 0
        # print('mp_grid', mp_grid)
        for i1 in range(0, int(mp_grid[0, 0])):
            for i2 in range(0, int(mp_grid[1, 1])):
                # 0?
                for i3 in range(0, int(mp_grid[2, 2])):
                    real = np.array([i1, i2, i3])
                    rr = np.matmul(invcoeffs, real + shifts)
                    # print('rr', rr, 'invcoeffs', invcoeffs, 'real', real)
                    # print('shifts', shifts)
                    if (all(rr >= minlim) and all(rr < maxlim)):
                        print('k', k, allkpoints[:, k])
                        allkpoints[:, k] = rr[:]
                        k += 1
        nallkpoint = k
        determinant33 = self.determinant33(coeffs)
        if (abs(nallkpoint*1.0 - determinant33) > tol):
            print("Monkhorst-Pack routine failed to find all K-points.")
        allkweights = np.empty(nallkpoint)
        allkweights[:] = 1.0/nallkpoint
        if treduce:
            irreducible = np.empty(nallkpoint)
            irreducible[:] = True
            for i1 in range(1, nallkpoint):
                if not irreducible[i1]:
                    pass
                rr[:] = modulo(-1.0 * allkpoints[:,i1], 1.0)
                for i2 in range(i1 + 1, nallkpoint):
                    if not irreducible[i2]:
                        pass
                    if all(abs(allkpoints[:, i2] - rr[:]) < tol):
                        irreducible[i2] = False
                        allkweights[i1] = allkweights[i1] + allkweights[i2]
            nkpoint = count(irreducible)
            i1 = 1
            i2 = 1
            while i2 <= nkpoint:
                if irreducible[i1]:
                    kpoints[:, i2] = allkpoints[:, i1]
                    kweights[i2] = allkweights[i1]
                    i2 = i2 + 1
                i1 = i1 + 1
        else:
            kpoints[:, :] = allkpoints
            kweights[:] = allkweights
        # print('kpoints', allkpoints, 'kweights', allkweights)
        return kpoints, kweights

    def matinv(self, aa, nn):
        lwork = nn**2
        ipiv = np.empty(nn)
        # work = np.empty(lwork)
        linalg.lapack.dgetrf(aa, ipiv)
        # if info != 0:
        #    print("zgetrf failed in matinv")
        linalg.lapack.dgetri(aa, lwork)
        # if info != 0:
        #    print("zgetri failed in matinv")

    def determinant33(self, matrix):
        tmp = matrix[0, 0]*matrix[1, 1]*matrix[2, 2]-matrix[2, 1]*matrix[1, 2]
        tmp = tmp-matrix[0, 1]*(matrix[1, 0]*matrix[2, 2]-matrix[2, 0])*matrix[1, 2]
        tmp = tmp+matrix[0, 2]*(matrix[1, 0]*matrix[2, 1]-matrix[2, 0])*matrix[1, 1]
        determinant33 = abs(tmp)
        return determinant33

    def gmatrixpe(self, generalpara):
        """"gmatrixpe is used to generate the gamma matrix in periodic
        condition"""
        # unitcell, reclat, vol, natom, coor, uhubb
        bohr = PUBPARA["BOHR"]
        unitcell = generalpara["latticepara"].transpose()
        reclat = generalpara["reclat"].transpose()
        vol = generalpara["volume0"]
        natom = generalpara["natom"]
        coor = generalpara["coor"]*bohr
        uhubb = generalpara["uhubb"]
        gmat = []
        nr = np.empty(3)
        nk = np.empty(3)
        rr = np.empty(3)
        tol_alpha = 1E-8
        tol = 1E-11
        alpha = self.getalpha(unitcell, reclat, vol, tol_alpha)
        # print('unitcell', unitcell, 'reclat', reclat, vol, tol_alpha)
        rcut = np.sqrt(-np.log(tol))/alpha
        kcut = 2*alpha*np.sqrt(-np.log(tol))
        # print('alpha', alpha, "rcut, kcut", rcut, kcut)
        for i in range(0, 3):
            nr[i] = int(rcut/np.sqrt(unitcell[i, 0]**2 + unitcell[i, 1]**2 +
                                     unitcell[i, 2]**2))
            nk[i] = int(kcut/np.sqrt(reclat[i, 0]**2 + reclat[i, 1]**2 +
                                     reclat[i, 2]**2))
        ngmat = 0
        for i in range(0, natom):
            for j in range(0, i+1):
                ngmat += 1
                # print('rr[0]', rr[0])
                # print(i, coor)
                rr[0] = coor[i, 1] - coor[j, 1]
                rr[1] = coor[i, 2] - coor[j, 2]
                rr[2] = coor[i, 3] - coor[j, 3]
                # print('rr', rr)
                rr = rr/0.5291772106712
                # print('rr', rr, 'unitcell', unitcell, 'reclat', reclat)
                # print('alpha', alpha, 'vol', vol, 'tol', tol)
                # print('uhubb[i]', uhubb[i], 'uhubb[j]', uhubb[j])
                glong = self.gmatllong(rr, unitcell, reclat, alpha, vol, tol)
                gshort = self.gmatlshort(rr, unitcell, uhubb[i], uhubb[j], tol)
                gval = glong + gshort
                # print('rr', rr)
                # print('i,j', i, j, 'rr', rr)
                # print('glong', glong, 'gshort', gshort)
                gmat.append(gval)
                # print(gval,gmat)
        return gmat

    def getalpha(self, unitcell, reclat, vol, tol):
        r_u1 = unitcell[0, 0]**2+unitcell[0, 1]**2+unitcell[0, 2]**2
        r_u2 = unitcell[1, 0]**2+unitcell[1, 1]**2+unitcell[1, 2]**2
        r_u3 = unitcell[2, 0]**2+unitcell[2, 1]**2+unitcell[2, 2]**2
        rsmall = np.sqrt(min(r_u1, r_u2, r_u3))
        r_r1 = reclat[0, 0]**2+reclat[0, 1]**2+reclat[0, 2]**2
        r_r2 = reclat[1, 0]**2+reclat[1, 1]**2+reclat[1, 2]**2
        r_r3 = reclat[2, 0]**2+reclat[2, 1]**2+reclat[2, 2]**2
        gsmall = np.sqrt(min(r_r1, r_r2, r_r3))
        # print('unitcell', unitcell)
        # print('reclat', reclat)
        # print('rsmall', rsmall, 'gsmall', gsmall)
        alpha = 1E-5
        while self.diffrecreal(alpha, gsmall, rsmall, vol) < tol:
            alphal = alpha
            alpha = alpha * 2.0
        print('alpha1', alpha)
        alpha = 1E+5
        while self.diffrecreal(alpha, gsmall, rsmall, vol) > tol:
            alphar = alpha
            alpha = alpha / 2.0
        print('alpha2', alpha)
        nopt = 0
        while abs(self.diffrecreal(alpha, gsmall, rsmall, vol)) > tol\
                and nopt <= 20:
            # print(self.diffrecreal(alpha, gsmall, rsmall, vol), tol, nopt)
            if self.diffrecreal(alpha, gsmall, rsmall, vol) < tol:
                alphal = alpha
            if self.diffrecreal(alpha, gsmall, rsmall, vol) > tol:
                alphar = alpha
            alpha = (alphal + alphar)/2.0
            nopt += 1
        print('alpha3', alpha)
        return alpha

    def diffrecreal(self, alpha, gg, rr, vol):
        diffrec = self.gspace(2.0*gg, alpha, vol) -\
                   self.gspace(3.0*gg, alpha, vol)
        diffreal = self.rspace(2.0*rr, alpha) - self.rspace(3.0*rr, alpha)
        diffrecreal = diffrec - diffreal
        # print('diffrec', diffrec, 'diffreal', diffreal)
        return diffrecreal

    def gspace(self, gg, alpha, vol):
        gsapce = 4*np.pi*np.exp(-gg**2/(4.0*alpha*alpha))/(gg**2)/vol
        return gsapce

    def rspace(self, rr, alpha):
        rspace = self.terfc(alpha*rr)/rr
        return rspace

    def terfc(self, xx):
        zz = abs(xx)
        rt = 1.0/(1.0+0.5*zz)
        terfc2 = rt*np.exp(-zz*zz-1.26551223+rt*(1.00002368+rt*(0.37409196+rt *
                          (0.09678418+rt*(-0.18628806+rt*(0.27886807+rt *
                          (-1.13520398+rt*(1.48851587+rt*(-0.82215223 +
                          rt*0.17087277)))))))))
        if xx < 0.0:
            terfc2 = 2.0 - terfc2
        return terfc2

    def gmatllong(self, rr, basis, recbasis, alpha, vol, tol):
        para = np.empty(4)
        garr = np.zeros(3)
        job2 = 2
        job3 = 3
        para[0] = alpha
        para[1:4] = rr[:]
        # print('job3', job3, 'rr', garr, 'recbasis', recbasis, 'para', para,
        #      'tol', tol)
        reciprocal = self.sumlattice(job3, garr, recbasis, para, tol)
        reciprocal = (4.0*np.pi*reciprocal)/vol
        # real space
        para[0] = alpha
        # print('job2', job2, 'rr', rr, 'basis', basis,
        #      'para', para, 'tol', tol)
        rcspace = self.sumlattice(job2, rr, basis, para, tol)
        cterm = -np.pi/(vol*alpha*alpha)
        # if r = 0 there is another constant to be added
        if rr[0]**2+rr[1]**2+rr[2]**2 < 1E-20:
            cterm = cterm-2.0*alpha/np.sqrt(np.pi)
        # print('reciprocal', reciprocal, 'rcspace', rcspace, 'cterm', cterm)
        potential = reciprocal + rcspace + cterm
        return potential

    def gmatlshort(self, rr, unitcell, uhubbi, uhubbj, tol):
        job = 4
        para = np.empty(2)
        para[0] = uhubbi
        para[1] = uhubbj
        sumresult = self.sumlattice(job, rr, unitcell, para, tol)
        # print('job', job, 'rr', rr, 'unitcell', unitcell, 'para', para)
        # print('tol', tol, 'sumresult', sumresult)
        return sumresult

    def sumlattice(self, job, disr, lattice, para, tol):
        nmax = 100
        cut_dist2 = 1E12
        sumresult = 0.0
        nrepeat = np.empty((3, 3))
        rvect = np.empty(3)
        for ii in range(0, 3):
            if disr[ii] > lattice[ii, ii]/2:
                disr[ii] = disr[ii] - lattice[ii, ii]
            if disr[ii] < -(lattice[ii, ii])/2:
                disr[ii] = disr[ii] + lattice[ii, ii]
        # print('lattice', lattice)
        # print('disr', disr)
        nrepeat[:, :] = nmax
        for i in range(0, 3):
            for j in range(0, 3, 2):
                for k in range(1, nmax+1):
                    rvect[0] = disr[0] + (j-1)*k*lattice[i, 0]
                    rvect[1] = disr[1] + (j-1)*k*lattice[i, 1]
                    rvect[2] = disr[2] + (j-1)*k*lattice[i, 2]
                    norm2 = rvect[0]**2 + rvect[1]**2 + rvect[2]**2
                    # print('rvect[:]', rvect[:])
                    if norm2 >= cut_dist2:
                        nrepeat[j, i] = k-1
                        # print('    nrepeat[j, i]', nrepeat[j, i], j, i)
                        break
                    # print('norm2', norm2, 'i, j, k', i, j, k)
                    # print('cut_dist2', cut_dist2)
                    value, decay = self.calpot(job, rvect, para, True)
                    # print('    nrepeat[j, i]', nrepeat[j, i])
                    # print('    cut_dist2', cut_dist2, 'value', value)
                    # print()
                    sumresult += value
                    # print('sumresult', sumresult)
                    if abs(decay) < tol:
                        cut_dist2 = min(cut_dist2, norm2)
                        nrepeat[j, i] = k-1
                        # print('    nrepeat[j, i]', nrepeat[j, i], j, i)
                        # print('    norm2', norm2)
                        break
                    # print('    nrepeat[0, 0]', nrepeat[0, 0], 'j, i', j, i)
        # print('        nrepeat', nrepeat, nrepeat[0, 0])
        # print('        disr', disr, 'rvect', rvect)
        # print('        lattic', lattice)
        # print('nrepeat[-1, 2], nrepeat[1, 2]', nrepeat[-1, 2], nrepeat[1, 2])
        # print('nrepeat[-1, 1], nrepeat[1, 1]', nrepeat[-1, 1], nrepeat[1, 1])
        for k in range(-int(nrepeat[0, 2]), int(nrepeat[2, 2])+1):
            for j in range(-int(nrepeat[0, 1]), int(nrepeat[2, 1])+1):
                for i in range(-int(nrepeat[0, 0]), int(nrepeat[2, 0])+1):
                    # print("    i, j, k", i, j, k)
                    if i != 0 and j == 0 and k == 0:
                        continue
                    elif i == 0 and j != 0 and k == 0:
                        continue
                    elif i == 0 and j == 0 and k != 0:
                        continue
                    rvect[0] = (disr[0]+i*lattice[0, 0]+j*lattice[1, 0]
                                + k*lattice[2, 0])
                    rvect[1] = (disr[1]+i*lattice[0, 1]+j*lattice[1, 1]
                                + k*lattice[2, 1])
                    rvect[2] = (disr[2]+i*lattice[0, 2]+j*lattice[1, 2]
                                + k*lattice[2, 2])
                    norm2 = rvect[0]**2 + rvect[1]**2 + rvect[2]**2
                    if norm2 >= cut_dist2:
                        continue
                    value = self.calpot(job, rvect, para, False)
                    sumresult += value
                    # print('    sumresult', sumresult, i, j, k)
        return sumresult

    def calpot(self, job, rvect, para, decay01):
        if job == 1:
            """"what is the para should be checked"""
            value = self.calshortpot(rvect, para[0])
            if decay01 == True:
                decay = value
        elif job == 4:
            value = self.calcshortgamma(rvect, para[0], para[1])
            if decay01 == True:
                decay = value
        elif job == 2:
            value = self.calclongreal(rvect, para[0])
            if decay01 == True:
                decay = value
        else:
            if decay01 == True:
                value, decay = self.calclongrecip(para[1:4], rvect, para[0],
                                                  decay01)
            else:
                value = self.calclongrecip(para[1:4], rvect, para[0], False)
        if decay01 == True:
            return value, decay
        else:
            return value

    def calshortpot(self, rv, uhubbi):
        norm = np.sqrt(rv[0]**2+rv[1]**2+rv[2]**2)
        if norm > 1.0E-20:
            value = -(np.exp(-3.2*uhubbi*norm))*(1.6*uhubbi+1/norm)
        else:
            value = 0.0
        return value

    def calcshortgamma(self, rv, uhubbi, uhubbj):
        rab = np.sqrt(rv[0]**2+rv[1]**2+rv[2]**2)
        taui = 3.2*uhubbi
        tauj = 3.2*uhubbj
        if rab < 1E-5:
            # on-site case with R~0
            if abs(uhubbi-uhubbj) < 1E-5:
                # same Hubbard U values, tolerance 1E-5 can be change
                expgamma = -0.50*(uhubbi + uhubbj)
            else:
                # Ui != Uj Hubbard U values
                expgamma = -0.5*((taui*tauj)/(taui+taui) +
                                 (taui*tauj)**2/(taui+tauj)**3)
        elif abs(uhubbi-uhubbj) < 1E-5:
            # R > 0 and same Hubbard U values
            taumean = 0.5*(taui+tauj)
            expgamma = np.exp(-taumean*rab)*(1.0/rab+0.6875*taumean +
                              0.18750*rab*(taumean**2) +
                              0.02083333333333333*(rab**2)*(taumean**3))
        else:
            gammasub1 = np.exp(-taui*rab)*((0.50*tauj**4*taui /
                              (taui**2-tauj**2)**2)-(tauj**6 -
                               3.0*tauj**4*taui**2) /
                              (rab*(taui**2-tauj**2)**3))
            gammasub2 = np.exp(-tauj*rab)*((0.5*taui**4*tauj /
                              (tauj**2-taui**2)**2)-(taui**6 -
                               3.0*taui**4*tauj**2) /
                              (rab*(tauj**2-taui**2)**3))
            expgamma = gammasub1+gammasub2
        return -expgamma

    def calclongreal(self, rvect, alpha):
        norm = np.sqrt(rvect[0]**2+rvect[1]**2+rvect[2]**2)
        if norm > 1E-20:
            value = self.terfc(alpha*norm)/norm
        else:
            value = 0.0
        return value

    def calclongrecip(self, rr, rvect, alpha, decay01):
        norm2 = rvect[0]**2+rvect[1]**2+rvect[2]**2
        if norm2 > 1E-20:
            norm2 = np.exp(-norm2/(4.0*alpha*alpha))/norm2
            value = np.cos(np.dot(rvect, rr))*norm2
            if decay01 == True:
                decay = norm2
        else:
            value = 0.0
            if decay01 == True:
                decay = 1.0
        if decay01 == True:
            return value, decay
        else:
            return value

    def formhspe(self, nk, generalpara, shift, eigvec, smat):
        # nk, xyz, natom, kpoint,shift,ncell, s_irvec,reclat,eigvec, overmat
        natom = generalpara["natom"]
        ncell = generalpara["ncell"]
        norbs = int(generalpara["norbs"])
        kpoint = generalpara["k_mesh"]
        s_irvec = generalpara["s_irvec"]
        reclat = generalpara["reclat"]
        xyz = generalpara["coor"][:, 1:]
        xyz = xyz.transpose()
        izp = generalpara["natomtype"]
        rvec = generalpara["latticepara"].transpose()
        xyzshift = np.zeros((3, natom))
        ceye = np.complex(0.0, 1.0)
        hmat = np.zeros((norbs, norbs), dtype=complex)
        smat = np.zeros((norbs, norbs), dtype=complex)
        a_h = np.zeros((norbs, norbs))
        a_s = np.zeros((norbs, norbs))
        # cone = np.complex(1.0, 0.0)
        # czero = np.complex(0.0, 0.0)
        # print('    nk', nk, 'shift', shift)
        # print('    kpoint', kpoint)
        # print('    s_irvec', s_irvec)
        # print('    xyz[:, :]', xyz[:, :])
        # print('    s_irvec[:, :]', s_irvec[:, :])
        # print('    rvec[:, :]', rvec[:, :])
        for icell in range(0, ncell):
            rdotk = 2.0*np.pi*np.dot(kpoint[:, nk], s_irvec[:, icell])
            # print('    icell', icell)
            # print("    rdotk", rdotk, 'icell', icell)
            # print('    kpoint[:, nk]', kpoint[:, nk])
            # print('    s_irvec[:, icell]', s_irvec[:, icell])
            fac = np.exp(ceye*rdotk)
            # print('fac', fac)
            for i in range(0, natom):
                xyzshift[:, i] = xyz[:, i]+s_irvec[0, icell]*rvec[:, 0] +\
                                 s_irvec[1, icell]*rvec[:, 1] +\
                                 s_irvec[2, icell]*rvec[:, 2]
            # print('    xyzshift[:, :]', xyzshift[:, :])
            # print('    xyz[:, i]', xyz[:, i])
            # print('    xyz[:, :]', xyz[:, :])
            # print('    s_irvec[:, :]', s_irvec[:, :])
            # print('    reclat[:, :]', reclat[:, :])
            # print('icell', icell)
            a_h, a_s = self.formhs(generalpara, izp, xyzshift, shift,
                                   a_s, a_h)
            # print('hmat1', hmat)
            a_h[0:norbs, 0:norbs] = a_h[0:norbs, 0:norbs]*PUBPARA["AUEV"]
            hmat = hmat + fac * a_h
            smat = smat + fac * a_s
            # print('a_h2', a_h)
            # print('hmat2', hmat, 'PUBPARA["AUEV"]', PUBPARA["AUEV"])
        # print('hmat in formhspe', hmat)
        return hmat, smat

    def formhs(self, generalpara, izp, xyz_shift, shift, overmat, hammat):
        natom = generalpara["natom"]
        ind1 = generalpara["atomind"]
        ind2 = generalpara["atomind"]
        xyz1 = generalpara["coor"]
        # atomind = generalpara["atomind"]
        ldim = PUBPARA["LDIM"]
        atomname = generalpara["atomnameall"]
        hams = np.empty((ldim, ldim))
        ovrs = np.empty((ldim, ldim))
        # atomind2 = int(atomind[natom]*(atomind[natom]+1)/2)
        # hammat = np.zeros((atomind2))
        # overmat = np.zeros((atomind2))
        rr = np.empty(3)
        # print("hammat in formHS", hammat)
        for i in range(0, natom):
            izpi = izp[i]
            lmaxi = VAL_ORB[atomname[i]]
            for j in range(0, natom):
                generalpara["skfnameij"] = atomname[i]+"-"+atomname[j]+".skf"
                izpj = izp[j]
                rr[0] = xyz_shift[0, j] - xyz1[i, 1]
                rr[1] = xyz_shift[1, j] - xyz1[i, 2]
                rr[2] = xyz_shift[2, j] - xyz1[i, 3]
                hams = np.zeros((9, 9))
                ovrs = np.zeros((9, 9))
                # print('rr in formHS', rr[:], 'izpi, izpj', izpi, izpj)
                # print('xyz_shift', xyz_shift, xyz_shift[0, j], xyz1[i, 1])
                # print('xyz1', xyz1)
                hams, ovrs = self.slkode(rr, izpi, izpj, generalpara, hams,
                                         ovrs, lmaxi, hammat, overmat)
                # print('hams', hams)
                for n in range(0, int(ind2[j+1] - ind2[j])):
                    nn = int(ind2[j] - ind2[0] + n)
                    for m in range(0, int(ind1[i+1] - ind1[i])):
                        mm = int(ind1[i] - ind1[0] + m)
                        # print("m, n, mm, nn", m, n, mm, nn)
                        overmat[mm, nn] = ovrs[n, m]
                        hammat[mm, nn] = hams[n, m]+0.5*ovrs[n, m] *\
                                         (shift[i] + shift[j])
                        # print('hammat[mm,nn]', hammat[mm, nn])
        # print('hammat2 in formHS', hammat)
        return hammat, overmat

    def hermitianize(self, mat, n):
        for j in range(0, int(n)):
            print("mat[j, j]", mat[j, j], type(mat[j, j]))
            mat[j, j] = np.complex(mat[j, j])
            for i in range(0, j-1):
                mat[i, j] = (mat[i, j]+mat[j, i].conjugate())/2
                mat[j, i] = mat[i, j].conjugate()

    def k_weight2int(self, generalpara):
        nkpt = len(generalpara["k_kweight"])
        weight = np.zeros(nkpt)
        weight[:] = generalpara["k_kweight"]
        wmin = weight[0]
        ik_weight = np.empty(nkpt)
        for k in range(0, nkpt):
            if weight[k] < wmin:
                wmin = weight[k]
                # kmin = k
        if wmin <= 0.0:
            print('Error in k_weight2Int, division by zero: ', wmin)
        for k in range(0, nkpt):
            weight[k] = weight[k] / wmin
        for k in range(0, nkpt):
            ik_weight[k] = int(weight[k])
            if abs(weight[k] - ik_weight[k]) > 1E-5:
                print('Error: failed to convert k_weight to integers: ',
                      weight[k])
        generalpara["ik_weight"] = ik_weight
        return generalpara

    def read_skf_file(self, GEN_PARA, skf_name):
        """read_skf_file: read .skf file
       get_grid: return the value of grid_dist, ngridpoint, skself
       line_spline: get which the line of Spline is
       """
        self.skfile_name = skf_name
        self.data_skflist = []
        read_file = open(self.skfile_name)
        try:
            for line in read_file:
                num_line = line.strip().split()
                self.data_skflist.append(num_line)
        finally:
            read_file.close()
        # print("the name of skf file is:",self.skfile_name,"\n")
        return self.data_skflist

    def get_grid(self, skf_name):
        """get the grid_dist,ngridpoint"""
        self.read_skf_file(GEN_PARA, skf_name)
        data_skflist0 = np.array(self.data_skflist[0])
        data_skflist1 = np.array(self.data_skflist[1])
        # print(self.data_skflist0)
        grid_dist = 0
        ngridpoint = 0
        grid_dist = float(''.join(list(filter(lambda x: x in "0123456789.-",
                                              data_skflist0[0]))))
        ngridpoint = int(''.join(list(filter(lambda x: x in "0123456789.-",
                                             data_skflist0[1]))))
        skself = []
        for i in range(0, len(data_skflist1)):
            uhubb = ''.join(list(filter(lambda x: x in "0123456789.-",
                                        data_skflist1[i])))
            skself.append(uhubb)
        for i in range(0, len(data_skflist1)):
            skself[i] = float(skself[i])
        # print("uhubb",uHubb,len(self.data_skflist1),skself)
        return grid_dist, ngridpoint, skself

    def get_skself(self, generalpara):
        atom_name_all = generalpara["atomnameall"]
        atom_name_once = set(atom_name_all)
        natomtype = len(atom_name_once)
        numarr = int(natomtype*3)
        skself_data = np.empty(numarr)
        iel = 0
        for element in atom_name_once:
            skf_name = element+'-'+element+'.skf'
            self.read_skf_file(GEN_PARA, skf_name)
            skself_data[iel] = float(self.data_skflist[1][0])
            skself_data[iel+1] = float((self.data_skflist[1])[1])
            skself_data[iel+2] = float((self.data_skflist[1])[2])
            iel += 3
        return skself_data

    def line_spline(self, skf_name):
        """get the line of spline"""
        self.read_skf_file(GEN_PARA, skf_name)
        line_spline = self.get_grid(skf_name)[1]
        for i in range(line_spline, line_spline+50):
            # can be revised
            if 'Spline' in self.data_skflist[i]:
                break
            else:
                line_spline += 1
        # print("the line number of Spline line ",self.line_spline)
        return line_spline

    def gmatrix(self, generalpara):
        distance = generalpara["distance"]
        uhubb = generalpara["uhubb"]
        natom = generalpara["natom"]
        print(distance)
        gmat = []
        for i in range(0, natom):
            for j in range(0, i+1):
                # if range(0,i), i=j=0 will not print
                """
                rr = np.sqrt(distance[i, j, 0] * distance[i, j, 0] +
                             distance[i, j, 1] * distance[i, j, 1] +
                             distance[i, j, 2] * distance[i, j, 2])"""
                rr = distance[i, j]
                rr = rr/0.5291772106712
                # print(r,k,uhubb)
                a1 = 3.2 * uhubb[i]
                a2 = 3.2 * uhubb[j]
                src = 1 / (a1 + a2)
                fac = a1 * a2 * src
                avg = 1.6 * (fac + fac * fac * src)
                fhbond = 1
                if rr < 1.0E-4:
                    gval = 0.3125*avg
                    # print(gval)
                else:
                    rrc = 1.0/rr
                    # rrc3 = rrc * rrc * rrc
                    # print(a1,a2)
                    if abs(a1 - a2) < 1.0E-5:
                        fac = avg*rr
                        fac2 = fac * fac
                        efac = np.exp(-fac) / 48.0
                        gval = (1.0 - fhbond * (48.0 + 33 * fac +
                                                fac2*(9.0+fac))*efac)*rrc
                    else:
                        val12 = self.gamsub(a1, a2, rr, rrc)
                        val21 = self.gamsub(a2, a1, rr, rrc)
                        gval = rrc - fhbond * val12 - fhbond * val21
                        print(rrc, fhbond, val12, val21)
                gmat.append(gval)
                # print(gval,gmat)
        return gmat

    def gamsub(self, a, b, rr, rrc):
        a2 = a*a
        b2 = b*b
        b4 = b2*b2
        b6 = b4*b2
        drc = 1.0/(a2-b2)
        drc2 = drc*drc
        efac = np.exp(-a*rr)
        fac = (b6-3*a2*b4)*drc2*drc*rrc
        gval = efac*(0.5*a*b4*drc2-fac)
        # gdrv = -a*gval+efac*fac*rrc
        return gval

    def shifthamgam(self, natom, qatom, qzero, gmat):
        qdiff = []
        shift = []
        for i in range(0, natom):
            qdiff.append(qatom[i] - qzero[i])
            # print("qatom, qzero, qfiff in shifthamgam",qatom, qzero, qdiff)
        for i in range(0, natom):
            shifti = 0
            for j in range(0, natom):
                if j > i:
                    k = j*(j + 1)/2 + i
                    gamma = gmat[int(k)]
                    # print(k,natom,i,j)
                else:
                    k = i*(i + 1)/2 + j
                    # print(k,natom,i,j)
                    gamma = gmat[int(k)]
                shifti = shifti+qdiff[j]*gamma
                # print("shifti, qdiff[j]", shifti, qdiff[j], gamma, i, j)
            shift.append(shifti)
        # print(shift)
        shift = np.array(shift)
        return shift

    def fermi(self, generalpara, nelect, norbs, eigval, occ):
        # telec, nelect
        telec = generalpara['tElec']
        ckbol = 3.16679E-6   # original from lodestar, with revision
        degtol = 1.0E-4
        racc = 2E-16
        dacc = 4*racc
        for i in range(1, norbs):
            occ[i] = 0.0
        if nelect > 1.0E-5:
            if nelect > 2*norbs:
                print('too many electrons')
                # seems break is wrong here
            elif telec > 5.0:
                beta = 1.0/(ckbol*telec)
                etol = ckbol*telec*(np.log(beta)-np.log(racc))
                tzero = False
            else:
                etol = degtol
                tzero = True
            if nelect > int(nelect):
                nef1 = int((nelect+2)/2)
                nef2 = int((nelect+2)/2)
            else:
                nef1 = int((nelect+1)/2)
                nef2 = int((nelect+2)/2)
            # print("nef1,2", nef1,nef2)
            # eBot = eigval[0]
            efermi = 0.5*(eigval[nef1-1] + eigval[nef2-1])
            # print("efermi", efermi, eigval[nef1-1], eigVal[nef2-1])
            nup = nef1
            ndown = nef1
            nup0 = nup
            ndown0 = ndown
            # print("nup,ndown before",nup,ndown)
            while nup0 < norbs:   #
                # print("eigval[nup]-efermi", eigval[nup], efermi, nup0)
                if abs(eigval[nup0]-efermi) < etol:
                    nup0 = nup0+1
                    # print("nup0",nup0,abs(eigval[nup]-efermi), etol)
                else:
                    break
            nup = nup0
            # print("nup",nup,"nef1,2  ",nef1,nef2)
            while ndown0 > 0:
                # print(ndown0,efermi,"abs",abs(eigval[ndown0-1]-efermi))
                if abs(eigval[ndown0-1]-efermi) < etol:
                    ndown0 = ndown0-1
                else:
                    break
            ndown = ndown0
            # print("ndown", ndown, nup, nelect)
            ndeg = nup-ndown    # check
            nocc2 = ndown
            for i in range(0, nocc2):
                occ[i] = 2.0
                # print("occ[i]",occ[i],"i",i,nocc2)
            if ndeg == 0:
                return occ, efermi

            if tzero:
                occdg = ndeg
                # print("ndeg",ndeg,nup,ndown)
                occdg = (nelect-2*nocc2)/occdg
                # print("occdg",occdg,nocc2)
                for i in range(nocc2, nocc2+ndeg):
                    occ[i] = occdg
                    # print("occdg", occdg, nelect, nocc2)
            else:
                chleft = nelect-2*nocc2
                istart = nocc2+1
                iend = istart+ndeg-1
                if ndeg == 1:
                    occ[istart] = chleft
                    return
                ef1 = efermi-etol-degtol
                ef2 = efermi+etol+degtol
                ceps = dacc*chleft
                eeps = dacc*max(abs(ef1), abs(ef2))
                efermi = 0.5*(ef1+ef2)  # check
                charge = 0.0
                for i in range(istart, iend):
                    occ[i] = 2.0/(1.0+np.exp(beta*(eigval[i]-efermi)))
                    charge = charge+occ[i]
                    if charge > chleft:
                        ef2 = efermi
                    else:
                        ef1 = efermi
                    if abs(charge-chleft) > ceps or abs(ef1-ef2) < eeps:
                        continue
                    else:
                        exit
                if abs(charge-chleft) < ceps:
                    return
                else:
                    fac = chleft/charge
                    for i in range(istart, iend):
                        occ[i] = occ[i]*fac
                        # print("occ[i] in fermi", occ[i],"i",i)
        else:
            print('electron number is zero!')
            return
        # print("occ",occ)
        return occ, efermi

    def fermipe(self, generalpara, nkpoint, ev, occ_k):
        # telec = generalpara["telect"]
        norbs = int(generalpara["norbs"])
        nelect = generalpara["nelectrons"]
        ik_weight = generalpara["ik_weight"]
        nkpoint = np.shape(generalpara["k_mesh"])[1]
        nktotal = sum(ik_weight[0: nkpoint])
        ndim = int(norbs*nktotal)
        ev_sorted = np.zeros(ndim)
        occ_sorted = np.zeros(ndim)
        occ_sorted2 = np.zeros(ndim)
        back_ind = np.zeros(ndim)
        map_ind = np.zeros((norbs, nkpoint))
        ev_ind = np.zeros(ndim)
        ii = 0
        for k in range(0, nkpoint):
            for j in range(0, int(ik_weight[k])):
                for i in range(0, norbs):
                    ev_sorted[ii] = ev[i, k]
                    map_ind[i, k] = ii
                    ev_ind[ii] = ii
                    ii += 1
        # print('ev_sorted', ev_sorted)
        # ev_sorted = self.qsortind(ev_sorted, ev_ind)
        ev_sorted2 = ev_sorted[np.argsort(ev_sorted)]
        back_ind = np.argsort(ev_sorted)
        # print('back_ind', back_ind)
        # print('ev_sorted2', ev_sorted2)
        occ_sorted, efermi = self.fermi(generalpara, int(nelect*nktotal),
                                        int(norbs*nktotal), ev_sorted2,
                                        occ_sorted)
        # for ii in range(0, ndim):
        #    back_ind[int(ev_ind[ii])] = ii
        nk = 0
        for ni in back_ind:
            occ_sorted2[ni] = occ_sorted[nk]
            nk += 1
        # print('occ_sorted', occ_sorted)
        # print('occ_sorted2', occ_sorted2)
        for k in range(0, nkpoint):
            for i in range(0, norbs):
                ii = int(map_ind[i, k])
                # print('ii', ii, 'i,k', i, k)
                occ_k[i, k] = occ_sorted2[ii]
        return occ_k, efermi

    def qsortind(self, aa, ind):
        if np.size(aa) > 1:
            aa, iq = self.partitionInd(aa, ind)
            print('aa in qsortind', aa)
            aa = self.qsortind(aa[:iq-1], ind[:iq-1])
            aa = self.qsortind(aa[iq:], ind[iq:])
        return aa

    def partitionInd(self, aa, ind):
        x = aa[0]
        i = 0
        j = np.size(aa)
        while True:
            j = j-1
            while True:
                if aa[j] <= x:
                    break
                j = j-1
            i += 1
            while True:
                if aa[i] >= x:
                    break
                i += 1
            if i < j:
                # exchange A(i) and A(j)
                temp = aa[i]
                aa[i] = aa[j]
                aa[j] = temp
                itmp = ind[i]
                ind[i] = ind[j]
                ind[j] = itmp
            elif i == j:
                marker = i+1
                return aa, marker
            else:
                marker = i
                return aa, marker
        return aa, marker

    def mulliken(self, natom, norbs, atomind, overmat, denmat, qat):
        """calculate Mulliken charge"""
        for ii in range(0, natom):
            qat[ii] = 0.0
            for i in range(int(atomind[ii]), int(atomind[ii+1])):
                for j in range(0, i):
                    k = i*(i+1)/2+j
                    qat[ii] = qat[ii]+denmat[int(k)]*overmat[int(k)]
                for j in range(i, norbs):
                    k = j*(j+1)/2+i
                    qat[ii] = qat[ii]+denmat[int(k)]*overmat[int(k)]
        # print("qat",qat)
        return qat

    def get_hs_skf(self, skf_name):
        """read H and S from skf file"""
        line_spline = self.line_spline(skf_name)
        self.read_skf_file(GEN_PARA, skf_name)
        grid_dist = self.get_grid(skf_name)[0]
        j = 3
        # j is the line where H and S begins, following several lines code
        # is to make sure the beginning line of H and S
        # data_hslist_test = np.zeros((5, 20))
        j_line = 0
        for line in self.data_skflist[0: 4]:
            # print(skf_name)
            jj = 0
            for sub_list in line:
                if '*' in sub_list:
                    abbr = ''.join(list(filter(lambda x: x in "0123456789.*",
                                               sub_list)))
                    abbr_arr = abbr.split("*")
                    # print(abbr,abbr_arr[0])
                    for _ in range(0, int(abbr_arr[0])):
                        jj += 1
                else:
                    jj += 1
            j_line += 1
            if jj == 20:
                j = j_line
                break
        self.data_hslist = np.zeros((line_spline - j + 1, 21))
        for line in self.data_skflist[j-1: line_spline]:
            # need to revise
            # print(line, line_spline)
            # self.data_hslist[:,:]
            jj = 1
            self.data_hslist[j-j_line, 0] = grid_dist*(j - j_line)
            for sub_list in line:
                # print("sub_list",sub_list)
                # if sum(map(lambda x : '*' in x , sub_list)) != 0:
                # if element in each line has *
                # if element in each line has *
                if '*' in sub_list:
                    abbr = ''.join(list(filter(lambda x: x in "0123456789.*",
                                               sub_list)))
                    abbr_arr = abbr.split("*")
                    # print(abbr,abbr_arr[0])
                    for _ in range(0, int(abbr_arr[0])):
                        self.data_hslist[j-j_line, jj] = np.array(abbr_arr[1])
                        # print(j,jj,abbr_arr[0],np.array(abbr_arr[1]))
                        jj += 1
                else:
                    self.data_hslist[j - j_line, jj] = np.array(sub_list)
                    jj += 1
                # print(self.data_hslist[j-3,jj-1])
            j += 1
        return self.data_hslist

    def spline_para(self):
        """readSkf().get_hs_skf()"""
        line_spline = self.line_spline()
        self.get_grid()
        skfile_name = GEN_PARA["skfile_name"]
        line = linecache.getline(skfile_name, self.line_spline+2)
        every_line = line.split()
        # print(every_line)
        self.nint_spline = int(every_line[0])
        # cutoff_spline = float(every_line[1])
        line = linecache.getline(skfile_name, self.line_spline + 3)
        every_line = line.split()
        self.a1_spline = float(every_line[0])
        self.a2_spline = float(every_line[1])
        self.a3_spline = float(every_line[2])
        # print(self.nint_spline, self.a3_spline)
        self.repulsive_c = np.zeros((self.nint_Spline, 8))
        # the last line is 8, the other line is 6 or 7
        jj = 0
        for line in open(skfile_name).readlines()[line_spline + 3:
                                                  line_spline + 4 +
                                                  self.nint_Spline - 1]:
            # need to revise
            every_line = line.strip().split()
            # print(every_line)
            j = 0
            for sub_list in every_line:
                # print(sub_list)
                self.repulsive_c[jj, j] = np.array(sub_list)
                j += 1
            # print(self.repulsive_c[jj,:])
            jj += 1
        # print("the repulsive parameter: \n",self.repulsive_c)
        return self.repulsive_c


    def broyden0(self, niter, unit31, unit32, vecin, vecout, generalpara):
        coor = generalpara["coor"]
        alpha = generalpara['mixFactor']
        jtop = generalpara["natom"]
        vector = np.zeros((jtop, 2))
        # df = np.zeros((jtop))
        # maxiter = self.get_maxiter(general)
        diff = np.zeros((jtop))
        dumvi = np.zeros((jtop))
        # qatom0 = self.get_electron(coor)[1]
        for i in range(0, jtop):
            vector[i, 0] = vecin[i]
            vector[i, 1] = vecout[i]
        # print("vector",vector)
        ivsiz = jtop
        # W0 = 0.01
        # uamix = np.zeros((maxiter))  #save the last cycle value
        # lastit = np.zeros((maxiter))
        # ilastit = np.zeros((maxiter))
        # lastit=1
        # print("lastit",lastit,niter)
        amix = alpha
        # ilastit=lastit
        for i in range(0, ivsiz):
            diff[i] = vector[i, 1]-vector[i, 0]
            print("F(i) in 0 iter", diff[i])
        for i in range(0, ivsiz):
            unit31[i, 0] = diff[i]
        for i in range(0, ivsiz):
            unit31[i, 1] = vector[i, 1]
        # print("unit31",unit31[:,0],"\n unit31[:,1]",unit31[:,1])
        # !!!!!!!!!unit31 not equal
        for i in range(0, ivsiz):
            dumvi[i] = vector[i, 0] + amix*diff[i]
            # print("dumvi[i]",dumvi[i],vector[i,0],amix,F[i])
        for i in range(0, ivsiz):
            vecin[i] = dumvi[i]
        return vecin, diff

    def broyden1(self, niter, unit31, unit32, vecin, vecout, f0, generalpara):
        coor = generalpara["coor"]
        alpha = generalpara["mixf"]
        jtop = generalpara["natom"]
        atomname = generalpara["atomnameall"]
        natom = generalpara["natom"]
        vector = np.zeros((jtop, 2))
        vector = np.zeros((jtop, 2))
        df = np.zeros((jtop))
        ui = np.zeros((jtop))
        vti = np.zeros((jtop))
        # maxiter = self.get_maxiter(general)
        diff = np.zeros((jtop))
        dumvi = np.zeros((jtop))
        qatom0 = self.get_electron(coor, natom, atomname)[1]
        for i in range(0, jtop):
            vector[i, 0] = vecin[i]
            vector[i, 1] = vecout[i]
        # print("vector",vector)
        ivsiz = jtop
        # W0 = 0.01
        amix = alpha
        lastit = niter
        # check
        # lastit=ilastit
        # for i in range(0,ivsiz):
        # F[i]=unit31[i,0]
        # for i in range(0,ivsiz):
        # dumvi[i]=unit31[i,1]
        for i in range(0, ivsiz):
            # print("dumvi[i]",dumvi[i])
            dumvi[i] = vector[i, 0] - qatom0[i]
            diff[i] = vector[i, 1] - vector[i, 0]
            df[i] = vector[i, 1] - vector[i, 0] - f0[i]
            # print(vector[i,0],vector[i,1],dumvi[i])
        print("dumvi", dumvi, f0, "\n df\n", df)
        dfnorm = 0
        fnorm = 0
        for i in range(0, ivsiz):
            dfnorm += df[i]*df[i]
            fnorm += diff[i]*diff[i]
        dfnorm = np.sqrt(dfnorm)
        fnorm = np.sqrt(fnorm)
        print("dfnorm,fnorm", dfnorm, fnorm)
        fac2 = 1/dfnorm   #
        # one=1?
        fac1 = amix*fac2
        for i in range(0, ivsiz):
            ui[i] = fac1*df[i] + fac2*dumvi[i]
            vti[i] = fac2*df[i]
        lastit = lastit+1
        # lastm1 = lastit-1
        # lastm2 = lastit-2
        return vecin, diff


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
