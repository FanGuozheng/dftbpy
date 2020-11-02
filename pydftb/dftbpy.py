"""DFTB python."""
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
    """"This is the main function, with different interface."""
    parser_cmd_args(para)

    initialization(para)

    DFTBpy(para)


def initialization(para):
    """Initialize DFTB."""
    ReadInput(para)

    if para['ty'] in ['dftbml']:
        read_sk_interp(para)
    elif para['ty'] in ['dftb', 'dftbpy']:
        read_sk(para)


def parser_cmd_args(para):
    """Parser."""
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
    """DFTB python code module."""

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
        '''Self consistent field calculation.'''
        natom = self.para['natom']
        qzero = self.para['qatom']
        atomind = self.para['atomind']
        nind = atomind[natom]

        ham_, over_ = slakotran.sk_tran(self.para)  # SK transfer

        # H, S 1D or 2D transfermation
        if ham_.ndim == 1 and over_.ndim == 1:
            hammat = np.zeros((nind, nind))
            overmat = np.zeros((nind, nind))
            ind = np.tril_indices_from(hammat)

            # the lower triagular indices, then upper triagular
            hammat[ind], overmat[ind] = ham_, over_
            hammat[ind[1], ind[0]], overmat[ind[1], ind[0]] = ham_, over_
        else:
            hammat, overmat = ham_, over_

        # construct gamma in second-order term
        gmat = self.elect.gmatrix()
        niter = self.para['maxIter'] if self.para['scc'] else 1

        energy = np.zeros((niter), dtype=float)
        qmix = qzero
        qdiff = []
        mix = Mix(self.para, qzero, qdiff)
        for iiter in range(niter):
            # get shift gamma and repeat according to orbital numbers
            shift_ = self.elect.shifthamgam(natom, qmix, qzero, gmat)
            shiftorb_ = np.repeat(shift_, np.diff(atomind))
            shiftorb_2d = np.expand_dims(shiftorb_, axis=0) + \
                np.expand_dims(shiftorb_, axis=1)

            # get Fock operator
            fockmat = hammat + 0.5 * overmat * shiftorb_2d

            # get eigenvector and eigenvalue
            eigval, eigvec = linalg.eigh(fockmat, overmat, lower=False,
                                         overwrite_a=True, overwrite_b=True)

            # calculate the occupation of electrons
            occ = self.elect.fermi(eigval)[0]

            # calculate the density matrix
            eigvec = np.sqrt(occ) * eigvec
            density_mat = linalg.blas.dgemm(alpha=1.0, a=eigvec, b=eigvec,
                                            beta=0.0, trans_b=1)

            # mulliken charge and charge mixing
            qnew = self.elect.mulliken(overmat, density_mat)
            qmix = mix.mix(iiter, qnew, qmix)

            # convergence
            energy[iiter] = sum(occ * eigval) - 0.5 * sum(shift_ * (qmix + qzero))
            self.print_energy(iiter, energy)
            if self.convergence(iiter, niter, qdiff):
                break

        if self.para['dipole']:
            get_dipole(self.para, qzero, qmix)
        self.para['humo_lumo'] = eigval[:] * PUBPARA['AUEV']

    def scf_pe(self, para):
        norbs = int(para["norbs"])
        natom = para["natom"]
        atomind = para["atomind"]
        atomnamlist = list(set(para["atomnameall"]))
        para["skself"] = self.get_skself(para)
        if len(atomnamlist) == 1:
            skf_name = atomnamlist+"-"+atomnamlist+".skf"
            para["datalist"+atomnamlist[0]] = self.get_hs_skf(skf_name)
            para["linespline"+atomnamlist[0]] = self.line_spline(skf_name)
            para["griddist"+atomnamlist[0]] = self.get_grid(skf_name)[0]
            para["ngrid"+atomnamlist[0]] = self.get_grid(skf_name)[1]
            # para["skself"+atomnamlist[0]] = self.get_grid(skf_name)[2]
        else:
            skfnamelist = []
            skfnamelist.append(atomnamlist[0]+"-"+atomnamlist[0]+".skf")
            skfnamelist.append(atomnamlist[0]+"-"+atomnamlist[1]+".skf")
            skfnamelist.append(atomnamlist[1]+"-"+atomnamlist[0]+".skf")
            skfnamelist.append(atomnamlist[1]+"-"+atomnamlist[1]+".skf")
            for skfnami in skfnamelist:
                print("datalist"+skfnami)
                para["datalist"+skfnami] = self.get_hs_skf(skfnami)
                para["linespline"+skfnami] = self.line_spline(skfnami)
                para["griddist"+skfnami] = self.get_grid(skfnami)[0]
                para["ngrid"+skfnami] = self.get_grid(skfnami)[1]
                # para["skself"+skfnami] = self.get_grid(skfnami)[2]
            para["ngridmax"] = max(para["ngrid"+skfnamelist[0]],
                                   para["ngrid"+skfnamelist[1]],
                                   para["ngrid"+skfnamelist[2]],
                                   para["ngrid"+skfnamelist[3]])
            para["gridall"] = [para["griddist"+skfnamelist[0]],
                               para["griddist"+skfnamelist[1]],
                               para["griddist"+skfnamelist[2]],
                               para["griddist"+skfnamelist[3]]]
        para = self.genmpgrids(para)
        nkpoint = np.shape(para["k_mesh"])[1]
        eigval_k = np.zeros((norbs, nkpoint))
        eigvec_k = np.zeros((norbs, norbs, nkpoint), dtype=complex)
        para = self.get_electron(para)
        gmat = self.gmatrixpe(para)
        para = self.k_weight2int(para)
        k_weight = para["k_kweight"]
        nelect = np.empty(natom)
        qatom = para["qatom"]
        qmix = qzero = para["qatom"]
        nelect[:] = qzero[:]
        occ_k = np.zeros((norbs, nkpoint))
        oldenergy = 0.0
        for niter in range(0, para["maxi"]):
            print("\n \n niter", niter, "\n")

            shift_ = self.elect.shifthamgam(natom, qmix, qzero, gmat)
            shiftorb_ = np.repeat(shift_, np.diff(atomind))
            shiftorb_2d = np.expand_dims(shiftorb_, axis=0) + \
                np.expand_dims(shiftorb_, axis=1)

            for nk in range(0, nkpoint):
                eigvec, overmat = self.formhspe(nk, para, shift,
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
            occ, efermi = self.fermipe(para, nkpoint, eigval_k, occ_k)
            # occ = self.fermi(int(atomind[natom]), eigval, occ)[0]
            # print("occ after fermi", occ)
            # sum of occupied eigenvalues
            energy = 0.0
            for nk in range(0, nkpoint):
                eigvec, overmat = self.formhspe(nk, para, shift,
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
                                         oldqatom, qatom, para)[0]
                f0 = self.broyden0(niter, broyden1, broyden2, oldqatom,
                                   qatom, para)[1]
            else:
                f0 = self.broyden1(niter, broyden1, broyden2, oldqatom, qatom,
                                   f0, para)qdiff[1]"""
            print("oldqatom", oldqatom, "qatom", qatom)
            for i in range(0, natom):
                qatom[i] = oldqatom[i]

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
    """Mix method"""

    def __init__(self, para, qzero, qdiff):
        '''
        call different mixing method
        '''
        self.para = para
        self.qzero = qzero
        self.qdiff = qdiff
        self.qatom_ = []
        self.qmix_ = []
        self.natom = self.para['natom']
        self.df, self.uu = [], []

    def mix(self, iiter, qnew, qmix):
        """Call different mixing methods."""
        self.qnew = qnew  # list of charge
        self.qmix = qmix
        self.qatom_.append(self.qnew)  # append mulliken charge
        self.qmix_.append(self.qmix)  # append mixing charge
        if iiter == 0:
            qmix = self.simple_mix()
        else:
            if self.para['mixMethod'] == 'simple':
                qmix = self.simple_mix()
            elif self.para['mixMethod'] == 'broyden':
                qmix = self.broyden_mix()
            elif self.para['mixMethod'] == 'anderson':
                qmix = self.anderson_mix()
        return qmix

    def simple_mix(self):
        """Simple mixing method."""
        mixf = self.para['mixFactor']
        self.qdiff.append(self.qnew - self.qmix)
        return self.qmix + mixf * self.qdiff[-1]

    def anderson_mix(self):
        """Anderson mixing method."""
        mixf = self.para['mixFactor']
        self.qdiff.append(self.qnew - self.qmix)
        df_iiter, df_prev = self.qdiff[-1], self.qdiff[-2]
        temp1 = np.dot(df_iiter, df_iiter - df_prev)
        temp2 = np.dot(df_iiter - df_prev, df_iiter - df_prev)
        beta = temp1 / temp2
        average_qin = (1.0 - beta) * self.qatom_[-1] + beta * self.qatom_[-2]
        average_qout = (1.0 - beta) * self.qmix_[-1] + beta * self.qmix_[-2]
        return (1 - mixf) * average_qin + mixf * average_qout

    def broyden_mix(self):
        """Broyden mixing."""
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
    """Calculate dipole moment."""
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
    """Call main function."""
    para = {}
    main(para)
