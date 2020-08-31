"""Main DFTB code.

implement pytorch to DFTB
"""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch as t
import bisect
from slakot import SKspline, SKTran
from electront import DFTBelect
import readt
from periodic import Periodic
from matht import EigenSolver
import parameters
import dftbtorch.parser as parser
from mixer import Simple, Anderson, Broyden
from mbd import MBD
from utils.save import SaveData
from write import Print
import DFTBMaLT.dftbmalt.dftb.dos as dos


def main(para):
    """Run main DFTB code.

    Initialize parameters and then run DFTB
    """
    # read input construct these data for next DFTB calculations
    Initialization(para)

    # with all necessary data, run DFTB calculation
    Rundftbpy(para)


class Initialization:
    """Initialize parameters for DFTB.

    this class aims to read input coordinates, calculation parameters and SK
    tables;
    Then with SK transformation, construct 1D Hamiltonian and overlap matrices
    if symmetry is defined as half, otherwise build 2D, symmetric matrices for
    the following DFTB calculations.

    """

    def __init__(self, para):
        """Interface for different applications.

        Args:
            LCmdArgs (optional): Get information from command line
            LReadInput (optional): Get information form a defined file

        """
        self.para = para

        # Read, process SK integral
        self.slako = SKspline(self.para)

        # read
        self.readsk = readt.ReadSlaKo(self.para)

        # read input
        self.readin = readt.ReadIn(self.para)

        # save data
        self.save = SaveData(self.para)

        # get the constant parameters for DFTB
        parameters.dftb_parameter(self.para)

        # check parameters from command line
        parser.parser_cmd_args(self.para)

        # if define input file, now only support json
        if 'LReadInput' in self.para.keys():
            if self.para['LReadInput']:

                # read input parameters
                self.readin.read_input(para)

                # read input coordinates
                self.readin.get_coor(para)

        # define LReadInput, if LReadInput is True, read from input
        elif 'LReadInput' not in self.para.keys():
            if self.para['LReadInput']:

                # read input parameters
                self.readin.read_input(para)

                # read input coordinates
                self.readin.get_coor()

        # generate vector, distance ... from coordinate
        self.readin.cal_coor()

        # read skf fike from normal skf or list of skf files
        self.read_sk()

        # get Hubbert for each if use gaussian density basis
        if self.para['scc_den_basis'] == 'gaussian':
            self.get_this_hubbert()

        # deal with SK transformation
        self.run_sk()

    def read_sk(self):
        """Read integrals and perform SK transformations.

        Read integrals from .skf
        direct offer integral
        get integrals by interpolation from a list of skf files.

        """
        # do not perform machine learning
        if not self.para['Lml']:

            # get integral from directly reading normal skf file
            if not self.para['LreadSKFinterp']:

                # read normal skf file by atom specie
                self.readsk.read_sk_specie()

            # get integral from a list of skf file (compr) by interpolation
            if self.para['LreadSKFinterp']:

                # read all corresponding skf files by defined parameters
                readt.interpskf(self.para,
                                self.para['typeSKinterpR'],
                                self.para['atomspecie'],
                                self.para['dire_interpSK'])

        # machine learning is on
        # read a list of skf files with various compression radius
        if self.para['Lml'] and self.para['LreadSKFinterp']:

            # ML variables is skf parameters (compression radius)
            # read all corresponding skf files (different compression radius)
            if self.para['Lml_skf']:

                # replace local specie with global specie
                self.para['atomspecie'] = self.para['specie_all']

                # read all corresponding skf files by defined parameters
                readt.interpskf(self.para,
                                self.para['typeSKinterpR'],
                                self.para['atomspecie'],
                                self.para['dire_interpSK'])

            # ML variables is ACSF parameters
            # read all corresponding skf files (different compression radius)
            elif self.para['Lml_acsf']:

                # replace local specie with global specie
                self.para['atomspecie'] = self.para['specie_all']

                # read all corresponding skf files by defined parameters
                readt.interpskf(self.para,
                                self.para['typeSKinterpR'],
                                self.para['atomspecie'],
                                self.para['dire_interpSK'])

            # get integrals directly from spline interpolation
            elif self.para['Lml_HS']:

                self.readsk.read_sk_specie()
                self.slako.get_sk_spldata()

    def get_this_hubbert(self):
        """Get Hubbert for current calculation, nou orbital resolved."""
        # create temporal Hubbert list
        this_U = []
        # only support not orbital resolved U
        if not self.para['Lorbres']:

            # get U hubbert (s orbital) for each atom
            [this_U.append(self.para['uhubb' + iname + iname][-1])
             for iname in self.para['atomnameall']]

        # transfer to tensor
        self.para['this_U'] = t.tensor(this_U, dtype=t.float64)

    def run_sk(self):
        """Read integrals and perform SK transformations.

        Read integrals from .skf
        direct offer integral
        get integrals by interpolation from a list of skf files.

        """
        # do not perform machine learning
        if not self.para['Lml']:

            # get integral from directly reading skf file
            if not self.para['LreadSKFinterp']:

                # SK transformations
                SKTran(self.para)


class Rundftbpy:
    """Run DFTB according to the Initialized parameters.

    DFTB task:
        solid or molecule;
        SCC, non-SCC or XLBOMD;
    """

    def __init__(self, para):
        """Run (SCC-) DFTB."""
        self.para = para

        # analyze DFTB result
        self.analysis = Analysis(self.para)

        # print DFTB calculation information
        self.print_ = Print(self.para)

        # print title before DFTB calculations
        self.print_.print_scf_title()

        # calculate (SCC-) DFTB
        self.runscf()

        # calculate repulsive term
        self.run_repulsive()

        self.run_analysis()

    def runscf(self):
        """Run DFTB with multi interface.

        solid or molecule
        SCC, non-SCC or XLBOMD
        """
        scf = SCF(self.para)

        # calculate solid
        if self.para['Lperiodic']:

            # SCC-DFTB
            if self.para['scc'] == 'scc':
                scf.scf_pe_scc()

        # calculate molecule
        else:

            # non-SCC-DFTB
            if self.para['scc'] == 'nonscc':
                scf.scf_npe_nscc()

            # SCC-DFTB
            elif self.para['scc'] == 'scc':
                scf.scf_npe_scc()

            # XLBOMD-DFTB
            elif self.para['scc'] == 'xlbomd':
                scf.scf_npe_xlbomd()

    def run_repulsive(self):
        """Calculate repulsive term."""
        if self.para['Lrepulsive']:
            Repulsive(self.para)

    def run_analysis(self):
        """Analyse the DFTB calculation results and print."""
        if self.para['scc'] == 'scc':
            self.analysis.dftb_energy(shift_=self.para['shift'],
                                      qatom=self.para['charge'])
        elif self.para['scc'] == 'nonscc':
            self.analysis.dftb_energy()

        # claculate physical properties
        self.analysis.sum_property()

        # print and write non-SCC DFTB results
        self.print_.print_dftb_tail()


class SCF:
    """For self-consistent field method.

    Args:
        Lperiodic: Ture or False
        scc: scc, nonscc or xlbomd
        hammat: this is H0 after SK transformations
        overm: this is overlap after SK transformations
        natom: number of atoms
        atomind: lmax of atoms (valence electrons)
        atomind2: sum of all lamx od atoms (the length of Hamiltonian matrix)
        HSsym: if write all / half of the matrices (due to symmetry)

    """

    def __init__(self, para):
        """Parameters for SCF."""
        self.para = para

        # number of atom in molecule
        self.nat = para['natom']

        # number of orbital of each atom
        self.atind = para['atomind']

        # number of total orbital number
        self.atind2 = para['atomind2']

        # H0 after SK transformation
        self.hmat = para['hammat']

        # S after SK transformation
        self.smat = para['overmat']

        # eigen solver
        self.eigen = EigenSolver(self.para)

        # analyze DFTB result
        self.analysis = Analysis(self.para)

        # electronic DFTB calculation
        self.elect = DFTBelect(self.para)

        # mixing of charge
        self.mix = Mixing(self.para)

        # print DFTB calculation information
        self.print_ = Print(self.para)

        # get electron information, such as initial charge
        self.analysis.get_qatom()

        # mixing method is simple method
        if para['mixMethod'] == 'simple':
            self.mixer = Simple(mix_param=self.para['mixFactor'])

        # mixing method is anderson
        if para['mixMethod'] == 'anderson':
            self.mixer = Anderson(mix_param=self.para['mixFactor'],
                                  init_mix_param=self.para['mixFactor'],
                                  generations=2)

        # mixing method is broyden
        if para['mixMethod'] == 'broyden':
            self.mixer = Broyden(mix_param=self.para['mixFactor'],
                                 init_mix_param=self.para['mixFactor'],
                                 generations=self.para['maxIter'])

    def scf_npe_nscc(self):
        """DFTB for non-SCC, non-perodic calculation."""
        # initial neutral charge
        self.para['qzero'] = self.para['qatom']

        # number of orbitals
        ind_nat = self.atind[self.nat]

        if self.para['HSsym'] == 'symhalf':

            # transfer H0, S from 1D to 2D
            ham = self.half_to_sym(self.hmat, ind_nat)
            over = self.half_to_sym(self.smat, ind_nat)

        elif self.para['HSsym'] == 'symall':

            # replace H0, S name for convenience
            ham, over = self.hmat, self.smat

        # get eigenvector and eigenvalue (and cholesky decomposition)
        self.para['eigenvalue'], C = self.eigen.eigen(ham, over)

        # calculate the occupation of electrons
        occupancies = self.elect.fermi(self.para['eigenvalue'])

        # build density according to occupancies and eigenvector
        C_scaled = t.sqrt(occupancies) * C
        self.para['denmat'] = C_scaled @ C_scaled.T

        # return the eigenvector
        self.para['eigenvec'] = C

        # calculate mulliken charges
        self.para['charge'] = self.elect.mulliken(
            self.para['HSsym'], over, self.para['denmat'])

    def half_to_sym(self, in_mat, dim_out):
        """Transfer 1D half H0, S to full, symmetric H0, S."""
        # build 2D full, symmetric H0 or S
        out_mat = t.zeros((dim_out, dim_out), dtype=t.float64)

        # transfer 1D to 2D
        icount = 0
        for iind in range(dim_out):
            for jind in range(iind + 1):
                out_mat[jind, iind] = out_mat[iind, jind] = in_mat[icount]
                icount += 1
        return out_mat

    def scf_npe_scc(self):
        """SCF for non-periodic-ML system with scc.

        atomind is the number of atom, for C, lmax is 2, therefore
        we need 2**2 orbitals (s, px, py, pz), then define atomind2
        """
        # todo: using __slots__ to help with speed and memory instead of dict
        # get the dimension of 2D H0, S
        ind_nat = self.atind[self.nat]

        # max iteration
        maxiter = self.para['maxIter']

        # define convergence list
        conv_ = []

        if self.para['HSsym'] == 'symhalf':

            # transfer H0, S from 1D to 2D
            ham = self.half_to_sym(self.hmat, ind_nat)
            over = self.half_to_sym(self.smat, ind_nat)

        elif self.para['HSsym'] == 'symall':

            # replace H0, S name for convenience
            ham, over = self.hmat, self.smat

        # choose density basis type: Gaussian profile
        if self.para['scc_den_basis'] == 'gaussian':
            gmat = self.elect._gamma_gaussian(self.para['this_U'],
                                              self.para['coor'][:, 1:])

        # expexponential normalized spherical charge density
        elif self.para['scc_den_basis'] == 'exp_spher':
            gmat = self.elect.gmatrix()

        # Warning the next line will creates linked references
        self.para['qzero'] = qzero = self.para['qatom']
        denmat = []

        q_mixed = qzero.clone()
        for iiter in range(maxiter):
            # calculate the sum of gamma * delta_q, the 1st cycle is zero

            # The "shift_" term is the a product of the gamma and dQ values
            shift_ = (q_mixed - qzero) @ gmat

            # "n_orbitals" should be a system constant which should not be
            # defined here.
            n_orbitals = t.tensor(np.diff(self.atind))
            shiftorb_ = shift_.repeat_interleave(n_orbitals)

            # To get the Fock matrix "F"; Construct the gamma matrix "G" then
            # H0 + 0.5 * S * G. Note: the unsqueeze axis should be made into a
            # relative value for true vectorisation. shift_mat is precomputed
            # to make the code easier to understand, however it will be removed
            # later in the development process to save on memory allocation.
            shift_mat = t.unsqueeze(shiftorb_, 1) + shiftorb_
            F = ham + 0.5 * over * shift_mat

            # Calculate the eigen-values & vectors via a Cholesky decomposition
            epsilon, C = self.eigen.eigen(F, over)

            # Calculate the occupation of electrons via the fermi method
            occupancies = self.elect.fermi(epsilon)

            # To get the density matrix "rho":
            #   1) Scale C by occupancies, which remain in the domain ∈[0, 2]
            #       rather than being remapped to [0, 1].
            #   2) Multiply the scaled coefficients by their transpose.
            C_scaled = t.sqrt(occupancies) * C
            rho = C_scaled @ C_scaled.T

            # Housekeeping functions:
            # Append the density matrix to "denmat", this is needed by MBD at
            # least.
            denmat.append(rho)

            # calculate mulliken charges
            q_new = self.elect.mulliken(self.para['HSsym'], over, rho)

            # Last mixed charge is the current step now
            q_mixed = self.mixer(q_new, q_mixed)

            if self.para['convergenceType'] == 'energy':

                # get energy: E0 + E_coul
                conv_.append(epsilon @ occupancies +
                             0.5 * shift_ @ (q_mixed + qzero))

                # print energy information
                diff_ = self.print_.print_energy(iiter, conv_)

            # use charge as convergence condition
            elif self.para['convergenceType'] == 'charge':
                conv_.append(q_mixed)
                diff_ = self.print_.print_charge(iiter, conv_, self.nat)

            # if reached convergence
            if self.convergence(iiter, maxiter, diff_):
                break

        # return eigenvalue and charge
        self.para['eigenvalue'], self.para['charge'] = epsilon, q_mixed

        # return density matrix
        self.para['denmat'] = denmat

        # return the eigenvector
        self.para['eigenvec'] = C

        # return the final shift
        self.para['shift'] = shift_

    def scf_pe_scc(self):
        """SCF for periodic."""
        pass

    def scf_npe_xlbomd(self):
        """SCF for non-periodic-ML system with scc."""
        gmat = self.elect.gmatrix()

        energy = 0
        self.para['qzero'] = qzero = self.para['qatom']
        qatom_xlbomd = self.para['qatom_xlbomd']
        ind_nat = self.atind[self.nat]

        # calculate the sum of gamma * delta_q, the 1st cycle is zero
        denmat_, qatom_ = t.zeros(self.atind2), t.zeros(self.nat)
        shift_, shiftorb_ = t.zeros(self.nat), t.zeros(ind_nat)
        occ_, work_ = t.zeros(ind_nat), t.zeros(ind_nat)

        shift_ = self.elect.shifthamgam(self.para, qatom_xlbomd, qzero, gmat)
        for iat in range(0, self.nat):
            for jind in range(self.atind[iat], self.atind[iat + 1]):
                shiftorb_[jind] = shift_[iat]

        icount = 0
        if self.para['HSsym'] == 'symall':
            eigm_ = t.zeros((ind_nat, ind_nat), dtype=t.float64)
            for iind in range(0, ind_nat):
                for j_i in range(0, ind_nat):
                    eigm_[iind, j_i] = self.hmat[iind, j_i] + 0.5 * \
                        self.smat[iind, j_i] * \
                        (shiftorb_[iind] + shiftorb_[j_i])
                    icount += 1
            oldsmat_ = self.hmat
        elif self.para['HSsym'] == 'symhalf':
            fockmat_ = t.zeros(self.atind2)
            eigm_ = t.zeros((ind_nat, ind_nat), dtype=t.float64)
            oldsmat_ = t.zeros((ind_nat, ind_nat), dtype=t.float64)
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
        eigval_, eigm_ch = self.eigen.eigen(eigm_, oldsmat_)

        # calculate the occupation of electrons
        occ_ = self.elect.fermi(eigval_)
        for iind in range(0, int(self.atind[self.nat])):
            if occ_[iind] > self.para['general_tol']:
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
        qatom_ = self.elect.mulliken(self.para['HSsym'], self.smat, denmat_)
        ecoul = 0.0
        for i in range(0, self.nat):
            ecoul = ecoul + shift_[i] * (qatom_[i] + qzero[i])
        energy = energy - 0.5 * ecoul

        # print and write non-SCC DFTB results
        self.para['eigenvalue'], self.para['charge'] = eigval_, qatom_
        self.para['denmat'] = denmat_

    def convergence(self, iiter, maxiter, diff_):
        """Convergence for SCC loops."""
        # use energy as convergence condition
        if self.para['convergenceType'] == 'energy':
            if abs(diff_) < self.para['convergence_tol']:
                reach_convergence = True

            # read max iterations, end DFTB calculation
            elif iiter + 1 >= maxiter:
                if abs(diff_) > self.para['convergence_tol']:
                    print('Warning: SCF donot reach required convergence')
                    reach_convergence = True
            else:
                reach_convergence = False

        # use charge as convergence condition
        elif self.para['convergenceType'] == 'charge':
            if abs(diff_) < self.para['convergence_tol']:
                reach_convergence = True

            # read max iterations, end DFTB calculation
            elif iiter + 1 >= maxiter:
                if abs(diff_) > self.para['convergence_tol']:
                    print('Warning: SCF donot reach required convergence')
                    reach_convergence = True
            else:
                reach_convergence = False
        return reach_convergence


class Repulsive():
    """Calculate repulsive for DFTB."""

    def __init__(self, para):
        """Initialize parameters."""
        self.para = para
        self.nat = self.para['natom']
        self.get_rep_para()
        self.cal_rep_energy()

    def get_rep_para(self):
        """Get neighbour number, usually in solid."""
        Periodic(self.para).get_neighbour(cutoff='repulsive')

    def cal_rep_energy(self):
        """Calculate repulsive energy."""
        self.rep_energy = t.zeros((self.nat), dtype=t.float64)
        atomnameall = self.para['atomnameall']
        for iat in range(0, self.nat):
            for jat in range(iat + 1, self.nat):
                nameij = atomnameall[iat] + atomnameall[jat]
                cutoff_ = self.para['cutoff_rep' + nameij]
                distanceij = self.para['distance'][iat, jat]
                if distanceij < cutoff_:
                    ienergy = self.cal_rep_atomij(distanceij, nameij)
                    self.rep_energy[iat] = self.rep_energy[iat] + ienergy
        sum_energy = t.sum(self.rep_energy[:])
        self.para['rep_energy'] = sum_energy

    def cal_rep_atomij(self, distanceij, nameij):
        """Calculate repulsive polynomials for atom pair."""
        nint = self.para['nint_rep' + nameij]
        alldist = t.zeros((nint + 1), dtype=t.float64)
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
    """Mixing method."""

    def __init__(self, para):
        """Initialize parameters."""
        self.para = para

        # max iteration loop
        self.nit = self.para['maxIter']

        # initialize broyden mixing parameters
        if self.para['mixMethod'] == 'broyden':

            # global diufference of charge difference
            self.df = []

            self.uu = []

            # weight parameter in broyden method
            self.ww = t.zeros((self.nit), dtype=t.float64)

            # global a parameter for broyden method
            self.aa = t.zeros((self.nit, self.nit), dtype=t.float64)

            # global c parameter for broyden method
            self.cc = t.zeros((self.nit, self.nit), dtype=t.float64)

            # global beta parameter for broyden method
            self.beta = t.zeros((self.nit, self.nit), dtype=t.float64)

    def mix(self, iiter, qzero, qatom, qmix, qdiff):
        """Deal with the first iteration."""
        if iiter == 0:
            qmix.append(qzero)
            if self.para['mixMethod'] == 'broyden':
                self.df.append(t.zeros((self.para['natom']), dtype=t.float64))
                self.uu.append(t.zeros((self.para['natom']), dtype=t.float64))
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
        self.para['charge'] = qatom

    def simple_mix(self, oldqatom, qatom, qdiff):
        """Simple mixing method."""
        mixf = self.para['mixFactor']
        qdiff.append(qatom - oldqatom)
        qmix_ = oldqatom + mixf * qdiff[-1]
        return qmix_

    def anderson_mix(self, iiter, qmix, qatom, qdiff):
        """Anderson mixing method."""
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
        """Broyden mixing method.

        Reference:
            D. D. Johnson, PRB, 38 (18), 1988.

        """
        # cc = t.zeros((iiter, iiter), dtype=t.float64)

        # temporal a parameter for current interation
        aa_ = []

        # temporal c parameter for current interation
        cc_ = []

        weightfrac = 1e-2

        omega0 = 1e-2

        # max weight
        maxweight = 1E5

        # min weight
        minweight = 1.0

        alpha = self.para['mixFactor']

        # get temporal parameter of last interation: <dF|dF>
        ww_ = t.sqrt(qdiff[-1] @ qdiff[-1])

        # build omega (ww) according to charge difference
        # if weight from last loop is larger than: weightfrac / maxweight
        if ww_ > weightfrac / maxweight:
            self.ww[iiter - 1] = weightfrac / ww_

        # here the gradient may break
        else:
            self.ww[iiter - 1] = maxweight

        # if weight is smaller than: minweight
        if self.ww[iiter - 1] < minweight:

            # here the gradient may break
            self.ww[iiter - 1] = minweight

        # get updated charge difference
        qdiff.append(qatom_ - qmix[-1])

        # temporal (latest) difference of charge difference
        df_ = qdiff[-1] - qdiff[-2]

        # get normalized difference of charge difference
        ndf = 1 / t.sqrt(df_ @ df_) * df_

        if iiter >= 2:
            # build loop from first loop to last loop
            [aa_.append(t.tensor(idf, dtype=t.float64) @ ndf) for idf in
             self.df[:-1]]

            # build loop from first loop to last loop
            [cc_.append(t.tensor(idf, dtype=t.float64) @ t.tensor(qdiff[-1],
             dtype=t.float64)) for idf in self.df[:-1]]

            # update last a parameter
            self.aa[: iiter - 1, iiter] = t.tensor(aa_, dtype=t.float64)
            self.aa[iiter, : iiter - 1] = t.tensor(aa_, dtype=t.float64)

            # update last c parameter
            self.cc[: iiter - 1, iiter] = t.tensor(cc_, dtype=t.float64)
            self.cc[iiter, : iiter - 1] = t.tensor(cc_, dtype=t.float64)

        self.aa[iiter - 1, iiter - 1] = 1.0

        # update last c parameter
        self.cc[iiter - 1, iiter - 1] = self.ww[iiter - 1] * (ndf @ qdiff[-1])

        for ii in range(iiter):
            self.beta[:iiter, ii] = self.ww[:iiter] * self.ww[ii] * \
                self.aa[:iiter, ii]

            self.beta[ii, ii] = self.beta[ii, ii] + omega0 ** 2

        self.beta[: iiter, : iiter] = t.inverse(self.beta[: iiter, : iiter])

        gamma = t.mm(self.cc[: iiter, : iiter], self.beta[: iiter, : iiter])

        # add difference of charge difference
        self.df.append(ndf)

        df = alpha * ndf + 1 / t.sqrt(df_ @ df_) * (qmix[-1] - qmix[-2])

        qmix_ = qmix[-1] + alpha * qdiff[-1]

        for ii in range(iiter):
            qmix_ = qmix_ - self.ww[ii] * gamma[0, ii] * self.uu[ii]

        qmix_ = qmix_ - self.ww[iiter - 1] * gamma[0, iiter - 1] * df

        self.uu.append(df)
        return qmix_


class Analysis:
    """Analysis of DFTB results, processing DFTB results."""

    def __init__(self, para):
        """Initialize parameters."""
        self.para = para
        self.nat = self.para['natom']

    def dftb_energy(self, shift_=None, qatom=None):
        """Get energy for DFTB with electronic and eigen results."""
        # get eigenvalue
        eigval = self.para['eigenvalue']

        # get occupancy
        occ = self.para['occ']

        # non-SCC DFTB energy
        if self.para['scc'] == 'nonscc':

            # product of occupancy and eigenvalue
            self.para['H0_energy'] = eigval @ occ

            # add up energy
            if self.para['Lrepulsive']:
                self.para['energy'] = self.para['H0_energy'] + \
                    self.para['rep_energy']
            else:
                self.para['energy'] = self.para['H0_energy']

        # SCC energy
        if self.para['scc'] == 'scc':
            qzero = self.para['qzero']

            # product of occupancy and eigenvalue
            self.para['H0_energy'] = eigval @ occ

            # get Coulomb energy
            self.para['coul_energy'] = shift_ @ (qatom + qzero) / 2

            # self.para
            self.para['electronic_energy'] = self.para['H0_energy'] - \
                self.para['coul_energy']

            # add up energy
            if self.para['Lrepulsive']:
                self.para['energy'] = self.para['electronic_energy'] + \
                    self.para['rep_energy']

    def sum_property(self):
        """Get alternative DFTB results."""
        nocc = self.para['nocc']
        eigval = self.para['eigenvalue']
        qzero, qatom = self.para['qzero'], self.para['charge']

        # get HOMO-LUMO, not orbital resolved
        self.para['homo_lumo'] = eigval[int(nocc) - 1:int(nocc) + 1] * \
            self.para['AUEV']

        # calculate dipole
        self.para['dipole'] = self.get_dipole(qzero, qatom)

        # calculate MBD-DFTB
        if self.para['LMBD_DFTB']:
            MBD(self.para)

        # calculate PDOS or not
        if self.para['Lpdos']:
            self.pdos()

    def get_qatom(self):
        """Get the basic electronic information of each atom."""
        # name of all atoms
        name = self.para['atomnameall']

        # define initial charge
        qat = []

        # get each intial atom charge
        [qat.append(self.para['val_' + name[iat]]) for iat in range(self.nat)]

        # sum up all atom charge
        self.para['nelectrons'] = sum(qat)

        # return charge information
        self.para['qatom'] = t.Tensor(qat).double()

    def get_dipole(self, qzero, qatom):
        """Read and process dipole data."""
        coor = self.para['coor']
        dipole = t.zeros((3), dtype=t.float64)
        for iatom in range(0, self.nat):
            if type(coor[iatom][:]) is list:
                coor_t = t.from_numpy(np.asarray(coor[iatom][1:]))
                dipole[:] = dipole[:] + (qzero[iatom] - qatom[iatom]) * coor_t
            else:
                dipole[:] = dipole[:] + (qzero[iatom] - qatom[iatom]) * \
                    coor[iatom][1:]
        return dipole

    def pdos(self):
        """Calculate PDOS."""
        # E: define for testing
        self.para['pdos_E'] = t.linspace(-20, 20, 1000, dtype=t.float64)
        # print("self.para['eigenvec']", self.para['eigenvec'])
        # print("self.para['overmat']", self.para['overmat'])
        # print("self.para['eigenvalue']", self.para['eigenvalue'])

        # calculate pdos
        self.para['pdos'] = dos.PDoS(
            # C eigen vector
            self.para['eigenvec'].T,

            # overlap
            self.para['overmat'],

            # PDOS energy
            self.para['pdos_E'],

            # eigenvalue, use eV
            self.para['eigenvalue'] * self.para['AUEV'],

            # gaussian smearing
            sigma=1E-1)
