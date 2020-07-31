"""Main DFTB code.

implement pytorch to DFTB
"""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch as t
import bisect
from slakot import SlaKo, SKTran
from electront import DFTBelect
from readt import ReadSlaKo, ReadInt, SkInterpolator
from periodic import Periodic
from matht import EigenSolver
import parameters
import dftbtorch.parser as parser
from mbd import MBD
GEN_PARA = {"inputfile_name": 'in.ground'}
VAL_ELEC = {"H": 1, "C": 4, "N": 5, "O": 6, "Ti": 4}
PUBPARA = {"tol": 1E-4}
ATOMNUM = {'H': 1, 'C': 6, 'N': 7, 'O': 8}


def main(para):
    """Run main DFTB code.

    Initialize parameters and then run DFTB
    """
    # read input construct these data for next DFTB calculations
    Initialization(para)

    # with all necessary data, run dftb calculation
    Rundftbpy(para)


class Initialization:
    """Initialize parameters for DFTB.

    this class aims to read input coor, calculation parameters and SK tables;
    Then with SK transformation, construct 1D Hamiltonian and overlap matrice
    for the following DFTB calculations
    """

    def __init__(self, para):
        """Interface for different applications.

        Args:
            LCmdArgs (optional): True or False
            LReadInput (optional): True or False

        """
        self.para = para
        parameters.dftb_parameter(para)
        self.slako = SlaKo(self.para)
        self.readsk = ReadSlaKo(self.para)
        self.readin = ReadInt(para)

        # step 1: whether get parameters from command line
        if 'LCmdArgs' in self.para.keys():
            if self.para['LCmdArgs']:  # define 'LCmdArgs' True
                parser.parser_cmd_args(self.para)
        elif 'LCmdArgs' not in self.para.keys():  # not define 'LCmdArgs'
            parser.parser_cmd_args(self.para)

        # step 2: if read para from dftb_in, if define yourself, set False
        if 'LReadInput' in self.para.keys():
            if self.para['LReadInput']:
                self.readin.get_task(para)
                self.readin.get_coor(para)
        elif 'LReadInput' not in self.para.keys():
            self.readin.get_task(para)
            self.readin.get_coor()

        # step 3: generate vector, distance ... from given geometry
        self.readin.cal_coor()

        # step 4: read SKF files and run SK transformation
        self.run_sk()

    def run_sk(self):
        """DFTB calculations, read integrals from .skf."""
        if not self.para['Lml']:
            if not self.para['LreadSKFinterp']:
                self.readsk.read_sk_specie()
                SKTran(self.para)
            if self.para['LreadSKFinterp']:
                self.interpskf()
        if self.para['Lml']:
            if self.para['Lml_skf'] and self.para['LreadSKFinterp']:
                # replace local specie with global specie
                self.para['atomspecie'] = self.para['specie_all']
                self.interpskf()
            elif self.para['Lml_acsf'] and self.para['LreadSKFinterp']:
                # replace local specie with global specie
                self.para['atomspecie'] = self.para['specie_all']
                self.interpskf()

    def form_sk_spline(self):
        """Use SK table data to build spline interpolation."""
        self.readsk.read_sk_specie()
        self.slako.get_sk_spldata()

    def interpskf(self):
        """Read .skf data from skgen with various compR."""
        print('** read skf file with all compR **')
        if self.para['typeSKinterpR'] == 'wavefunction':
            nametail = '_wav'
        elif self.para['typeSKinterpR'] == 'density':
            nametail = '_den'
        elif self.para['typeSKinterpR'] == 'all':
            nametail = '_all'
        for namei in self.para['atomspecie']:
            for namej in self.para['atomspecie']:
                if ATOMNUM[namei] <= ATOMNUM[namej]:  # this is just nanestyle
                    dire = self.para['dire_interpSK'] + '/' + namei + \
                        '_' + namej + nametail
                    SkInterpolator(self.para, gridmesh=0.2).readskffile(
                        namei, namej, dire)
                else:
                    dire = self.para['dire_interpSK'] + '/' + namej + \
                        '_' + namei + nametail
                    SkInterpolator(self.para, gridmesh=0.2).readskffile(
                        namei, namej, dire)


class Rundftbpy:
    """Run DFTB according to the Initialized parameters.

    DFTB task:
        solid or molecule
        SCC, non-SCC or XLBOMD

    """

    def __init__(self, para):
        """Run DFTB with multi interface.

        you can read from .skf file, or interpolate integrals from a
        list of .skf files, or directly offer integrals
        """
        self.para = para
        self.cal_repulsive()
        self.rundftbplus()
        # self.sum_dftb()

    def rundftbplus(self):
        """Run DFTB with multi interface.

        solid or molecule
        SCC, non-SCC or XLBOMD
        """
        scf = SCF(self.para)
        Print(self.para).print_scf_title()

        if self.para['Lperiodic']:
            if self.para['scc'] == 'scc':
                scf.scf_pe_scc()
        else:
            if self.para['scc'] == 'nonscc':
                scf.scf_npe_nscc()
            elif self.para['scc'] == 'scc':
                if self.para['HSsym'] == 'symhalf':
                    scf.scf_npe_scc()
                elif self.para['HSsym'] in ['symall', 'symall_chol']:
                    scf.scf_npe_scc_symall()
            elif self.para['scc'] == 'xlbomd':
                scf.scf_npe_xlbomd()

    def cal_repulsive(self):
        """Run DFTB repulsive part."""
        if self.para['Lrepulsive']:
            Repulsive(self.para)

    def sum_dftb(self):
        """Make a summary of DFTB."""
        Analysis(self.para).dftb_energy()


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
        self.eigen = EigenSolver(self.para)
        self.nat = para['natom']
        self.atind = para['atomind']
        self.atind2 = para['atomind2']
        self.hmat = para['hammat']
        self.smat = para['overmat']

    def scf_npe_nscc(self):
        """Atomind is the number of atom, for C, lmax is 2, therefore.

        we need 2**2 orbitals (s, px, py, pz), then define atomind2
        """
        analysis, elect = Analysis(self.para), DFTBelect(self.para)

        analysis.get_qatom()
        print_ = Print(self.para)
        self.para['qzero'], ind_nat = self.para['qatom'], self.atind[self.nat]

        icount = 0
        if self.para['HSsym'] in ['symall', 'symall_chol']:
            eigm, overm = self.hmat, self.smat
        elif self.para['HSsym'] == 'symhalf':
            eigm = t.zeros((ind_nat, ind_nat), dtype=t.float64)
            overm = t.zeros((ind_nat, ind_nat), dtype=t.float64)
            for iind in range(ind_nat):
                for jind in range(iind + 1):
                    eigm[jind, iind] = self.hmat[icount]
                    overm[jind, iind] = self.smat[icount]
                    eigm[iind, jind] = self.hmat[icount]
                    overm[iind, jind] = self.smat[icount]
                    icount += 1

        # get eigenvector and eigenvalue (and cholesky decomposition)
        eigval_ch, eigm_ch = self.eigen.eigen(eigm, overm)

        # calculate the occupation of electrons
        energy = 0
        occ = elect.fermi(eigval_ch)
        for iind in range(0, int(self.atind[self.nat])):
            if occ[iind] > self.para['general_tol']:
                energy = energy + occ[iind] * eigval_ch[iind]

        # density matrix, work controls the unoccupied eigm as 0!!
        work = t.sqrt(occ)
        for jind in range(0, ind_nat):  # n = no. of occupied orbitals
            for iind in range(0, ind_nat):
                eigm_ch[iind, jind] = eigm_ch[iind, jind] * work[jind]
        denmat = t.mm(eigm_ch, eigm_ch.t())

        # calculate mulliken charges
        if self.para['HSsym'] == 'symhalf':
            denmat_ = t.zeros((self.atind2), dtype=t.float64)
            for iind in range(0, ind_nat):
                for j_i in range(0, iind + 1):
                    inum = int(iind * (iind + 1) / 2 + j_i)
                    denmat_[inum] = denmat[j_i, iind]
            qatom = elect.mulliken(self.para['HSsym'], self.smat, denmat_)
        elif self.para['HSsym'] in ['symall', 'symall_chol']:
            qatom = elect.mulliken(self.para['HSsym'], self.smat, denmat)

        # print and write non-SCC DFTB results
        self.para['eigenvalue'], self.para['qatomall'] = eigval_ch, qatom
        self.para['denmat'] = denmat
        analysis.dftb_energy()
        analysis.sum_property()
        print_.print_dftb_caltail()

    def scf_npe_scc(self):
        """SCF for non-periodic-ML system with scc.

        atomind is the number of atom, for C, lmax is 2, therefore
        we need 2**2 orbitals (s, px, py, pz), then define atomind2
        """
        elect = DFTBelect(self.para)
        mix = Mixing(self.para)
        elect = DFTBelect(self.para)
        analysis = Analysis(self.para)

        print_ = Print(self.para)
        maxiter = self.para['maxIter']
        analysis.get_qatom()
        gmat = elect.gmatrix()

        energy = t.zeros((maxiter), dtype=t.float64)
        self.para['qzero'] = qzero = self.para['qatom']
        eigm, eigval, qatom, qmix, qdiff = [], [], [], [], []
        denmat, denmat_2d = [], []
        ind_nat = self.atind[self.nat]

        for iiter in range(0, maxiter):
            # calculate the sum of gamma * delta_q, the 1st cycle is zero
            eigm_ = t.zeros((ind_nat, ind_nat), dtype=t.float64)
            oldsmat_ = t.zeros((ind_nat, ind_nat), dtype=t.float64)
            denmat_ = t.zeros((self.atind2), dtype=t.float64)
            qatom_ = t.zeros((self.nat), dtype=t.float64)
            fockmat_ = t.zeros((self.atind2), dtype=t.float64)
            shift_ = t.zeros((self.nat), dtype=t.float64)
            shiftorb_ = t.zeros((ind_nat), dtype=t.float64)
            occ_ = t.zeros((ind_nat), dtype=t.float64)
            work_ = t.zeros((ind_nat), dtype=t.float64)

            if iiter > 0:
                shift_ = elect.shifthamgam(self.para, qmix[-1], qzero, gmat)
            for iat in range(0, self.nat):
                for jind in range(self.atind[iat], self.atind[iat + 1]):
                    shiftorb_[jind] = shift_[iat]

            # Hamiltonian = H0 + H2, where
            # H2 = 0.5 * sum(overlap * (gamma_IK + gamma_JK))
            icount = 0
            for iind in range(int(self.atind[self.nat])):
                for j_i in range(iind + 1):
                    fockmat_[icount] = self.hmat[icount] + 0.5 * \
                        self.smat[icount] * (shiftorb_[iind] + shiftorb_[j_i])
                    #icount += 1

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
            eigval_, eigm_ch = self.eigen.eigen(eigm_, oldsmat_)
            eigval.append(eigval_), eigm.append(eigm_ch)

            # calculate the occupation of electrons
            occ_ = elect.fermi(eigval_)
            for iind in range(0, int(self.atind[self.nat])):
                if occ_[iind] > self.para['general_tol']:
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
            self.dE = print_.print_energy(iiter, energy)
            reach_convergence = self.convergence(iiter, maxiter, qdiff)
            if reach_convergence:
                break

        # print and write non-SCC DFTB results
        self.para['eigenvalue'], self.para['qatomall'] = eigval_, qatom[-1]
        self.para['denmat'] = denmat
        analysis.sum_property()
        print_.print_dftb_caltail()

    def scf_npe_scc_symall(self):
        """SCF for non-periodic-ML system with scc.

        atomind is the number of atom, for C, lmax is 2, therefore
        we need 2**2 orbitals (s, px, py, pz), then define atomind2
        """
        """
        Will need to clean up what should be a class instance and what should
        be a parameter. A fixed data structure will need to be decided on.
        """
        # A lot of information here is being stored to a dictionary, it is likely
        # more efficient to assign these to a class using __slots__ to help with
        # speed and memory. This might help compartmentalise the code a little more.
        from mixer import Anderson
        mixer = Anderson(mix_param=0.2, init_mix_param=0.2, generations=2)

        # Looks as if we are assuming there is not orbital resolved DFTB
        # Primary SCC function
        elect = DFTBelect(self.para)
        mix = Mixing(self.para)
        elect = DFTBelect(self.para)
        analysis = Analysis(self.para)

        print_ = Print(self.para)
        maxiter = self.para['maxIter']
        analysis.get_qatom()
        gmat = elect.gmatrix().double()

        energy = t.zeros((maxiter), dtype=t.float64)
        # Warning the next line will creates linked references, is this intended
        self.para['qzero'] = qzero = self.para['qatom']
        #eigm, eigval, qatom, qmix, qdiff, denmat = [], [], [], [], [], []
        denmat = []
        ind_nat = self.atind[self.nat]
        # print('hamt:', self.hmat)

        q_mixed = qzero.clone()
        for iiter in range(self.para['maxIter']):
            # calculate the sum of gamma * delta_q, the 1st cycle is zero


            # Should move the atomic -> orbital expansion to an external function
            # to avoid code recitation. We can call this even when we have no
            # charge fluctuations yet.
            # The "shift_" term is the a product of the gamma and dQ values
            #shift_2 = (q_mixed - qzero) @ gmat.double()
            shift_ = elect.shifthamgam(self.para, q_mixed, qzero, gmat)

            # "n_orbitals" should be a system constant which should not be
            # defined here.
            n_orbitals = t.tensor(np.diff(self.atind))
            shiftorb_ = shift_.repeat_interleave(n_orbitals)

            pause = 10
            # To get the Fock matrix "F"; Construct the gamma matrix "G" then
            # H0 + 0.5 * S * G. Note: the unsqueeze axis should be made into a
            # relative value for true vectorisation. shift_mat is precomputed to
            # make the code easier to understand, however it will be removed later
            # in the development process to save on memory allocation.
            shift_mat = t.unsqueeze(shiftorb_, 1) + shiftorb_
            F = self.hmat + 0.5 * self.smat * shift_mat

            # Calculate the eigen-values & vectors via a Cholesky decomposition
            epsilon, C = self.eigen.eigen(F, self.smat)

            # Calculate the occupation of electrons via the fermi method
            occupancies = elect.fermi(epsilon)

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
            q_new = elect.mulliken(self.para['HSsym'], self.smat[:], rho)
            # Last mixed charge is the current step now
            q_mixed = mixer(q_new, q_mixed)

            # This is needed for "analysis" we really don't want this in a loop
            self.para['eigenvalue'], self.para['shift_'] = epsilon, shift_
            #self.para['eigenvalue'] = epsilon
            # This should be done outside of the SCC loop, we really want to avoid
            # frequent dictionary calls. Why is this the unmixed charge? Why not
            # pass as options to the function that needs it?
            self.para['qatom_'] = q_new

            # if reached convergence
            analysis.dftb_energy()
            energy[iiter] = self.para['energy']
            self.dE = print_.print_energy(iiter, energy)
            if self.convergence(iiter, maxiter, q_mixed-q_new):
                break

            # General notes:
            #   1) Will need to introduce dynamic error handling




        # print and write non-SCC DFTB results
        self.para['eigenvalue'], self.para['qatomall'] = epsilon, q_mixed
        self.para['denmat'] = denmat
        analysis.sum_property()
        print_.print_dftb_caltail()

    def scf_pe_scc(self):
        """SCF for periodic."""
        pass

    def scf_npe_xlbomd(self):
        """SCF for non-periodic-ML system with scc."""
        elect = DFTBelect(self.para)
        gmat = elect.gmatrix()
        elect = DFTBelect(self.para)
        analysis = Analysis(self.para)
        print_ = Print(self.para)
        analysis.get_qatom()

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
        occ_ = elect.fermi(eigval_)
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
        qatom_ = elect.mulliken(self.para['HSsym'], self.smat, denmat_)
        ecoul = 0.0
        for i in range(0, self.nat):
            ecoul = ecoul + shift_[i] * (qatom_[i] + qzero[i])
        energy = energy - 0.5 * ecoul

        # print and write non-SCC DFTB results
        self.para['eigenvalue'], self.para['qatomall'] = eigval_, qatom_
        self.para['denmat'] = denmat_
        analysis.sum_property(), print_.print_dftb_caltail()

    def convergence(self, iiter, maxiter, qdiff):
        """Convergence for SCC loops."""
        if self.para['convergenceType'] == 'energy':
            if abs(self.dE) < self.para['energy_tol']:
                reach_convergence = True
            elif iiter + 1 >= maxiter:
                if abs(self.dE) > self.para['energy_tol']:
                    print('Warning: SCF donot reach required convergence')
                    reach_convergence = True
            else:
                reach_convergence = False
        elif self.para['convergenceType'] == 'charge':
            qdiff_ = t.sum(qdiff[-1]) / len(qdiff[-1])
            if abs(qdiff_) < self.para['charge_tol']:
                reach_convergence = True
            elif iiter + 1 >= maxiter:
                if abs(qdiff_) > PUBPARA['tol']:
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
        self.rep_energy = t.zeros((self.nat), dtype=t.float64)
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
        if self.para['mixMethod'] == 'broyden':
            self.df, self.uu = [], []
            self.ww = t.zeros((self.para['maxIter']), dtype=t.float64)

    def mix(self, iiter, qzero, qatom, qmix, qdiff):
        """Call different mixing method."""
        """
        There is code here to deal with the zeroth iteration yet it is not used.
        If possible each type of mixer should be in a separate class all of
        which should inherent from a an abstract base meta class.   
        """
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
        self.para['qatomall'] = qatom

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
        """Broyden mixing method."""
        aa = t.zeros((iiter, iiter), dtype=t.float64)
        cc = t.zeros((iiter, iiter), dtype=t.float64)
        beta = t.zeros((iiter, iiter), dtype=t.float64)
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
    """Write DFTB results."""

    def __init__(self, para):
        """Initialize parameters."""
        self.para = para

    def write(self):
        pass


class Print:
    """Print DFTB results."""

    def __init__(self, para):
        """Initialize parameters."""
        self.para = para

    def print_scf_title(self):
        """Print DFTB type."""
        if not self.para['Lperiodic'] and self.para['scc'] == 'nonscc':
            print('*' * 35, 'Non-periodic Non-SCC-DFTB', '*' * 35)
        elif not self.para['Lperiodic'] and self.para['scc'] == 'scc':
            print('*' * 35, 'Non-periodic SCC-DFTB', '*' * 35)
        elif not self.para['Lperiodic'] and self.para['scc'] == 'xlbomd':
            print('*' * 35, 'Non-periodic xlbomd-DFTB', '*' * 35)
        elif self.para['Lperiodic'] and self.para['scc'] == 'scc':
            print('*' * 35, 'Periodic SCC-DFTB', '*' * 35)

    def print_energy(self, iiter, energy):
        """Print energy for SCC loops."""
        if iiter == 0:
            dE = energy[iiter].detach()
            print('iteration', ' '*8, 'energy', ' '*20, 'dE')
            print(f'{iiter:5} {energy[iiter].detach():25}', f'{dE:25}')
            return dE
        elif iiter >= 1:
            dE = energy[iiter].detach() - energy[iiter - 1].detach()
            print(f'{iiter:5} {energy[iiter].detach():25}', f'{dE:25}')
            return dE

    def print_dftb_caltail(self):
        """Print DFTB results."""
        t.set_printoptions(precision=10)
        print('charge (e): \n', self.para['qatomall'].detach())
        print('dipole (eAng): \n', self.para['dipole'].detach())
        print('energy (Hartree): \n', self.para['energy'].detach())
        print('TS energy (Hartree): \n', self.para['H0_energy'].detach())
        if self.para['LMBD_DFTB']:
            print('CPA: \n', self.para['cpa'].detach())
            print('polarizability: \n', self.para['alpha_mbd'].detach())
        if self.para['scc'] == 'scc':
            print('Coulomb energy (Hartree): \n',
                  -self.para['coul_energy'].detach())
        if self.para['Lrepulsive']:
            print('repulsive energy (Hartree): \n',
                  self.para['rep_energy'].detach())


class Analysis:
    """Analysis of DFTB results, processing DFTB results."""

    def __init__(self, para):
        """Initialize parameters."""
        self.para = para
        self.nat = self.para['natom']

    def dftb_energy(self):
        """Get energy for DFTB with electronic and eigen results."""
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
            # qatom_[i] + qzero[i] ? qatom_[i] + qqatom_[j]?
            # look up later (AJM)
            for i in range(0, self.nat):
                ecoul = ecoul + shift_[i] * (qatom_[i] + qzero[i])
            self.para['coul_energy'] = ecoul / 2.0
            if self.para['Lrepulsive']:
                self.para['energy'] = self.para['H0_energy'] + \
                    self.para['rep_energy'] - self.para['coul_energy']
            else:
                self.para['energy'] = self.para['H0_energy'] + \
                    self.para['coul_energy']

    def sum_property(self):
        """Get alternative DFTB results."""
        nocc = self.para['nocc']
        eigval = self.para['eigenvalue']
        qzero, qatom = self.para['qzero'], self.para['qatomall']
        self.para['homo_lumo'] = eigval[int(nocc) - 1:int(nocc) + 1] * \
            self.para['AUEV']
        self.para['dipole'] = self.get_dipole(qzero, qatom)
        if self.para['LMBD_DFTB']:
            MBD(self.para)

    def get_qatom(self):
        """Get the basic electronic information of each atom."""
        atomname = self.para['atomnameall']
        num_electrons = 0
        qatom = t.zeros((self.nat), dtype=t.float64)
        for i in range(0, self.nat):
            qatom[i] = VAL_ELEC[atomname[i]]
            num_electrons += qatom[i]
        self.para['qatom'] = qatom
        self.para['nelectrons'] = num_electrons

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
