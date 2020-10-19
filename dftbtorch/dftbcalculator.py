"""DFTB calculator.

implement pytorch to DFTB
"""
import numpy as np
import torch as t
import bisect
from dftbtorch.slakot import SKTran
from dftbtorch.electront import DFTBelect
import IO.readt as readt
from dftbtorch.periodic import Periodic
from dftbtorch.matht import EigenSolver
import dftbtorch.parameters as parameters
import dftbtorch.parser as parser
from DFTBMaLT.dftbmalt.dftb.mixer import Simple, Anderson  # Broyden
from dftbtorch.mbd import MBD
from IO.write import Print
import DFTBMaLT.dftbmalt.dftb.dos as dos
from ml.padding import pad1d, pad2d
import dftbtorch.init_parameter as initpara


class DFTBCalculator:
    """DFTB calculator."""

    def __init__(self, parameter=None, dataset=None, skf=None, ml=None):
        """Collect data, run DFTB calculations and return results.

        Args:
            parameter (dict, optional): a general dictionary which includes
                DFTB parameters, general environment parameters.
            dataset (dict, optional): dataset, geomreic parameters.
            skf (dict, optional): Slater-Koster parameters.
            ml (dict, optional): machine learning parameters.

        Returns:
            result: DFTB calculation results.

        """
        # define general DFTB parameters dictionary
        self.parameter = [parameter, {}][parameter is None]

        # define dataset and geometric dictionary
        self.dataset = [dataset, {}][dataset is None]

        # define slater-koster dictionary
        self.skf = [skf, {}][skf is None]

        # define machine learning dictionary, this is optional for DFTB
        self.ml = ml

        # return hamiltonian, overlap from input parameters and skf data
        self.initialization()

        # run DFTB calculations
        self.run_dftb()

    def initialization(self):
        """Initialize DFTB, geometric, skf, dataset, ML parametes."""
        init = Initialization(self.parameter, self.dataset, self.skf, self.ml)
        init.initialization_dftb()

    def run_dftb(self):
        """Run DFTB code."""
        Rundftbpy(self.parameter, self.dataset, self.skf)

    def get_result(self):
        pass


class Initialization:
    """Initialize parameters for DFTB.

    This class aims to read input parameters, coordinates, and SK integrals.
    Then with SK transformation, get 3D Hamiltonian and overlap matrices, the
    extra dimension is designed for batch calculations.
    Reading input parameters section: firstly it will run constant_parameter,
    dftb_parameter, skf_parameter, init_dataset, init_ml (optional). After
    running the above functions, you will get default parameters. If you run
    code by command line and define a input file named 'dftb_in', the code will
    automatically read dftb_in and replace the default values. If you do not
    want to read dftb_in, set LReadInput as False.

    """
    def __init__(self, parameter, dataset=None, skf=None, ml=None, Lreadskf=True):
        """Interface for different applications."""
        self.Lreadskf = Lreadskf
        # get the constant DFTB parameters for DFTB
        self.parameter = parameters.constant_parameter(parameter)

        # get DFTB calculation parameters dictionary
        self.parameter = initpara.dftb_parameter(self.parameter)

        # get SKF parameters dictionary
        self.skf = initpara.skf_parameter(skf)

        # get dataset parameters dictionary
        self.dataset = initpara.init_dataset(dataset)

        # get machine learning parameters dictionary, optional
        self.ml = ml

    def initialization_dftb(self):

        # return/update DFTB, geometric, skf parameters from input
        readt.ReadInput(self.parameter, self.dataset, self.skf)

        # check all parameters before interpolation of integrals and
        # DFTB calculations
        self.pre_check()

        # read skf fike from normal skf or list of skf files
        if self.Lreadskf:
            self.read_sk()

        # get Hubbert for each if use gaussian density basis
        if self.parameter['densityProfile'] == 'gaussian':
            self.get_this_hubbert()

        # deal with SK transformation
        for ib in range(self.parameter['nbatch']):
            self.skf = self.run_sk(ib)

    def pre_check(self):
        """Check every parameters used in DFTB calculations."""
        # a single system
        if not self.parameter['Lbatch']:

            # number of batch, single system will be one
            self.parameter['nbatch'] = 1

            # add 1 dimension to tensor
            if self.dataset['distance'].dim() == 2:
                self.dataset['distance'] = self.dataset['distance'].unsqueeze(0)
            if self.dataset['coordinate'].dim() == 2:
                self.dataset['coordinate'] = self.dataset['coordinate'].unsqueeze(0)
            self.ibatch = 0
        else:
            # number of batch, single system will be one
            self.parameter['nbatch'] = self.parameter['nfile']

    def read_sk(self):
        """Read integrals and perform SK transformations.

        Read integrals from .skf
        direct offer integral
        get integrals by interpolation from a list of skf files.

        """
        # do not perform machine learning
        if not self.parameter['Lml']:

            # get integral from directly reading normal skf file
            if not self.dataset['LSKFinterpolation']:

                # read normal skf file by atom specie
                readt.ReadSlaKo(self.parameter,
                                self.dataset, self.skf,
                                self.ibatch).read_sk_specie()

            # get integral from a list of skf file (compr) by interpolation
            if self.dataset['LSKFinterpolation']:

                # read all corresponding skf files by defined dftb_parametermeters
                readt.interpskf(self.parameter,
                                self.parameter['typeSKinterpR'],
                                self.parameter['atomspecie'],
                                self.parameter['dire_interpSK'])

        # perform machine learning
        # read a list of skf files with various compression radius
        elif self.parameter['Lml'] and self.dataset['LSKFinterpolation']:

            # ML variables is skf parameters (compression radius)
            # read all corresponding skf files (different compression radius)
            if self.parameter['Lml_skf']:

                # replace local specie with global specie
                self.parameter['atomspecie'] = self.parameter['specie_all']

                # read all corresponding skf files by defined parameters
                readt.interpskf(self.parameter,
                                self.parameter['typeSKinterpR'],
                                self.parameter['atomspecie'],
                                self.parameter['dire_interpSK'])

            # ML variables is ACSF parameters
            # read all corresponding skf files (different compression radius)
            elif self.parameter['Lml_acsf']:

                # replace local specie with global specie
                self.parameter['atomspecie'] = self.parameter['specie_all']

                # read all corresponding skf files by defined parameters
                readt.interpskf(self.parameter,
                                self.parameter['typeSKinterpR'],
                                self.parameter['atomspecie'],
                                self.parameter['dire_interpSK'])

            # get integrals directly from spline interpolation
            elif self.parameter['Lml_HS']:

                self.readsk.read_sk_specie()
                # SKspline(self.parameter).get_sk_spldata()

    def get_this_hubbert(self):
        """Get Hubbert for current calculation, nou orbital resolved."""
        # create temporal Hubbert list
        this_U = []
        # only support not orbital resolved U
        if not self.parameter['Lorbres']:

            # get U hubbert (s orbital) for each atom
            [this_U.append(self.parameter['uhubb' + iname + iname][-1])
             for iname in self.parameter['atomnameall']]

        # transfer to tensor
        self.parameter['this_U'] = t.tensor(this_U, dtype=t.float64)

    def run_sk(self, ibatch=None):
        """Read integrals and perform SK transformations.

        Read integrals from .skf
        direct offer integral
        get integrals by interpolation from a list of skf files.

        """
        # do not perform machine learning
        if not self.parameter['Lml']:

            # get integral from directly reading skf file
            if not self.dataset['LSKFinterpolation']:

                # SK transformations
                SKTran(self.parameter, self.dataset, self.skf, self.ml, ibatch)
        return self.skf


class Rundftbpy:
    """Run DFTB according to the Initialized parameters.

    DFTB task:
        solid or molecule;
        SCC, non-SCC or XLBOMD;
    """

    def __init__(self, para, dataset, skf, ibatch=None):
        """Run (SCC-) DFTB."""
        self.para = para
        self.dataset = dataset
        self.skf = skf

        # for single system
        if not self.para['Lbatch']:

            # single calculation, do not define ibatch, ibatch is always 0
            if ibatch is None:
                self.ibatch = [0]

            # this is for single calculation in batch, the nth calculation
            else:
                self.ibatch = [ibatch]

        # batch calculation, return a list
        elif self.para['Lbatch']:
            self.ibatch = [ib for ib in range(ibatch)]

        # analyze DFTB result
        self.analysis = Analysis(self.para, self.dataset, self.skf)

        # print DFTB calculation information
        self.print_ = Print(self.para, self.dataset, self.skf)

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
        scf = SCF(self.para, self.dataset, self.skf)

        # calculate solid
        if self.para['Lperiodic']:

            # SCC-DFTB
            if self.para['scc'] == 'scc':
                scf.scf_pe_scc()

        # calculate molecule
        else:

            # non-SCC-DFTB
            if self.para['scc'] == 'nonscc':
                scf.scf_npe_nscc(self.ibatch)

            # SCC-DFTB
            elif self.para['scc'] == 'scc':
                scf.scf_npe_scc(self.ibatch)

            # XLBOMD-DFTB
            elif self.para['scc'] == 'xlbomd':
                scf.scf_npe_xlbomd(self.ibatch)

    def run_repulsive(self):
        """Calculate repulsive term."""
        if self.para['Lrepulsive']:
            Repulsive(self.para, self.dataset, self.skf)

    def run_analysis(self):
        """Analyse the DFTB calculation results and print."""
        if self.para['scc'] == 'scc':
            self.analysis.dftb_energy(shift_=self.para['shift'],
                                      qatom=self.para['charge'])
        elif self.para['scc'] == 'nonscc':
            self.analysis.dftb_energy()

        # claculate physical properties
        self.analysis.sum_property(self.ibatch)

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
        HSSymmetry: if write all / half of the matrices (due to symmetry)

    """

    def __init__(self, para, dataset, skf):
        """Parameters for SCF."""
        self.para = para
        self.dataset = dataset
        self.skf = skf

        # single systems
        self.batch = self.para['Lbatch']
        if not self.batch:

            # H0 after SK transformation
            self.hmat = self.skf['hammat'].unsqueeze(0)

            # S after SK transformation
            self.smat = self.skf['overmat'].unsqueeze(0)

        # multi systems
        elif self.batch:

            # H0 after SK transformation
            self.hmat = self.skf['hammat_']

            # S after SK transformation
            self.smat = self.skf['overmat_']

        self.nb = self.para['nbatch']

        # number of atom in molecule
        self.nat = self.dataset['natomAll']

        # number of orbital of each atom
        self.atind = self.dataset['atomind']

        # number of total orbital number if flatten H, S into 1D
        self.atind2 = self.dataset['atomind2']

        # the name of all atoms
        self.atomname = self.dataset['atomnameall']

        # eigen solver
        self.eigen = EigenSolver(self.para['eigenMethod'])

        # analyze DFTB result
        self.analysis = Analysis(self.para, self.dataset, self.skf)

        # electronic DFTB calculation
        self.elect = DFTBelect(self.para, self.dataset, self.skf)

        # mixing of charge
        self.mix = Mixing(self.para)

        # print DFTB calculation information
        self.print_ = Print(self.para, self.dataset, self.skf)

        # mixing method is simple method
        if para['mixMethod'] == 'simple':
            self.mixer = Simple(mix_param=self.para['mixFactor'])

        # mixing method is anderson
        elif para['mixMethod'] == 'anderson':
            self.mixer = Anderson(mix_param=self.para['mixFactor'],
                                  init_mix_param=self.para['mixFactor'],
                                  generations=2)

        # mixing method is broyden
        elif para['mixMethod'] == 'broyden':
            self.mixer = Broyden(mix_param=self.para['mixFactor'],
                                 init_mix_param=self.para['mixFactor'],
                                 generations=self.para['maxIter'])

        # get the dimension of 2D H0, S
        ind_nat = self.dataset['norbital']

        if self.para['HSSymmetry'] == 'half':

            # transfer H0, S from 1D to 2D
            self.ham = self.half_to_sym(self.hmat, ind_nat)
            self.over = self.half_to_sym(self.smat, ind_nat)

        elif self.para['HSSymmetry'] == 'all':

            # replace H0, S name for convenience
            self.ham, self.over = self.hmat, self.smat

    def scf_npe_nscc(self, ibatch=0):
        """DFTB for non-SCC, non-perodic calculation."""
        # get electron information, such as initial charge
        qatom = self.analysis.get_qatom(self.atomname, ibatch)
        # qatom = t.stack(
        #    [self.analysis.get_qatom() for i in range(self.nb)])

        # qatom here is 2D, add up the along the rows
        nelectron = qatom.sum(axis=1)

        # initial neutral charge
        self.para['qzero'] = qatom

        # get eigenvector and eigenvalue (and cholesky decomposition)
        epsilon, C = self.eigen.eigen(self.ham, self.over, self.batch, self.atind, ibatch)

        # batch calculation of the occupation of electrons
        occ, nocc = self.elect.fermi(epsilon, nelectron, self.para['tElec'])

        # build density according to occ and eigenvector
        # ==> t.stack([t.sqrt(occ[i]) * C[i] for i in range(self.nb)])
        C_scaled = t.sqrt(occ).unsqueeze(1).expand_as(C) * C

        # batch calculation of density, normal code is: C_scaled @ C_scaled.T
        self.para['denmat'] = t.matmul(C_scaled, C_scaled.transpose(1, 2))

        # return the eigenvector, eigenvalue
        self.para['eigenvec'] = C
        self.para['eigenvalue'] = epsilon
        self.para['occ'] = occ
        self.para['nocc'] = nocc

        # calculate mulliken charges
        '''self.para['charge'] = pad1d([self.elect.mulliken(
            self.over[i], self.para['denmat'][i], self.atind[i], self.nat[i])
            for i in range(self.nb)])'''
        self.para['charge'] = pad1d([self.elect.mulliken(i, j, m, n)
            for i, j, m, n in zip(self.over, self.para['denmat'], self.atind, self.nat)])

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

    def scf_npe_scc(self, ibatch=[0]):
        """SCF for non-periodic-ML system with scc.

        atomind is the number of atom, for C, lmax is 2, therefore
        we need 2**2 orbitals (s, px, py, pz), then define atomind2
        """
        # todo: using __slots__ to help with speed and memory instead of dict
        # max iteration
        maxiter = self.para['maxIteration']

        # define convergence list, append charge or energy to convergencelist
        convergencelist = []

        if self.para['densityProfile'] == 'spherical':
            gmat_ = [self.elect.gmatrix(
                self.dataset['distance'][i], self.nat[i],
                self.dataset['atomnameall'][i]) for i in ibatch]

            # pad a list of 2D gmat with different size
            gmat = pad2d(gmat_)

        elif self.para['scc_den_basis'] == 'gaussian':
            gmat = self.elect._gamma_gaussian(self.para['this_U'],
                                              self.para['coordinate'])

        qatom = self.analysis.get_qatom(self.atomname, ibatch)
        # qatom = t.stack(
        #    [self.analysis.get_qatom() for i in range(self.nb)])

        # qatom here is 2D, add up the along the rows
        nelectron = qatom.sum(axis=1)
        self.para['qzero'] = qzero = qatom

        q_mixed = qzero.clone()
        self.para['reach_convergence'] = False
        for iiter in range(maxiter):

            # The "shift_" term is the a product of the gamma and dQ values
            # 2D @ 3D, the normal single system code: (q_mixed - qzero) @ gmat
            # unstable: RuntimeError: Function 'BmmBackward' returned ...
            # shift_ = t.einsum('ij, ijk-> ik', q_mixed - qzero, gmat)
            shift_ = t.stack([(q_mixed[i] - qzero[i]) @ gmat[i] for i in range(self.nb)])

            # "n_orbitals" should be a system constant which should not be
            # defined here.
            '''n_orbitals = pad1d([t.tensor(np.diff(self.atind[i]))
                                for i in range(self.nb)])
            shiftorb_ = pad1d([shift_[i].repeat_interleave(n_orbitals[i])
                               for i in range(self.nb)])
            shift_mat = t.stack([t.unsqueeze(shiftorb_[i], 1) + shiftorb_[i]
                                 for i in range(self.nb)])'''
            n_orbitals = pad1d([t.tensor(np.diff(self.atind[i]))
                                for i in ibatch])
            shiftorb_ = pad1d([ishif.repeat_interleave(iorb)
                               for iorb, ishif in zip(n_orbitals, shift_)])
            shift_mat = t.stack([t.unsqueeze(ishift, 1) + ishift
                                 for ishift in shiftorb_])

            # To get the Fock matrix "F"; Construct the gamma matrix "G" then
            # H0 + 0.5 * S * G. Note: the unsqueeze axis should be made into a
            # relative value for true vectorisation. shift_mat is precomputed
            # to make the code easier to understand, however it will be removed
            # later in the development process to save on memory allocation.
            F = self.ham + 0.5 * self.over * shift_mat

            # Calculate the eigen-values & vectors via a Cholesky decomposition
            epsilon, C = self.eigen.eigen(F, self.over, self.batch, self.atind, ibatch)

            # Calculate the occupation of electrons via the fermi method
            occ, nocc = self.elect.fermi(epsilon, nelectron, self.para['tElec'])

            # build density according to occ and eigenvector
            # t.stack([t.sqrt(occ[i]) * C[i] for i in range(self.nb)])
            C_scaled = t.sqrt(occ).unsqueeze(1).expand_as(C) * C

            # batch calculation of density, normal code: C_scaled @ C_scaled.T
            rho = t.matmul(C_scaled, C_scaled.transpose(1, 2))

            # calculate mulliken charges for each system in batch
            q_new = pad1d([self.elect.mulliken(i, j, m, n)
                           for i, j, m, n in zip(self.over, rho, self.atind, self.nat)])

            # Last mixed charge is the current step now
            if not self.batch:
                natom = self.dataset['natomAll'][ibatch[0]]
                q_new_ = q_new[0][: natom]
                q_mixed = self.mixer(q_new_.squeeze(), q_mixed.squeeze()).unsqueeze(0)
            else:
                q_mixed = self.mixer(q_new, q_mixed)

            if self.para['convergenceType'] == 'energy':

                # get energy: E0 + E_coul
                convergencelist.append(
                    t.stack([iep @ iocc + 0.5 * ish @ (iqm + iqz)
                             for iep, iocc, ish, iqm, iqz in
                             zip(epsilon, occ, shift_, q_mixed, qzero)]))

                # print energy information
                dif = self.print_.print_energy(iiter, convergencelist,
                                               self.batch, self.nb)

            # use charge as convergence condition
            elif self.para['convergenceType'] == 'charge':
                convergencelist.append(q_mixed)
                dif = self.print_.print_charge(iiter, convergencelist, self.nat)

            # if reached convergence
            if self.convergence(iiter, maxiter, dif, self.batch,
                                self.para['convergenceTolerance']):
                self.para['reach_convergence'] = True
                break

            # delete the previous charge or energy to save memory
            del convergencelist[:-2]

        # return eigenvalue and charge
        self.para['eigenvalue'], self.para['charge'] = epsilon, q_mixed

        # return density matrix
        self.para['denmat'] = rho

        # return the eigenvector
        self.para['eigenvec'] = C

        # return the final shift
        self.para['shift'] = shift_
        self.para['occ'] = occ
        self.para['nocc'] = nocc

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
        if self.para['HSSymmetry'] == 'all':
            eigm_ = t.zeros((ind_nat, ind_nat), dtype=t.float64)
            for iind in range(0, ind_nat):
                for j_i in range(0, ind_nat):
                    eigm_[iind, j_i] = self.hmat[iind, j_i] + 0.5 * \
                        self.smat[iind, j_i] * \
                        (shiftorb_[iind] + shiftorb_[j_i])
                    icount += 1
            oldsmat_ = self.hmat
        elif self.para['HSSymmetry'] == 'half':
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
        qatom_ = self.elect.mulliken(self.para['HSSymmetry'], self.smat, denmat_)
        ecoul = 0.0
        for i in range(0, self.nat):
            ecoul = ecoul + shift_[i] * (qatom_[i] + qzero[i])
        energy = energy - 0.5 * ecoul

        # print and write non-SCC DFTB results
        self.para['eigenvalue'], self.para['charge'] = eigval_, qatom_
        self.para['denmat'] = denmat_

    def convergence(self, iiter, maxiter, dif, batch=False, tolerance=1E-6):
        """Convergence for SCC loops."""
        # for multi system, the max of difference will be chosen instead
        if batch:
            dif = dif.max()

        # use energy as convergence condition
        if self.para['convergenceType'] == 'energy':

            if abs(dif) < tolerance:
                return True

            # read max iterations, end DFTB calculation
            elif iiter + 1 >= maxiter:
                if abs(dif) > tolerance:
                    print('Warning: SCF donot reach required convergence')
                    return False

            # do not reach convergence and iiter < maxiter
            else:
                return False

        # use charge as convergence condition
        elif self.para['convergenceType'] == 'charge':
            if abs(dif) < tolerance:
                return True

            # read max iterations, end DFTB calculation
            elif iiter + 1 >= maxiter:
                if abs(dif) > tolerance:
                    print('Warning: SCF donot reach required convergence')
                    return True
            else:
                return False


class Repulsive():
    """Calculate repulsive for DFTB."""

    def __init__(self, para, dataset, skf):
        """Initialize parameters."""
        self.para = para
        self.dataset = dataset
        self.skf = skf
        self.nat = self.dataset['natomall']
        self.get_rep_para()
        self.cal_rep_energy()

    def get_rep_para(self):
        """Get neighbour number, usually in solid."""
        Periodic(self.para).get_neighbour(cutoff='repulsive')

    def cal_rep_energy(self):
        """Calculate repulsive energy."""
        self.rep_energy = t.zeros((self.nat), dtype=t.float64)
        atomnameall = self.para['atomnameall']

        # repulsive cutoff not atom specie resolved
        if not self.para['cutoff_atom_resolve']:
            cutoff_ = self.para['cutoff_rep']

        for iat in range(self.nat):
            for jat in range(iat + 1, self.nat):
                nameij = atomnameall[iat] + atomnameall[jat]

                # repulsive cutoff atom specie resolved
                if self.para['cutoff_atom_resolve']:
                    cutoff_ = self.para['cutoff_rep' + nameij]

                # compare distance and cutoff
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
        self.nit = self.para['maxIteration']

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

    def __init__(self, para, dataset, skf):
        """Initialize parameters."""
        self.para = para
        self.dataset = dataset
        self.skf = skf

        # number of atom in batch
        self.nat = self.dataset['natomAll']

    def dftb_energy(self, shift_=None, qatom=None):
        """Get energy for DFTB with electronic and eigen results."""
        # get eigenvalue
        eigval = self.para['eigenvalue']

        # get occupancy
        occ = self.para['occ']

        # product of occupancy and eigenvalue
        self.para['H0_energy'] = t.stack(
            [eigval[i] @ occ[i] for i in range(eigval.shape[0])])

        # non-SCC DFTB energy
        if self.para['scc'] == 'nonscc':
            self.para['electronic_energy'] = self.para['H0_energy']

        # SCC energy
        if self.para['scc'] == 'scc':
            qzero = self.para['qzero']

            # get Coulomb energy, normal code: shift_ @ (qatom + qzero) / 2
            self.para['coul_energy'] = t.bmm(
                shift_.unsqueeze(1), (qatom + qzero).unsqueeze(2)).squeeze() / 2

            # self.para
            self.para['electronic_energy'] = self.para['H0_energy'] - \
                self.para['coul_energy']

        # add repulsive energy
        if self.para['Lrepulsive']:
            self.para['energy'] = self.para['electronic_energy'] + \
                self.para['rep_energy']

        # total energy without repulsive energy
        else:
            self.para['energy'] = self.para['electronic_energy']

    def sum_property(self, ibatch=None):
        """Get alternative DFTB results."""
        nocc = self.para['nocc']
        eigval = self.para['eigenvalue']
        qzero, qatom = self.para['qzero'], self.para['charge']

        # get HOMO-LUMO, not orbital resolved
        '''self.para['homo_lumo'] = t.stack([
            eigval[i][int(nocc[i]) - 1:int(nocc[i]) + 1] * self.para['AUEV']
            for i in range(self.para['nbatch'])])'''
        self.para['homo_lumo'] = t.stack([
            ieig[int(iocc) - 1:int(iocc) + 1] * self.para['AUEV']
            for ieig, iocc in zip(eigval, nocc)])

        # calculate dipole
        self.para['dipole'] = t.stack(
            [self.get_dipole(iqz, iqa, self.dataset['coordinate'][i], self.nat[i])
             for iqz, iqa, i in zip(qzero, qatom, ibatch)])

        # calculate MBD-DFTB
        if self.para['LMBD_DFTB']:
            MBD(self.para, self.dataset)

        # calculate PDOS or not
        if self.para['Lpdos']:
            self.pdos()

    def get_qatom(self, atomname, batch):
        """Get the basic electronic information of each atom."""
        # get each intial atom charge
        qat = [[self.para['val_' + atomname[ib][iat]]
                for iat in range(self.nat[ib])] for ib in batch]

        # return charge information
        return pad1d([t.tensor(iq, dtype=t.float64) for iq in qat])

    def get_dipole(self, qzero, qatom, coor, natom):
        """Read and process dipole data."""
        dipole = t.zeros((3), dtype=t.float64)
        for iatom in range(natom):
            if type(coor[iatom][:]) is list:
                coor_t = t.from_numpy(np.asarray(coor[iatom]))
                dipole[:] = dipole[:] + (qzero[iatom] - qatom[iatom]) * coor_t
            else:
                dipole[:] = dipole[:] + (qzero[iatom] - qatom[iatom]) * \
                    coor[iatom]
        return dipole

    def pdos(self):
        """Calculate PDOS."""
        # calculate pdos
        self.para['pdos_E'] = t.linspace(-20, 20, 1000, dtype=t.float64)

        self.para['pdos'] = dos.PDoS(
            # C eigen vector, the 1st dimension is batch dimension
            self.para['eigenvec'].transpose(1, 2),

            # overlap
            self.para['overmat'],

            # PDOS energy
            self.para['pdos_E'],

            # eigenvalue, use eV
            self.para['eigenvalue'] * self.para['AUEV'],

            # gaussian smearing
            sigma=1E-1)
