"""DFTB calculator.

implement pytorch to DFTB
"""
import numpy as np
import torch as t
import bisect
from dftbtorch.sk import SKTran, GetSKTable, GetSK_, skt
from dftbtorch.electront import DFTBelect
import IO.readt as readt
from dftbtorch.periodic import Periodic
from dftbtorch.matht import EigenSolver
import dftbtorch.parameters as parameters
import dftbtorch.parser as parser
from IO.systems import System
from DFTBMaLT.dftbmalt.dftb.mixer import Simple, Anderson  # Broyden
from dftbtorch.mbd import MBD
from IO.write import Print
import DFTBMaLT.dftbmalt.dftb.dos as dos
from ml.padding import pad1d, pad2d
import dftbtorch.initparams as initpara
_CORE = {"H": 0., "C": 2., "N": 2., "O": 2.}


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
        # initialize parameters
        self.init = Initialization(parameter, dataset, skf, ml)

        # return DFTB parameters and skf data
        self.initialization()

        # update parameter, dataset, skf if init is None
        self.parameter = self.init.parameter
        self.dataset = self.init.dataset
        self.skf = self.init.skf
        self.ml = self.init.ml

        # run DFTB calculations
        self.run_dftb()

    def initialization(self):
        """Initialize DFTB, geometric, skf, dataset, ML parametes."""
        self.init.initialize_parameter()
        self.init.initialize_dftb()

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

    def run_dftb(self):
        """Run DFTB code."""
        if not self.parameter['Lbatch']:
            Rundftbpy(self.parameter, self.dataset, self.skf)
        else:
            Rundftbpy(self.parameter, self.dataset, self.skf, self.dataset['nfile'])

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

    def __init__(self, parameter, dataset=None, skf=None, ml=None):
        """Interface for different applications."""
        # get the constant DFTB parameters for DFTB
        self.parameter = parameters.constant_parameter(parameter)
        self.parameter = parser.parser_cmd_args(self.parameter)
        self.dataset = [dataset, {}][dataset is None]
        self.skf = [skf, {}][skf is None]
        self.ml = [ml, {}][ml is None]

    def initialize_parameter(self):
        # get DFTB calculation parameters dictionary
        self.parameter = initpara.dftb_parameter(self.parameter)

        # set the default tesnor dtype
        if self.parameter['device'] == 'cpu':
            if self.parameter['precision'] in (t.float64, t.float32):
                t.set_default_dtype(self.parameter['precision'])  # cpu device
            else:
                raise ValueError('device is cpu, please select float64 or float32')
        elif self.parameter['device'] == 'cuda':
            if self.parameter['precision'] in (t.cuda.DoubleTensor, t.cuda.FloatTensor):
                t.set_default_tensor_type(self.parameter['precision'])  # gpu device
            else:
                raise ValueError('device is cuda, please select cuda.FloatTensor or cuda.DoubleTensor')
        else:
            raise ValueError('device only support cpu and cuda')
 
        # get SKF parameters dictionary
        self.skf = initpara.skf_parameter(self.parameter, self.skf)

        # get dataset parameters dictionary
        self.dataset = initpara.init_dataset(self.dataset)

    def initialize_dftb(self):
        # get geometric, systematic information
        if type(self.dataset['numbers'][0]) is list:
            self.dataset['numbers'] = pad1d([t.tensor(ii) for ii in self.dataset['numbers']])

        self.sys = System(self.dataset['numbers'], self.dataset['positions'])
        self.dataset['positions'] = self.sys.positions
        self.dataset['positions_vec'] = self.sys.get_positions_vec()
        self.dataset['distances'] = self.sys.distances
        self.dataset['symbols'] = self.sys.symbols
        self.dataset['natomAll'] = self.sys.size_system
        self.dataset['lmaxall'] = self.sys.get_l_numbers()
        self.dataset['atomind'], self.dataset['atomindcumsum'], self.dataset['norbital'] = \
            self.sys.get_accumulated_orbital_numbers()
        self.dataset['globalSpecies'] = self.sys.get_global_species()
        self.specie_res, self.l_res = self.sys.get_resolved_orbital()

        # check all parameters before interpolation of integrals
        self.pre_check()

        # get Hubbert for each if use gaussian density basis
        if self.parameter['densityProfile'] == 'gaussian':
            self.get_this_hubbert()

        # Get SK table from normal skf, or hdf according to input parameters
        # GetSK(self.parameter, self.dataset, self.skf, self.ml)
        if self.skf['ReadSKType'] == 'normal':
            self.skf = GetSKTable.read(self.parameter['directorySK'],
                                       self.dataset['globalSpecies'],
                                       orbresolve=self.skf['LOrbitalResolve'],
                                       skf=self.skf)
            # deal with SK transformation
            for ib in range(self.dataset['nbatch']):
                # SK transformations
                SKTran(self.parameter, self.dataset, self.skf, self.ml, ib)

        elif self.skf['ReadSKType'] == 'mask':
            from IO.system import Basis, Bases
            from dftbtorch.sk import SKIntegralGenerator
            max_l_key = {1: 0, 6: 1, 7: 1, 8: 1, 79: 2}
            skf_path = '/home/gz_fan/Documents/ML/dftb/slko/test'
            sk_integral_generator = SKIntegralGenerator.from_dir(skf_path)
            basis_info_list = [Basis(self.sys.numbers[i], max_l_key) for i in range(self.dataset['nbatch'])]
            bases_info = Bases(basis_info_list, max_l_key)
            self.skf['hammat'] = skt(self.sys, bases_info, sk_integral_generator, mat_type='H')
            self.skf['overmat'] = skt(self.sys, bases_info, sk_integral_generator, mat_type='S')
            mask = pad2d([t.eye(*iover.shape).bool() for iover in self.skf['overmat']])
            self.skf['overmat'].masked_fill_(mask, 1.)
            onsite = t.tensor([[t.flip(self.skf['onsite'+iispe+iispe], [0])[iil]
                                for iispe, iil in zip(ispe, il)]
                               for ispe, il in zip(self.specie_res, self.l_res)])
            idx = t.arange(0, len(self.skf['overmat'][0]), out=t.LongTensor())
            self.skf['hammat'][:, idx, idx] = onsite

    def pre_check(self):
        """Check every parameters used in DFTB calculations."""
        # a single system
        if not self.parameter['Lbatch']:
            # number of batch, single system will be one
            self.dataset['nbatch'] = 1

            # add 1 dimension to tensor
            if self.sys.distances.dim() == 2:
                self.sys.distances.unsqueeze_(0)
            self.ibatch = 0
        else:
            # number of batch, single system will be one
            self.dataset['nbatch'] = self.dataset['nfile']

    def get_this_hubbert(self):
        """Get Hubbert for current calculation, nou orbital resolved."""
        # create temporal Hubbert list
        this_U = []
        # only support not orbital resolved U
        if not self.parameter['Lorbres']:

            # get U hubbert (s orbital) for each atom
            [this_U.append(self.parameter['uhubb' + iname + iname][-1])
             for iname in self.parameter['atomNameAll']]

        # transfer to tensor
        self.parameter['this_U'] = t.tensor(this_U)


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

        # batch size
        if self.para['task'] in ('mlCompressionR', 'mlIntegral', 'dftb'):
            self.nb = self.dataset['nbatch']
        elif self.para['task'] in ('testCompressionR', 'testIntegral'):
            self.nb = self.dataset['ntest']

        self.mask = [[True] * self.nb]  # mask for convergence in batch

        # batch calculations for multi systems
        if self.batch:
            self.ham_ = self.skf['hammat_']
            self.over_ = self.skf['overmat_']

        # single system calculations
        else:
            ham_, over_ = self.skf['hammat'], self.skf['overmat']
            self.ham_ = ham_ if ham_.dim() == 3 else ham_.unsqueeze(0)
            self.over_ = over_ if over_.dim() == 3 else over_.unsqueeze(0)

        # transfer lower or upper H0, S to full symmetric H0, S
        ind_nat = sum(self.dataset['atomind'])  # self.dataset['norbital']
        if self.para['HSSymmetry'] == 'half':
            self.ham = self.half_to_sym(self.ham_, ind_nat)
            self.over = self.half_to_sym(self.over_, ind_nat)

        # replace H0, S name for convenience
        elif self.para['HSSymmetry'] == 'all':
            self.ham, self.over = self.ham_, self.over_

        # number of atom in molecule
        self.nat = self.dataset['natomAll']

        # number of orbital of each atom
        self.atind = self.dataset['atomind']

        # the name of all atoms
        self.atomname = self.dataset['symbols']

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

    def scf_npe_nscc(self, ibatch=[0]):
        """DFTB for non-SCC, non-perodic calculation."""
        # get electron information, such as initial charge
        qatom = self.analysis.get_qatom(self.atomname, ibatch)

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
        self.para['charge'] = pad1d([self.elect.mulliken(i, j, m, n)
            for i, j, m, n in zip(self.over, self.para['denmat'], self.atind, self.nat)])
        self.para['fullCharge'] = self.analysis.to_full_electron_charge(
            self.atomname, self.para['charge'], ibatch)

    def half_to_sym(self, in_mat, dim_out):
        """Transfer 1D half H0, S to full, symmetric H0, S."""
        # build 2D full, symmetric H0 or S
        out_mat = t.zeros(dim_out, dim_out)

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

        # set initial reachConvergence as False
        self.para['reachConvergence'] = False

        # define convergence list, append charge or energy to convergencelist
        convergencelist = []

        if self.para['densityProfile'] == 'spherical':
            gmat_ = [self.elect.gmatrix(
                self.dataset['distances'][i], self.nat[i],
                self.dataset['symbols'][i]) for i in ibatch]

            # pad a list of 2D gmat with different size
            gmat = pad2d(gmat_)

        elif self.para['scc_den_basis'] == 'gaussian':
            gmat = self.elect._gamma_gaussian(self.para['this_U'],
                                              self.para['positions'])

        qatom = self.analysis.get_qatom(self.atomname, ibatch)

        # qatom here is 2D, add up the along the rows
        nelectron = qatom.sum(axis=1)
        self.para['qzero'] = qzero = qatom

        q_mixed = qzero.clone()  # q_mixed will maintain the shape unchanged
        for iiter in range(maxiter):
            # get index of mask where is True
            ind_mask = list(np.where(np.array(self.mask[-1]) == True)[0])

            # The "shift_" term is the a product of the gamma and dQ values
            # 2D @ 3D, the normal single system code: (q_mixed - qzero) @ gmat
            # t.einsum('ij, ijk-> ik', q_mixed - qzero, gmat) is much faster,
            # but unstable: RuntimeError: Function 'BmmBackward' returned ...
            shift_ = t.stack([(q_mixed[i] - qzero[i]) @ gmat[i] for i in ind_mask])

            # repeat shift according to number of orbitals
            # for convinience, we will keep the size of shiftorb_ unchanged
            # during the dynamic SCC-DFTB batch calculations
            shiftorb_ = pad1d([
                ishif.repeat_interleave(iorb) for iorb, ishif in zip(
                    self.atind[self.mask[-1]], shift_)])
            shift_mat = t.stack([t.unsqueeze(ishift, 1) + ishift
                                 for ishift in shiftorb_])

            # To get the Fock matrix "fock"; Construct the gamma matrix "G" then
            # H0 + 0.5 * S * G. Note: the unsqueeze axis should be made into a
            # relative value for true vectorisation. shift_mat is precomputed
            # to make the code easier to understand, however it will be removed
            # later in the development process to save on memory allocation.
            dim_ = shift_mat.shape[-1]   # the new dimension of max orbitals
            fock = self.ham[self.mask[-1]][:, :dim_, :dim_] + \
                0.5 * self.over[self.mask[-1]][:, :dim_, :dim_] * shift_mat

            # Calculate the eigen-values & vectors via a Cholesky decomposition
            epsilon, C = self.eigen.eigen(
                fock, self.over[self.mask[-1]][:, :dim_, :dim_], self.batch,
                self.atind[self.mask[-1]], t.tensor(ibatch)[self.mask[-1]], inverse=self.para['inverse'])
            # Calculate the occupation of electrons via the fermi method
            occ, nocc = self.elect.fermi(epsilon, nelectron[self.mask[-1]],
                                         self.para['tElec'])

            # build density according to occ and eigenvector
            # t.stack([t.sqrt(occ[i]) * C[i] for i in range(self.nb)])
            C_scaled = t.sqrt(occ).unsqueeze(1).expand_as(C) * C

            # batch calculation of density, normal code: C_scaled @ C_scaled.T
            rho = t.matmul(C_scaled, C_scaled.transpose(1, 2))

            # calculate mulliken charges for each system in batch
            q_new = pad1d([
                self.elect.mulliken(i, j, m, n) for i, j, m, n in zip(
                    self.over[self.mask[-1]][:, :dim_, :dim_], rho, self.atind[self.mask[-1]],
                    t.tensor(self.nat)[self.mask[-1]])])

            # Last mixed charge is the current step now
            if not self.batch:
                if q_new.squeeze().dim() == 1:  # single atom
                    q_new_ = q_new[:, :self.dataset['natomAll'][ibatch[0]]]
                else:
                    q_new_ = q_new[0][: self.dataset['natomAll'][ibatch[0]]]
                q_mixed = self.mixer(q_new_.squeeze(), q_mixed.squeeze()).unsqueeze(0)
            else:
                q_mixed = self.mixer(q_new, q_mixed[self.mask[-1]], self.mask[-1])

            if self.para['convergenceType'] == 'energy':
                # get energy: E0 + E_coul
                convergencelist.append(
                    t.stack([iep @ iocc + 0.5 * ish @ (iqm + iqz)
                             for iep, iocc, ish, iqm, iqz in
                             zip(epsilon, occ, shift_, q_mixed, qzero[self.mask[-1]])]))

                # return energy difference and print energy information
                dif = self.print_.print_energy(iiter, convergencelist,
                                               self.batch, self.nb, self.mask)

            # use charge as convergence condition
            elif self.para['convergenceType'] == 'charge':
                convergencelist.append(q_mixed)
                dif = self.print_.print_charge(iiter, convergencelist, self.nat)

            # if reached convergence, and append mask
            conver_, self.mask = self.convergence(
                iiter, maxiter, dif, self.batch, self.para['convergenceTolerance'], self.mask, ind_mask)
            if conver_:
                self.para['reachConvergence'] = True
                break

            # delete the previous charge or energy to save memory
            del convergencelist[:-2]

        # return eigenvalue and charge
        self.para['charge'] = q_mixed
        self.para['fullCharge'] = self.analysis.to_full_electron_charge(
            self.atomname, self.para['charge'], ibatch)

        # return the final shift
        self.para['shift'] = t.stack([(iqm - iqz) @ igm for iqm, iqz, igm
                                      in zip(q_mixed, qzero, gmat)])
        shiftorb = pad1d([
                ishif.repeat_interleave(iorb) for iorb, ishif in zip(
                    self.atind, self.para['shift'])])
        shift_mat = t.stack([t.unsqueeze(ish, 1) + ish for ish in shiftorb])
        fock = self.ham + 0.5 * self.over * shift_mat
        self.para['eigenvalue'], self.para['eigenvec'] = \
            self.eigen.eigen(fock, self.over, self.batch, self.atind)

        # return occupied states
        self.para['occ'], self.para['nocc'] = \
            self.elect.fermi(self.para['eigenvalue'], nelectron, self.para['tElec'])

        # return density matrix
        C_scaled = t.sqrt(self.para['occ']).unsqueeze(1).expand_as(
            self.para['eigenvec']) * self.para['eigenvec']
        self.para['denmat'] = t.matmul(C_scaled, C_scaled.transpose(1, 2))


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
            eigm_ = t.zeros(ind_nat, ind_nat)
            for iind in range(0, ind_nat):
                for j_i in range(0, ind_nat):
                    eigm_[iind, j_i] = self.hmat[iind, j_i] + 0.5 * \
                        self.smat[iind, j_i] * \
                        (shiftorb_[iind] + shiftorb_[j_i])
                    icount += 1
            oldsmat_ = self.hmat
        elif self.para['HSSymmetry'] == 'half':
            fockmat_ = t.zeros(self.atind2)
            eigm_ = t.zeros(ind_nat, ind_nat)
            oldsmat_ = t.zeros(ind_nat, ind_nat)
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

    def convergence(self, iiter, maxiter, dif, batch=False, tolerance=1E-6, mask=None, ind=None):
        """Convergence for SCC loops."""
        # for multi system, the max of difference will be chosen instead
        difmax = dif.max() if batch else None

        # use energy as convergence condition
        if self.para['convergenceType'] == 'energy':
            if not batch:
                if abs(dif) < tolerance:
                    return True, mask

                # read max iterations, end DFTB calculation
                elif iiter + 1 >= maxiter:
                    if abs(dif) > tolerance:
                        print('Warning: SCF donot reach required convergence')
                        return False, mask
                # do not reach convergence and iiter < maxiter
                else:
                    return False, mask

            # batch calculations
            elif batch:
                # if mask.append(mask[-1]), all columns will change
                mask.append([True if ii is True else False for ii in mask[-1]])
                for ii, iind in enumerate(ind):
                    if abs(dif[ii]) < tolerance:  # ii is index in not convergenced
                        mask[-1][iind] = False  # iind is index in whole batch
                if abs(difmax) < tolerance:
                    print('All systems reached convergence')
                    return True, mask

                # read max iterations, end DFTB calculation
                elif iiter + 1 >= maxiter:
                    if abs(difmax) > tolerance:
                        print('Warning: SCF donot reach required convergence')
                        return False, mask
                # do not reach convergence and iiter < maxiter
                else:
                    return False, mask

        # use charge as convergence condition
        elif self.para['convergenceType'] == 'charge':
            if not batch:
                if abs(dif) < tolerance:
                    return True, mask

                # read max iterations, end DFTB calculation
                elif iiter + 1 >= maxiter:
                    if abs(dif) > tolerance:
                        print('Warning: SCF donot reach required convergence')
                        return False, mask
                else:
                    return False, mask

            # batch calculations
            elif batch:
                for ii, iind in enumerate(ind):
                    mask.append(mask[-1])
                    if abs(dif[ii]) < tolerance:  # ii ==> not convergenced
                        mask[-1][iind] = False  # iind ==> whole batch
                if abs(difmax) < tolerance:
                    print('All systems reached convergence')
                    return True, mask

                # read max iterations, end DFTB calculation
                elif iiter + 1 >= maxiter:
                    if abs(difmax) > tolerance:
                        print('Warning: SCF donot reach required convergence')
                        return False, mask
                # do not reach convergence and iiter < maxiter
                else:
                    return False, mask

class Repulsive():
    """Calculate repulsive for DFTB."""

    def __init__(self, para, dataset, skf):
        """Initialize parameters."""
        self.para = para
        self.dataset = dataset
        self.skf = skf
        self.nat = self.dataset['natomAll']
        self.get_rep_para()
        self.cal_rep_energy()

    def get_rep_para(self):
        """Get neighbour number, usually in solid."""
        Periodic(self.para).get_neighbour(cutoff='repulsive')

    def cal_rep_energy(self):
        """Calculate repulsive energy."""
        self.rep_energy = t.zeros(self.nat)
        atomnameall = self.para['atomNameAll']

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
        alldist = t.zeros(nint + 1)
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
            self.ww = t.zeros(self.nit)

            # global a parameter for broyden method
            self.aa = t.zeros(self.nit, self.nit)

            # global c parameter for broyden method
            self.cc = t.zeros(self.nit, self.nit)

            # global beta parameter for broyden method
            self.beta = t.zeros(self.nit, self.nit)

    def mix(self, iiter, qzero, qatom, qmix, qdiff):
        """Deal with the first iteration."""
        if iiter == 0:
            qmix.append(qzero)
            if self.para['mixMethod'] == 'broyden':
                self.df.append(t.zeros((self.para['natom'])))
                self.uu.append(t.zeros((self.para['natom'])))
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
            [aa_.append(idf @ ndf) for idf in  self.df[:-1]]

            # build loop from first loop to last loop
            [cc_.append(idf @ qdiff[-1]) for idf in self.df[:-1]]

            # update last a parameter
            self.aa[: iiter - 1, iiter] = aa_
            self.aa[iiter, : iiter - 1] = aa_

            # update last c parameter
            self.cc[: iiter - 1, iiter] = cc_
            self.cc[iiter, : iiter - 1] = cc_

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

            # Coulomb energy, single system code: shift_ @ (qatom + qzero) / 2
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
        self.para['homo_lumo'] = t.stack([
            ieig[int(iocc) - 1:int(iocc) + 1] * self.para['AUEV']
            for ieig, iocc in zip(eigval, nocc)])

        # calculate dipole
        self.para['dipole'] = t.stack(
            [self.get_dipole(iqz, iqa, self.dataset['positions'][i], self.nat[i])
             for iqz, iqa, i in zip(qzero, qatom, ibatch)])

        # calculate MBD-DFTB
        if self.para['LCPA']:
            MBD(self.para, self.dataset)

        # calculate PDOS or not
        if self.para['Lpdos']:
            self.pdos()

    def get_qatom(self, atomname, ibatch=[0]):
        """Get the basic electronic information of each atom."""
        # get each intial atom charge
        qat = [[self.para['val_' + atomname[ib][iat]]
                for iat in range(self.nat[ib])] for ib in ibatch]
        return pad1d([t.tensor(iq).type(self.para['precision']) for iq in qat])

    def to_full_electron_charge(self, atomname, charge, ibatch=[0]):
        """Get the basic electronic information of each atom."""
        # add core electrons
        qat = [[_CORE[atomname[ib][iat]]
                for iat in range(self.nat[ib])] for ib in ibatch]
        # return charge information
        return pad1d([t.tensor(iq) for iq in qat]) + charge

    def get_dipole(self, qzero, qatom, coor, natom):
        """Read and process dipole data."""
        return sum([(qzero[iat] - qatom[iat]) * coor[iat]
                    for iat in range(natom)])

    def pdos(self):
        """Calculate PDOS."""
        # calculate pdos
        self.para['pdos_E'] = t.linspace(5, 10, 1000)

        self.para['pdos'] = dos.PDoS(
            # C eigen vector, the 1st dimension is batch dimension
            self.para['eigenvec'].transpose(1, 2),

            # overlap
            self.skf['overmat'],

            # PDOS energy
            self.para['pdos_E'],

            # eigenvalue, use eV
            self.para['eigenvalue'] * self.para['AUEV'],

            # gaussian smearing
            sigma=1E-1)
        import matplotlib.pyplot as plt
        plt.plot(self.para['pdos_E'], self.para['pdos'].squeeze())
        print('dos', self.para['pdos'].squeeze().shape)
