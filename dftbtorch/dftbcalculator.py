"""DFTB calculator.

implement pytorch to DFTB
"""
import numpy as np
import torch as t
import bisect
from dftbtorch.sk import SKTran, GetSKTable, GetSK_, skt
import common.maths as maths
from dftbtorch.electront import DFTBelect
from dftbtorch.periodic import Periodic
from dftbtorch.electrons import fermi
from dftbtorch.properties import mulliken
import dftbtorch.parameters as parameters
import dftbtorch.parser as parser
from IO.systems import System
from dftbtorch.mixer import Anderson
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
        if self.parameter['precision'] in (t.float64, t.float32):
            t.set_default_dtype(self.parameter['precision'])  # cpu device
        else:
            raise ValueError('device is cpu, please select float64 or float32')

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
            self.sktran = SKTran(self.parameter, self.dataset, self.skf, self.ml)
            # deal with SK transformation
            for ib in range(self.dataset['nbatch']):
                # SK transformations
                self.sktran(ib)
                # SKTran(self.parameter, self.dataset, self.skf, self.ml, ib)

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
            scf.scf_npe_scc(self.ibatch)

    def run_repulsive(self):
        """Calculate repulsive term."""
        if self.para['Lrepulsive']:
            Repulsive(self.para, self.dataset, self.skf)

    def run_analysis(self):
        """Analyse the DFTB calculation results and print."""
        self.analysis.dftb_energy(shift_=self.para['shift'],
                                  qatom=self.para['charge'])

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

        self.nb = self.dataset['nfile']

        self.mask = [[True] * self.nb]  # mask for convergence in batch

        # batch calculations for multi systems
        self.ham = self.skf['hammat_']
        self.over = self.skf['overmat_']

        # number of atom in molecule
        self.nat = self.dataset['natomAll']

        # number of orbital of each atom
        self.atind = self.dataset['atomind']

        # the name of all atoms
        self.atomname = self.dataset['symbols']

        # analyze DFTB result
        self.analysis = Analysis(self.para, self.dataset, self.skf)

        # electronic DFTB calculation
        self.elect = DFTBelect(self.para, self.dataset, self.skf)

        # print DFTB calculation information
        self.print_ = Print(self.para, self.dataset, self.skf)

    def scf_npe_scc(self, ibatch=[0]):
        """SCF for non-periodic-ML system with scc.

        atomind is the number of atom, for C, lmax is 2, therefore
        we need 2**2 orbitals (s, px, py, pz), then define atomind2
        """
        # todo: using __slots__ to help with speed and memory instead of dict
        # max iteration
        self.atind = pad1d([self.atind[ii] for ii in ibatch])
        self.nat_ = [self.nat[ii] for ii in ibatch]
        maxiter = self.para['maxIteration']

        self.Lmask = self.para['dynamicSCC']

        # set initial reachConvergence as False
        self.para['reachConvergence'] = False

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
        self.mixer = Anderson(qatom, return_convergence=True)

        # qatom here is 2D, add up the along the rows
        nelectron = qatom.sum(axis=1)
        self.para['qzero'] = qzero = qatom
        q_mixed = qzero.clone()
        for iiter in range(maxiter):
            shift_ = t.stack(
                [(im - iz) @ ig for im, iz, ig in zip(
                    q_mixed[self.mask[-1]], qzero[self.mask[-1]], gmat[self.mask[-1]])])

            # repeat shift according to number of orbitals
            shiftorb_ = pad1d([ishif.repeat_interleave(iorb) for iorb, ishif in
                              zip(self.atind[self.mask[-1]], shift_)])
            shift_mat = t.stack([t.unsqueeze(ishift, 1) + ishift
                                 for ishift in shiftorb_])

            # To get the Fock matrix "fock"; Construct the gamma matrix "G" then
            dim_ = shift_mat.shape[-1]   # the new dimension of max orbitals
            fock = self.ham[self.mask[-1]][:, :dim_, :dim_] + \
                0.5 * self.over[self.mask[-1]][:, :dim_, :dim_] * shift_mat

            # Calculate the eigen-values & vectors via a Cholesky decomposition
            epsilon, C = maths.eighb(fock, self.over[self.mask[-1]][:, :dim_, :dim_])
            occ, nocc = fermi(epsilon, nelectron[self.mask[-1]])

            # build density according to occ and eigenvector
            C_scaled = t.sqrt(occ).unsqueeze(1).expand_as(C) * C

            # batch calculation of density, normal code: C_scaled @ C_scaled.T
            rho = t.matmul(C_scaled, C_scaled.transpose(1, 2))

            # calculate mulliken charges for each system in batch
            q_new = mulliken(self.over[self.mask[-1], :dim_, :dim_],
                             rho, self.atind[self.mask[-1]])

            # Last mixed charge is the current step now
            q_mixed[self.mask[-1]], conv = self.mixer(q_new)

            self.mask.append(~conv)
            if conv.all():
                self.para['reachConvergence'] = True
                break

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
        self.para['eigenvalue'], self.para['eigenvec'] = maths.eighb(fock, self.over)

        # return occupied states
        self.para['occ'], self.para['nocc'] = \
            self.elect.fermi(self.para['eigenvalue'], nelectron, self.para['tElec'])

        # return density matrix
        C_scaled = t.sqrt(self.para['occ']).unsqueeze(1).expand_as(
            self.para['eigenvec']) * self.para['eigenvec']
        self.para['denmat'] = t.matmul(C_scaled, C_scaled.transpose(1, 2))


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
