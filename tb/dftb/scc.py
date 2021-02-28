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
from dftbtorch.mixer import Anderson
from dftbtorch.mbd import MBD
from tb.electrons import Gamma
from IO.write import Print
import DFTBMaLT.dftbmalt.dftb.dos as dos
from common.batch import pack
import dftbtorch.initparams as initpara
_CORE = {"H": 0., "C": 2., "N": 2., "O": 2.}


class Initialization:
    """Initialize parameters for DFTB."""

    def __init__(self, parameter, skf=None, ml=None):
        """Interface for different applications."""
        # get the constant DFTB parameters for DFTB
        self.parameter = parameters.constant_parameter(parameter)
        self.skf = [skf, {}][skf is None]
        self.ml = [ml, {}][ml is None]

    def initialize_parameter(self):
        # get DFTB calculation parameters dictionary
        self.parameter = initpara.dftb_parameter(self.parameter)

        # get SKF parameters dictionary
        self.skf = initpara.skf_parameter(self.parameter, self.skf)


class Rundftbpy:
    """Run DFTB according to the Initialized parameters.

    DFTB task:
        solid or molecule;
        SCC, non-SCC or XLBOMD;
    """

    def __init__(self, para, skf, sys, ibatch=None):
        """Run (SCC-) DFTB."""
        self.para = para
        self.skf = skf
        self.sys = sys

        self.ibatch = [ib for ib in range(ibatch)]

        # analyze DFTB result
        self.analysis = Analysis(self.para, self.skf, self.sys)

        # print DFTB calculation information
        self.print_ = Print(self.para, self.skf)

        # print title before DFTB calculations
        self.print_.print_scf_title()

        # calculate (SCC-) DFTB
        self.runscf()

        self.run_analysis()

    def runscf(self):
        """Run DFTB with multi interface.

        solid or molecule
        SCC, non-SCC or XLBOMD
        """
        scf = SCF(self.para, self.skf, self.sys)
        # non-SCC-DFTB
        scf.scf_npe_scc(self.ibatch)

    def run_analysis(self):
        """Analyse the DFTB calculation results and print."""
        self.analysis.dftb_energy(shift_=self.para['shift'],
                                  qatom=self.para['charge'])

        # claculate physical properties
        self.analysis.sum_property(self.ibatch)

        # print and write non-SCC DFTB results
        # self.print_.print_dftb_tail()


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

    def __init__(self, para, skf, sys):
        """Parameters for SCF."""
        self.para = para
        self.skf = skf
        self.sys = sys

        # single systems
        # self.batch = self.para['Lbatch']

        self.nb = self.sys.size_batch

        self.mask = [[True] * self.nb]  # mask for convergence in batch

        # batch calculations for multi systems
        self.ham = self.skf['hammat_']
        self.over = self.skf['overmat_']

        # number of atom in molecule
        self.nat = self.sys.size_system

        # number of orbital of each atom
        self.atom_orbitals = self.sys.atom_orbitals

        # the name of all atoms
        self.atomname = self.sys.symbols

        # analyze DFTB result
        self.analysis = Analysis(self.para, self.skf, self.sys)

        # electronic DFTB calculation
        self.elect = DFTBelect(self.para, self.skf)

        # print DFTB calculation information
        self.print_ = Print(self.para, self.skf)

    def scf_npe_scc(self, ibatch=[0]):
        """SCF for non-periodic-ML system with scc.

        atomind is the number of atom, for C, lmax is 2, therefore
        we need 2**2 orbitals (s, px, py, pz), then define atomind2
        """
        self.nat_ = [self.nat[ii] for ii in ibatch]
        maxiter = self.para['maxIteration']

        self.Lmask = self.para['dynamicSCC']

        # set initial reachConvergence as False
        self.para['reachConvergence'] = False

        gmat = pack([self.elect.gmatrix(
                self.sys.distances[i], self.nat[i],
                self.atomname[i]) for i in ibatch])
        # gmat = Gamma(U, distance, kwargs)
        qatom = self.sys.get_valence_electrons(self.ham.dtype)
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
            shiftorb_ = pack([ishif.repeat_interleave(iorb) for iorb, ishif in
                              zip(self.atom_orbitals[self.mask[-1]], shift_)])
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
                             rho, self.atom_orbitals[self.mask[-1]])

            # Last mixed charge is the current step now
            q_mixed[self.mask[-1]], conv = self.mixer(q_new)
            self.mask.append(~conv)
            if conv.all():
                self.para['reachConvergence'] = True
                break

        # return eigenvalue and charge
        self.para['charge'] = q_mixed
        print('charge', q_mixed)

        # return the final shift
        self.para['shift'] = t.stack([(iqm - iqz) @ igm for iqm, iqz, igm
                                      in zip(q_mixed, qzero, gmat)])
        shiftorb = pack([
            ishif.repeat_interleave(iorb) for iorb, ishif in zip(
                self.atom_orbitals, self.para['shift'])])
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


class Analysis:
    """Analysis of DFTB results, processing DFTB results."""

    def __init__(self, para, skf, sys):
        """Initialize parameters."""
        self.para = para
        self.skf = skf
        self.sys = sys

        # number of atom in batch
        self.nat = sys.size_system

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
            [self.get_dipole(iqz, iqa, self.sys.positions[i], self.nat[i])
             for iqz, iqa, i in zip(qzero, qatom, ibatch)])

        # # calculate MBD-DFTB
        # if self.para['LCPA']:
        #     MBD(self.para, self.dataset)

        # calculate PDOS or not
        if self.para['Lpdos']:
            self.pdos()

    def to_full_electron_charge(self, atomname, charge, ibatch=[0]):
        """Get the basic electronic information of each atom."""
        # add core electrons
        qat = [[_CORE[atomname[ib][iat]]
                for iat in range(self.nat[ib])] for ib in ibatch]
        # return charge information
        return pack([t.tensor(iq) for iq in qat]) + charge

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
