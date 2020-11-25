"""Interface to ASE."""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import os
import numpy as np
import torch as t
from ase import Atoms
import subprocess
from ase.calculators.dftb import Dftb
from ase.calculators.aims import Aims
ATOMNUM = {'H': 1, 'C': 6, 'N': 7, 'O': 8}
DFTB_ENERGY = {"H": -0.238600544, "C": -1.398493891, "N": -2.0621839400,
               "O": -3.0861916005}
AIMS_ENERGY = {"H": -0.45891649, "C": -37.77330663, "N": -54.46973501,
               "O": -75.03140052}
_AUEV = 27.2113845


class DFTB:
    """Run DFTB+, return results, write into hdf5."""

    def __init__(self, dftbplus, directory_sk):
        """Initialize parameters.

        Args:
            dftbplus: binary executable DFTB+
            directory_sk: path to SKF files

        """
        # path to dftb+
        self.dftb = dftbplus

        # check if skf dataset exists
        if not os.path.isfile(self.dftb):
            raise FileNotFoundError('%s not found' % self.dftb)

        # SKF path
        self.slko_path = directory_sk

        # set environment before calculations
        self.set_env()

    def set_env(self):
        """Set the environment before DFTB calculations with ase."""
        # copy binary to current path
        os.system('cp ' + self.dftb + ' ./dftb+')

        # get the current binary path and name
        path_bin = os.path.join(os.getcwd(), 'dftb+')

        # set ase environemt
        os.environ['ASE_DFTB_COMMAND'] = path_bin + ' > PREFIX.out'
        os.environ['DFTB_PREFIX'] = self.slko_path

    def run_dftb(self, nbatch, coorall, group, symbols, begin=0, write_hdf=True):
        """Run batch systems with ASE-DFTB."""
        self.symbols = symbols
        for ibatch in range(begin, nbatch):
            # transfer specie style, e.g., ['C', 'H', 'H', 'H', 'H'] to 'CHHHH'
            # so that satisfy ase.Atoms style
            ispecie = group.attrs['specie']
            print('ibatch', ispecie, ibatch)

            # get coordinates of a single molecule
            self.coor = coorall[ibatch]
            self.nat = len(self.coor)

            # run each molecule in batches
            self.ase_idftb(ispecie, self.coor)

            # process each result (overmat, eigenvalue, eigenvect, dipole)
            self.process_iresult(ibatch)

            # creat or write  reference data as hdf type
            if write_hdf:
                self.write_hdf5(ibatch, ispecie, group=group)

        # remove DFTB files after DFTB calculations
        self.remove()

    def ase_idftb(self, moleculespecie, coor):
        """Build DFTB input by ASE."""
        # set Atoms with molecule specie and coordinates
        mol = Atoms(moleculespecie, positions=coor)

        # set DFTB caulation parameters
        cal = Dftb(Hamiltonian_='DFTB',
                   Hamiltonian_SCC='No',
                   Hamiltonian_SCCTolerance=1e-8,
                   Hamiltonian_MaxAngularMomentum_='',
                   Hamiltonian_MaxAngularMomentum_H='s',
                   Hamiltonian_MaxAngularMomentum_C='p',
                   Hamiltonian_MaxAngularMomentum_N='p',
                   Hamiltonian_MaxAngularMomentum_O='p',
                   Options_='',
                   Options_WriteHS='Yes',
                   Analysis_='',
                   # Analysis_CalculateForces='Yes',
                   Analysis_MullikenAnalysis='Yes',
                   Analysis_WriteEigenvectors='Yes',
                   Analysis_EigenvectorsAsText='Yes',
                   ParserOptions_='',
                   ParserOptions_IgnoreUnprocessedNodes='Yes')

        # get calculators
        mol.calc = cal
        try:
            mol.get_potential_energy()
        except UnboundLocalError:
            mol.calc.__dict__["parameters"]['Options_WriteHS'] = 'No'
            mol.get_potential_energy()

    def process_iresult(self, ibatch):
        """Process result for each calculation."""
        # read in the H (hamsqr1.dat) and S (oversqr.dat) matrices now
        self.overmat = get_matrix('oversqr.dat')

        # read final eigenvector
        self.eigenvec = get_eigenvec('eigenvec.out')

        # read final eigenvalue
        self.eigenvalue, occ, self.hl = get_eigenvalue('band.out')

        # read detailed.out and return energy, charge
        self.E_tot, self.Q_, self.dip = read_detailed_out(self.nat)

        # calculate formation energy
        self.E_f = self.cal_optfor_energy(self.E_tot, ibatch)

    def write_hdf5(self, ibatch, ispecie, group=None):
        """Write each molecule DFTB calculation results to hdf type data."""
        coor_name = str(ibatch) + 'positions'
        group.create_dataset(coor_name, data=self.coor)

        # eigenvalue
        eigval_name = str(ibatch) + 'eigenValue'
        group.create_dataset(eigval_name, data=self.hl)

        # HOMO LUMO
        eigval_name = str(ibatch) + 'HOMOLUMO'
        group.create_dataset(eigval_name, data=self.hl)

        # dipole name
        dip_name = str(ibatch) + 'dipole'
        group.create_dataset(dip_name, data=self.dip)

        # formation energy
        ener_name = str(ibatch) + 'formationEnergy'
        group.create_dataset(ener_name, data=self.E_f)

        # total energy
        ener_name = str(ibatch) + 'totalEnergy'
        group.create_dataset(ener_name, data=self.E_tot)

        # total charge
        q_name = str(ibatch) + 'charge'
        group.create_dataset(q_name, data=self.Q_)

    def remove(self):
        """Remove all DFTB data after calculations."""
        os.system('rm band.out charges.bin detailed.out')
        os.system('rm dftb_in.hsd dftb.out dftb_pin.hsd eigenvec.bin')
        os.system('rm eigenvec.out geo_end.gen hamsqr1.dat oversqr.dat')

    def cal_optfor_energy(self, energy, ibatch):
        """Calculate formation energy."""
        return energy - sum([DFTB_ENERGY[self.symbols[ibatch][iat]]
                             for iat in range(self.nat)])


class AseAims:
    """RunASEAims will run FHI-aims with both batch or single calculations."""

    def __init__(self, aims, aims_specie):
        """Initialize parameters.

        Args:
            aims: binary executable FHI-aims
            aims_specie: path to species_defaults

        """
        self.aims = aims
        self.aims_specie = aims_specie

        # check if aims exists
        if not os.path.isfile(self.aims):
            raise FileNotFoundError('%s not found' % self.aims)

        # set environment before calculations
        self.set_env()

    def set_env(self):
        """Set the environment before DFTB calculations with ase."""
        # copy binary to current path
        os.system('cp ' + self.aims + ' ./aims.x')

        # get the current binary path and name
        path_bin = os.path.join(os.getcwd(), 'aims.x')
        self.aimsout = os.path.join(os.getcwd(), 'aims.out')

        # set ase environemt
        os.environ['ASE_AIMS_COMMAND'] = path_bin + ' > PREFIX.out'
        os.environ['AIMS_SPECIES_DIR'] = self.aims_specie

    def run_aims(self, nbatch, positions, group, symbols, begin=0, write_hdf=True):
        """Run batch systems with ASE-DFTB."""
        self.symbols = symbols
        for ibatch in range(begin, nbatch):
            # transfer specie style, e.g., ['C', 'H', 'H', 'H', 'H'] to 'CHHHH'
            # so that satisfy ase.Atoms style
            ispecie = group.attrs['specie']
            print('ibatch', ibatch, ispecie, 'size', nbatch)

            # get coordinates of a single molecule
            self.coor = positions[ibatch]

            # run each molecule in batches
            self.ase_iaims(ispecie, self.coor)

            # process each result (overmat, eigenvalue, eigenvect, dipole)
            self.process_iresult(ibatch)

            # creat or write  reference data as hdf type
            if write_hdf:
                self.write_hdf5(ibatch, ispecie, group=group)

        # remove DFTB files
        self.remove()

    def ase_iaims(self, moleculespecie, coor):
        """Build Aims input by ASE."""
        # set Atoms with molecule specie and coordinates
        mol = Atoms(moleculespecie, positions=coor)

        cal = Aims(xc='PBE',
                   output=['dipole', 'mulliken'],
                   sc_accuracy_etot=1e-6,
                   sc_accuracy_eev=1e-3,
                   sc_accuracy_rho=1e-6,
                   sc_accuracy_forces=1e-4,
                   many_body_dispersion=' ',
                   command = "mpirun aims.x > aims.out")

        # get calculators
        mol.calc = cal
        try:
            mol.get_potential_energy()
        except UnboundLocalError:
            mol.calc.__dict__["parameters"]['Options_WriteHS'] = 'No'
            mol.get_potential_energy()

    def process_iresult(self, ibatch):
        """Process result for each calculation."""
        # get number of atoms in molecule
        self.nat = len(self.coor)

        # read HUMO, LUMO
        commh = "grep 'Highest occupied state (VBM) at' " + \
            self.aimsout + " | tail -n 1 | awk '{print $6}'"
        self.homo = subprocess.check_output(commh, shell=True).decode('utf-8')
        comml = "grep 'Lowest unoccupied state (CBM) at' " + \
            self.aimsout + " | tail -n 1 | awk '{print $6}'"
        self.lumo = subprocess.check_output(comml, shell=True).decode('utf-8')

        # read dipole
        commdip = "grep 'Total dipole moment' "
        cdx = commdip + self.aimsout + " | awk '{print $7}'"
        cdy = commdip + self.aimsout + " | awk '{print $8}'"
        cdz = commdip + self.aimsout + " | awk '{print $9}'"
        idipx = float(subprocess.check_output(cdx, shell=True).decode('utf-8'))
        idipy = float(subprocess.check_output(cdy, shell=True).decode('utf-8'))
        idipz = float(subprocess.check_output(cdz, shell=True).decode('utf-8'))
        self.dip = np.asarray([idipx, idipy, idipz])

        # get total energy, formation energy
        comme = "grep 'Total energy                  :' " + self.aimsout + \
            " | tail -n 1 | awk '{print $5}'"
        self.E_tot = float(
            subprocess.check_output(comme, shell=True).decode('utf-8'))
        self.E_f = self.cal_optfor_energy(self.E_tot, ibatch)

        # read polarizability
        commp = "grep -A " + str(self.nat + 1) + \
            " 'C6 coefficients and polarizabilities' " + self.aimsout + \
                " | tail -n " + str(self.nat) + " | awk '{print $6}'"
        ipol = subprocess.check_output(commp, shell=True).decode('utf-8')
        self.alpha_mbd = np.asarray([float(i) for i in ipol.split('\n')[:-1]])

        # read mulliken charge
        commc = "grep -A " + str(self.nat) + \
            " 'atom       electrons          charge' " + self.aimsout + \
                " | tail -n " + str(self.nat) + " | awk '{print $3}'"
        icharge = subprocess.check_output(commc, shell=True).decode('utf-8')
        self.charge = np.asarray([float(i) for i in icharge.split('\n')[:-1]])

        # read Hirshfeld volume
        cvol = "grep 'Hirshfeld volume        :' " + self.aimsout + \
            " | awk '{print $5}'"
        ivol = subprocess.check_output(cvol, shell=True).decode('utf-8')
        self.hirshfeldvolume = np.asarray([float(i) for i in ivol.split('\n')[:-1]])

    def write_hdf5(self, ibatch, ispecie, group=None):
        """Write each molecule DFTB calculation results to group (metadata)."""
        # coordinate name
        coor_name = str(ibatch) + 'positions'
        group.create_dataset(coor_name, data=self.coor)

        # eigenvalue
        hl_name = str(ibatch) + 'HOMOLUMO'
        group.create_dataset(hl_name, data=np.asarray([float(self.homo),
                                                       float(self.lumo)]))

        # dipole name
        dip_name = str(ibatch) + 'dipole'
        group.create_dataset(dip_name, data=self.dip)

        # formation energy
        ener_name = str(ibatch) + 'formationEnergy'
        group.create_dataset(ener_name, data=self.E_f)

        # total energy
        ener_name = str(ibatch) + 'totalEnergy'
        group.create_dataset(ener_name, data=self.E_tot)

        # total charge
        c_name = str(ibatch) + 'charge'
        group.create_dataset(c_name, data=self.charge)

        # Hirshfeld volume
        v_name = str(ibatch) + 'hirshfeldVolume'
        group.create_dataset(v_name, data=self.hirshfeldvolume)

        # alpha_mbd, polarizability
        p_name = str(ibatch) + 'alphaMBD'
        group.create_dataset(p_name, data=self.alpha_mbd)

    def remove(self):
        """Remove all DFTB data after calculations."""
        os.system('rm aims.out control.in geometry.in Mulliken.out parameters.ase')

    def cal_optfor_energy(self, energy, ibatch):
        """Calculate formation energy."""
        return energy - sum([AIMS_ENERGY[self.symbols[ibatch][iat]]
                             for iat in range(self.nat)])


def get_matrix(filename):
    """Read DFTB+ hamsqr1.dat and oversqr.dat."""
    text = ''.join(open(filename, 'r').readlines())
    string = re.search('(?<=MATRIX\n).+(?=\n)',text, flags = re.DOTALL).group(0)
    out = np.array([[float(i) for i in row.split()] for row in string.split('\n')])
    return t.from_numpy(out)


def get_eigenvec(filename):
    """Read DFTB+ eigenvec.out."""
    string = []
    text = ''.join(open(filename, 'r').readlines())

    # only read float
    string_ = re.findall(r"[-+]?\d*\.\d+", text)

    # delete even column
    del string_[1::2]
    [string.append(float(ii)) for ii in string_]
    nstr = int(np.sqrt(len(string)))

    # transfer list to ==> numpy(float64) ==> torch
    eigenvec = np.asarray(string).reshape(nstr, nstr)
    return t.from_numpy(eigenvec).T


def get_eigenvalue(filename):
    """Read DFTB+ band.out."""
    text = ''.join(open(filename, 'r').readlines())

    # only read float
    string = re.findall(r"[-+]?\d*\.\d+", text)

    # delete even column
    eigenval_ = [float(ii) for ii in string[1::2]]
    occ_ = [float(ii) for ii in string[0::2]][1:]  # remove the first value

    # transfer list to ==> numpy(float64) ==> torch
    eigenval = t.from_numpy(np.asarray(eigenval_)) / _AUEV
    occ = t.from_numpy(np.asarray(occ_))
    humolumo = np.asarray([eigenval[np.where(occ != 0)[0]][-1],
                           eigenval[np.where(occ == 0)[0]][0]])
    return eigenval, occ, humolumo


def read_detailed_out(natom):
    """Read DFTB+ output file detailed.out."""
    qatom, dip = [], []
    text = ''.join(open('detailed.out', 'r').readlines())
    E_tot_ = re.search('(?<=Total energy:).+(?=\n)',
                       text, flags=re.DOTALL | re.MULTILINE).group(0)
    E_tot = re.findall(r"[-+]?\d*\.\d+", E_tot_)[0]

    # read charge
    text2 = re.search('(?<=Atom       Population\n).+(?=\n)',
                      text, flags=re.DOTALL | re.MULTILINE).group(0)
    qatom_ = re.findall(r"[-+]?\d*\.\d+", text2)[:natom]
    [qatom.append(float(ii)) for ii in qatom_]

    # read dipole (Debye)
    text3 = re.search('(?<=Dipole moment:).+(?=\n)',
                      text, flags=re.DOTALL | re.MULTILINE).group(0)
    # if tail is [-3::], read Debye dipole, [:3] will read au dipole
    dip_ = re.findall(r"[-+]?\d*\.\d+", text3)[:3]
    [dip.append(float(ii)) for ii in dip_]

    return float(E_tot), \
        t.from_numpy(np.asarray(qatom)), t.from_numpy(np.asarray(dip))
