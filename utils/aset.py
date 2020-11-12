"""Interface to ASE."""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import os
import numpy as np
import torch as t
from ase import Atoms
import subprocess
from ase.build import molecule
from ase.calculators.dftb import Dftb
from ase.calculators.aims import Aims
import dftbtorch.dftbcalculator as dftbcalculator
from IO.save import Save1D, Save2D
ATOMNUM = {'H': 1, 'C': 6, 'N': 7, 'O': 8}
DFTB_ENERGY = {"H": -0.238600544, "C": -1.398493891, "N": -2.0621839400,
               "O": -3.0861916005}
AIMS_ENERGY = {"H": -0.45891649, "C": -37.77330663, "N": -54.46973501,
               "O": -75.03140052}

class DFTB:
    """"""

    def __init__(self, para, dataset, ml, setenv=True):
        self.para = para
        self.dataset = dataset
        self.ml = ml

        # path to dftb+
        self.dftb = self.ml['dftbplus']

        # check if skf dataset exists
        if not os.path.isfile(self.dftb):
            raise FileNotFoundError('%s not found' % self.dftb)

        # SKF path
        self.slko_path = self.para['directorySK']

        # set environment before calculations
        if setenv:
            self.set_env()

        # transfer all eV to H
        self.ev_hat = self.para["AUEV"]

    def set_env(self):
        """Set the environment before DFTB calculations with ase."""
        # copy binary to current path
        os.system('cp ' + self.dftb + ' ./dftb+')

        # get the current binary path and name
        path_bin = os.path.join(os.getcwd(), 'dftb+')

        # set ase environemt
        os.environ['ASE_DFTB_COMMAND'] = path_bin + ' > PREFIX.out'
        os.environ['DFTB_PREFIX'] = self.slko_path

    def run_dftb(self, nbatch, coorall, begin=0, hdf=None, group=None):
        """Run batch systems with ASE-DFTB."""
        # if calculate, deal with pdos or not
        if self.para['Lpdos']:
            self.dataset['pdosdftbplus'] = []

        # save eigenvalue as reference eigenvalue for ML
        if 'eigval' in self.ml['target']:
            self.dataset['refEigval'] = []

        # create reference list for following ML
        self.dataset['refFormEnergy'], self.dataset['refCharge'] = [], []
        self.dataset['refDipole'], self.dataset['refHomoLumo'] = [], []

        for ibatch in range(begin, nbatch):
            # transfer specie style, e.g., ['C', 'H', 'H', 'H', 'H'] to 'CHHHH'
            # so that satisfy ase.Atoms style
            if group is None:
                self.dataset['natom'] = self.dataset['natomAll'][ibatch]
                ispecie = ''.join(self.dataset['symbols'][ibatch])
            else:
                ispecie = group.attrs['specie']
                self.dataset['natom'] = group.attrs['natom']
            print('ibatch', ispecie, ibatch)

            # get coordinates of a single molecule
            self.coor = coorall[ibatch]

            # run each molecule in batches
            self.ase_idftb(ispecie, self.coor)

            # process each result (overmat, eigenvalue, eigenvect, dipole)
            self.process_iresult(ibatch)

            # creat or write  reference data as hdf type
            if self.para['task'] == 'get_dftb_hdf':
                self.write_hdf5(hdf, ibatch, ispecie, begin, group=group)

        # deal with DFTB data
        self.process_results()

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
        self.dataset['overmat'] = get_matrix('oversqr.dat')

        # read final eigenvector
        self.dataset['eigenvec'] = get_eigenvec('eigenvec.out')

        # read final eigenvalue
        self.dataset['eigenvalue'], occ, self.hl = get_eigenvalue('band.out', self.ev_hat)

        # read detailed.out and return energy, charge
        self.E_tot, self.Q_, self.dip = read_detailed_out(self.dataset['natom'])

        # calculate formation energy
        self.E_f = self.cal_optfor_energy(self.E_tot, ibatch)

        # add each molecule properties
        self.dataset['refFormEnergy'].append(self.E_f)
        self.dataset['refCharge'].append(self.Q_)
        self.dataset['refDipole'].append(self.dip)
        self.dataset['refHomoLumo'].append(self.hl)

    def write_hdf5(self, hdf, ibatch, ispecie, begin, group=None):
        """Write each molecule DFTB calculation results to hdf type data."""
        # get the name of each molecule and its properties
        num = ibatch - begin
        # coordinate name
        if group is None:
            coor_name = ispecie + str(num) + 'positions'
            hdf.create_dataset(coor_name, data=self.coor)
        else:
            coor_name = str(num) + 'positions'
            group.create_dataset(coor_name, data=self.coor)

        # eigenvalue
        if group is None:
            eigval_name = ispecie + str(num) + 'eigenvalue'
            eigval = self.para['eigenvalue']
            hdf.create_dataset(eigval_name, data=eigval)
        else:
            eigval_name = str(num) + 'eigenvalue'
            eigval = self.dataset['eigenvalue']
            group.create_dataset(eigval_name, data=eigval)

        # HOMO LUMO
        if group is None:
            eigval_name = ispecie + str(num) + 'humolumo'
            eigval = self.para['refHomoLumo']
            hdf.create_dataset(eigval_name, data=eigval)
        else:
            eigval_name = str(num) + 'humolumo'
            eigval = self.dataset['refHomoLumo']
            group.create_dataset(eigval_name, data=eigval)

        # dipole name
        if group is None:
            dip_name = ispecie + str(num) + 'dipole'
            hdf.create_dataset(dip_name, data=self.dip)
        else:
            dip_name = str(num) + 'dipole'
            group.create_dataset(dip_name, data=self.dip)

        # formation energy
        if group is None:
            ener_name = ispecie + str(num) + 'formationenergy'
            hdf.create_dataset(ener_name, data=self.E_f)
        else:
            ener_name = str(num) + 'formationenergy'
            group.create_dataset(ener_name, data=self.E_f)

        # total energy
        if group is None:
            ener_name = ispecie + str(num) + 'totalenergy'
            hdf.create_dataset(ener_name, data=self.E_tot)
        else:
            ener_name = str(num) + 'totalenergy'
            group.create_dataset(ener_name, data=self.E_tot)

        # total charge
        if group is None:
            q_name = ispecie + str(num) + 'charge'
            hdf.create_dataset(q_name, data=self.Q_)
        else:
            q_name = str(num) + 'charge'
            group.create_dataset(q_name, data=self.Q_)

    def remove(self):
        """Remove all DFTB data after calculations."""
        os.system('rm dftb+ band.out charges.bin detailed.out')
        os.system('rm dftb_in.hsd dftb.out dftb_pin.hsd eigenvec.bin')
        os.system('rm eigenvec.out geo_end.gen hamsqr1.dat oversqr.dat')

    def process_results(self):
        pass

    def cal_optfor_energy(self, energy, ibatch):
        natom = self.dataset['natom']
        return energy - sum([DFTB_ENERGY[self.dataset['symbols'][ibatch][iat]]
                             for iat in range(natom)])

class AseAims:
    """RunASEAims will run FHI-aims with both batch or single calculations."""

    def __init__(self, para, dataset, ml, setenv=False):
        self.para = para
        self.ml = ml
        self.dataset = dataset

        # set environment before calculations
        if setenv:
            self.set_env()

    def set_env(self):
        """Set the environment before DFTB calculations with ase."""

        # copy binary to current path
        os.system('cp ' + self.ml['aims'] + ' ./aims.x')

        # get the current binary path and name
        path_bin = os.path.join(os.getcwd(), 'aims.x')
        self.aimsout = os.path.join(os.getcwd(), 'aims.out')

        # set ase environemt
        os.environ['ASE_AIMS_COMMAND'] = path_bin + ' > PREFIX.out'
        os.environ['AIMS_SPECIES_DIR'] = self.ml['aimsSpecie']

    def run_aims(self, nbatch, begin=None, hdf=None, group=None):
        """Run batch systems with ASE-DFTB."""
        coorall = self.dataset['positions']
        if begin is None:
            begin = 0

        # if calculate, deal with pdos or not
        if self.para['Lpdos']:
            self.dataset['pdosdftbplus'] = []

        # save eigenvalue as reference eigenvalue for ML
        if 'eigval' in self.ml['target']:
            self.dataset['refEigval'] = []

        # create reference list for following ML
        self.dataset['refEnergy'] = []
        self.dataset['refDipole'] = []
        self.dataset['refMBDAlpha'] = []
        self.dataset['refHirshfeldVolume'] = []
        self.dataset['refCharge'] = []

        for ibatch in range(begin, nbatch):
            # transfer specie style, e.g., ['C', 'H', 'H', 'H', 'H'] to 'CHHHH'
            # so that satisfy ase.Atoms style
            if group is None:
                self.dataset['natom'] = self.dataset['natomAll'][ibatch]
                ispecie = ''.join(self.dataset['symbols'][ibatch])
            else:
                ispecie = group.attrs['specie']
                self.dataset['natom'] = group.attrs['natom']
            print('ibatch', ibatch, ispecie, 'size', nbatch)

            # get coordinates of a single molecule
            self.coor = coorall[ibatch]

            # run each molecule in batches
            self.ase_iaims(ispecie, self.coor)

            # process each result (overmat, eigenvalue, eigenvect, dipole)
            self.process_iresult(ibatch)

            # creat or write  reference data as hdf type
            if self.para['task'] == 'get_aims_hdf':
                self.write_hdf5(hdf, ibatch, ispecie, begin, group=group)

            # save dftb results as reference .txt
            elif self.para['task'] == 'opt':
                pass

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
                   command = "mpirun -np 4 aims.x > aims.out")

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
        ihomo = subprocess.check_output(commh, shell=True).decode('utf-8')
        comml = "grep 'Lowest unoccupied state (CBM) at' " + \
            self.aimsout + " | tail -n 1 | awk '{print $6}'"
        ilumo = subprocess.check_output(comml, shell=True).decode('utf-8')
        self.para['humolumo'] = np.asarray([float(ihomo), float(ilumo)])

        # read dipole
        commdip = "grep 'Total dipole moment' "
        cdx = commdip + self.aimsout + " | awk '{print $7}'"
        cdy = commdip + self.aimsout + " | awk '{print $8}'"
        cdz = commdip + self.aimsout + " | awk '{print $9}'"
        idipx = float(subprocess.check_output(cdx, shell=True).decode('utf-8'))
        idipy = float(subprocess.check_output(cdy, shell=True).decode('utf-8'))
        idipz = float(subprocess.check_output(cdz, shell=True).decode('utf-8'))

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

        # read mulliken charge
        commc = "grep -A " + str(self.nat) + \
            " 'atom       electrons          charge' " + self.aimsout + \
                " | tail -n " + str(self.nat) + " | awk '{print $3}'"
        icharge = subprocess.check_output(commc, shell=True).decode('utf-8')

        # read Hirshfeld volume
        cvol = "grep 'Hirshfeld volume        :' " + self.aimsout + \
            " | awk '{print $5}'"
        ivol = subprocess.check_output(cvol, shell=True).decode('utf-8')

        # add each molecule properties
        self.alpha_mbd = np.asarray([float(i) for i in ipol.split('\n')[:-1]])
        self.dataset['refMBDAlpha'].append(t.from_numpy(self.alpha_mbd))
        self.dip = np.asarray([idipx, idipy, idipz])
        self.dataset['refDipole'].append(t.from_numpy(self.dip))
        self.hirshfeldvolume = np.asarray([float(i) for i in ivol.split('\n')[:-1]])
        self.dataset['refHirshfeldVolume'].append(t.from_numpy(self.hirshfeldvolume))
        self.charge = np.asarray([float(i) for i in icharge.split('\n')[:-1]])
        self.dataset['refCharge'].append(t.from_numpy(self.charge))

    def write_hdf5(self, hdf, ibatch, ispecie, begin, group=None):
        """Write each molecule DFTB calculation results to hdf type data."""
        # get the name of each molecule and its properties
        num = ibatch - begin
        # coordinate name
        if group is None:
            coor_name = ispecie + str(num) + 'positions'
            hdf.create_dataset(coor_name, data=self.coor)
        else:
            coor_name = str(num) + 'positions'
            group.create_dataset(coor_name, data=self.coor)

        # eigenvalue
        if group is None:
            hl_name = ispecie + str(num) + 'humolumo'
            hl = self.para['humolumo']
            hdf.create_dataset(hl_name, data=hl)
        else:
            hl_name = str(num) + 'humolumo'
            hl = self.para['humolumo']
            group.create_dataset(hl_name, data=hl)

        # dipole name
        if group is None:
            dip_name = ispecie + str(num) + 'dipole'
            hdf.create_dataset(dip_name, data=self.dip)
        else:
            dip_name = str(num) + 'dipole'
            group.create_dataset(dip_name, data=self.dip)

        # formation energy
        if group is None:
            ener_name = ispecie + str(num) + 'formationenergy'
            hdf.create_dataset(ener_name, data=self.E_f)
        else:
            ener_name = str(num) + 'formationenergy'
            group.create_dataset(ener_name, data=self.E_f)

        # total energy
        if group is None:
            ener_name = ispecie + str(num) + 'totalenergy'
            hdf.create_dataset(ener_name, data=self.E_tot)
        else:
            ener_name = str(num) + 'totalenergy'
            group.create_dataset(ener_name, data=self.E_tot)

        # total charge
        if group is None:
            c_name = ispecie + str(num) + 'charge'
            hdf.create_dataset(c_name, data=self.charge)
        else:
            c_name = str(num) + 'charge'
            group.create_dataset(c_name, data=self.charge)

        # Hirshfeld volume
        if group is None:
            v_name = ispecie + str(num) + 'hirshfeldvolume'
            hdf.create_dataset(v_name, data=self.hirshfeldvolume)
        else:
            v_name = str(num) + 'hirshfeldvolume'
            group.create_dataset(v_name, data=self.hirshfeldvolume)
        # alpha_mbd, polarizability
        if group is None:
            p_name = ispecie + str(num) + 'alpha_mbd'
            hdf.create_dataset(p_name, data=self.alpha_mbd)
        else:
            p_name = str(num) + 'alpha_mbd'
            group.create_dataset(p_name, data=self.alpha_mbd)

    def remove(self):
        """Remove all DFTB data after calculations."""
        os.system('rm aims.x aims.out control.in geometry.in')

    def cal_optfor_energy(self, energy, ibatch):
        natom = self.dataset['natom']
        return energy - sum([AIMS_ENERGY[self.dataset['symbols'][ibatch][iat]]
                             for iat in range(natom)])


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


def get_eigenvalue(filename, eV_Hat):
    """Read DFTB+ band.out."""
    text = ''.join(open(filename, 'r').readlines())

    # only read float
    string = re.findall(r"[-+]?\d*\.\d+", text)

    # delete even column
    eigenval_ = [float(ii) for ii in string[1::2]]
    occ_ = [float(ii) for ii in string[0::2]][1:]  # remove the first value

    # transfer list to ==> numpy(float64) ==> torch
    eigenval = t.from_numpy(np.asarray(eigenval_)) / eV_Hat
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
    dip_ = re.findall(r"[-+]?\d*\.\d+", text3)[-3::]
    [dip.append(float(ii)) for ii in dip_]

    return float(E_tot), \
        t.from_numpy(np.asarray(qatom)), t.from_numpy(np.asarray(dip))
