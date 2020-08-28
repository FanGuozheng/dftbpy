"""Interface to ASE"""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import os
import numpy as np
import torch as t
from ase import Atoms
import h5py
from ase.build import molecule
from ase.calculators.dftb import Dftb
from ase.optimize import QuasiNewton
from ase.io import write
import dftbtorch.dftb_torch as dftb_torch
from utils.save import SaveData
ATOMNUM = {'H': 1, 'C': 6, 'N': 7, 'O': 8}
DFTB_ENERGY = {"H": -0.238600544, "C": -1.398493891, "N": -2.0621839400,
               "O": -3.0861916005}


class DFTB:
    """"""

    def __init__(self, para, setenv=False):
        self.para = para

        # DFTB+ binary name, normally dftb+
        self.dftb_bin = self.para['dftb_bin']

        # DFTB+ binary path
        self.dftb_path = para['dftb_ase_path']

        # SKF path
        self.slko_path = self.para['skf_ase_path']

        # set environment before calculations
        self.setenv = setenv
        if self.setenv:
            self.set_env()

        # transfer all eV to H
        self.ev_hat = self.para["AUEV"]

    def set_env(self):
        """Set the environment before DFTB calculations with ase."""
        # merge DFTB+ binary path and name
        path_bin_org = os.path.join(self.dftb_path, self.dftb_bin)

        # copy binary to current path
        os.system('cp ' + path_bin_org + ' .')

        # get the current binary path and name
        path_bin = os.path.join(os.getcwd(), self.dftb_bin)

        # set ase environemt
        with open('ase_env.sh', 'w') as fp:
            fp.write('export ASE_DFTB_COMMAND="' + path_bin + ' > PREFIX.out"')
            fp.write('\n')
            fp.write('export DFTB_PREFIX=' + self.slko_path)
            fp.write('\n')
        fp.close()

        # source the ase_env.sh bash file
        os.system('source ase_env.sh')

    def run_dftb(self, nbatch, coorall, begin=None, hdf=None, group=None):
        """Run batch systems with ASE-DFTB."""
        if begin is None:
            begin = 0

        # source and set environment for python, ase before calculations
        self.save = SaveData(self.para)

        # if calculate, deal with pdos or not
        if self.para['Lpdos']:
            self.para['pdosdftbplus'] = []

        # save eigenvalue as reference eigenvalue for ML
        if 'eigval' in self.para['target']:
            self.para['refeigval'] = []

        # create reference list for following ML
        self.para['refenergy'], self.para['refqatom'] = [], []
        self.para['refdipole'] = []

        for ibatch in range(begin, nbatch):
            # transfer specie style, e.g., ['C', 'H', 'H', 'H', 'H'] to 'CH4'
            # so that satisfy ase.Atoms style
            if group is None:
                self.para['natom'] = self.para['natomall'][ibatch]
                ispecie = ''.join(self.para['symbols'][ibatch])
            else:
                ispecie = group.attrs['specie']
                self.para['natom'] = group.attrs['natom']

            # get coordinates of a single molecule
            self.coor = coorall[ibatch]

            # run each molecule in batches
            print("ibatch", ibatch, "specie", ispecie)
            self.ase_idftb(ispecie, self.coor[:, 1:])

            # process each result (overmat, eigenvalue, eigenvect, dipole)
            self.process_iresult()

            # creat reference data as hdf type
            if self.para['task'] == 'get_hdf_data':
                self.write_hdf5(hdf, ibatch, ispecie, begin, group=group)

            # save dftb results as reference .txt
            elif self.para['task'] == 'opt':
                if 'eigval' in self.para['target']:
                    self.para['refeigval'].append(self.para['eigenvalue'])

                # calculate PDOS or not
                if self.para['Lpdos']:

                    # calculate PDOS
                    dftb_torch.Analysis(self.para).pdos()
                    self.para['pdosdftbplus'].append(self.para['pdos'])

                    # save PDOS
                    self.save.save2D(self.para['pdos'].numpy(),
                                     name='pdosref.dat', dire='.data', ty='a')

                    # save charge
                    self.save.save1D(self.para['refqatom'][ibatch].numpy(),
                                     name='refqatom.dat', dire='.data', ty='a')

                    # save eigenvalue
                    self.save.save1D(self.para['eigenvalue'],
                                     name='HLdftbplus.dat', dire='.data', ty='a')

                # save energy
                self.save.save1D(np.asarray(self.para['refenergy']),
                                 name='energydftbplus.dat', dire='.data', ty='a')

        # deal with DFTB data
        self.process_results()

        # remove DFTB files
        self.remove()

    def ase_idftb(self, moleculespecie, coor):
        """Build DFTB input by ASE."""
        # set Atoms with molecule specie and coordinates
        mol = Atoms(moleculespecie, positions=coor)

        # set DFTB caulation parameters
        cal = Dftb(Hamiltonian_='DFTB',
                   Hamiltonian_SCC='Yes',
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
            mol.calc.__dict__["parameters"]['Options_WriteHS']='No'
            mol.get_potential_energy()

    def process_iresult(self):
        """Process result for each calculation."""
        # read in the H (hamsqr1.dat) and S (oversqr.dat) matrices now
        self.para['overmat'] = get_matrix('oversqr.dat')

        # read final eigenvector
        self.para['eigenvec'] = get_eigenvec('eigenvec.out')

        # read final eigenvalue
        self.para['eigenvalue'], occ = get_eigenvalue('band.out', self.ev_hat)

        # read detailed.out and return energy, charge
        self.E_tot, self.Q_, self.dip = read_detailed_out(self.para['natom'])

        # calculate formation energy
        self.E_f = self.cal_optfor_energy(self.E_tot)

        # add each molecule properties
        self.para['refenergy'].append(self.E_f)
        self.para['refqatom'].append(self.Q_)
        self.para['refdipole'].append(self.dip)

    def write_hdf5(self, hdf, ibatch, ispecie, begin, group=None):
        """Write each molecule DFTB calculation results to hdf type data."""
        # get the name of each molecule and its properties
        num = ibatch - begin
        # coordinate name
        if group is None:
            coor_name = ispecie + str(num) + 'coordinate'
            hdf.create_dataset(coor_name, data=self.coor)
        else:
            coor_name = str(num) + 'coordinate'
            group.create_dataset(coor_name, data=self.coor)

        # eigenvalue
        if group is None:
            eigval_name = ispecie + str(num) + 'eigenvalue'
            eigval = self.para['eigenvalue']
            hdf.create_dataset(eigval_name, data=eigval)
        else:
            eigval_name = str(num) + 'eigenvalue'
            eigval = self.para['eigenvalue']
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
        os.system('rm dftb+ ase_env.sh band.out charges.bin detailed.out')
        os.system('rm dftb_in.hsd dftb.out dftb_pin.hsd eigenvec.bin')
        os.system('rm eigenvec.out geo_end.gen hamsqr1.dat oversqr.dat')

    def process_results(self):
        pass

    def cal_optfor_energy(self, energy):
        natom = self.para['natom']
        for iat in range(0, natom):
            idx = int(self.coor[iat, 0])
            iname = list(ATOMNUM.keys())[list(ATOMNUM.values()).index(idx)]
            energy = energy - DFTB_ENERGY[iname]
        return energy

class Aims:

    def __init__(self, para, setenv=False):
        self.para = para

        # FHI-aims binary name, normally aims
        self.dftb_bin = self.para['aims_bin']

        # GHI-aims binary path
        self.dftb_path = para['aims_ase_path']

        # set environment before calculations
        self.setenv = setenv
        if self.setenv:
            self.set_env()

    def run_aims(self, nbatch, coorall):
        pass

    def ase_iaims(self):
        pass


def get_matrix(filename):
    """Read hamsqr1.dat and oversqr.dat."""
    text = ''.join(open(filename, 'r').readlines())
    string = re.search('(?<=MATRIX\n).+(?=\n)',text, flags = re.DOTALL).group(0)
    out = np.array([[float(i) for i in row.split()] for row in string.split('\n')])
    return t.from_numpy(out)


def get_eigenvec(filename):
    """Read eigenvec.out."""
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
    """Read band.out."""
    eigenval_, occ_ = [], []
    text = ''.join(open(filename, 'r').readlines())

    # only read float
    string = re.findall(r"[-+]?\d*\.\d+", text)

    # delete even column
    [eigenval_.append(float(ii)) for ii in string[1::2]]
    [occ_.append(float(ii)) for ii in string[0::2]]

    # transfer list to ==> numpy(float64) ==> torch
    eigenval = t.from_numpy(np.asarray(eigenval_)) / eV_Hat
    occ = t.from_numpy(np.asarray(occ_))
    return eigenval, occ


def read_detailed_out(natom):
    """Read output file detailed.out."""
    qatom, dip = [], []
    text = ''.join(open('detailed.out', 'r').readlines())
    E_tot_ = re.search('(?<=Total energy:).+(?=\n)',
                       text, flags = re.DOTALL | re.MULTILINE).group(0)
    E_tot = re.findall(r"[-+]?\d*\.\d+", E_tot_)[0]

    # read charge
    text2 = re.search('(?<=Atom       Population\n).+(?=\n)',
                      text, flags = re.DOTALL | re.MULTILINE).group(0)
    qatom_ = re.findall(r"[-+]?\d*\.\d+", text2)[:natom]
    [qatom.append(float(ii)) for ii in qatom_]

    # read dipole (Debye)
    text3 = re.search('(?<=Dipole moment:).+(?=\n)',
                      text, flags = re.DOTALL | re.MULTILINE).group(0)
    dip_ = re.findall(r"[-+]?\d*\.\d+", text3)[-3::]
    [dip.append(float(ii)) for ii in dip_]

    return float(E_tot), \
        t.from_numpy(np.asarray(qatom)), \
            t.from_numpy(np.asarray(dip))