"""Interface to ASE"""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import numpy as np
import torch as t
from ase import Atoms
from ase.build import molecule
from ase.calculators.dftb import Dftb
from ase.optimize import QuasiNewton
from ase.io import write
import os
import dftbplus.asedftb as asedftb
import dftbtorch.dftb_torch as dftb_torch
from utils.save import SaveData


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

    def run_dftb(self, nbatch, coorall):
        """Run batch systems with ASE-DFTB."""
        # source and set environment for python, ase before calculations
        self.save = SaveData(self.para)
        os.system('source ase_env.sh')
        if self.para['Lpdos']:
            self.para['pdosdftbplus'] = []

        for ibatch in range(nbatch):
            # transfer specie style, e.g., ['C', 'H', 'H', 'H', 'H'] to 'CH4'
            # so that satisfy ase.Atoms style
            ispecie = ''.join(self.para['symbols'][ibatch])

            # run each molecule in batches
            self.ase_idftb(coorall[ibatch], ispecie)

            # process result data for each calculations
            self.process_iresult()

            # calculate PDOS or not
            if self.para['Lpdos']:
                self.para['natom'] = len(coorall[ibatch])

                # calculate PDOS
                dftb_torch.Analysis(self.para).pdos()
                self.para['pdosdftbplus'].append(self.para['pdos'])

                # save PDOS
                self.save.save2D(self.para['pdos'].numpy(),
                                 name='pdosref.dat', dire='.data', ty='a')


        # deal with DFTB data
        self.process_results()

        # remove DFTB files
        self.remove()

    def ase_idftb(self, coor, moleculespecie):
        """Build DFTB input by ASE."""
        # set Atoms with molecule specie and coordinates
        mol = Atoms(moleculespecie, positions=coor[:, 1:])

        # set DFTB caulation parameters
        cal = Dftb(Hamiltonian_='DFTB',
                   Hamiltonian_SCC='Yes',
                   Hamiltonian_SCCTolerance=1e-8,
                   Hamiltonian_MaxAngularMomentum_='',
                   Hamiltonian_MaxAngularMomentum_H='s',
                   Hamiltonian_MaxAngularMomentum_C='p',
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
        self.save.save1D(self.para['eigenvalue'],
                         name='HLdftbplus.dat', dire='.data', ty='a')

    def remove(self):
        """Remove all DFTB data after calculations."""
        os.system('rm dftb+ ase_env.sh band.out charges.bin detailed.out')
        os.system('rm dftb_in.hsd dftb.out dftb_pin.hsd eigenvec.bin')
        os.system('rm eigenvec.out geo_end.gen hamsqr1.dat oversqr.dat')

    def process_results(self):
        pass


class Aims:

    def __init__(self, para, setenv=False):
        self.para = para

        # DFTB+ binary name, normally dftb+
        self.dftb_bin = self.para['aims_bin']

        # DFTB+ binary path
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
    return t.from_numpy(eigenvec)


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
