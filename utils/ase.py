"""Interface to ASE"""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import numpy as np
from ase import Atoms
from ase.build import molecule
from ase.calculators.dftb import Dftb
from ase.optimize import QuasiNewton
from ase.io import write
import os
import dftbplus.asedftb as asedftb


class DFTBASE:
    """Run DFTB calculations by ase."""

    def __init__(self, para):
        self.para = para

        # DFTB+ binary name, normally dftb+
        self.dftb_bin = self.para['dftb_bin']

        # DFTB+ binary path
        self.dftb_path = para['dftb_ase_path']

        # SKF path
        self.slko_path = self.para['skf_path']

        # set environment before calculations
        self.set_env()

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

    def run_batch(self, nbatch, coorall):
        """Run batch systems with ASE-DFTB."""
        # source and set environment for python, ase before calculations
        os.system('source ase_env.sh')

        for ibatch in range(nbatch):
            # transfer specie style, e.g., ['C', 'H', 'H', 'H', 'H'] to 'CH4'
            # so that satisfy ase.Atoms style
            ispecie = ''.join(self.para['symbols'][ibatch])

            # run each molecule in batches
            asedftb.ase_dftb(coorall[ibatch], ispecie)

            # process result data for each calculations
            self.process_iresult()

        # deal with DFTB data
        self.process_results()

        # remove DFTB files
        self.remove()

    def ase_dftb(moleculespecie, coor):
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
            x = mol.get_potential_energy()

    def process_iresult(self):
        """Process result for each calculation."""
        # read in the H (hamsqr1.dat) and S (oversqr.dat) matrices now
        S_mat = get_matrix('oversqr.dat')

        # read final eigenvector
        get_eigenvec('eigenvec.out')

        # read final eigenvalue
        get_eigenvec('band.out')

    def remove(self):
        """Remove all DFTB data after calculations."""
        os.system('rm dftb+ ase_env.sh band.out charges.bin detailed.out')
        os.system('rm dftb_in.hsd dftb.out dftb_pin.hsd eigenvec.bin')
        os.system('rm eigenvec.out geo_end.gen hamsqr1.dat oversqr.dat')


    def process_results(self):
        pass


def get_matrix(filename):
    """Read hamsqr1.dat and oversqr.dat."""
    text = ''.join(open(filename, 'r').readlines())
    string = re.search('(?<=MATRIX\n).+(?=\n)',text, flags = re.DOTALL).group(0)
    return np.array([[float(i) for i in row.split()] for row in string.split('\n')])


def get_eigenvec(filename):
    """Read eigenvec.out."""
    text = ''.join(open(filename, 'r').readlines())
    string = re.search('(?<=Eigenvector\n).+(?=\n)',text, flags = re.DOTALL)
    print(text, "string", string, type(text))
    [x.split(' ')[3] for x in text]

def get_eigenvalue(filename):
    """Read band.out."""
    pass
