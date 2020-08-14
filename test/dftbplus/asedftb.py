""" """
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
from ase import Atoms
from ase.build import molecule
from ase.calculators.dftb import Dftb
from ase import Atoms


def ase_dftb(coor, moleculespecie):
    os.system('source ase_env.sh')
    # mol = Atoms(moleculespecie, positions=coor[:, 1:])
    atoms = molecule("CH4")
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
    atoms.calc = cal
    try:
        atoms.get_potential_energy()
    except UnboundLocalError:
        atoms.calc.__dict__["parameters"]['Options_WriteHS']='No'
        x = atoms.get_potential_energy()


ase_dftb(coor=[], moleculespecie=[])