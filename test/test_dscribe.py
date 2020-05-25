#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 23:49:08 2020

@author: gz_fan
"""
import numpy as np
from ase.io import read
from ml.dscribe.descriptors import SOAP
from ase import Atoms
from ase.build import molecule
from ml.dscribe.descriptors.acsf import ACSF


def test_acsf():
    # ------------------- read in different way ------------------------
    # structure1 = read('water.xyz')
    structure2 = molecule('H2O')
    structure3 = Atoms(
            symbols=['C', 'O'], positions=[[0, 0, 0], [1.128, 0, 0]])
    structures = [structure2, structure3]

    species = set()
    for structure in structures:
        species.update(structure.get_chemical_symbols())
    soap = SOAP(
            species=species,
            periodic=False,
            rcut=5,
            nmax=8,
            lmax=8,
            average=True,
            sparse=False
            )

    feature_vector = soap.create(structures, n_jobs=1)
    print('feature_vector: \n', feature_vector[:], structure2.positions)

    # ------------------------ H2O and CO test -------------------------
    acsf = ACSF(
            species=["H", "O"],
            rcut=6.0,
            g2_params=[[1, 1]],
            # g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]],
            g4_params=[[1, 1, 1]],
            )
    acsf2 = ACSF(
            species=["C", "O"],
            rcut=6.0,
            g2_params=[[1, 1]],
            # g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]],
            g4_params=[[1, 1, 1]],
            )

    water = molecule("H2O")
    co = Atoms(symbols=['C', 'O'], positions=[[0, 0, 0], [1.128, 0, 0]])
    acsf_water = acsf.create(water, positions=np.array([0, 1, 2]))
    acsf_co = acsf2.create(co, positions=np.array([0, 1]))
    print('acsf_water: \n', acsf_water)
    print('acsf_co: \n', acsf_co)
    print(acsf_water.shape, acsf_co.shape)


def test_coulomb():
    pass


test_acsf()
