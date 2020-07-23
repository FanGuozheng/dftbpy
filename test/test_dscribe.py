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
import test.test_grad_compr as compr


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


def test_acsf_for_dftb_ml():
    r"""Test the G2 and G4 parameters:

    Results of pred_error / dftb+_error when only using G2:
        G2: 0.5354001357166923, 0.5389358895492274, 0.5325853892664808,
        0.5301861033428172, 0.5287115327921316, 0.5336270663192335,
        0.5349144878694986, 0.5355534784120733, 0.5323373740153409,
        0.6126326134749273, 0.536918635881971, 0.5262552719336869,
        0.5265829666575547, 0.5464805594796515, 0.6256814124565421,
        0.5387164476713776, 0.5248554831501793, 0.542438780967216,
        0.5713781924496826, 0.6269527757182325, 0.5480595382122667,
        0.5522754902518822, 0.571847467014786, 0.6263858532245615,
        0.6297959913734027
        therefore [2, 2] is determined for G2 (\eta, R_s)
    Results of pred_error / dftb+_error when only using G4:
        0.5687255744444507, 0.5700938495741285, 0.5763594398321703,
        0.5942085006483682, 0.587053663882241, 0.5797089687823177,
        0.5857651721797287, 0.5803530533297074, 0.6146661055139869,
        0.5777667287380277, 0.5774239123043253, 0.5724250240366471,
        0.6202037863514859, 0.5652300358793572, 0.5612864648012391,
        0.5570177334911115, 0.566216642718892, 0.5732978575496268,
        0.5786621780161461, 0.59156157643023, 0.5853260332026019,
        0.582163832567171, 0.5866197312036721, 0.5797911273139243,
        0.6146723927958654, 0.5859229465780758, 0.5724734660695092,
        0.5704670440720648, 0.6275017290055176, 0.5754486320956607,
        0.5611192681435139, 0.5564739555239681, 0.5763438620553747,
        0.5983969516117763, 0.5978794886309906, 0.5990198955528153,
        0.5888879892839599, 0.5921142392247912, 0.5945209847597304,
        0.5875749140345606, 0.6143773145055769, 0.5925897947871739,
        0.5849159052016264, 0.5693502164187623, 0.6253260286898451,
        0.5712919885611055, 0.5628299142416171, 0.5594006360933076,
        0.6206007087795117, 0.5866410613236664, 0.5953198335928769,
        0.6014136299688317, 0.6146130165220798, 0.5936647925481664,
        0.5957972710964444, 0.5943997004188302, 0.6197311111591821,
        0.5959610769225566, 0.5840804857072008, 0.5826261524606692,
        0.6203812283041704, 0.5792527892700453, 0.5800716370720163,
        0.5701608295349253
        therefore [[0.02, 10, 5]] is determined for G4 (\eta, \sigma, \lamda)
    """
    para = {}
    para['task'] = 'test'
    para['dire_data'] = '../data/results/200718compr_300mol_dip'
    para['Lacsf_g2'] = False
    para['Lacsf_g4'] = True
    pred_dftb_ratio = []
    para['acsf_g2_all'] = [[0.1, 0.5], [0.1, 2], [0.1, 3], [0.1, 5], [0.1, 10],
                           [0.5, 0.5], [0.5, 2], [0.5, 3], [0.5, 5], [0.5, 10],
                           [1, 0.5], [1, 2], [1, 3], [1, 5], [1, 10],
                           [2, 0.5], [2, 2], [2, 3], [2, 5], [2, 10],
                           [5, 0.5], [5, 2], [5, 3], [5, 5], [5, 10]]
    para['acsf_g4_all'] = [[0.02, 2, 1]]
    para['acsf_g4_all'] = [[0.01, 1, -1], [0.01, 1, 1], [0.01, 1, 2],
                           [0.01, 1, 5], [0.01, 2, -1], [0.01, 2, 1],
                           [0.01, 2, 2], [0.01, 2, 5], [0.01, 5, -1],
                           [0.01, 5, 1], [0.01, 5, 2], [0.01, 5, 5],
                           [0.01, 10, -1], [0.01, 10, 1], [0.01, 10, 2],
                           [0.01, 10, 5], [0.02, 1, -1], [0.02, 1, 1],
                           [0.02, 1, 2], [0.02, 1, 5], [0.02, 2, -1],
                           [0.02, 2, 1], [0.02, 2, 2], [0.02, 2, 5],
                           [0.02, 5, -1], [0.02, 5, 1], [0.02, 5, 2],
                           [0.02, 5, 5], [0.02, 10, -1], [0.02, 10, 1],
                           [0.02, 10, 2], [0.02, 10, 5], [0.1, 1, -1],
                           [0.1, 1, 1], [0.1, 1, 2], [0.1, 1, 5], [0.1, 2, -1],
                           [0.1, 2, 1], [0.1, 2, 2], [0.1, 2, 5], [0.1, 5, -1],
                           [0.1, 5, 1], [0.1, 5, 2], [0.1, 5, 5],
                           [0.1, 10, -1], [0.1, 10, 1], [0.1, 10, 2],
                           [0.1, 10, 5], [0.5, 1, -1], [0.5, 1, 1],
                           [0.5, 1, 2], [0.5, 1, 5], [0.5, 2, -1], [0.5, 2, 1],
                           [0.5, 2, 2], [0.5, 2, 5], [0.5, 5, -1], [0.5, 5, 1],
                           [0.5, 5, 2], [0.5, 5, 5], [0.5, 10, -1],
                           [0.5, 10, 1], [0.5, 10, 2], [0.5, 10, 5]]
    if para['Lacsf_g2']:
        for g2 in para['acsf_g2_all']:
            para['acsf_g2'] = []
            para['acsf_g2'].append(g2)
            compr.testml(para)
            pred_dftb_ratio.append(para['dip_ratio_pred_dftb'])
    if para['Lacsf_g4']:
        for g4 in para['acsf_g4_all']:
            para['acsf_g4'] = []
            para['acsf_g4'].append(g4)
            compr.testml(para)
            pred_dftb_ratio.append(para['dip_ratio_pred_dftb'])
    print(pred_dftb_ratio)


test_acsf_for_dftb_ml()
