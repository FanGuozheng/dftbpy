#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
init the parameters for DFTB-ML optimization and dftb calculation
normally, you can read this parameters from specified file or define here
"""
import torch as t
import os


def init_dftb_ml(para):
    '''
    In general, you have to define:
        dataType and related parameters
        DFTB-ML parameters and realted
        DFTB parameters
        Others, such as plotting parameters
    '''
    # --------------------------- load data -------------------------------
    para['dataType'] = 'hdf'  # optional datatype: hdf, json
    path = os.getcwd()  # get the current path
    if para['dataType'] == 'json':
        para['pythondata_dire'] = '../data'  # path of data
        para['pythondata_file'] = 'CH4_data'  # name of data in defined path
        para['n_dataset'] = ['2']
        para['dire_interpSK'] = os.path.join(path, '../slko')
    elif para['dataType'] == 'hdf':
        hdffilelist = []
        hdffilelist.append(os.path.join(path, 'data/an1/ani_gdb_s01.h5'))
        para['hdffile'] = hdffilelist
        para['hdf_num'] = 3  # this will determine read which type of molecule
        para['n_dataset'] = ['50']  # how many molecules
        # para['dire_interpSK'] = os.path.join(path, '../slko')
    # para['optim_para'] = ['Hamiltonian']

    # ------------------  ML and environment parameters -------------------
    # ML training
    para['testMLmodel'] = 'linear'
    para['feature'] = 'rad'
    para['direfeature'] = '.'

    para['rcut'] = 15
    para['r_s'] = 5
    para['eta'] = 0.1
    para['tol'] = 1E-4
    para['zeta'] = 1
    para['lambda'] = 1
    para['ang_paraall'] = []
    para['rad_paraall'] = []

    # ----------------------------- DFTB-ML -----------------------------
    # splinetype: Bspline, Polyspline
    para['ref'] = 'aims'  # optional reference: aims, dftbplus, dftb
    para['target'] = ['dipole']  # dipole, homo_lumo, gap, eigval, qatomall
    para['mlsteps'] = 100  # how many steps for optimizing in DFTB-ML
    para['save_steps'] = 10  # how many steps to save the DFTB-ML data
    para['Lml'] = True  # is DFTB-ML, if not, it will perform normal DFTB
    para['lr'] = 8e-1  # learning rate

    # the follwing is ML target, if optimize compression radius, integrals...
    para['Lml_skf'] = True  # if use interp to gen .skf with compress_r
    para['Lml_HS'] = False  # if use interp to gen HS mat (e.g Polyspline)
    para['Lml_compr'] = False  # test gradients of interp of SK table
    para['Lml_compr_global'] = False  # each spiece has the same compress_r
    if para['Lml_HS']:
        para['interptype'] = 'Polyspline'
        para['zero_threshold'] = 5E-3
        para['rand_threshold'] = 5E-2
    para['interpdist'] = 0.4
    para['interpcutoff'] = 10
    para['atomspecie_old'] = []
    if para['Lml_skf']:
        para['LreadSKFinterp'] = True
        para['Lonsite'] = False

        para['typeSKinterp'] = 'uniform'  # if the grid of compr is uniform
        if para['typeSKinterp'] == 'nonuniform':
            para['dire_interpSK'] = os.path.join(path, '../slko/nonuniform')
        elif para['typeSKinterp'] == 'uniform':
            para['dire_interpSK'] = os.path.join(path, '../slko/uniform')
        para['ncompr'] = 10
        para['H_compr_grid'] = t.tensor((
                [2.00, 2.50, 3.00, 3.50, 4.00, 4.50, 5.00, 6.00, 8.00, 10.00]),
                dtype=t.float64)
        para['C_compr_grid'] = t.tensor((
                [2.00, 2.50, 3.00, 3.50, 4.00, 4.50, 5.00, 6.00, 8.00, 10.00]),
                dtype=t.float64)
        para['N_compr_grid'] = t.tensor((
                [2.00, 2.50, 3.00, 3.50, 4.00, 4.50, 5.00, 6.00, 8.00, 10.00]),
                dtype=t.float64)
        para['O_compr_grid'] = t.tensor((
                [2.00, 2.50, 3.00, 3.50, 4.00, 4.50, 5.00, 6.00, 8.00, 10.00]),
                dtype=t.float64)
        assert len(para['H_compr_grid']) == para['ncompr']
        assert len(para['C_compr_grid']) == para['ncompr']
        assert len(para['N_compr_grid']) == para['ncompr']
        assert len(para['O_compr_grid']) == para['ncompr']

        para['H_init_compr'] = 3.0
        para['C_init_compr'] = 3.0
        para['N_init_compr'] = 3.0
        para['O_init_compr'] = 3.0
        para['onsiteH'] = t.tensor((
                [0.0E+00, 0.0E+00, -2.386005440483E-01]), dtype=t.float64)
        para['onsiteC'] = t.tensor((
                [0.0E+00, -1.943551799182E-01, -5.048917654803E-01]),
                dtype=t.float64)
        para['onsiteN'] = t.tensor((
                [0.0E+00, -2.607280834222E-01, -6.400000000000E-01]),
                dtype=t.float64)
        para['onsiteO'] = t.tensor((
                [0.0E+00, -3.321317735288E-01, -8.788325840767E-01]),
                dtype=t.float64)
        para['uhubbH'] = t.tensor((
                [0.0E+00, 0.0E+00, 4.196174261214E-01]), dtype=t.float64)
        para['uhubbC'] = t.tensor((
                [0.0E+00, 3.646664973641E-01, 3.646664973641E-01]),
                dtype=t.float64)
        para['uhubbN'] = t.tensor((
                [0.0E+00, 4.308879578818E-01, 4.308879578818E-01]),
                dtype=t.float64)
        para['uhubbO'] = t.tensor((
                [0.0E+00, 4.954041702122E-01, 4.954041702122E-01]),
                dtype=t.float64)

    # ----------------------------- DFTB -----------------------------
    para['LReadInput'] = False
    para['LMBD_DFTB'] = False
    para['convergenceType'], para['energy_tol'] = 'energy', 1e-6
    para['scf'] = True
    para['scc'] = 'scc'
    para['HSsym'] = 'symall_chol'  # symall, symhalf. important!!!!!!
    para['ninterp'] = 8
    para['dist_tailskf'] = 1.0
    para['mixMethod'], para['mixFactor'] = 'anderson', 0.2
    para['tElec'] = 0
    para['maxIter'] = 60
    para['Lperiodic'] = False
    para['Ldipole'] = True
    para['Lrepulsive'] = False
    para['coorType'] = 'C'
    para['filename'] = 'dftb_in'
    para['direInput'] = os.path.join(path, 'dftbtorch')

    # ---------------------- plotting and others ----------------------
    para['plot_ham'] = True
    para['hamold'] = 0
    return para


def init_dftb(para):
    pass
