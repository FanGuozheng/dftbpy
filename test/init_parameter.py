"""This documents offer parameters definition for DFTB-ML.

What is included:
    DFTB-ML parameters for optimization
    Testing parameters for DFTB-ML
    Only DFTB parameters
"""

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch as t
import os


def init_dftb_ml(para):
    """Initialize the optimization of certain physical properties.

    In general, you have to define:
        dataType and related parameters
        DFTB-ML parameters and realted
        DFTB parameters
        Others, such as plotting parameters
    """
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
        para['hdf_num'] = 1  # this will determine read which type of molecule
        para['n_dataset'] = ['30']  # how many molecules used to optimize
        para['n_test'] = ['20']  # how many used to test
        assert len(para['n_dataset']) == len(para['n_test'])
    # para['optim_para'] = ['Hamiltonian']

    # ------------------  ML and environment parameters -------------------
    para['testMLmodel'] = 'linear'  # linear, svm, schnet, nn
    para['featureType'] = 'cm'  # rad, cm (CoulombMatrix), acsf
    if para['featureType'] == 'acsf':
        para['Lg2'] = True
        para['Lg3'] = True
        para['Lg4'] = True
        para['Lg5'] = True
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
    # dipole, homo_lumo, gap, eigval, qatomall, polarizability, cpa
    para['target'] = ['cpa']
    para['mlsteps'] = 30  # how many steps for optimizing in DFTB-ML
    para['save_steps'] = 5  # how many steps to save the DFTB-ML data
    para['Lml'] = True  # is DFTB-ML, if not, it will perform normal DFTB
    para['lr'] = 1.5  # learning rate

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

    para['LMBD_DFTB'] = True
    para['n_omega_grid'] = 15  # mbd_vdw_n_quad_pts = para['n_omega_grid']
    para['vdw_self_consistent'] = False
    para['beta'] = 1.05

    para['scf'] = True
    para['scc'] = 'scc'
    para['convergenceType'], para['energy_tol'] = 'energy', 1E-7
    para['delta_r_skf'] = 1E-5
    para['general_tol'] = 1E-4
    para['mixMethod'], para['mixFactor'] = 'anderson', 0.2
    para['HSsym'] = 'symall_chol'  # symall, symhalf. important!!!!!!
    para['ninterp'] = 8
    para['dist_tailskf'] = 1.0
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
    """Initialize the parameters for DFTB.

    In general, you have to define:
        Path of input (if you do not offer all in python)
        DFTB parameters
        Others, such as plotting parameters
    """
    para['LReadInput'] = False  # define parameters in python, not read input
    para['LreadSKFinterp'] = False
    para['Lml_HS'] = False  # donot perform ML process
    para['scf'] = True  # perform scf
    para['Lml'] = False  # only perform DFTB part without ML
    para['Lperiodic'] = False  # solid or molecule
    para['Ldipole'] = True  # if calculate dipole
    para['Lml_skf'] = False  # ML type SKF or not
    para['Lrepulsive'] = False  # if calculate repulsive
    para['mixMethod'], para['mixFactor'] = 'anderson', 0.2
    para['convergenceType'], para['energy_tol'] = 'energy',  1e-7
    para['delta_r_skf'] = 1E-5
    para['general_tol'] = 1E-4
    para['tElec'] = 0
    para['maxIter'] = 60
    para['HSsym'] = 'symall_chol'  # symhalf, symall, symall_chol
    para['dist_tailskf'] = 1.0
    para['ninterp'] = 8
    para['direSK'] = '../slko/test'


def init_dftb_interp(para):
    """Initialize the parameters for DFTB with interpolation of SKF.

    In general, you have to define:
        Path of input (if you do not offer all in python)
        DFTB parameters
        Others, such as plotting parameters
    """
    para['Lml'] = False  # only perform DFTB part without ML
    para['LReadInput'] = False  # define parameters in python, not read input
    para['LreadSKFinterp'] = True
    para['Lperiodic'] = False
    para['mixMethod'], para['mixFactor'] = 'anderson', 0.2
    para['convergenceType'], para['energy_tol'] = 'energy',  1e-7
    para['general_tol'] = 1E-4
    para['delta_r_skf'] = 1E-5
    para['tElec'] = 0
    para['maxIter'] = 60
    para['Ldipole'] = True
    para['HSsym'] = 'symall_chol'  # symhalf, symall, symall_chol
    para['dist_tailskf'] = 1.0
    para['ninterp'] = 8
    para['interpcutoff'] = 4.0
    para['Lml_skf'] = True
    para['Lml_HS'] = False
    para['Lrepulsive'] = False
    para['Lml_compr_global'] = False
    para['Lonsite'] = False
    para['atomspecie_old'] = []
    para['dire_interpSK'] = os.path.join(os.getcwd(), '../slko/uniform')
    para['H_init_compr'] = 2.5
    para['C_init_compr'] = 3.0
    para['H_compr_grid'] = t.tensor(([2.00, 2.50, 3.00, 3.50, 4.00, 4.50,
                                     5.00, 5.50, 6.00]), dtype=t.float64)
    para['C_compr_grid'] = t.tensor(([2.00, 2.50, 3.00, 3.50, 4.00, 4.50,
                                     5.00, 5.50, 6.00]), dtype=t.float64)
    para['onsiteH'] = t.tensor((
            [0.0E+00, 0.0E+00, -2.386005440483E-01]), dtype=t.float64)
    para['onsiteC'] = t.tensor((
            [0.0E+00, -1.943551799182E-01, -5.048917654803E-01]),
            dtype=t.float64)
    para['uhubbH'] = t.tensor((
            [0.0E+00, 0.0E+00, 4.196174261214E-01]), dtype=t.float64)
    para['uhubbC'] = t.tensor((
            [0.0E+00, 3.646664973641E-01, 3.646664973641E-01]),
            dtype=t.float64)
