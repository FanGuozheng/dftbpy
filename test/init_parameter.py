"""This documents offer parameters definition for DFTB-ML.

What is included:
    init_dftb_ml: DFTB-ML parameters for optimization and testing
    init_dftb: only DFTB parameters
    init_dftb_interp: DFTB with SKF interpolation
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
    # ************************** load data **************************

    # optional datatype: ani, json, qm7
    para['dataType'] = 'json'

    # get the current path
    path = os.getcwd()
    dire_data = '../data/dataset/'
    if para['dataType'] == 'json':
        para['pythondata_dire'] = '../data/json'  # path of data
        para['pythondata_file'] = 'CH4_data'  # name of data in defined path
        para['n_dataset'] = [1]
        para['dire_interpSK'] = os.path.join(path, '../slko')
    elif para['dataType'] == 'ani':
        hdffilelist = []
        hdffilelist.append(os.path.join(dire_data, 'an1/ani_gdb_s01.h5'))
        para['hdffile'] = hdffilelist
        para['hdf_num'] = [['all']]  # determine the type of molecule!!!!!
        para['n_dataset'] = ['500']  # how many molecules used to optimize!!!!!
        para['n_test'] = ['500']  # used to test!!!!! n_test >= n_dataset!!!!!
        para['hdf_mixture'] = True  # mix different molecule type
        assert len(para['n_dataset']) == len(para['n_test'])
    elif para['dataType'] == 'qm7':
        para['qm7_data'] = os.path.join(dire_data, 'qm7.mat')
        para['train_specie'] = [1, 6, 8]
        para['n_dataset'] = ['5']  # how many molecules used to optimize!!!!!
        para['n_test'] = ['5']  # how many used to test!!!!!
        assert len(para['n_dataset']) == len(para['n_test'])

    # ------------------  ML and environment parameters -------------------
    para['testMLmodel'] = 'linear'  # linear, svm, schnet, nn...!!!!!
    para['featureType'] = 'acsf'  # rad, cm (CoulombMatrix), acsf!!!!!
    if para['featureType'] == 'acsf':
        para['acsf_rcut'] = 6.0
        para['Lacsf_g2'] = True
        para['acsf_g2'] = [[1, 1]]
        para['Lacsf_g3'] = False
        para['Lacsf_g4'] = True
        para['acsf_g4'] = [[0.02, 1, -1]]
        para['Lacsf_g5'] = False
    # for test, where to read the optimized parameters, 1 means read the last
    # optimized para in dire_data, 0 is the unoptimized !!!!!
    para['opt_para_test'] = 1.0
    para['direfeature'] = '.'
    # get ACSF by hand, if featureType is rad
    para['rcut'] = 15
    para['r_s'] = 5
    para['eta'] = 0.1
    para['tol'] = 1E-4
    para['zeta'] = 1
    para['lambda'] = 1
    para['ang_paraall'] = []
    para['rad_paraall'] = []

    # ----------------------------- DFTB-ML -----------------------------
    para['reference'] = 'aims'  # optional reference: aims, dftbplus, dftb!!!!!
    # dipole, homo_lumo, gap, eigval, qatomall, polarizability, cpa...!!!!!
    para['target'] = ['dipole']
    para['dipole_loss_ratio'] = 1
    para['polarizability_loss_ratio'] = 0.25
    para['mlsteps'] = 2  # how many steps for optimize in DFTB-ML!!!!!
    para['save_steps'] = 1  # how many steps to save the DFTB-ML data!!!!!
    para['opt_step_min'] = 2
    para['lr'] = 5E-3  # learning rate !!!!!
    para['loss_function'] = 'MSELoss'  # MSELoss, L1Loss

    para['Lml'] = True  # is DFTB-ML, if not, it will perform normal DFTB
    para['Lml_skf'] = False  # if optimize compress_r!!!!!
    para['Lml_HS'] = False  # if optimize HS mat (e.g Polyspline)
    para['Lml_compr'] = False  # test gradients of interp of SK table
    para['Lml_compr_global'] = False  # each spiece has the same compress_r
    para['Lml_acsf'] = True
    if para['Lml_HS']:
        para['interptype'] = 'Polyspline'
        para['zero_threshold'] = 5E-3
        para['rand_threshold'] = 5E-2
    para['interpdist'] = 0.4
    para['atomspecie_old'] = []

    if para['Lml_skf'] or para['Lml_acsf']:
        # if read SKF from a list of files with interpolation
        para['LreadSKFinterp'] = True
        para['Lonsite'] = False  # if optimize onsite in DFTB-ML
        para['typeSKinterp'] = 'uniform'  # grid of compr is uniform or ?!!!!!
        para['typeSKinterpR'] = 'all'  # all, wavefunction, density...
        if para['typeSKinterp'] == 'nonuniform':
            para['dire_interpSK'] = os.path.join(path, '../slko/nonuniform')
        elif para['typeSKinterp'] == 'uniform':
            para['dire_interpSK'] = os.path.join(path, '../slko/uniform')
        para['ncompr'] = 10  # should be equal to atom_compr_grid
        # if fix the optimization step and set convergence condition
        para['Lopt_step'] = True
        para['opt_ml_tol'] = 1E-3
        para['Lopt_ml_compr'] = False  # if predict compR during DFTB-ML!!!!!
        # after opt_ml_step*nbatch molecule, perform ML predict compR!!!!!!
        para['opt_ml_step'] = 0.5
        para['opt_ml_all'] = False
        para['compr_min'] = 2.2  # if any compR < 2.2, break DFTB-ML loop
        para['compr_max'] = 9  # if any compR > 9, break DFTB-ML loop
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

        para['H_init_compr'] = 3.5
        para['C_init_compr'] = 3.5
        para['N_init_compr'] = 3.5
        para['O_init_compr'] = 3.5
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
    para['LReadInput'] = False  # if read parameter from dftb_in file
    para['LMBD_DFTB'] = True  # if perform MBD-DFTB calculation
    para['n_omega_grid'] = 15  # mbd_vdw_n_quad_pts = para['n_omega_grid']
    para['vdw_self_consistent'] = False
    para['beta'] = 1.05
    # general eigenvalue methodin DFTB-ML: cholesky, lowdin_qr!!!!!
    para['eigenmethod'] = 'cholesky'
    para['scf'] = True
    para['scc'] = 'scc'
    para['convergenceType'], para['energy_tol'] = 'energy', 1E-6
    para['delta_r_skf'] = 1E-5
    para['general_tol'] = 1E-4
    para['mixMethod'], para['mixFactor'] = 'anderson', 0.2
    # if build half H0, S or build whole H0, S: symall, symhalf.. !!!!!
    para['HSsym'] = 'symall'
    para['ninterp'] = 8  # interpolation integrals when read SKF with distance
    para['dist_tailskf'] = 1.0  # smooth tail of SKF
    para['tElec'] = 0
    para['maxIter'] = 60  # max of SCF loop
    para['Lperiodic'] = False
    para['Ldipole'] = True
    para['Lrepulsive'] = False
    para['coorType'] = 'C'  # cartesian...
    para['filename'] = 'dftb_in'
    para['direInput'] = os.path.join(path, 'dftbtorch')

    # ---------------------- plotting and others ----------------------
    para['Lplot_ham'] = True
    para['Lplot_feature'] = False
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
    para['HS_spline'] = False
    para['mixMethod'], para['mixFactor'] = 'anderson', 0.2
    para['convergenceType'], para['energy_tol'] = 'energy',  1e-7
    para['delta_r_skf'] = 1E-5
    para['general_tol'] = 1E-4
    para['tElec'] = 0
    para['maxIter'] = 60
    para['HSsym'] = 'symall'  # symhalf, symall
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
    para['HSsym'] = 'symall'  # symhalf, symall
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
