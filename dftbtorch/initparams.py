"""Example of defining parameters for DFTB-ML by python code."""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import torch as t

# get global path
path = os.getcwd()


def dftb_parameter(parameter=None):
    """Return the general parameters for DFTB calculations.

    This function demonstrates the definition of general DFTB parameters.
    You can define from the very beginning otherwise the default parameters
    will be used for the following training or calculations.

    Args:
        parameter (dict, optional): a general dictionary which includes
            DFTB parameters, general environment parameters.

    Returns:
        parameter(dict, optional): default general parameters if not
            defined in advance.

    """
    parameter = {} if parameter is None else parameter

    # build temporal dictionary, the input parameter will override parameter_
    parameter_ = {
        # task: dftb, mlCompressionR, mlIntegral
        'task': 'dftb',

        # precision control: t.float64, t.float32
        'precision': t.float64,

        # if True, read parameters from input (default name: dftb_in)
        'LReadInput': False,

        # SCC, non-SCC DFTB calculation
        'scc': 'scc',

        # if the system perodic condition, with lattice parameter
        'Lperiodic': False,

        # if calculate repulsive or not
        'Lrepulsive': False,

        # plot results during or after calculations
        'Lplot': False,

        # input (dftb_in) directory
        'directory': path,

        # SKF directory
        'directorySK': path,

        # input file name, default is dftb_in
        'inputName': 'dftb_in',

        # mixing method: simple. anderson, broyden
        'mixMethod': 'anderson',

        # mixing factor
        'mixFactor': 0.2,

        # convergence method: energy, charge
        'convergenceType': 'energy',

        # convergence tolerance
        'convergenceTolerance': 1e-8,

        # system temperature
        'tElec': 0,

        # max interation of SCC loop
        'maxIteration': 60,

        # density basis: spherical, gaussian
        'densityProfile': 'spherical',

        # coordinate type: 'C': cartesian...
        'coordinateType': 'C',

        # write H0, S: half (write upper or lower HS), all (write whole HS)
        'HSSymmetry': 'all',

        # general eigenvalue method: cholesky, lowdinQR
        'eigenMethod': 'cholesky',

        # MBD-DFTB
        'LMBD_DFTB': False,

        # omega grid
        'nOmegaGrid': 15,
        'vdwConsistent': False,
        'beta': 1.05,

        # calculate dipole
        'Ldipole': True,

        # if calculate PDOS
        'Lpdos': False,
        'Leigval': False,
        'Lenergy': True,

        # calculate HOMO-LUMO
        'LHomoLumo': True}

    # update temporal parameter_ with input parameter
    parameter_.update(parameter)

    # is machine learning is on, it means that the task is machine learning
    parameter_['Lml'] = True if parameter_['task'] in ('mlCompressionR', 'mlIntegral') else False

    # batch calculation, usually True for machine learning
    parameter_['Lbatch'] = True if parameter_['Lml'] is True else False

    # dire of skf dataset (write SKF as binary file)
    if parameter_['task'] == 'mlCompressionR':
        parameter_['SKDataset'] = '../slko/hdf/skf.hdf5'
    elif parameter_['task'] == 'mlIntegral':
        parameter_['SKDataset'] = '../slko/hdf/skfmio.hdf5'

    # return DFTB calculation parameters
    return parameter_


def init_dataset(dataset=None):
    """Return the dataset or geometric parameters for DFTB calculations.

    If parameters in dataset have nothing to do with ML, the parameters will be
    in dict dataset, else it will be in ml.
    If parameters used in both non-ML and ML, it will be in dict dataset.

    Args:
        dataset (dict, optional): a dictionary which includes dataset,
            geometric parameters.

    Returns:
        dataset (dict, optional): default dataset, geometric parameters if not
            defined in advance.

    """
    dataset = {} if dataset is None else dataset

    dataset_ = {
        # optional datatype: ani, json, hdf
        'datasetType': 'ani',

        # get the dataset path
        'directoryDataset': '../data/dataset/',

        # directly read SKF or interpolate from a list of skf files
        'LSKFinterpolation': False,

        # define path of files with feature information in machine learning
        'pathFeature': '.',

        # how many molecules for each molecule specie !!
        'sizeDataset': ['2'],

        # used to test (optimize ML algorithm parameters) !!
        'sizeTest': ['2'],

        # mix different molecule specie type
        'LdatasetMixture': True}

    # update temporal dataset_ with input dataset
    dataset_.update(dataset)

    # read json type geometry
    if 'json' in dataset_['datasetType']:
        if 'Dataset' not in dataset_.keys():
            dataset_['Dataset'] = '../data/json/H2_data'

    # read ANI dataset
    elif 'ani' in dataset_['datasetType']:
        # add hdf data: ani_gdb_s01.h5 ... ani_gdb_s08.h5
        hdffilelist = os.path.join(dataset_['directoryDataset'], 'an1/ani_gdb_s01.h5')

        # transfer to list
        if 'Dataset' not in dataset_.keys():
            dataset_['Dataset'] = [hdffilelist]

    # read QM7 dataset
    elif 'qm7' in dataset_['datasetType']:
        # define path and dataset name
        dataset_['Dataset'] = os.path.join(dataset_['directoryDataset'], 'qm7.mat')

    # test the molecule specie is the same (the length means speices size)
    assert len(dataset_['sizeDataset']) == len(dataset_['sizeTest'])

    return dataset_


def init_ml(para=None, dataset=None, skf=None, ml=None):
    """Return the machine learning parameters for DFTB calculations.

    Args:
        dataset (dict, optional): a dictionary which includes dataset,
            geometric parameters.

    Returns:
        dataset (dict, optional): default dataset, geometric parameters if not
            defined in advance.

    """
    para = {} if para is None else para
    ml = {} if ml is None else ml
    dataset = {} if dataset is None else dataset
    skf = {} if skf is None else skf

    # machine learning algorithm: linear, svm, schnet, nn...!!!!
    if 'MLmodel' not in ml.keys():
        ml['MLmodel'] = 'linear'

    if para['task'] == 'mlCompressionR':
        skf['ReadSKType'] = 'compressionRadii'

    if para['task'] == 'mlIntegral':
        skf['ReadSKType'] = 'mlIntegral'

    # define atomic representation: cm (CoulombMatrix), acsf!!!!!
    if 'featureType' not in ml.keys():
        ml['featureType'] = 'acsf'

    # define ACSF parameter
    if ml['featureType'] == 'acsf':

        # cutoff, for G1
        if 'acsfRcut' not in ml.keys():
            ml['acsfRcut'] = 6.

        # G2 parameters
        if 'LacsfG2' not in ml.keys():
            ml['LacsfG2'] = True
        if 'acsfG2' not in ml.keys():
            ml['acsfG2'] = [[1., 1.]]

        # G3 parameters
        if 'LacsfG3' not in ml.keys():
            ml['LacsfG3'] = False

        # G4 parameters
        if 'LacsfG4' not in ml.keys():
            ml['LacsfG4'] = True
        if 'acsfG4' not in ml.keys():
            ml['acsfG4'] = [[0.02, 1., -1.]]

        # G5 parameters
        if 'LacsfG5' not in ml.keys():
            ml['LacsfG5'] = False

    # *********************************************************************
    #                              DFTB-ML
    # *********************************************************************
    # run referecne calculations or directly get read reference properties
    if 'runReference' not in ml.keys():
        ml['runReference'] = False

    # optional reference: aims, dftbplus, dftb, dftbase, aimsase !!
    if 'reference' not in ml.keys():
        ml['reference'] = 'hdf'

    # path to dataset data
    if 'referenceDataset' not in ml.keys():
        ml['referenceDataset'] = '../data/dataset/testfile.hdf5'

    # read hdf (with coordinates, reference physical properties) type
    if ml['reference'] == 'hdf':

        # if read SKF from a list of files with interpolation, instead from hdf
        if 'LSKFInterpolation' not in ml.keys():
            dataset['LSKFInterpolation'] = False

    if ml['reference'] in ('dftbase', 'dftbplus'):

        # path of binary, executable DFTB file
        if 'dftbplus' not in ml.keys():
            ml['dftbplus'] = '../test/bin/dftb+'

    if ml['reference'] in ('aimsase', 'aims'):

        # path of binary, executable FHI-aims file
        if 'aims' not in ml.keys():
            ml['aims'] = '../test/bin/aims.171221_1.scalapack.mpi.x'

        # path of atom specie parameters
        if 'aimsSpecie' not in ml.keys():
            ml['aimsSpecie'] = '../test/species_defaults/tight/'

    # dipole, homo_lumo, gap, eigval, polarizability, cpa, pdos, charge
    if 'target' not in ml.keys():
        ml['target'] = ['dipole']

    # If turn on some calculations related to these physical properties
    # turn on anyway
    if 'energy' in ml['target']:
        para['Lrepulsive'] = True

    # the machine learning energy type
    if 'mlEnergyType' not in ml.keys():
        para['mlEnergyType'] = 'formationEnergy'

    if 'cpa' in ml['target']:
        para['LMBD_DFTB'] = True

    # define weight in loss function
    if 'LossRatio' not in ml.keys():
        ml['LossRatio'] = [1]

    # how many steps for optimize in DFTB-ML !!
    if 'mlSteps' not in ml.keys():
        ml['mlSteps'] = 5

    # how many steps to save the DFTB-ML data !!
    if 'saveSteps' not in ml.keys():
        ml['saveSteps'] = 2

    # minimum ML steps
    if 'stepMin' not in ml.keys():
        ml['stepMin'] = 2

    # learning rate !!
    if 'lr' not in ml.keys():
        ml['lr'] = 3E-2

    # optimizer
    if 'optimizer' not in ml.keys():
        ml['optimizer'] = 'Adam'

    # define loss function: MSELoss, L1Loss
    if 'lossFunction' not in ml.keys():
        ml['lossFunction'] = 'MSELoss'

    # optimize integral directly
    if para['task'] == 'mlIntegral':

        # spline type to generate integral
        ml['interpolationType'] = 'Polyspline'

    # optimize compression radius: by interpolation or by ML prediction
    if para['task'] in ('mlCompressionR', 'ACSF'):

        # interpolation of compression radius: BiCub, BiCubVec
        if 'interpolationType' not in ml.keys():
            ml['interpolationType'] = 'BiCubVec'

    # grid of compression radius is uniform or not !!
    if 'typeSKinterp' not in ml.keys():
        ml['typeSKinterp'] = 'uniform'

    # skgen compression radius parameters: all, wavefunction, density
    if 'typeSKinterpR' not in ml.keys():
        ml['typeSKinterpR'] = 'all'

    # number of grid points, should be equal to atom_compr_grid
    if 'nCompressionR' not in ml.keys():
        ml['nCompressionR'] = 10

    # if any compR < 2.2, break DFTB-ML loop
    if 'compressionRMin' not in ml.keys():
        ml['compressionRMin'] = 2.2

    # if any compR > 9, break DFTB-ML loop
    if 'compressionRMax' not in ml.keys():
        ml['compressionRMax'] = 9

    # compression radius of H
    ml['H_compr_grid'] = t.tensor((
        [2., 2.5, 3., 3.5, 4., 4.5, 5., 6., 8., 10.]), dtype=t.float64)

    # compression radius of C
    ml['C_compr_grid'] = t.tensor((
        [2., 2.5, 3., 3.5, 4., 4.5, 5., 6., 8., 10.]), dtype=t.float64)

    # compression radius of N
    ml['N_compr_grid'] = t.tensor((
        [2., 2.5, 3., 3.5, 4., 4.5, 5., 6., 8., 10.]), dtype=t.float64)

    # compression radius of O
    ml['O_compr_grid'] = t.tensor((
        [2., 2.5, 3., 3.5, 4., 4.5, 5., 6., 8., 10.]), dtype=t.float64)

    # test the grid points length
    assert len(ml['H_compr_grid']) == ml['nCompressionR']
    assert len(ml['C_compr_grid']) == ml['nCompressionR']
    assert len(ml['N_compr_grid']) == ml['nCompressionR']
    assert len(ml['O_compr_grid']) == ml['nCompressionR']

    # set initial compression radius
    ml['H_init_compr'] = 3.5
    ml['C_init_compr'] = 3.5
    ml['N_init_compr'] = 3.5
    ml['O_init_compr'] = 3.5
    return para, dataset, skf, ml


def skf_parameter(skf=None):
    """Return the default parameters for skf.

    Returns:
    ----
        skf: `dictionary`
        only for normal skf files, dataset skf parameters is in dataset.
    """
    if skf is None:
        skf = {}

    skf_ = {
        # SKF type: normal, mask, compressionRadii, hdf
        'ReadSKType': 'normal',

        # smooth the tail when read the skf
        'LSmoothTail': True,

        # skf file tail distance
        'tailSKDistance': 1.0,

        # skf integral interpolation points number
        'sizeInterpolationPoints': 8,

        # SK transformation method
        'transformationSK': 'new',

        # delta distance when interpolate SKF integral when calculate gradient
        'deltaSK': 1E-5,

        # orbital resolved: if Ture, only use Hubbert of s orbital
        'LOrbitalResolve': False,

        # if optimize (True) or fix (False) onsite in DFTB-ML
        'Lonsite': False}

    # the parameters from skf will overwrite skf_ default parameters
    skf_.update(skf)

    # define onsite if not orbital resolved
    if not skf_['LOrbitalResolve']:
        skf_['onsiteHH'] = t.tensor((
            [0.0E+00, 0.0E+00, -2.386005440483E-01]), dtype=t.float64)
        skf_['onsiteCC'] = t.tensor((
            [0.0E+00, -1.943551799182E-01, -5.048917654803E-01]),
            dtype=t.float64)
        skf_['onsiteNN'] = t.tensor((
            [0.0E+00, -2.607280834222E-01, -6.400000000000E-01]),
            dtype=t.float64)
        skf_['onsiteOO'] = t.tensor((
            [0.0E+00, -3.321317735288E-01, -8.788325840767E-01]),
            dtype=t.float64)

        # Hubbert is not orbital resolved, value from skgen
        skf_['uhubbHH'] = t.tensor(([4.196174261214E-01,
                                     4.196174261214E-01,
                                     4.196174261214E-01]), dtype=t.float64)
        skf_['uhubbCC'] = t.tensor(([3.646664973641E-01,
                                     3.646664973641E-01,
                                     3.646664973641E-01]), dtype=t.float64)
        skf_['uhubbNN'] = t.tensor(([4.308879578818E-01,
                                     4.308879578818E-01,
                                     4.308879578818E-01]), dtype=t.float64)
        skf_['uhubbOO'] = t.tensor(([4.954041702122E-01,
                                     4.954041702122E-01,
                                     4.954041702122E-01]), dtype=t.float64)

    # Hubbert is orbital resolved
    # if use different parametrization method, remember revise value here
    elif skf_['LOrbitalResolve']:
        skf_['uhubbHH'] = t.tensor((
            [0.0E+00, 0.0E+00, 4.196174261214E-01]), dtype=t.float64)
        skf_['uhubbCC'] = t.tensor((
            [0.0E+00, 3.646664973641E-01, 3.646664973641E-01]), dtype=t.float64)
        skf_['uhubbNN'] = t.tensor((
            [0.0E+00, 4.308879578818E-01, 4.308879578818E-01]), dtype=t.float64)
        skf_['uhubbOO'] = t.tensor((
            [0.0E+00, 4.954041702122E-01, 4.954041702122E-01]), dtype=t.float64)

    # return skf
    return skf_
