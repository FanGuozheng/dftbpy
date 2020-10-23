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

    # Main task: dftb, mlCompressionR, mlIntegral
    if 'task' not in parameter.keys():
        parameter['task'] = 'dftb'

    # is machine learning is on, it means that the task is machine learning
    parameter['Lml'] = True if parameter['task'] in ('mlCompressionR', 'mlIntegral') else False

    # precision control: t.float64, t.float32
    if 'precision' not in parameter.keys():
        parameter['precision'] = t.float64

    # batch calculation, usually True for machine learning
    if 'Lbatch' not in parameter.keys():
        parameter['Lbatch'] = True if parameter['Lml'] is True else False

    # if True, read parameters from input (default name: dftb_in)
    if 'LReadInput' not in parameter.keys():
        parameter['LReadInput'] = False

    # SCC, non-SCC DFTB calculation
    if 'scc' not in parameter.keys():
        parameter['scc'] = 'scc'

    # if the system perodic condition, with lattice parameter, switch to True
    if 'Lperiodic' not in parameter.keys():
        parameter['Lperiodic'] = False

    # if calculate repulsive or not
    if 'Lrepulsive' not in parameter.keys():
        parameter['Lrepulsive'] = False

    # plot results during or after calculations
    if 'Lplot' not in parameter.keys():
        parameter['Lplot'] = False

    # input (dftb_in) directory
    if 'directory' not in parameter.keys():
        parameter['directory'] = path

    # SKF directory
    if 'directorySK' not in parameter.keys():
        parameter['directorySK'] = path

    # dire of skf dataset (write SKF as binary file)
    if 'SKDataset' not in parameter.keys():
        if parameter['task'] == 'mlCompressionR':
            parameter['SKDataset'] = '../slko/hdf/skf.hdf5'
        elif parameter['task'] == 'mlIntegral':
            parameter['SKDataset'] = '../slko/hdf/skfmio.hdf5'

    # input file name, default is dftb_in
    if 'inputName' not in parameter.keys():
        parameter['inputName'] = 'dftb_in'

    # mixing method: simple. anderson, broyden
    if 'mixMethod' not in parameter.keys():
        parameter['mixMethod'] = 'anderson'

    # mixing factor
    if 'mixFactor' not in parameter.keys():
        parameter['mixFactor'] = 0.2

    # convergence method: energy, charge
    if 'convergenceType' not in parameter.keys():
        parameter['convergenceType'] = 'energy'

    # convergence tolerance
    if 'convergenceTolerance' not in parameter.keys():
        parameter['convergenceTolerance'] = 1e-8

    # system temperature
    if 'tElec' not in parameter.keys():
        parameter['tElec'] = 0

    # max interation of SCC loop
    if 'maxIteration' not in parameter.keys():
        parameter['maxIteration'] = 60

    # density basis: spherical, gaussian
    if 'densityProfile' not in parameter.keys():
        parameter['densityProfile'] = 'spherical'

    # coordinate type: 'C': cartesian...
    if 'coordinateType' not in parameter.keys():
        parameter['coordinateType'] = 'C'

    # ******************************** H, S ********************************
    # how to write H0, S: half (write upper or lower HS), all (write whole HS)
    if 'HSSymmetry' not in parameter.keys():
        parameter['HSSymmetry'] = 'all'

    # general eigenvalue method: cholesky, lowdinQR
    if 'eigenMethod' not in parameter.keys():
        parameter['eigenMethod'] = 'cholesky'

    # ****************************** MBD-DFTB ******************************
    if 'LMBD_DFTB' not in parameter.keys():
        parameter['LMBD_DFTB'] = False

    # omega grid
    if 'nOmegaGrid' not in parameter.keys():
        parameter['nOmegaGrid'] = 15
    if 'vdwConsistent' not in parameter.keys():
        parameter['vdwConsistent'] = False
    if 'beta' not in parameter.keys():
        parameter['beta'] = 1.05

    # **************************** Result *********************************
    # calculate dipole
    if 'Ldipole' not in parameter.keys():
        parameter['Ldipole'] = True

    # if calculate PDOS
    if 'Lpdos' not in parameter.keys():
        parameter['Lpdos'] = False

    if 'Leigval' not in parameter.keys():
        parameter['Leigval'] = False

    if 'Lenergy' not in parameter.keys():
        parameter['Lenergy'] = True

    # calculate HOMO-LUMO
    if 'LHomoLumo' not in parameter.keys():
        parameter['LHomoLumo'] = True

    # return DFTB calculation parameters
    return parameter


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

    # optional datatype: ani, json, hdf
    if 'datasetType' not in dataset.keys():
        dataset['datasetType'] = 'ani'

    # get the dataset path
    if 'directoryDataset' not in dataset.keys():
        dataset['directoryDataset'] = '../data/dataset/'

    # directly read SKF or interpolate from a list of skf files
    if 'LSKFinterpolation' not in dataset.keys():
        dataset['LSKFinterpolation'] = False

    # define path of files with feature information in machine learning
    if 'pathFeature' not in dataset.keys():
        dataset['pathFeature'] = '.'

    # how many molecules for each molecule specie !!
    if 'sizeDataset' not in dataset.keys():
        dataset['sizeDataset'] = ['1']

    # used to test (optimize ML algorithm parameters) !!
    if 'sizeTest' not in dataset.keys():
        dataset['sizeTest'] = ['2']

    # mix different molecule specie type
    if 'LdatasetMixture' not in dataset.keys():
        dataset['LdatasetMixture'] = True

    # read json type geometry
    if 'json' in dataset['datasetType']:

        # path of data
        if 'Dataset' not in dataset.keys():
            dataset['Dataset'] = '../data/json/H2_data'

    # read ANI dataset
    elif 'ani' in dataset['datasetType']:

        # add hdf data: ani_gdb_s01.h5 ... ani_gdb_s08.h5
        hdffilelist = os.path.join(dataset['directoryDataset'], 'an1/ani_gdb_s01.h5')

        # transfer to list
        if 'Dataset' not in dataset.keys():
            dataset['Dataset'] = [hdffilelist]

    # read QM7 dataset
    elif 'qm7' in dataset['datasetType']:

        # define path and dataset name
        dataset['Dataset'] = os.path.join(dataset['directoryDataset'], 'qm7.mat')

    # test the molecule specie is the same (the length means speices size)
    assert len(dataset['sizeDataset']) == len(dataset['sizeTest'])

    return dataset


def init_ml(para=None, dataset=None, ml=None):
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

    # machine learning algorithm: linear, svm, schnet, nn...!!!!
    if 'MLmodel' not in ml.keys():
        ml['MLmodel'] = 'linear'

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

        # the grid point of compression radius is not uniform
        if 'directoryInterpSK' not in ml.keys():
            ml['directoryInterpSK'] = path

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

    return para, ml, dataset


def skf_parameter(skf=None):
    """Return the default parameters for skf.

    Returns:
    ----
        skf: `dictionary`
        only for normal skf files, dataset skf parameters is in dataset.
    """
    if skf is None:
        skf = {}

    # smooth the tail when read the skf
    if 'LSmoothTail' not in skf.keys():
        skf['LSmoothTail'] = True

    # skf file tail distance
    if 'tailSKDistance' not in skf.keys():
        skf['tailSKDistance'] = 1.0

    # skf integral interpolation points number
    if 'sizeInterpolationPoints' not in skf.keys():
        skf['sizeInterpolationPoints'] = 8

    # SK transformation method
    if 'transformationSK' not in skf.keys():
        skf['transformationSK'] = 'new'

    # delta distance when interpolate SKF integral when calculate gradient
    if 'deltaSK' not in skf.keys():
        skf['deltaSK'] = 1E-5

    # orbital resolved: if Ture, only use Hubbert of s orbital
    if 'LOrbitalResolve' not in skf.keys():
        skf['LOrbitalResolve'] = False

    # if optimize (True) or fix (False) onsite in DFTB-ML
    if 'Lonsite' not in skf.keys():
        skf['Lonsite'] = False

    # define onsite
    if not skf['LOrbitalResolve']:
        skf['onsiteHH'] = t.tensor((
            [0.0E+00, 0.0E+00, -2.386005440483E-01]), dtype=t.float64)
        skf['onsiteCC'] = t.tensor((
            [0.0E+00, -1.943551799182E-01, -5.048917654803E-01]),
            dtype=t.float64)
        skf['onsiteNN'] = t.tensor((
            [0.0E+00, -2.607280834222E-01, -6.400000000000E-01]),
            dtype=t.float64)
        skf['onsiteOO'] = t.tensor((
            [0.0E+00, -3.321317735288E-01, -8.788325840767E-01]),
            dtype=t.float64)

        # Hubbert is not orbital resolved, value from skgen
        skf['uhubbHH'] = t.tensor(([4.196174261214E-01,
                                    4.196174261214E-01,
                                    4.196174261214E-01]), dtype=t.float64)
        skf['uhubbCC'] = t.tensor(([3.646664973641E-01,
                                    3.646664973641E-01,
                                    3.646664973641E-01]), dtype=t.float64)
        skf['uhubbNN'] = t.tensor(([4.308879578818E-01,
                                    4.308879578818E-01,
                                    4.308879578818E-01]), dtype=t.float64)
        skf['uhubbOO'] = t.tensor(([4.954041702122E-01,
                                    4.954041702122E-01,
                                    4.954041702122E-01]), dtype=t.float64)

    # Hubbert is orbital resolved
    # if use different parametrization method, remember revise value here
    elif skf['LOrbitalResolve']:
        skf['uhubbHH'] = t.tensor((
            [0.0E+00, 0.0E+00, 4.196174261214E-01]), dtype=t.float64)
        skf['uhubbCC'] = t.tensor((
            [0.0E+00, 3.646664973641E-01, 3.646664973641E-01]), dtype=t.float64)
        skf['uhubbNN'] = t.tensor((
            [0.0E+00, 4.308879578818E-01, 4.308879578818E-01]), dtype=t.float64)
        skf['uhubbOO'] = t.tensor((
            [0.0E+00, 4.954041702122E-01, 4.954041702122E-01]), dtype=t.float64)

    # return skf
    return skf
