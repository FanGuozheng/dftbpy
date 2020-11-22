"""Example of defining parameters for DFTB-ML by python code."""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import torch as t

# get global path
path = os.getcwd()


def dftb_parameter(parameter_=None):
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
    parameter_ = {} if parameter_ is None else parameter_

    # build new dictionary, the input parameter_ will override parameter
    parameter = {
        # dataset of skf
        'datasetSK': '../slko/hdf/skf.hdf5',

        # task: dftb, mlCompressionR, mlIntegral
        'task': 'dftb',

        # precision control: t.float64, t.float32, cuda.DoubleTensor
        'precision': t.cuda.DoubleTensor,

        # device
        'device': 'cpu',

        # if open pytorch profiler
        'profiler': False,

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

        # get inverse of tensor directly or from torch.solve or np.solve
        'inverse': True,

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

        # MBD-DFTB'
        'LCPA': False,
        'LMBD': False,

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
    parameter.update(parameter_)
    if parameter['precision'] == 't.float64':
        parameter['precision'] = t.float64
    elif parameter['precision'] == 't.float32':
        parameter['precision'] = t.float32
    elif parameter['precision'] == 't.cuda.DoubleTensor':
        parameter['precision'] = t.cuda.DoubleTensor
    elif parameter['precision'] == 't.cuda.FloatTensor':
        parameter['precision'] = t.cuda.FloatTensor
    if parameter['device'] == 'cpu':
        if parameter['precision'] not in (t.float64, t.float32):
            parameter['precision'] = t.float64
            print('convert precision to cpu type')
    elif parameter['device'] == 'cuda':
        if parameter['precision'] not in (t.cuda.DoubleTensor, t.cuda.FloatTensor):
            parameter['precision'] = t.cuda.DoubleTensor
            print('convert precison to cuda type')

    # is machine learning is on, it means that the task is machine learning
    parameter['Lml'] = True if parameter['task'] in (
        'mlCompressionR', 'mlIntegral', 'testCompressionR', 'testIntegral') else False

    # batch calculation, usually True for machine learning
    parameter['Lbatch'] = True if parameter['Lml'] is True else False

    # return DFTB calculation parameters
    return parameter


def init_dataset(dataset_=None):
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
    dataset_ = {} if dataset_ is None else dataset_

    dataset = {
        # default dataset is ani
        'dataset': ['../data/dataset/an1/ani_gdb_s01.h5'],

        # optional datatype: ani, json, hdf, runani
        'datasetType': 'ani',

        # get the dataset path
        'directoryDataset': '../data/dataset/',

        # directly read SKF or interpolate from a list of skf files
        'LSKFinterpolation': False,

        # define path of files with feature information in machine learning
        'pathFeature': '.',

        # how many molecules for each molecule specie !!
        'sizeDataset': [6, 6, 6],

        # used to test (optimize ML algorithm parameters) !!
        'sizeTest': [6, 6, 6],

        # mix different molecule specie type
        'LdatasetMixture': True}

    # update temporal dataset_ with input dataset
    dataset.update(dataset_)

    return dataset


def init_ml(para=None, dataset=None, skf=None, ml_=None):
    """Return the machine learning parameters for DFTB calculations.

    Args:
        dataset (dict, optional): a dictionary which includes dataset,
            geometric parameters.

    Returns:
        dataset (dict, optional): default dataset, geometric parameters if not
            defined in advance.

    """
    para = {} if para is None else para
    ml_ = {} if ml_ is None else ml_
    dataset = {} if dataset is None else dataset
    skf = {} if skf is None else skf

    ml = {
        # dipole, HOMOLUMO, gap, eigval, polarizability, cpa, pdos, charge
        'target': 'dipole',

        # path to dataset data
        'referenceDataset': '../data/dataset/ani01_100.hdf5',

        # define weight in loss function
        'LossRatio': [1],

        # how many steps for optimize in DFTB-ML !!
        'mlSteps': 3,

        # how many steps to save the DFTB-ML data !!
        'saveSteps': 2,

        # minimum ML steps
        'stepMin': 2,

        # learning rate !!
        'lr': 3E-2,

        # optimizer
        'optimizer': 'Adam',

        # define loss function: MSELoss, L1Loss
        'lossFunction': 'MSELoss',

        # ML energy type: total energy with offset or formation energy
        'mlEnergyType': 'formationEnergy',

        # machine learning algorithm: linear, svm, schnet, nn...!!!!
        'MLmodel': 'linear',

        # define atomic representation: cm (CoulombMatrix), acsf!!!!!
        'featureType': 'acsf',

        # do not run DFTB or DFT to get reference data
        'runReference': False,

        # optional reference: aims, dftbplus, dftb, dftbase, aimsase !!
        'reference': 'hdf',

        # grid of compression radius is uniform or not !!
        'typeSKinterp': 'uniform',

        # skgen compression radius parameters: all, wavefunction, density
        'typeSKinterpR': 'all',

        # if any compR < 2.2, break DFTB-ML loop
        'compressionRMin': 1.2,

        # if any compR > 9, break DFTB-ML loop
        'compressionRMax': 9,

        # multi interpolation method
        'interpolationType': 'BiCubVec',

        # set initial compression radius
        'H_init_compr': 3.5,
        'C_init_compr': 3.5,
        'N_init_compr': 3.5,
        'O_init_compr': 3.5,

        # compression radius of H
        'H_compr_grid': t.tensor([1., 1.5, 2., 2.5, 3., 3.5, 4., 5., 6., 8., 10.]),

        # compression radius of C
        'C_compr_grid': t.tensor([1., 1.5, 2., 2.5, 3., 3.5, 4., 5., 6., 8., 10.]),

        # compression radius of N
        'N_compr_grid': t.tensor([1., 1.5, 2., 2.5, 3., 3.5, 4., 5., 6., 8., 10.]),

        # compression radius of O
        'O_compr_grid': t.tensor([1., 1.5, 2., 2.5, 3., 3.5, 4., 5., 6., 8., 10.])
        }

    # update ml with input ml_
    ml.update(ml_)

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

    # If turn on some calculations related to these physical properties
    # turn on anyway
    if 'energy' in ml['target']:
        para['Lrepulsive'] = True

    if 'cpa' in ml['target']:
        para['LCPA'] = True
    if 'polarizability' in ml['target']:
        para['LCPA'], para['LMBD'] = True, True

    # optimize integral directly
    if para['task'] == ('mlIntegral', 'testIntegral'):
        # spline type to generate integral
        if 'interpolationType' not in ml.keys():
            ml['interpolationType'] = 'Polyspline'

    # optimize compression radius: by interpolation or by ML prediction
    if para['task'] in ('mlCompressionR', 'testCompressionR'):
        # interpolation of compression radius: BiCub, BiCubVec
        if 'interpolationType' not in ml.keys():
            ml['interpolationType'] = 'BiCubVec'

    return para, dataset, skf, ml


def skf_parameter(para, skf_=None):
    """Return the default parameters for skf.

    Returns:
    ----
        skf: `dictionary`
        only for normal skf files, dataset skf parameters is in dataset.
    """
    skf_ = {} if skf_ is None else skf_

    skf = {
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
        'Lonsite': False,

        # define onsite
        'onsiteHH': t.tensor([0.0E+0, 0.0E+0, -2.386005440483E-01]),
        'onsiteCC': t.tensor([0.0E+0, -1.943551799182E-01, -5.048917654803E-01]),
        'onsiteNN': t.tensor([0.0E+0, -2.607280834222E-01, -6.400000000000E-01]),
        'onsiteOO': t.tensor([0.0E+00, -3.321317735288E-01, -8.788325840767E-01])
        }

    # the parameters from skf will overwrite skf_ default parameters
    skf.update(skf_)

    if not skf['LOrbitalResolve']:
        # Hubbert is not orbital resolved, value from skgen
        skf['uhubbHH'] = t.tensor([4.196174261214E-01, 4.196174261214E-01, 4.196174261214E-01])
        skf['uhubbCC'] = t.tensor([3.646664973641E-01, 3.646664973641E-01, 3.646664973641E-01])
        skf['uhubbNN'] = t.tensor([4.308879578818E-01, 4.308879578818E-01, 4.308879578818E-01])
        skf['uhubbOO'] = t.tensor([4.954041702122E-01, 4.954041702122E-01, 4.954041702122E-01])

    # Hubbert is orbital resolved
    # if use different parametrization method, remember revise value here
    elif skf['LOrbitalResolve']:
        skf['uhubbHH'] = t.tensor([0.0E+00, 0.0E+00, 4.196174261214E-01])
        skf['uhubbCC'] = t.tensor([0.0E+00, 3.646664973641E-01, 3.646664973641E-01])
        skf['uhubbNN'] = t.tensor([0.0E+00, 4.308879578818E-01, 4.308879578818E-01])
        skf['uhubbOO'] = t.tensor([0.0E+00, 4.954041702122E-01, 4.954041702122E-01])

    if para['task'] in ('mlCompressionR', 'testCompressionR'):
        skf['ReadSKType'] = 'compressionRadii'
    elif para['task'] in ('mlIntegral', 'testIntegral'):
        skf['ReadSKType'] = 'mlIntegral'

    # return skf
    return skf
