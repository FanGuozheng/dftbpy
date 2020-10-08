"""Example of defining parameters for DFTB-ML by python code."""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import torch as t


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

    # batch calculation, usually True for machine learning
    if 'Lbatch' not in parameter.keys():
        parameter['Lbatch'] = False

    # read parameters from input file, if False, you can define in python
    if 'LReadInput' not in parameter.keys():
        parameter['LReadInput'] = False

    # SCC, non-SCC DFTB calculation
    if 'scc' not in parameter.keys():
        parameter['scc'] = 'scc'

    # if the system perodic condition, with lattice parameter, switch to True
    if 'Lperiodic' not in parameter.keys():
        parameter['Lperiodic'] = False

    # calculate repulsive term or not, if True, the code will read repulsive
    # parameter from skf
    if 'Lrepulsive' not in parameter.keys():
        parameter['Lrepulsive'] = False

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

    # electron temperature
    if 'tElec' not in parameter.keys():
        parameter['tElec'] = 0

    # max interation of SCC loop
    if 'maxIteration' not in parameter.keys():
        parameter['maxIteration'] = 60

    # density basis: spherical, gaussian
    if 'density_profile' not in parameter.keys():
        parameter['density_profile'] = 'spherical'

    # coordinate type: 'C': cartesian...
    if 'coordinateType' not in parameter.keys():
        parameter['coordinateType'] = 'C'

    # ******************************** H, S ********************************
    # how to write H0, S: symhalf (write upper or lower), symall (write whole)
    if 'HSsym' not in parameter.keys():
        parameter['HSsym'] = 'symall'

    # general eigenvalue method: cholesky, lowdin_qr
    if 'eigenmethod' not in parameter.keys():
        parameter['eigenmethod'] = 'cholesky'

    # ****************************** MBD-DFTB ******************************
    if 'LMBD_DFTB' not in parameter.keys():
        parameter['LMBD_DFTB'] = False

    # omega grid
    if 'n_omega_grid' not in parameter.keys():
        parameter['n_omega_grid'] = 15
    if 'vdw_self_consistent' not in parameter.keys():
        parameter['vdw_self_consistent'] = False
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

    # return DFTB calculation parameters
    return parameter


def init_dataset(dataset=None):
    """Return the dataset or geometric parameters for DFTB calculations.

    Args:
        dataset (dict, optional): a dictionary which includes dataset,
            geometric parameters.

    Returns:
        dataset (dict, optional): default dataset, geometric parameters if not
            defined in advance.

    """
    dataset = {} if dataset is None else dataset

    # optional datatype: ani, json, hdf
    if 'dataType' not in dataset.keys():
        dataset['dataType'] = 'hdf'

    # get the current path
    if 'path_dataset' not in dataset.keys():
        dataset['path_dataset'] = '../data/dataset/'

    # skf: directly read or interpolate from a list of skf files
    if 'LSKFinterpolation' not in dataset.keys():
        dataset['LSKFinterpolation'] = False

    # define path of files with feature information in machine learning
    if 'pathFeature' not in dataset.keys():
        dataset['pathFeature'] = '.'

    # how many molecules for each molecule specie !!
    if 'n_dataset' not in dataset.keys():
        dataset['n_dataset'] = ['2']

    # used to test (optimize ML algorithm parameters) !!
    if 'n_test' not in dataset.keys():
        dataset['n_test'] = ['2']

    # determine the type of molecule specie: str(integer), 'all' !!
    if 'hdf_num' not in dataset.keys():
        dataset['hdf_num'] = ['all']

    # mix different molecule specie type
    dataset['LdatasetMixture'] = True

    # read json type geometry
    if dataset['dataType'] == 'json':

        # path of data
        dataset['path_dataset'] = '../data/json'

        # name of data in defined path
        dataset['name_dataset'] = 'H2_data'

    # read ANI dataset
    elif dataset['dataType'] == 'ani':
        hdffilelist = []

        # add hdf data: ani_gdb_s01.h5 ... ani_gdb_s08.h5
        hdffilelist.append(os.path.join(dataset['path_dataset'], 'an1/ani_gdb_s01.h5'))

        # transfer to list
        dataset['hdffile'] = hdffilelist

        # test the molecule specie is the same
        assert len(dataset['n_dataset']) == len(dataset['n_test'])

    # read QM7 dataset
    elif dataset['dataType'] == 'qm7':

        # define path and dataset name
        dataset['qm7_data'] = os.path.join(dataset['path_dataset'], 'qm7.mat')

        # define dataset specie
        dataset['train_specie'] = [1, 6, 8]

        # test the molecule specie is the same
        assert len(dataset['n_dataset']) == len(dataset['n_test'])
    return dataset


def init_ml(para=None, ml=None, dataset=None):
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

    # get current path
    path = os.getcwd()

    # is machine learning is on, the following is machine learning target
    if 'Lml' not in ml.keys():
        ml['Lml'] = True

    # ML target: compressionRadius, integral
    if 'mlType' not in ml.keys():
        ml['mlType'] = 'compressionRadius'

    ml['MLmodel'] = 'linear'  # linear, svm, schnet, nn...!!!!!

    # define atomic representation: rad, cm (CoulombMatrix), acsf!!!!!
    ml['featureType'] = 'acsf'

    # define ACSF parameter
    if ml['featureType'] == 'acsf':

        # cutoff, for G1
        ml['acsf_rcut'] = 6.0

        # G2 parameters
        ml['Lacsf_g2'] = True
        ml['acsf_g2'] = [[1, 1]]

        # G3 parameters
        ml['Lacsf_g3'] = False

        # G4 parameters
        ml['Lacsf_g4'] = True
        ml['acsf_g4'] = [[0.02, 1, -1]]

        # G5 parameters
        ml['Lacsf_g5'] = False

    # for test, where to read the optimized parameters, 1 means read the last
    # optimized para in path_dataset, 0 is the unoptimized !!!!!
    ml['opt_para_test'] = 1.0

    # *********************************************************************
    #                              DFTB-ML
    # *********************************************************************
    # optional reference: aims, dftbplus, dftb, dftbase, aimsase !!
    if 'reference' not in ml.keys():
        ml['reference'] = 'hdf'

    # read hdf (with coordinates, reference physical properties) type
    if ml['reference'] == 'hdf':
        # run referecne calculations or directly get read reference properties
        ml['run_reference'] = False

        # path of data
        dataset['path_dataset'] = '../data/dataset'

        # name of data in defined path
        dataset['name_dataset'] = 'testfile.hdf5'

        # if read SKF from a list of files with interpolation, instead from hdf
        dataset['LSKFinterpolation'] = False

        # dire of skf with hdf type
        ml['dire_hdfSK'] = '../slko/hdf'

        # name of skf with hdf type
        ml['name_hdfSK'] = 'skf.hdf5'

    if ml['reference'] in ('dftbase', 'dftbplus'):

        # path of binary, executable DFTB file
        ml['dftb_ase_path'] = '../test/dftbplus'

        # name of binary, executable DFTB file
        ml['dftb_bin'] = 'dftb+'

        # path slater-koster file
        ml['skf_ase_path'] = '../slko/mio'

    if ml['reference'] in ('aimsase', 'aims'):

        # path of binary, executable FHI-aims file
        ml['aims_ase_path'] = '/home/gz_fan/Downloads/software/fhiaims/fhiaims/bin'

        # name of binary, executable FHI-aims file
        ml['aims_bin'] = 'aims.171221_1.scalapack.mpi.x'

        # path of atom specie parameters
        ml['aims_specie_path'] = '/home/gz_fan/Downloads/software/fhiaims/fhiaims/species_defaults/tight/'

    # dipole, homo_lumo, gap, eigval, polarizability, cpa, pdos, charge
    ml['target'] = ['formationenergy']

    # If turn on some calculations related to these physical properties
    # turn on anyway
    if 'energy' in ml['target']:
        para['Lrepulsive'] = False
    else:
        para['Lrepulsive'] = False

    # the machine learning energy type
    para['mlenergy'] = 'formationenergy'

    # calculate, read, save the HOMO LUMO
    if 'homo_lumo' in ml['target']:
        para['LHL'] = True
    else:
        para['LHL'] = False

    if 'cpa' in ml['target']:
        para['LMBD_DFTB'] = True

    # define weight in loss function
    ml['dipole_loss_ratio'] = 1
    ml['polarizability_loss_ratio'] = 0.15

    # how many steps for optimize in DFTB-ML !!
    ml['mlsteps'] = 2

    # how many steps to save the DFTB-ML data !!
    ml['save_steps'] = 2

    # minimum steps
    ml['opt_step_min'] = 2

    # learning rate !!
    ml['lr'] = 1E-2

    # optimizer
    ml['optimizer'] = 'Adam'

    # define loss function: MSELoss, L1Loss
    ml['loss_function'] = 'MSELoss'

    # optimize integral directly
    if ml['mlType'] == 'integral':

        # type to generate integral
        ml['interptype'] = 'Polyspline'
        ml['zero_threshold'] = 5E-3
        ml['rand_threshold'] = 5E-2

    # optimize compression radius: by interpolation or by ML prediction
    if ml['mlType'] in ('compressionRadius', 'ACSF'):

        # interpolation of compression radius: BiCub, BiCubVec
        ml['interp_compr_type'] = 'BiCub'

        # grid of compression radius is uniform or not !!
        ml['typeSKinterp'] = 'uniform'

        # skgen compression radius parameters: all, wavefunction, density
        ml['typeSKinterpR'] = 'all'

        # the grid point of compression radius is not uniform
        if ml['typeSKinterp'] == 'nonuniform':
            ml['dire_interpSK'] = os.path.join(path, '../slko/nonuniform')

        # the grid point of compression radius is uniform
        elif ml['typeSKinterp'] == 'uniform':
            ml['dire_interpSK'] = os.path.join(path, '../slko/uniform')

        # number of grid points, should be equal to atom_compr_grid
        ml['ncompr'] = 10

        # if fix the optimization step and set convergence condition
        ml['Lopt_step'] = True
        ml['opt_ml_tol'] = 1E-3

        # after opt_ml_step*nbatch molecule, perform ML predict compR!!!!!!
        ml['opt_ml_step'] = 0.5
        ml['opt_ml_all'] = False

        # if any compR < 2.2, break DFTB-ML loop
        ml['compr_min'] = 2.2

        # if any compR > 9, break DFTB-ML loop
        ml['compr_max'] = 9

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
        assert len(ml['H_compr_grid']) == ml['ncompr']
        assert len(ml['C_compr_grid']) == ml['ncompr']
        assert len(ml['N_compr_grid']) == ml['ncompr']
        assert len(ml['O_compr_grid']) == ml['ncompr']

        # set initial compression radius
        ml['H_init_compr'] = 3.5
        ml['C_init_compr'] = 3.5
        ml['N_init_compr'] = 3.5
        ml['O_init_compr'] = 3.5

    # get input path
    para['direInput'] = os.path.join(path, 'dftbtorch')

    # ---------------------- plotting and others ----------------------
    para['Lplot_ham'] = True
    para['Lplot_feature'] = False
    para['hamold'] = 0

    # save log during DFTB-ML as: txt, hdf (binary)
    para['log_save_type'] = 'hdf'

    # generate reference
    para['ref_save_type'] = 'hdf'
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
    if 'dist_tailskf' not in skf.keys():
        skf['dist_tailskf'] = 1.0

    # skf interpolation number
    if 'ninterp' not in skf.keys():
        skf['ninterp'] = 8

    # skf directory
    if 'direSK' not in skf.keys():
        skf['direSK'] = '../slko/test'

    # SK transformation method
    if 'sk_tran' not in skf.keys():
        skf['sk_tran'] = 'new'

    # delta distance when interpolate SKF integral
    if 'deltaRskf' not in skf.keys():
        skf['deltaRskf'] = 1E-5

    # orbital resolved: if Ture, only use Hubbert of s orbital
    if 'Lorbres' not in skf.keys():
        skf['Lorbres'] = False

    # if optimize (True) or fix (False) onsite in DFTB-ML
    if 'Lonsite' not in skf.keys():
        skf['Lonsite'] = False

    # define onsite
    if not skf['Lorbres']:
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
    elif skf['Lorbres']:
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
