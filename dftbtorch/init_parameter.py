"""Example of defining parameters for DFTB-ML by python code.

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
    # **********************************************************************
    #                              load data
    # **********************************************************************
    # optional datatype: ani, json, hdf
    para['dataType'] = 'ani'

    # get the current path
    path = os.getcwd()
    dire_data = '../data/dataset/'

    # define path of feature related files
    para['direfeature'] = '.'

    # how many molecules for each molecule specie !!
    para['n_dataset'] = ['2']

    # used to test (optimize ML algorithm parameters) !!
    para['n_test'] = ['2']

    # determine the type of molecule specie: str(integer), 'all' !!
    para['hdf_num'] = ['all']

    # mix different molecule specie type
    para['hdf_mixture'] = True

    # read json type geometry
    if para['dataType'] == 'json':

        # path of data
        para['pythondata_dire'] = '../data/json'

        # name of data in defined path
        para['pythondata_file'] = 'H2_data'

        # if read SKF from a list of files with interpolation
        para['LreadSKFinterp'] = True

    # read ANI dataset
    elif para['dataType'] == 'ani':
        # run referecne calculations or directly get read reference properties
        para['run_reference'] = True

        hdffilelist = []

        # add hdf data: ani_gdb_s01.h5 ... ani_gdb_s08.h5
        hdffilelist.append(os.path.join(dire_data, 'an1/ani_gdb_s01.h5'))

        # transfer to list
        para['hdffile'] = hdffilelist

        # test the molecule specie is the same
        assert len(para['n_dataset']) == len(para['n_test'])

        # if read SKF from a list of files with interpolation
        para['LreadSKFinterp'] = True

    # read QM7 dataset
    elif para['dataType'] == 'qm7':

        # define path and dataset name
        para['qm7_data'] = os.path.join(dire_data, 'qm7.mat')

        # define dataset specie
        para['train_specie'] = [1, 6, 8]

        # if read SKF from a list of files with interpolation
        para['LreadSKFinterp'] = True

        # test the molecule specie is the same
        assert len(para['n_dataset']) == len(para['n_test'])

    # **********************************************************************
    #             machine learning and environment parameters
    # **********************************************************************
    # is machine learning is on, the following is machine learning target
    para['Lml'] = True

    # optimize compression radius
    para['Lml_skf'] = True

    # optimize integral (e.g Polyspline)
    para['Lml_HS'] = False

    # test gradients of interp of SK table
    para['Lml_compr'] = False

    # each spiece has the same compress_r
    para['Lml_compr_global'] = False

    # optimize ACSF parameters
    para['Lml_acsf'] = False

    para['testMLmodel'] = 'linear'  # linear, svm, schnet, nn...!!!!!

    # define atomic representation: rad, cm (CoulombMatrix), acsf!!!!!
    para['featureType'] = 'acsf'

    # define ACSF parameter
    if para['featureType'] == 'acsf':

        # cutoff, for G1
        para['acsf_rcut'] = 6.0

        # G2 parameters
        para['Lacsf_g2'] = True
        para['acsf_g2'] = [[1, 1]]

        # G3 parameters
        para['Lacsf_g3'] = False

        # G4 parameters
        para['Lacsf_g4'] = True
        para['acsf_g4'] = [[0.02, 1, -1]]

        # G5 parameters
        para['Lacsf_g5'] = False

    # for test, where to read the optimized parameters, 1 means read the last
    # optimized para in dire_data, 0 is the unoptimized !!!!!
    para['opt_para_test'] = 1.0

    # get ACSF by hand, if featureType is rad
    para['rcut'] = 15
    para['r_s'] = 5
    para['eta'] = 0.1
    para['tol'] = 1E-4
    para['zeta'] = 1
    para['lambda'] = 1
    para['ang_paraall'] = []
    para['rad_paraall'] = []

    # *********************************************************************
    #                              DFTB-ML
    # *********************************************************************
    # optional reference: aims, dftbplus, dftb, dftbase, aimsase !!
    para['reference'] = 'hdf'

    # read hdf (with coordinates, reference physical properties) type
    if para['reference'] == 'hdf':
        # run referecne calculations or directly get read reference properties
        para['run_reference'] = False

        # path of data
        para['pythondata_dire'] = '../data/dataset'

        # name of data in defined path
        para['pythondata_file'] = 'testfile.hdf5'

        # if read SKF from a list of files with interpolation, instead from hdf
        para['LreadSKFinterp'] = False

        # dire of skf with hdf type
        para['dire_hdfSK'] = '/home/gz_fan/Documents/ML/dftb/slko/hdf'

        # name of skf with hdf type
        para['name_hdfSK'] = 'skf.hdf5'

    if para['reference'] in ('dftbase', 'dftbplus'):

        # path of binary, executable DFTB file
        para['dftb_ase_path'] = '/home/gz_fan/Documents/ML/dftb/test/dftbplus'

        # name of binary, executable DFTB file
        para['dftb_bin'] = 'dftb+'

        # path slater-koster file
        para['skf_ase_path'] = '/home/gz_fan/Documents/ML/dftb/slko/mio'

    if para['reference'] in ('aimsase', 'aims'):

        # path of binary, executable FHI-aims file
        para['aims_ase_path'] = '/home/gz_fan/Downloads/software/fhiaims/fhiaims/bin'

        # name of binary, executable FHI-aims file
        para['aims_bin'] = 'aims.171221_1.scalapack.mpi.x'

        # path of atom specie parameters
        para['aims_specie_path'] = '/home/gz_fan/Downloads/software/fhiaims/fhiaims/species_defaults/tight/'

    # dipole, homo_lumo, gap, eigval, qatomall, polarizability, cpa, pdos !!
    para['target'] = ['dipole']

    # If turn on some calculations related to these physical properties
    # turn on anyway
    if 'energy' in para['target']:
        para['Lenergy'] = True
        para['Lrepulsive'] = False
    else:
        para['Lenergy'] = True
        para['Lrepulsive'] = False

    # the machine learning energy type
    para['mlenergy'] = 'formationenergy'

    # calculate, read, save the eigenvalue
    if 'eigval' in para['target']:
        para['Leigval'] = True
    else:
        para['Leigval'] = False

    # calculate, read, save the MBD-DFTB
    if 'polarizability' in para['target']:
        para['LMBD_DFTB'] = True
        # mbd_vdw_n_quad_pts = para['n_omega_grid']
        para['n_omega_grid'] = 15
        para['vdw_self_consistent'] = False
        para['beta'] = 1.05
    else:
        para['LMBD_DFTB'] = False

    # calculate, read, save the PDOS
    if 'pdos' in para['target']:
        para['Lpdos'] = True
    else:
        para['Lpdos'] = False

    # calculate, read, save the dipole
    if 'dipole' in para['target']:
        para['Ldipole'] = True
    else:
        para['Ldipole'] = False

    # calculate, read, save the HOMO LUMO
    if 'homo_lumo' in para['target']:
        para['LHL'] = True
    else:
        para['LHL'] = False

    # define weight in loss function
    para['dipole_loss_ratio'] = 1
    para['polarizability_loss_ratio'] = 0.15

    # how many steps for optimize in DFTB-ML !!
    para['mlsteps'] = 2

    # how many steps to save the DFTB-ML data !!
    para['save_steps'] = 2

    # minimum steps
    para['opt_step_min'] = 2

    # learning rate !!
    para['lr'] = 5E-3

    # optimizer
    para['optimizer'] = 'Adam'

    # define loss function: MSELoss, L1Loss
    para['loss_function'] = 'MSELoss'

    # optimize integral directly
    if para['Lml_HS']:

        # type to generate integral
        para['interptype'] = 'Polyspline'
        para['zero_threshold'] = 5E-3
        para['rand_threshold'] = 5E-2
    para['interpdist'] = 0.4
    para['atomspecie_old'] = []

    # optimize compression radius: by interpolation or by ML prediction
    if para['Lml_skf'] or para['Lml_acsf']:

        # interpolation of compression radius: BiCub, BiCubVec
        para['interp_compr_type'] = 'BiCubVec'

        # if optimize onsite in DFTB-ML
        para['Lonsite'] = True

        # grid of compression radius is uniform or not !!
        para['typeSKinterp'] = 'uniform'

        # skgen compression radius parameters: all, wavefunction, density
        para['typeSKinterpR'] = 'all'

        # the grid point of compression radius is not uniform
        if para['typeSKinterp'] == 'nonuniform':
            para['dire_interpSK'] = os.path.join(path, '../slko/nonuniform')

        # the grid point of compression radius is uniform
        elif para['typeSKinterp'] == 'uniform':
            para['dire_interpSK'] = os.path.join(path, '../slko/uniform')

        # number of grid points, should be equal to atom_compr_grid
        para['ncompr'] = 10

        # if fix the optimization step and set convergence condition
        para['Lopt_step'] = True
        para['opt_ml_tol'] = 1E-3

        # if predict compression radius during DFTB-ML
        para['Lopt_ml_compr'] = False

        # after opt_ml_step*nbatch molecule, perform ML predict compR!!!!!!
        para['opt_ml_step'] = 0.5
        para['opt_ml_all'] = False

        # smooth the tail when read the skf
        para['smooth_tail'] = True

        # if any compR < 2.2, break DFTB-ML loop
        para['compr_min'] = 2.2

        # if any compR > 9, break DFTB-ML loop
        para['compr_max'] = 9

        # compression radius of H
        para['H_compr_grid'] = t.tensor((
                [2., 2.5, 3., 3.5, 4., 4.5, 5., 6., 8., 10.]), dtype=t.float64)

        # compression radius of C
        para['C_compr_grid'] = t.tensor((
                [2., 2.5, 3., 3.5, 4., 4.5, 5., 6., 8., 10.]), dtype=t.float64)

        # compression radius of N
        para['N_compr_grid'] = t.tensor((
                [2., 2.5, 3., 3.5, 4., 4.5, 5., 6., 8., 10.]), dtype=t.float64)

        # compression radius of O
        para['O_compr_grid'] = t.tensor((
                [2., 2.5, 3., 3.5, 4., 4.5, 5., 6., 8., 10.]), dtype=t.float64)

        # test the grid points length
        assert len(para['H_compr_grid']) == para['ncompr']
        assert len(para['C_compr_grid']) == para['ncompr']
        assert len(para['N_compr_grid']) == para['ncompr']
        assert len(para['O_compr_grid']) == para['ncompr']

        # set initial compression radius
        para['H_init_compr'] = 3.6
        para['C_init_compr'] = 3.5
        para['N_init_compr'] = 3.5
        para['O_init_compr'] = 3.5

        # define onsite
        para['onsiteHH'] = t.tensor((
                [0.0E+00, 0.0E+00, -2.386005440483E-01]), dtype=t.float64)
        para['onsiteCC'] = t.tensor((
                [0.0E+00, -1.943551799182E-01, -5.048917654803E-01]),
                dtype=t.float64)
        para['onsiteNN'] = t.tensor((
                [0.0E+00, -2.607280834222E-01, -6.400000000000E-01]),
                dtype=t.float64)
        para['onsiteOO'] = t.tensor((
                [0.0E+00, -3.321317735288E-01, -8.788325840767E-01]),
                dtype=t.float64)

        # deine Hubbert is orbital resolved or not
        para['Lorbres'] = False

        # Hubbert is orbital resolved
        # if use different parametrization method, remember revise value here
        if para['Lorbres']:
            para['uhubbHH'] = t.tensor((
                    [0.0E+00, 0.0E+00, 4.196174261214E-01]), dtype=t.float64)
            para['uhubbCC'] = t.tensor((
                    [0.0E+00, 3.646664973641E-01, 3.646664973641E-01]),
                    dtype=t.float64)
            para['uhubbNN'] = t.tensor((
                    [0.0E+00, 4.308879578818E-01, 4.308879578818E-01]),
                    dtype=t.float64)
            para['uhubbOO'] = t.tensor((
                    [0.0E+00, 4.954041702122E-01, 4.954041702122E-01]),
                    dtype=t.float64)

        # Hubbert is not orbital resolved, value from skgen
        elif not para['Lorbres']:
            para['uhubbHH'] = t.tensor(([4.196174261214E-01,
                                         4.196174261214E-01,
                                         4.196174261214E-01]), dtype=t.float64)
            para['uhubbCC'] = t.tensor(([3.646664973641E-01,
                                         3.646664973641E-01,
                                         3.646664973641E-01]), dtype=t.float64)
            para['uhubbNN'] = t.tensor(([4.308879578818E-01,
                                         4.308879578818E-01,
                                         4.308879578818E-01]), dtype=t.float64)
            para['uhubbOO'] = t.tensor(([4.954041702122E-01,
                                         4.954041702122E-01,
                                         4.954041702122E-01]), dtype=t.float64)

    # *********************************************************************
    #                          DFTB parameter                             #
    # *********************************************************************
    # if read parameter from dftb_in (default input name)
    para['LReadInput'] = False

    # general eigenvalue methodin DFTB-ML: cholesky, lowdin_qr !!
    para['eigenmethod'] = 'cholesky'

    # perform (non-) SCC DFTB: nonscc, scc
    para['scc'] = 'scc'

    # batch calculation
    para['Lbatch'] = True

    # convergence type
    para['convergenceType'] = 'energy'

    # tolerance
    para['convergence_tol'] = 1E-6

    # delta r when interpolating i9ntegral
    para['delta_r_skf'] = 1E-5
    para['general_tol'] = 1E-4
    para['sk_tran'] = 'new'

    # mix method: simple, anderson, broyden
    para['mixMethod'] = 'anderson'

    # mixing fraction
    para['mixFactor'] = 0.2

    # if build half H0, S or build whole H0, S: symall, symhalf !!
    para['HSsym'] = 'symall'

    # interpolation integrals when read SKF with distance
    para['ninterp'] = 8

    # distance when smoothing tail of SKF
    para['dist_tailskf'] = 1.0

    # temperature
    para['tElec'] = 0

    # max of SCF loop
    para['maxIter'] = 60

    # density basis: exp_spher, gaussian
    para['scc_den_basis'] = 'exp_spher'

    # if smaller than t_zero_max, treated as zero
    para['t_zero_max'] = 5

    # if for different atom species, cutoff will be set separately
    para['cutoff_atom_resolve'] = False
    para['cutoff_rep'] = 4.5

    # periodic condition
    para['Lperiodic'] = False

    # coordinate type: 'C': cartesian...
    para['coorType'] = 'C'

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


def init_dftb(para):
    """Initialize the parameters for normal DFTB.

    Applications:
        DFTB without gradient
        SKF from normal SKF file
    Returns:
        Path of input (if you do not offer all in python)
        DFTB parameters
        Others, such as plotting parameters
    """
    # batch calculation
    para['Lbatch'] = False

    # do not get parameters from input file, from python code
    para['LReadInput'] = False

    # system perodic condition
    para['Lperiodic'] = False

    # calculate dipole
    para['Ldipole'] = True

    # calculate repulsive term or not
    para['Lrepulsive'] = False

    # mixing method: simple. anderson, broyden
    para['mixMethod'] = 'anderson'

    # mixing factor
    para['mixFactor'] = 0.2

    # convergence method: energy, charge
    para['convergenceType'] = 'energy'

    # convergence precision
    para['convergence_tol'] = 1e-8

    # delta distance when interpolate SKF integral
    para['delta_r_skf'] = 1E-5

    # electron temperature
    para['tElec'] = 0

    # if smaller than t_zero_max, temperature treated as zero
    para['t_zero_max'] = 5

    # max interation of SCC loop
    para['maxIter'] = 60

    # density basis: exp_spher, gaussian
    para['scc_den_basis'] = 'exp_spher'

    # ****************************** SKF, H, S ******************************
    # skf: directly read or interpolate from a list of skf files
    para['LreadSKFinterp'] = False

    # if calculate PDOS
    para['Lpdos'] = False

    # orbital resolved: if Ture, only use Hubbert of s orbital
    para['Lorbres'] = False

    # skf file tail distance
    para['dist_tailskf'] = 1.0

    # skf interpolation number
    para['ninterp'] = 8

    # skf directory
    para['direSK'] = '../slko/test'
    para['sk_tran'] = 'new'

    # how to write H0, S: symhalf (write upper or lower), symall (write whole)
    para['HSsym'] = 'symall'

    # general eigenvalue method: cholesky, lowdin_qr
    para['eigenmethod'] = 'cholesky'

    # ****************************** MBD-DFTB ******************************
    para['LMBD_DFTB'] = False

    # omega grid
    para['n_omega_grid'] = 15
    para['vdw_self_consistent'] = False
    para['beta'] = 1.05

    # **************************** ML parameter ****************************
    para['Lml'] = False

    # machine learning optimize SKF parameters (compression radius...)
    para['Lml_skf'] = False

    # machine learning optimize integral
    para['Lml_HS'] = False


def init_dftb_interp(para):
    """Initialize the parameters for DFTB with interpolation of SKF.

    In general, you have to define:
        Path of input (if you do not offer all in python)
        DFTB parameters
        Others, such as plotting parameters
    """
    # batch calculation
    para['Lbatch'] = False

    # define parameters in python, or read from input
    para['LReadInput'] = False

    # periodic condition
    para['Lperiodic'] = False

    # if calculate PDOS
    para['Lpdos'] = True

    # mixing method: simple, anderson, broyden
    para['mixMethod'] = 'anderson'

    # mixing factor
    para['mixFactor'] = 0.2

    # if smaller than t_zero_max, treated as zero
    para['t_zero_max'] = 5

    # convergence_tol method: energy, charge
    para['convergenceType'] = 'energy'

    # convergence precision
    para['convergence_tol'] = 1e-7

    # delta distance when interpolating SKF integral
    para['delta_r_skf'] = 1E-5

    # temperature
    para['tElec'] = 0

    # max interation step
    para['maxIter'] = 60

    # density basis: exp_spher, gaussian
    para['scc_den_basis'] = 'exp_spher'

    # general eigenvalue method: cholesky, lowdin_qr
    para['eigenmethod'] = 'cholesky'

    # if calculate dipole
    para['Ldipole'] = True

    # if calculate repulsive
    para['Lrepulsive'] = False

    # write half, or full H0, S: symhalf, symall
    para['HSsym'] = 'symall'

    # smooth SKF integral tail
    para['dist_tailskf'] = 1.0

    # integral interpolation grid points
    para['ninterp'] = 8

    # cutoff
    # para['interpcutoff'] = 4.0

    # **************************** ML parameter ****************************
    para['Lml'] = False

    # machine learning optimize SKF parameters (compression radius...)
    para['Lml_skf'] = True

    # machine learning optimize integral
    para['Lml_HS'] = False

    # The following is for MBD-DFTB
    para['LMBD_DFTB'] = False
    para['n_omega_grid'] = 15
    para['vdw_self_consistent'] = False
    para['beta'] = 1.05

    para['Lml_compr_global'] = False
    para['sk_tran'] = 'new'

    # total atom specie in last step
    para['atomspecie_old'] = []

    # get integral from interpolation
    para['LreadSKFinterp'] = True

    # deine Hubbert is orbital resolved or not
    para['Lorbres'] = False

    # SKF compression radius: all (all radius the same), wavefunction
    para['typeSKinterpR'] = 'all'

    # smooth the tail when read the skf
    para['smooth_tail'] = True

    # interpolation of compression radius: BiCub, BiCubVec
    para['interp_compr_type'] = 'BiCubVec'

    # if optimize onsite in DFTB-ML
    para['Lonsite'] = True

    # grid of compression radius is uniform or not !!
    para['typeSKinterp'] = 'uniform'

    # skgen compression radius parameters: all, wavefunction, density
    para['typeSKinterpR'] = 'all'

    # the grid point of compression radius is not uniform
    if para['typeSKinterp'] == 'nonuniform':
        para['dire_interpSK'] = os.path.join(os.getcwd(), '../slko/nonuniform')

    # the grid point of compression radius is uniform
    elif para['typeSKinterp'] == 'uniform':
        para['dire_interpSK'] = os.path.join(os.getcwd(), '../slko/uniform')

    # number of grid points, should be equal to atom_compr_grid
    para['ncompr'] = 10

    # if any compR < 2.2, break DFTB-ML loop
    para['compr_min'] = 2.2

    # if any compR > 9, break DFTB-ML loop
    para['compr_max'] = 9

    # grid points of compression radius of H
    para['H_compr_grid'] = t.tensor(([
        2., 2.5, 3., 3.5, 4., 4.5, 5., 6., 8., 10.]), dtype=t.float64)

    # grid points of compression radius of C
    para['C_compr_grid'] = t.tensor(([
        2., 2.5, 3., 3.5, 4., 4.5, 5., 6., 8., 10.]), dtype=t.float64)

    # onsite of H
    para['onsiteHH'] = t.tensor((
            [0.0E+00, 0.0E+00, -2.386005440483E-01]), dtype=t.float64)

    # onsite of C
    para['onsiteCC'] = t.tensor((
            [0.0E+00, -1.943551799182E-01, -5.048917654803E-01]),
            dtype=t.float64)

    # Hubbert of H
    para['uhubbHH'] = t.tensor((
            [0.0E+00, 0.0E+00, 4.196174261214E-01]), dtype=t.float64)

    # Hubber of C
    para['uhubbCC'] = t.tensor((
            [0.0E+00, 3.646664973641E-01, 3.646664973641E-01]),
            dtype=t.float64)
