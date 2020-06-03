'''test dftbpy'''
from __future__ import absolute_import
import torch as t
import numpy as np
import os
import dftbtorch.slakot as slakot
import dftbtorch.dftb_torch as dftb_torch
from dftb_torch import main
import test_grad_compr


def test_accuracy(para, name, dire, H0=None, S=None, q=None):
    '''
    Read the corresponding data from normal DFTB+ code
    Required:
        H0: CH4_H0_half.dat
        S: CH4_S_half.dat
    '''
    print('-' * 35, 'test accuracy:', name, '-' * 35)
    read_dftbplus_data(para, dire, H0=H0, S=S, q=q)
    nat = para['natom']

    if H0 is not None:
        dataH0 = para['dataH']
        if abs(dataH0 - para['hammat']).sum() < 1e-6 * nat ** 2:
            print('average H0 error is smaller than 1E-6')
        elif abs(dataH0 - para['hammat']).sum() < 1e-5 * nat ** 2:
            print('average H0 error is smaller than 1E-5')
        elif abs(dataH0 - para['hammat']).sum() < 1e-4 * nat ** 2:
            print('average H0 error is smaller than 1E-4')
        else:
            print('average H0 error is larger than 1E-4')
    if S is not None:
        dataS = para['dataS']
        if abs(dataS - para['overmat']).sum() < 1e-6 * nat ** 2:
            print('average S error is smaller than 1E-6')
        elif abs(dataS - para['overmat']).sum() < 1e-5 * nat ** 2:
            print('average S error is smaller than 1E-5')
        elif abs(dataS - para['overmat']).sum() < 1e-4 * nat ** 2:
            print('average S error is smaller than 1E-4')
        else:
            print('average S error is larger than 1E-4')
    if q is not None:
        dataq = para['dataq']
        if abs(dataq - para['qatomall']).sum() < 1e-6 * nat:
            print('average charge error is smaller than 1E-6')
        elif abs(dataq - para['qatomall']).sum() < 1e-5 * nat:
            print('average charge error is smaller than 1E-5')
        elif abs(dataq - para['qatomall']).sum() < 1e-4 * nat:
            print('average charge error is smaller than 1E-4')
        else:
            print('average charge error is larger than 1E-4')
    print('-' * 35, 'end test accuracy:', name, '-' * 35)


def read_dftbplus_data(para, dire, H0=None, S=None, q=None):
    '''
    Return according to Input Agrs'''
    natom = para['natom']
    ind_nat = para['atomind'][natom]

    if H0 is not None:
        fpH = open(os.path.join(dire, H0))
        para['dataH'] = np.zeros((ind_nat, ind_nat))
    if S is not None:
        fpS = open(os.path.join(dire, S))
        para['dataS'] = np.zeros((ind_nat, ind_nat))
    if q is not None:
        fpq = open(os.path.join(dire, q))
        para['dataq'] = np.zeros(natom)

    for iind in range(ind_nat):
        for jind in range(ind_nat):
            if H0 is not None:
                para['dataH'][iind, jind] = \
                    np.fromfile(fpH, dtype=float, count=1, sep=' ')
            if S is not None:
                para['dataS'][iind, jind] = \
                    np.fromfile(fpS, dtype=float, count=1, sep=' ')
    if q is not None:
        for ii in range(natom):
            para['dataq'][ii] = np.fromfile(fpq, dtype=float, count=1, sep=' ')

    if H0 is not None:
        para['dataH'] = t.from_numpy(para['dataH'])
    if S is not None:
        para['dataS'] = t.from_numpy(para['dataS'])
    if q is not None:
        para['dataq'] = t.from_numpy(para['dataq'])


def nonscc_CH4(para):
    '''
    Test eigen values, charges of CH4 by using Non-SCC DFTB;
    Before DFTB calculations, we will also test H0 and S;
    '''
    para['Lml'] = False  # only perform DFTB part without ML
    para['Lperiodic'] = False
    para['task'] = 'ground'
    para['mixMethod'], para['mixFactor'] = 'anderson', 0.2
    para['tElec'] = 0
    para['maxIter'] = 60
    para['Ldipole'] = True
    para['symbols'] = ['C', 'H', 'H', 'H', 'H']
    para['HSsym'] = 'symall_chol'  # symhalf, symall, symall_chol
    para['dist_tailskf'] = 1.0
    para['ninterp'] = 8
    para['Lml_skf'] = False
    para['Lrepulsive'] = False
    para['direSK'] = '../slko/test'
    para['coor'] = t.tensor(([
            [6, 0.0000000000, 0.0000000000, 0.0000000000],
            [1, 0.6287614522, 0.6287614522, 0.6287614522],
            [1, -0.6287614522, -0.6287614522, 0.6287614522],
            [1, -0.6287614522, 0.6287614522, -0.6287614522],
            [1, 0.6287614522, -0.6287614522, -0.6287614522]]), dtype=t.float64)
    para['atomNumber'] = para['coor'][:, 0]
    main(para)
    test_accuracy(para, 'CH4', './data', q='CH4_nonscc_sym_q.dat')


def scc_CH4(para):
    '''
    Test eigen values, charges of CH4 by using SCC DFTB;
    Before DFTB calculations, we will also test H0 and S;
    '''
    para['scc'] = 'scc'  # nonscc, scc, xlbomd
    para['Lml'] = False  # only perform DFTB part without ML
    para['Lperiodic'] = False
    para['task'] = 'ground'
    para['mixMethod'], para['mixFactor'] = 'anderson', 0.2
    para['convergenceType'], para['energy_tol'] = 'energy',  1e-6
    para['tElec'] = 0
    para['maxIter'] = 60
    para['Ldipole'] = True
    para['symbols'] = ['C', 'H', 'H', 'H', 'H']
    para['HSsym'] = 'symall_chol'  # symhalf, symall, symall_chol
    para['dist_tailskf'] = 1.0
    para['ninterp'] = 8
    para['Lml_skf'] = False
    para['Lrepulsive'] = False
    para['direSK'] = '../slko/test'
    para['coor'] = t.tensor(([
            [6, 0.0000000000, 0.0000000000, 0.0000000000],
            [1, 0.6287614522, 0.6287614522, 0.6287614522],
            [1, -0.6287614522, -0.6287614522, 0.6287614522],
            [1, -0.6287614522, 0.6287614522, -0.6287614522],
            [1, 0.6287614522, -0.6287614522, -0.6287614522]]), dtype=t.float64)
    para['atomNumber'] = para['coor'][:, 0]
    main(para)
    test_accuracy(para, 'CH4', './data', q='CH4_scc_sym_q.dat')


def nonscc_CH4_nonsym(para):
    '''
    Test eigen values, charges of CH4 by using Non-SCC DFTB;
    Before DFTB calculations, we will also test H0 and S;
    '''
    para['scc'] = 'nonscc'  # nonscc, scc, xlbomd
    para['Lml'] = False  # only perform DFTB part without ML
    para['Lperiodic'] = False
    para['task'] = 'ground'
    para['mixMethod'], para['mixFactor'] = 'anderson', 0.2
    para['tElec'] = 0
    para['maxIter'] = 60
    para['Ldipole'] = True
    para['symbols'] = ['C', 'H', 'H', 'H', 'H']
    para['HSsym'] = 'symall_chol'  # symhalf, symall, symall_chol
    para['dist_tailskf'] = 1.0
    para['ninterp'] = 8
    para['Lml_skf'] = False
    para['Lrepulsive'] = False
    para['direSK'] = '../slko/test'
    para['coor'] = t.tensor((
            [[6, 3.5390060395e-02, -1.7719925381e-03, -8.0449748784e-03],
             [1, -9.5395135880e-01,  5.7158148289e-01, -1.5887808800e-01],
             [1, -6.3309413195e-01, -9.2448824644e-01,  2.2396698594e-01],
             [1, 4.5421713591e-01,  5.9006392956e-01, 7.5088745356e-01],
             [1, 7.1141016483e-01, -2.1603724360e-01, -7.2022646666e-01]]),
            dtype=t.float64)
    para['atomNumber'] = para['coor'][:, 0]
    main(para)
    test_accuracy(para, 'CH4_nonsym', './data', q='CH4_nonscc_nonsym_q.dat')


def scc_CH4_nonsym(para):
    '''
    Test eigen values, charges of CH4 by using Non-SCC DFTB;
    Before DFTB calculations, we will also test H0 and S;
    '''
    para['scc'] = 'scc'  # nonscc, scc, xlbomd
    para['Lml'] = False  # only perform DFTB part without ML
    para['Lperiodic'] = False
    para['task'] = 'ground'
    para['mixMethod'], para['mixFactor'] = 'anderson', 0.2
    para['tElec'] = 0
    para['maxIter'] = 60
    para['Ldipole'] = True
    para['symbols'] = ['C', 'H', 'H', 'H', 'H']
    para['HSsym'] = 'symall_chol'  # symhalf, symall, symall_chol
    para['dist_tailskf'] = 1.0
    para['ninterp'] = 8
    para['Lml_skf'] = False
    para['Lrepulsive'] = False
    para['direSK'] = '../slko/test'
    para['coor'] = t.tensor((
            [[6, 3.5390060395e-02, -1.7719925381e-03, -8.0449748784e-03],
             [1, -9.5395135880e-01,  5.7158148289e-01, -1.5887808800e-01],
             [1, -6.3309413195e-01, -9.2448824644e-01,  2.2396698594e-01],
             [1, 4.5421713591e-01,  5.9006392956e-01, 7.5088745356e-01],
             [1, 7.1141016483e-01, -2.1603724360e-01, -7.2022646666e-01]]),
            dtype=t.float64)
    para['atomNumber'] = para['coor'][:, 0]
    main(para)
    test_accuracy(para, 'CH4_nonsym', './data', q='CH4_scc_nonsym_q.dat')


def nonscc_H2(para):
    '''
    Test eigen values, charges of CH4 by using Non-SCC DFTB;
    Before DFTB calculations, we will also test H0 and S;
    '''
    para['scc'] = 'nonscc'  # nonscc, scc, xlbomd
    para['Lml'] = False  # only perform DFTB part without ML
    para['Lperiodic'] = False
    para['task'] = 'ground'
    para['mixMethod'], para['mixFactor'] = 'anderson', 0.2
    para['tElec'] = 0
    para['maxIter'] = 60
    para['Ldipole'] = True
    para['symbols'] = ['H', 'H']
    para['HSsym'] = 'symall_chol'  # symhalf, symall, symall_chol
    para['dist_tailskf'] = 1.0
    para['ninterp'] = 8
    para['Lml_skf'] = True
    para['Lrepulsive'] = True
    para['direSK'] = '../slko/test'
    para['coor'] = t.tensor(([
            [1, 0.0000000000, 0.0000000000, 0.0000000000],
            [1, 0.5000000000, 0.5000000000, 0.5000000000]]), dtype=t.float64)
    para['atomNumber'] = para['coor'][:, 0]
    main(para)
    test_accuracy(para, 'H2', './data', q='H2_nonscc_q.dat')


def scc_H2(para):
    '''
    Test eigen values, charges of CH4 by using SCC DFTB;
    Before DFTB calculations, we will also test H0 and S;
    '''
    para['scc'] = 'scc'  # nonscc, scc, xlbomd
    para['Lml'] = False  # only perform DFTB part without ML
    para['Lperiodic'] = False
    para['task'] = 'ground'
    para['mixMethod'], para['mixFactor'] = 'anderson', 0.2
    para['convergenceType'], para['energy_tol'] = 'energy',  1e-6
    para['tElec'] = 0
    para['maxIter'] = 60
    para['Ldipole'] = True
    para['symbols'] = ['H', 'H']
    para['HSsym'] = 'symall_chol'  # symhalf, symall, symall_chol
    para['dist_tailskf'] = 1.0
    para['ninterp'] = 8
    para['Lml_skf'] = False
    para['Lrepulsive'] = False
    para['direSK'] = '../slko/test'
    para['coor'] = t.tensor(([
            [1, 0.0000000000, 0.0000000000, 0.0000000000],
            [1, 0.5000000000, 0.5000000000, 0.5000000000]]), dtype=t.float64)
    para['atomNumber'] = para['coor'][:, 0]
    main(para)
    test_accuracy(para, 'H2', './data', q='H2_scc_q.dat')


def nonscc_CO(para):
    '''
    Test eigen values, charges of CH4 by using Non-SCC DFTB;
    Before DFTB calculations, we will also test H0 and S;
    '''
    para['scc'] = 'nonscc'  # nonscc, scc, xlbomd
    para['Lml'] = False  # only perform DFTB part without ML
    para['Lperiodic'] = False
    para['task'] = 'ground'
    para['mixMethod'], para['mixFactor'] = 'anderson', 0.2
    para['tElec'] = 0
    para['maxIter'] = 60
    para['Ldipole'] = True
    para['symbols'] = ['C', 'O']
    para['HSsym'] = 'symall_chol'  # symhalf, symall, symall_chol
    para['dist_tailskf'] = 1.0
    para['ninterp'] = 8
    para['Lml_skf'] = False
    para['Lrepulsive'] = False
    para['direSK'] = '../slko/test'
    para['coor'] = t.tensor(([
            [6, 0.0000000000, 0.0000000000, 0.0000000000],
            [8, 0.6512511036458978, -0.6512511036458978, 0.6512511036458978]]),
            dtype=t.float64)
    para['atomNumber'] = para['coor'][:, 0]
    main(para)
    test_accuracy(para, 'CO', './data', q='CO_nonscc_q.dat')


def scc_CO(para):
    '''
    Test eigen values, charges of CH4 by using SCC DFTB;
    Before DFTB calculations, we will also test H0 and S;
    '''
    para['scc'] = 'scc'  # nonscc, scc, xlbomd
    para['Lml'] = False  # only perform DFTB part without ML
    para['Lperiodic'] = False
    para['task'] = 'ground'
    para['mixMethod'], para['mixFactor'] = 'anderson', 0.2
    para['convergenceType'], para['energy_tol'] = 'energy',  1e-6
    para['tElec'] = 0
    para['maxIter'] = 60
    para['Ldipole'] = True
    para['symbols'] = ['C', 'O']
    para['HSsym'] = 'symall_chol'  # symhalf, symall, symall_chol
    para['dist_tailskf'] = 1.0
    para['ninterp'] = 8
    para['Lml_skf'] = False
    para['Lrepulsive'] = False
    para['direSK'] = '../slko/test'
    para['coor'] = t.tensor(([
            [6, 0.0000000000, 0.0000000000, 0.0000000000],
            [8, 0.6512511036458978, -0.6512511036458978, 0.6512511036458978]]),
            dtype=t.float64)
    para['atomNumber'] = para['coor'][:, 0]
    main(para)
    test_accuracy(para, 'CO', './data', q='CO_scc_q.dat')


def nonscc_CO2(para):
    '''
    Test eigen values, charges of CH4 by using Non-SCC DFTB;
    Before DFTB calculations, we will also test H0 and S;
    '''
    para['scc'] = 'nonscc'  # nonscc, scc, xlbomd
    para['Lml'] = False  # only perform DFTB part without ML
    para['Lperiodic'] = False
    para['task'] = 'ground'
    para['mixMethod'], para['mixFactor'] = 'anderson', 0.2
    para['tElec'] = 0
    para['maxIter'] = 60
    para['Ldipole'] = True
    para['symbols'] = ['O', 'C', 'O']
    para['HSsym'] = 'symall_chol'  # symhalf, symall, symall_chol
    para['dist_tailskf'] = 1.0
    para['ninterp'] = 8
    para['Lml_skf'] = False
    para['Lrepulsive'] = False
    para['direSK'] = '../slko/test'
    para['coor'] = t.tensor(([
            [8, -2.0357279573e-03, -1.7878314480e-02, 1.1467019320e+00],
            [6,  5.4268823005e-03,  4.7660354525e-02, 7.7558560297e-03],
            [8, -2.0357279573e-03, -1.7878314480e-02, -1.1525206566e+00]]),
            dtype=t.float64)
    para['atomNumber'] = para['coor'][:, 0]
    main(para)
    test_accuracy(para, 'CO2', './data', q='CO2_nonscc_q.dat')


def scc_CO2(para):
    '''
    Test eigen values, charges of CH4 by using Non-SCC DFTB;
    Before DFTB calculations, we will also test H0 and S;
    '''
    para['scc'] = 'scc'  # nonscc, scc, xlbomd
    para['Lml'] = False  # only perform DFTB part without ML
    para['Lperiodic'] = False
    para['task'] = 'ground'
    para['mixMethod'], para['mixFactor'] = 'anderson', 0.2
    para['tElec'] = 0
    para['maxIter'] = 60
    para['Ldipole'] = True
    para['symbols'] = ['O', 'C', 'O']
    para['HSsym'] = 'symall_chol'  # symhalf, symall, symall_chol
    para['dist_tailskf'] = 1.0
    para['ninterp'] = 8
    para['Lml_skf'] = False
    para['Lrepulsive'] = False
    para['direSK'] = '../slko/test'
    para['coor'] = t.tensor(([
            [8, -2.0357279573e-03, -1.7878314480e-02, 1.1467019320e+00],
            [6,  5.4268823005e-03,  4.7660354525e-02, 7.7558560297e-03],
            [8, -2.0357279573e-03, -1.7878314480e-02, -1.1525206566e+00]]),
            dtype=t.float64)
    para['atomNumber'] = para['coor'][:, 0]
    main(para)
    test_accuracy(para, 'CO2', './data', q='CO2_scc_q.dat')


def nonscc_C2H6(para):
    '''
    Test eigen values, charges of CH4 by using SCC DFTB;
    Before DFTB calculations, we will also test H0 and S;
    '''
    para['scc'] = 'nonscc'  # nonscc, scc, xlbomd
    para['Lml'] = False  # only perform DFTB part without ML
    para['Lperiodic'] = False
    para['task'] = 'ground'
    para['mixMethod'], para['mixFactor'] = 'anderson', 0.2
    para['convergenceType'], para['energy_tol'] = 'energy',  1e-6
    para['tElec'] = 0
    para['maxIter'] = 60
    para['Ldipole'] = True
    para['symbols'] = ['C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']
    para['HSsym'] = 'symall_chol'  # symhalf, symall, symall_chol
    para['dist_tailskf'] = 1.0
    para['ninterp'] = 8
    para['Lml_skf'] = False
    para['Lrepulsive'] = False
    para['direSK'] = '../slko/test'
    para['coor'] = t.tensor((
            [[6, 7.8179776669e-01,  1.5335133066e-03, 2.6934888214e-02],
             [6, -7.9243135452e-01, -3.9727156982e-03, -1.3786645606e-02],
             [1, 1.1178507805e+00,  9.4269967079e-01, -2.2845230997e-01],
             [1, 1.3574218750e+00, -7.3365643620e-02,  8.7523090839e-01],
             [1,  1.0803720951e+00, -8.7363147736e-01, -8.4418308735e-01],
             [1, -1.2459375858e+00,  7.0729362965e-01, 6.3562983274e-01],
             [1, -1.1666057110e+00, -1.0699002743e+00, 5.0889712572e-01],
             [1, -1.0138797760e+00,  3.6464625597e-01, -1.0678678751e+00]]),
            dtype=t.float64)
    para['atomNumber'] = para['coor'][:, 0]
    main(para)
    test_accuracy(para, 'C2H6', './data', q='C2H6_nonscc_q.dat')


def scc_C2H6(para):
    '''
    Test eigen values, charges of CH4 by using SCC DFTB;
    Before DFTB calculations, we will also test H0 and S;
    '''
    para['scc'] = 'scc'  # nonscc, scc, xlbomd
    para['Lml'] = False  # only perform DFTB part without ML
    para['Lperiodic'] = False
    para['task'] = 'ground'
    para['mixMethod'], para['mixFactor'] = 'anderson', 0.2
    para['convergenceType'], para['energy_tol'] = 'energy',  1e-6
    para['tElec'] = 0
    para['maxIter'] = 60
    para['Ldipole'] = True
    para['symbols'] = ['C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']
    para['HSsym'] = 'symall_chol'  # symhalf, symall, symall_chol
    para['dist_tailskf'] = 1.0
    para['ninterp'] = 8
    para['Lml_skf'] = False
    para['Lrepulsive'] = False
    para['direSK'] = '../slko/test'
    para['coor'] = t.tensor((
            [[6, 7.8179776669e-01,  1.5335133066e-03, 2.6934888214e-02],
             [6, -7.9243135452e-01, -3.9727156982e-03, -1.3786645606e-02],
             [1, 1.1178507805e+00,  9.4269967079e-01, -2.2845230997e-01],
             [1, 1.3574218750e+00, -7.3365643620e-02,  8.7523090839e-01],
             [1,  1.0803720951e+00, -8.7363147736e-01, -8.4418308735e-01],
             [1, -1.2459375858e+00,  7.0729362965e-01, 6.3562983274e-01],
             [1, -1.1666057110e+00, -1.0699002743e+00, 5.0889712572e-01],
             [1, -1.0138797760e+00,  3.6464625597e-01, -1.0678678751e+00]]),
            dtype=t.float64)
    para['atomNumber'] = para['coor'][:, 0]
    main(para)
    test_accuracy(para, 'C2H6', './data', q='C2H6_scc_q.dat')


def nonscc_C2H6O(para):
    '''
    Test eigen values, charges of CH4 by using SCC DFTB;
    Before DFTB calculations, we will also test H0 and S;
    '''
    para['scc'] = 'nonscc'  # nonscc, scc, xlbomd
    para['Lml'] = False  # only perform DFTB part without ML
    para['Lperiodic'] = False
    para['task'] = 'ground'
    para['mixMethod'], para['mixFactor'] = 'anderson', 0.2
    para['convergenceType'], para['energy_tol'] = 'energy',  1e-6
    para['tElec'] = 0
    para['maxIter'] = 60
    para['Ldipole'] = True
    para['symbols'] = ['C', 'C', 'O', 'H', 'H', 'H', 'H', 'H', 'H']
    para['HSsym'] = 'symall_chol'  # symhalf, symall, symall_chol
    para['dist_tailskf'] = 1.0
    para['ninterp'] = 8
    para['Lml_skf'] = False
    para['Lrepulsive'] = False
    para['direSK'] = '../slko/test'
    para['coor'] = t.tensor((
            [[6, -1.1924011707, -0.2497887760, -0.0266653895],
             [6,  0.1042943373,  0.5966255069,  0.0842601359],
             [8,  1.2215454578, -0.2753631771, -0.1334060133],
             [1, -2.1476652622,  0.2638695240, -0.1319026351],
             [1, -1.4004095793, -0.8033137321,  0.8916190267],
             [1, -1.1552665234, -0.8966175318, -0.9011813998],
             [1,  0.0644546151,  1.2907464504, -0.7243113518],
             [1,  0.1250893772,  1.1639704704,  1.0000016689],
             [1,  1.1436977386, -0.9601760507,  0.5842682123]]),
            dtype=t.float64)
    para['atomNumber'] = para['coor'][:, 0]
    main(para)
    test_accuracy(para, 'C2H6O', './data', q='C2H6O_nonscc_q.dat')


def scc_C2H6O(para):
    '''
    Test eigen values, charges of CH4 by using SCC DFTB;
    Before DFTB calculations, we will also test H0 and S;
    '''
    para['scc'] = 'scc'  # nonscc, scc, xlbomd
    para['Lml'] = False  # only perform DFTB part without ML
    para['Lperiodic'] = False
    para['task'] = 'ground'
    para['mixMethod'], para['mixFactor'] = 'anderson', 0.2
    para['convergenceType'], para['energy_tol'] = 'energy',  1e-6
    para['tElec'] = 0
    para['maxIter'] = 60
    para['Ldipole'] = True
    para['symbols'] = ['C', 'C', 'O', 'H', 'H', 'H', 'H', 'H', 'H']
    para['HSsym'] = 'symall_chol'  # symhalf, symall, symall_chol
    para['dist_tailskf'] = 1.0
    para['ninterp'] = 8
    para['Lml_skf'] = False
    para['Lrepulsive'] = False
    para['direSK'] = '../slko/test'
    para['coor'] = t.tensor((
            [[6, -1.1924011707, -0.2497887760, -0.0266653895],
             [6,  0.1042943373,  0.5966255069,  0.0842601359],
             [8,  1.2215454578, -0.2753631771, -0.1334060133],
             [1, -2.1476652622,  0.2638695240, -0.1319026351],
             [1, -1.4004095793, -0.8033137321,  0.8916190267],
             [1, -1.1552665234, -0.8966175318, -0.9011813998],
             [1,  0.0644546151,  1.2907464504, -0.7243113518],
             [1,  0.1250893772,  1.1639704704,  1.0000016689],
             [1,  1.1436977386, -0.9601760507,  0.5842682123]]),
            dtype=t.float64)
    para['atomNumber'] = para['coor'][:, 0]
    main(para)
    test_accuracy(para, 'C2H6O', './data', q='C2H6O_scc_q.dat')


def scc_CH4_compr(para):
    '''
    Test eigen values, charges of CH4 by using SCC DFTB;
    Before DFTB calculations, we will also test H0 and S;
    '''
    para['scc'] = 'scc'  # nonscc, scc, xlbomd
    para['Lml'] = True  # only perform DFTB part without ML
    para['Lperiodic'] = False
    para['task'] = 'ground'
    para['mixMethod'], para['mixFactor'] = 'anderson', 0.2
    para['convergenceType'], para['energy_tol'] = 'energy',  1e-6
    para['tElec'] = 0
    para['maxIter'] = 60
    para['Ldipole'] = True
    para['symbols'] = ['C', 'H', 'H', 'H', 'H']
    para['HSsym'] = 'symall_chol'  # symhalf, symall, symall_chol
    para['dist_tailskf'] = 1.0
    para['ninterp'] = 8
    para['grid0'] = 0.4
    para['interpcutoff'] = 4.0
    para['Lml_skf'] = True
    para['Lrepulsive'] = True
    para['Lml_compr_global'] = False
    para['atomspecie_old'] = []
    para['dire_interpSK'] = os.path.join(os.getcwd(), '../slko/sk_den3')
    para['n_dataset'] = 1
    para['coor'] = t.tensor(([
            [6, 0.0000000000, 0.0000000000, 0.0000000000],
            [1, 0.6287614522, 0.6287614522, 0.6287614522],
            [1, -0.6287614522, -0.6287614522, 0.6287614522],
            [1, -0.6287614522, 0.6287614522, -0.6287614522],
            [1, 0.6287614522, -0.6287614522, -0.6287614522]]), dtype=t.float64)
    para['atomNumber'] = para['coor'][:, 0]
    para['H_init_compr'] = 2.5
    para['C_init_compr'] = 3.0
    para['H_compr_grid'] = t.tensor(([2.00, 2.50, 3.00, 3.50, 4.00, 4.50,
                                     5.00, 5.50, 6.00]), dtype=t.float64)
    para['C_compr_grid'] = t.tensor(([2.00, 2.50, 3.00, 3.50, 4.00, 4.50,
                                     5.00, 5.50, 6.00]), dtype=t.float64)
    dftb_torch.Initialization(para)
    test_grad_compr.GenMLPara(para).get_spllabel()
    test_grad_compr.interpskf(para)
    test_grad_compr.RunML(para).get_compr_specie()

    # build the ref data
    para['compr_ml'] = para['compr_init']
    slakot.SlaKo(para).genskf_interp_compr()
    test_grad_compr.RunCalc(para).idftb_torchspline()
    test_accuracy(para, 'CH4_compr', './data', q='CH4_scc_sym_q.dat')


def nonscc_CH4_compr(para):
    '''
    Test eigen values, charges of CH4 by using SCC DFTB;
    Before DFTB calculations, we will also test H0 and S;
    '''
    para['scc'] = 'nonscc'  # nonscc, scc, xlbomd
    para['Lml'] = True  # only perform DFTB part without ML
    para['Lperiodic'] = False
    para['task'] = 'ground'
    para['mixMethod'], para['mixFactor'] = 'anderson', 0.2
    para['convergenceType'], para['energy_tol'] = 'energy',  1e-6
    para['tElec'] = 0
    para['maxIter'] = 60
    para['Ldipole'] = True
    para['symbols'] = ['C', 'H', 'H', 'H', 'H']
    para['HSsym'] = 'symall_chol'  # symhalf, symall, symall_chol
    para['dist_tailskf'] = 1.0
    para['ninterp'] = 8
    para['grid0'] = 0.4
    para['interpcutoff'] = 4.0
    para['Lml_skf'] = True
    para['Lrepulsive'] = True
    para['Lml_compr_global'] = False
    para['atomspecie_old'] = []
    para['dire_interpSK'] = os.path.join(os.getcwd(), '../slko/sk_den3')
    para['n_dataset'] = 1
    para['coor'] = t.tensor(([
            [6, 0.0000000000, 0.0000000000, 0.0000000000],
            [1, 0.6287614522, 0.6287614522, 0.6287614522],
            [1, -0.6287614522, -0.6287614522, 0.6287614522],
            [1, -0.6287614522, 0.6287614522, -0.6287614522],
            [1, 0.6287614522, -0.6287614522, -0.6287614522]]), dtype=t.float64)
    para['atomNumber'] = para['coor'][:, 0]
    para['H_init_compr'] = 3.34
    para['C_init_compr'] = 4.07
    para['H_compr_grid'] = t.tensor(([2.00, 2.50, 3.00, 3.50, 4.00, 4.50,
                                      5.00, 5.50, 6.00]), dtype=t.float64)
    para['C_compr_grid'] = t.tensor(([2.00, 2.50, 3.00, 3.50, 4.00, 4.50,
                                     5.00, 5.50, 6.00]), dtype=t.float64)
    dftb_torch.Initialization(para)
    test_grad_compr.GenMLPara(para).get_spllabel()
    test_grad_compr.interpskf(para)
    test_grad_compr.RunML(para).get_compr_specie()

    # build the ref data
    para['compr_ml'] = para['compr_init']
    slakot.SlaKo(para).genskf_interp_compr()
    test_grad_compr.RunCalc(para).idftb_torchspline()
    test_accuracy(para, 'CH4_compr', './data', q='CH4_nonscc_sym_q.dat')


def scc_H(para):
    '''
    Test eigen values, charges of CH4 by using SCC DFTB;
    Before DFTB calculations, we will also test H0 and S;
    '''
    para['scc'] = 'scc'  # nonscc, scc, xlbomd
    para['Lml'] = False  # only perform DFTB part without ML
    para['Lperiodic'] = False
    para['Lrepulsive'] = True
    para['task'] = 'ground'
    para['mixMethod'], para['mixFactor'] = 'anderson', 0.2
    para['convergenceType'], para['energy_tol'] = 'energy',  1e-6
    para['tElec'] = 0
    para['maxIter'] = 60
    para['Ldipole'] = True
    para['symbols'] = ['H']
    para['HSsym'] = 'symall_chol'  # symhalf, symall, symall_chol
    para['dist_tailskf'] = 1.0
    para['ninterp'] = 8
    para['grid0'] = 0.4
    para['direSK'] = '/home/gz_fan/Documents/ML/dftb/slko/test'
    para['coor'] = t.tensor(([[1, 0.0000000000, 0.0000000000, 0.0000000000]]),
                            dtype=t.float64)
    para['atomNumber'] = para['coor'][:, 0]
    main(para)


def scc_C(para):
    '''
    Test eigen values, charges of CH4 by using SCC DFTB;
    Before DFTB calculations, we will also test H0 and S;
    '''
    para['scc'] = 'scc'  # nonscc, scc, xlbomd
    para['Lml'] = False  # only perform DFTB part without ML
    para['Lperiodic'] = False
    para['Lrepulsive'] = True
    para['task'] = 'ground'
    para['mixMethod'], para['mixFactor'] = 'anderson', 0.2
    para['convergenceType'], para['energy_tol'] = 'energy',  1e-6
    para['tElec'] = 0
    para['maxIter'] = 60
    para['Ldipole'] = True
    para['symbols'] = ['C']
    para['HSsym'] = 'symall_chol'  # symhalf, symall, symall_chol
    para['dist_tailskf'] = 1.0
    para['ninterp'] = 8
    para['grid0'] = 0.4
    para['direSK'] = '/home/gz_fan/Documents/ML/dftb/slko'
    para['coor'] = t.tensor((
            [[6, 0.0000000000, 0.0000000000, 0.0000000000]]), dtype=t.float64)
    para['atomNumber'] = para['coor'][:, 0]
    main(para)


def test_nonsccCO(para):
    qatom = t.tensor(([3.7336089389600122, 6.266391061039996]),
                     dtype=t.float64)

    if t.all(abs(qatom - para['qatomall']) < 1e-4):
        print('Test: qatomall correct')
    else:
        print('Test: qatomall wrong')


if __name__ == '__main__':
    '''
    Attention:
        if read test H0, S, compression R of H: 3.34, C: 4.07
    '''
    t.set_printoptions(precision=15)
    para = {}
    testlist = ['scc_CH4', 'nonscc_CH4', 'scc_H2', 'nonscc_H2', 'scc_CO',
                'nonscc_CO', 'scc_CO2', 'nonscc_CO2', 'scc_C2H6',
                'nonscc_C2H6', 'scc_CH4_nonsym', 'nonscc_CH4_nonsym',
                'scc_C2H6O', 'nonscc_C2H6O']
    para['LReadInput'] = False  # define parameters in python, not read input
    para['Lml_HS'] = False  # donot perform ML process
    para['scf'] = True
    para['LMBD_DFTB'] = False
    if para['LMBD_DFTB']:
        para['n_omega_grid'] = 15  # mbd_vdw_n_quad_pts = para['n_omega_grid']
        para['vdw_self_consistent'] = False
    if 'scc_CH4' in testlist:
        scc_CH4(para)
    if 'nonscc_CH4' in testlist:
        nonscc_CH4(para)
    if 'scc_H2' in testlist:
        scc_H2(para)
    if 'nonscc_H2' in testlist:
        nonscc_H2(para)
    if 'scc_CO' in testlist:
        scc_CO(para)
    if 'nonscc_CO' in testlist:
        nonscc_CO(para)
    if 'scc_CO2' in testlist:
        scc_CO2(para)
    if 'nonscc_CO2' in testlist:
        nonscc_CO2(para)
    if 'scc_C2H6' in testlist:
        scc_C2H6(para)
    if 'nonscc_C2H6' in testlist:
        nonscc_C2H6(para)
    if 'scc_CH4_nonsym' in testlist:
        scc_CH4_nonsym(para)
    if 'nonscc_CH4_nonsym' in testlist:
        nonscc_CH4_nonsym(para)
    if 'scc_C2H6O' in testlist:
        scc_C2H6O(para)
    if 'nonscc_C2H6O' in testlist:
        nonscc_C2H6O(para)

    testlist_compr = ['scc_CH4', 'nonscc_CH4', 'scc_CO', 'nonscc_CO']
    if 'scc_CH4' in testlist:
        scc_CH4_compr(para)
    if 'nonscc_CH4' in testlist:
        nonscc_CH4_compr(para)
