'''test dftbpy'''
from __future__ import absolute_import
import os
import torch as t
import numpy as np
import matplotlib.pyplot as plt
import dftbtorch.slakot as slakot
import dftbtorch.dftb_torch as dftb_torch
from dftb_torch import main
import test_grad_compr
import init_parameter as initpara


def test_accuracy(para, name, dire,
                  LH0=False, H0=None,
                  LS0=False, S=None,
                  Lq=False, q=None,
                  Lp=False, p=None):
    """Read the corresponding data from normal DFTB+ code

    Required:
        H0: CH4_H0_half.dat
        S: CH4_S_half.dat

    """
    print('-' * 35, 'test accuracy:', name, '-' * 35)
    read_dftbplus_data(para, dire, H0=H0, S=S, q=q, p=None)
    nat = para['natom']

    if LH0:
        dataH0 = para['dataH']
        data_ = abs((dataH0 - para['hammat']) / abs(dataH0)).sum()
        if data_ < 1e-14 * nat ** 2:
            print('average H0 error is smaller than 1E-14')
        elif data_ < 1e-12 * nat ** 2:
            print('average H0 error is smaller than 1E-12')
        elif data_ < 1e-10 * nat ** 2:
            print('average H0 error is smaller than 1E-10')
        elif data_ < 1e-6 * nat ** 2:
            print('average H0 error is smaller than 1E-6')
        elif data_ < 1e-4 * nat ** 2:
            print('average H0 error is smaller than 1E-4')
        else:
            print('Warning: average H0 error is larger than 1E-4: {}'.format(
                    abs(dataH0 - para['qatomall']).sum() / nat))
    if LS0:
        dataS = para['dataS']
        data_ = abs((dataS - para['overmat']) / abs(dataS)).sum()
        if data_ < 1e-14 * nat ** 2:
            print('average S error is smaller than 1E-14')
        elif data_ < 1e-12 * nat ** 2:
            print('average S error is smaller than 1E-12')
        elif data_ < 1e-10 * nat ** 2:
            print('average S error is smaller than 1E-10')
        elif data_ < 1e-6 * nat ** 2:
            print('average S error is smaller than 1E-6')
        elif data_ < 1e-4 * nat ** 2:
            print('average S error is smaller than 1E-4')
        else:
            print('Warning: average S error is larger than 1E-4: {}'.format(
                    abs(dataS - para['qatomall']).sum() / nat))
    if Lq:
        dataq = para['dataq']
        data_ = abs((dataq - para['qatomall']) / abs(dataq)).sum()
        if data_ < 1e-14 * nat:
            print('average charge error is smaller than 1E-14')
        elif data_ < 1e-12 * nat:
            print('average charge error is smaller than 1E-12')
        elif data_ < 1e-10 * nat:
            print('average charge error is smaller than 1E-10')
        elif data_ < 1e-6 * nat:
            print('average charge error is smaller than 1E-6')
        elif data_ < 1e-4 * nat:
            print('average charge error is smaller than 1E-4')
        else:
            print('Warning: average charge error is larger than 1E-4: ',
                  '{}'.format(abs(dataq - para['qatomall']).sum() / nat))

    if Lp:
        datats = para['datats']
        data_ = abs((datats - para['alpha_ts']) / abs(datats)).sum()
        if data_ < 1e-14 * nat:
            print('average alpha_ts error is smaller than 1E-14')
        elif data_ < 1e-12 * nat:
            print('average alpha_ts error is smaller than 1E-12')
        elif data_ < 1e-10 * nat:
            print('average alpha_ts error is smaller than 1E-10')
        elif data_ < 1e-6 * nat:
            print('average alpha_ts error is smaller than 1E-6')
        elif data_ < 1e-4 * nat:
            print('average alpha_ts error is smaller than 1E-4')
        else:
            print('Warning: average alpha_ts0 error is larger than 1E-4: ',
                  '{}'.format(abs(datats - para['alpha_ts']).sum() / nat))
        datambd = para['datambd']
        data_ = abs((datambd - para['alpha_mbd']) / abs(datambd)).sum()
        if data_ < 1e-14 * nat:
            print('average alpha_mbd error is smaller than 1E-14')
        elif data_ < 1e-12 * nat:
            print('average alpha_mbd error is smaller than 1E-12')
        elif data_ < 1e-10 * nat:
            print('average alpha_mbd error is smaller than 1E-10')
        elif data_ < 1e-6 * nat:
            print('average alpha_mbd error is smaller than 1E-6')
        elif data_ < 1e-4 * nat:
            print('average alpha_mbd error is smaller than 1E-4')
        else:
            print('Warning: average alpha_mbd error is larger than 1E-4: ',
                  '{}'.format(abs(datambd - para['alpha_mbd']).sum() / nat))
    print('-' * 35, 'end test accuracy:', name, '-' * 35)


def read_dftbplus_data(para, dire, H0=None, S=None, q=None, p=None):
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
    """Test CH4 with non-scc DFTB

    What cab be test: dipole, charge, polarizability, H0, S
    """
    para['scc'] = 'nonscc'  # nonscc, scc, xlbomd
    para['coor'] = t.tensor(([
            [6, 0.0000000000, 0.0000000000, 0.0000000000],
            [1, 0.6287614522, 0.6287614522, 0.6287614522],
            [1, -0.6287614522, -0.6287614522, 0.6287614522],
            [1, -0.6287614522, 0.6287614522, -0.6287614522],
            [1, 0.6287614522, -0.6287614522, -0.6287614522]]), dtype=t.float64)
    para['atomNumber'] = para['coor'][:, 0]
    main(para)
    para['dataq'] = t.tensor([4.4496774784067616, 0.88758063039831014,
                              0.88758063039831003, 0.88758063039830970,
                              0.88758063039831003], dtype=t.float64)
    test_accuracy(para, 'CH4', './data', Lq=True)


def scc_CH4(para):
    """Test CH4 with scc DFTB

    What cab be test: dipole, charge, polarizability, H0, S
    """
    para['scc'] = 'scc'  # nonscc, scc, xlbomd
    para['LMBD_DFTB'] = True
    para['coor'] = t.tensor(([
            [6, 0.0000000000, 0.0000000000, 0.0000000000],
            [1, 0.6287614522, 0.6287614522, 0.6287614522],
            [1, -0.6287614522, -0.6287614522, 0.6287614522],
            [1, -0.6287614522, 0.6287614522, -0.6287614522],
            [1, 0.6287614522, -0.6287614522, -0.6287614522]]), dtype=t.float64)
    para['atomNumber'] = para['coor'][:, 0]
    main(para)
    para['dataq'] = t.tensor([4.36460632, 0.90884842, 0.90884842, 0.90884842,
                              0.90884842], dtype=t.float64)
    para['datats'] = t.tensor([9.79705420433358, 2.57029836887912,
                               2.57029836887912, 2.57029836887912,
                               2.57029836887912], dtype=t.float64)
    para['datambd'] = t.tensor([10.5834157921756, 1.82998716394802,
                                1.82998716394802, 1.82998716394802,
                                1.82998716394802], dtype=t.float64)
    test_accuracy(para, 'CH4', './data', Lq=True, Lp=True)


def nonscc_CH4_nonsym(para):
    """Test non-symmetric CH4 with non-scc DFTB

    What cab be test: dipole, charge, polarizability, H0, S
    """
    para['scc'] = 'nonscc'  # nonscc, scc, xlbomd
    para['coor'] = t.tensor((
            [[6, 3.5390060395e-02, -1.7719925381e-03, -8.0449748784e-03],
             [1, -9.5395135880e-01,  5.7158148289e-01, -1.5887808800e-01],
             [1, -6.3309413195e-01, -9.2448824644e-01,  2.2396698594e-01],
             [1, 4.5421713591e-01,  5.9006392956e-01, 7.5088745356e-01],
             [1, 7.1141016483e-01, -2.1603724360e-01, -7.2022646666e-01]]),
            dtype=t.float64)
    para['atomNumber'] = para['coor'][:, 0]
    para['dataq'] = t.tensor([4.49973615, 0.90054924, 0.89537501,
                              0.86417155, 0.84016805], dtype=t.float64)
    main(para)
    test_accuracy(para, 'CH4_nonsym', './data', Lq=True)


def scc_CH4_nonsym(para):
    """Test non-symmetric CH4 with scc DFTB

    What cab be test: dipole, charge, polarizability, H0, S
    """
    para['scc'] = 'scc'  # nonscc, scc, xlbomd
    para['LMBD_DFTB'] = True
    para['coor'] = t.tensor((
            [[6, 3.5390060395e-02, -1.7719925381e-03, -8.0449748784e-03],
             [1, -9.5395135880e-01,  5.7158148289e-01, -1.5887808800e-01],
             [1, -6.3309413195e-01, -9.2448824644e-01,  2.2396698594e-01],
             [1, 4.5421713591e-01,  5.9006392956e-01, 7.5088745356e-01],
             [1, 7.1141016483e-01, -2.1603724360e-01, -7.2022646666e-01]]),
            dtype=t.float64)
    para['atomNumber'] = para['coor'][:, 0]
    main(para)
    para['dataq'] = t.tensor([4.40465870, 0.92431392, 0.91890342,
                              0.88767461, 0.86444935], dtype=t.float64)
    para['datats'] = t.tensor([9.93812348342835, 2.76774226013437,
                               2.73426821500725, 2.45225760746137,
                               2.29442053432681], dtype=t.float64)
    para['datambd'] = t.tensor([10.6544331300661, 2.13683704440973,
                                2.15230148694062, 1.63880440230659,
                                1.42140268339990], dtype=t.float64)
    test_accuracy(para, 'CH4_nonsym', './data', Lq=True, Lp=True)


def nonscc_H2(para):
    '''
    Test eigen values, charges of CH4 by using Non-SCC DFTB;
    Before DFTB calculations, we will also test H0 and S;
    '''
    para['scc'] = 'nonscc'  # nonscc, scc, xlbomd
    para['coor'] = t.tensor(([
            [1, 0.0000000000, 0.0000000000, 0.0000000000],
            [1, 0.5000000000, 0.5000000000, 0.5000000000]]), dtype=t.float64)
    para['atomNumber'] = para['coor'][:, 0]
    main(para)
    para['dataq'] = t.tensor([1.00000000, 1.00000000], dtype=t.float64)
    test_accuracy(para, 'H2', './data', Lq=True)


def scc_H2(para):
    '''Test eigen values, charges of CH4 by using SCC DFTB;

    Before DFTB calculations, we will also test H0 and S;
    '''
    para['scc'] = 'scc'  # nonscc, scc, xlbomd
    para['convergenceType'], para['energy_tol'] = 'energy',  1e-6
    para['maxIter'] = 60
    para['LMBD_DFTB'] = True
    para['coor'] = t.tensor(([
            [1, 0.0000000000, 0.0000000000, 0.0000000000],
            [1, 0.5000000000, 0.5000000000, 0.5000000000]]), dtype=t.float64)
    para['atomNumber'] = para['coor'][:, 0]
    main(para)
    para['dataq'] = t.tensor([1.00000000, 1.00000000], dtype=t.float64)
    para['datats'] = t.tensor([2.96106578790466, 2.96106578790466],
                              dtype=t.float64)
    para['datambd'] = t.tensor([2.66188170934323, 2.66188170934323],
                               dtype=t.float64)
    test_accuracy(para, 'H2', './data', Lq=True, Lp=True)


def nonscc_CO(para):
    '''
    Test eigen values, charges of CH4 by using Non-SCC DFTB;
    Before DFTB calculations, we will also test H0 and S;
    '''
    para['scc'] = 'nonscc'  # nonscc, scc, xlbomd
    para['coor'] = t.tensor(([
            [6, 0.0000000000, 0.0000000000, 0.0000000000],
            [8, 0.6512511036458978, -0.6512511036458978, 0.6512511036458978]]),
            dtype=t.float64)
    para['atomNumber'] = para['coor'][:, 0]
    main(para)
    para['dataq'] = t.tensor([3.72849700, 6.27150300], dtype=t.float64)
    test_accuracy(para, 'CO', './data', Lq=True)


def scc_CO(para):
    '''
    Test eigen values, charges of CH4 by using SCC DFTB;
    Before DFTB calculations, we will also test H0 and S;
    '''
    para['scc'] = 'scc'  # nonscc, scc, xlbomd
    para['LMBD_DFTB'] = True
    para['coor'] = t.tensor(([
            [6, 0.0000000000, 0.0000000000, 0.0000000000],
            [8, 0.6512511036458978, -0.6512511036458978, 0.6512511036458978]]),
            dtype=t.float64)
    para['atomNumber'] = para['coor'][:, 0]
    main(para)
    para['dataq'] = t.tensor([3.88266540, 6.11733460], dtype=t.float64)
    para['datats'] = t.tensor([10.4472195378344, 5.03433839483291],
                              dtype=t.float64)
    para['datambd'] = t.tensor([10.0464154654307, 3.75102069485937],
                               dtype=t.float64)
    test_accuracy(para, 'CO', './data', Lq=True, Lp=True)


def nonscc_CO2(para):
    '''
    Test eigen values, charges of CH4 by using Non-SCC DFTB;
    Before DFTB calculations, we will also test H0 and S;
    '''
    para['scc'] = 'nonscc'  # nonscc, scc, xlbomd
    para['coor'] = t.tensor(([
            [8, -2.0357279573e-03, -1.7878314480e-02, 1.1467019320e+00],
            [6,  5.4268823005e-03,  4.7660354525e-02, 7.7558560297e-03],
            [8, -2.0357279573e-03, -1.7878314480e-02, -1.1525206566e+00]]),
            dtype=t.float64)
    para['atomNumber'] = para['coor'][:, 0]
    main(para)
    para['dataq'] = t.tensor([6.61355580, 2.71090713, 6.67553708],
                             dtype=t.float64)
    test_accuracy(para, 'CO2', './data', Lq=True)


def scc_CO2(para):
    '''
    Test eigen values, charges of CH4 by using Non-SCC DFTB;
    Before DFTB calculations, we will also test H0 and S;
    '''
    para['scc'] = 'scc'  # nonscc, scc, xlbomd
    para['LMBD_DFTB'] = True
    para['coor'] = t.tensor(([
            [8, -2.0357279573e-03, -1.7878314480e-02, 1.1467019320e+00],
            [6,  5.4268823005e-03,  4.7660354525e-02, 7.7558560297e-03],
            [8, -2.0357279573e-03, -1.7878314480e-02, -1.1525206566e+00]]),
            dtype=t.float64)
    para['atomNumber'] = para['coor'][:, 0]
    main(para)
    para['dataq'] = t.tensor([6.42707925, 3.12445809, 6.44846266],
                             dtype=t.float64)
    para['datats'] = t.tensor([5.29721592768678, 7.83915424274904,
                               5.32206102771691], dtype=t.float64)
    para['datambd'] = t.tensor([5.33032172678994, 7.09741229651905,
                                5.41673616918169], dtype=t.float64)
    test_accuracy(para, 'CO2', './data', Lq=True, Lp=True)


def nonscc_C2H6(para):
    '''
    Test eigen values, charges of CH4 by using SCC DFTB;
    Before DFTB calculations, we will also test H0 and S;
    '''
    para['scc'] = 'nonscc'  # nonscc, scc, xlbomd
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
    para['dataq'] = t.tensor([4.33252930, 4.28299142, 0.88460016, 0.86030025,
                              0.94836504, 0.89385542, 0.92004628, 0.87731213],
                             dtype=t.float64)
    test_accuracy(para, 'C2H6', './data', Lq=True)


def scc_C2H6(para):
    '''
    Test eigenvalue, charges of CH4 by using SCC DFTB;
    Before DFTB calculations, we will also test H0 and S;
    '''
    para['scc'] = 'scc'  # nonscc, scc, xlbomd
    para['LMBD_DFTB'] = True
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
    para['dataq'] = t.tensor([4.24124866, 4.19400827, 0.91722161, 0.89639943,
                              0.96568099, 0.92643924, 0.94624107, 0.91276073],
                             dtype=t.float64)
    para['datats'] = t.tensor([9.74943687855973, 9.71254658994863,
                               2.60683883074831, 2.51041218529889,
                               3.03792177278874, 2.65582313391292,
                               2.93939838062613, 2.66016640041776],
                              dtype=t.float64)
    para['datambd'] = t.tensor([10.7462967440674, 10.9812609031112,
                                1.49177182425342, 1.58170558886044,
                                2.48102887608234, 1.63630423450018,
                                2.37498195132055, 1.85438880204967],
                               dtype=t.float64)
    test_accuracy(para, 'C2H6', './data', Lq=True, Lp=True)


def nonscc_C2H6O(para):
    '''
    Test eigen values, charges of CH4 by using SCC DFTB;
    Before DFTB calculations, we will also test H0 and S;
    '''
    para['scc'] = 'nonscc'  # nonscc, scc, xlbomd
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
    para['dataq'] = t.tensor([4.24124866, 4.19400827, 0.91722161, 0.89639943,
                              0.96568099, 0.92643924, 0.94624107, 0.91276073],
                             dtype=t.float64)
    para['dataq'] = t.tensor([4.36754301, 3.78261076, 6.76038683, 0.89693636,
                              0.90218089, 0.89338497, 0.92167158, 0.90619654,
                              0.56908907], dtype=t.float64)
    test_accuracy(para, 'C2H6O', './data', Lq=True)


def scc_C2H6O(para):
    '''
    Test eigen values, charges of CH4 by using SCC DFTB;
    Before DFTB calculations, we will also test H0 and S;
    '''
    para['scc'] = 'scc'  # nonscc, scc, xlbomd
    para['LMBD_DFTB'] = True
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
    para['dataq'] = t.tensor([4.30451375, 3.82910185, 6.50280956, 0.92965294,
                              0.93426888, 0.90367817, 0.94812820, 0.96598390,
                              0.68186275], dtype=t.float64)
    para['datats'] = t.tensor([9.84660891982198, 8.98857943866821,
                               5.35847589393927, 2.70480456707182,
                               2.72124081467918, 2.56503906488386,
                               2.77880305439828, 2.88633663053240,
                               1.76563961688359], dtype=t.float64)
    para['datambd'] = t.tensor([10.7704481153938, 10.4206620417722,
                                5.17107415784681, 1.87077049923922,
                                1.75948649387994, 1.62631948159624,
                                1.72253860663783, 1.87086245531241,
                                1.29692091396141], dtype=t.float64)
    test_accuracy(para, 'C2H6O', './data', Lq=True, Lp=True)


def scc_CH4_compr(para):
    '''
    Test eigen values, charges of CH4 by using SCC DFTB;
    Before DFTB calculations, we will also test H0 and S;
    '''
    initpara.init_dftb_interp(para)
    para['scc'] = 'scc'  # nonscc, scc, xlbomd
    para['coor'] = t.tensor(([
            [6, 0.0000000000, 0.0000000000, 0.0000000000],
            [1, 0.6287614522, 0.6287614522, 0.6287614522],
            [1, -0.6287614522, -0.6287614522, 0.6287614522],
            [1, -0.6287614522, 0.6287614522, -0.6287614522],
            [1, 0.6287614522, -0.6287614522, -0.6287614522]]), dtype=t.float64)
    para['atomNumber'] = para['coor'][:, 0]
    dftb_torch.Initialization(para)
    test_grad_compr.GenMLPara(para).get_spllabel()
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
    initpara.init_dftb_interp(para)
    para['scc'] = 'nonscc'  # nonscc, scc, xlbomd
    para['coor'] = t.tensor(([
            [6, 0.0000000000, 0.0000000000, 0.0000000000],
            [1, 0.6287614522, 0.6287614522, 0.6287614522],
            [1, -0.6287614522, -0.6287614522, 0.6287614522],
            [1, -0.6287614522, 0.6287614522, -0.6287614522],
            [1, 0.6287614522, -0.6287614522, -0.6287614522]]), dtype=t.float64)
    para['atomNumber'] = para['coor'][:, 0]
    dftb_torch.Initialization(para)
    test_grad_compr.GenMLPara(para).get_spllabel()
    test_grad_compr.RunML(para).get_compr_specie()

    # build the ref data
    para['compr_ml'] = para['compr_init']
    slakot.SlaKo(para).genskf_interp_compr()
    test_grad_compr.RunCalc(para).idftb_torchspline()
    test_accuracy(para, 'CH4_compr', './data', q='CH4_nonscc_sym_q.dat')


def nonscc_CH4_compr_nongrid(para):
    '''
    Test eigen values, charges of CH4 by using SCC DFTB;
    Before DFTB calculations, we will also test H0 and S;
    '''
    initpara.init_dftb_interp(para)
    para['scc'] = 'nonscc'  # nonscc, scc, xlbomd
    para['coor'] = t.tensor((
            [[6, 3.5390060395e-02, -1.7719925381e-03, -8.0449748784e-03],
             [1, -9.5395135880e-01,  5.7158148289e-01, -1.5887808800e-01],
             [1, -6.3309413195e-01, -9.2448824644e-01,  2.2396698594e-01],
             [1, 4.5421713591e-01,  5.9006392956e-01, 7.5088745356e-01],
             [1, 7.1141016483e-01, -2.1603724360e-01, -7.2022646666e-01]]),
            dtype=t.float64)
    para['atomNumber'] = para['coor'][:, 0]
    dftb_torch.Initialization(para)
    test_grad_compr.GenMLPara(para).get_spllabel()
    test_grad_compr.RunML(para).get_compr_specie()

    # build the ref data
    para['compr_ml'] = para['compr_init']
    slakot.SlaKo(para).genskf_interp_compr()
    test_grad_compr.RunCalc(para).idftb_torchspline()
    test_accuracy(para, 'CH4_nonscc_nonsym_C2.2_H2.2', './data',
                  q='CH4_nonscc_9compr.dat')


def scc_CH4_compr_nongrid(para):
    '''
    Test eigen values, charges of CH4 by using SCC DFTB;
    Before DFTB calculations, we will also test H0 and S;
    '''
    initpara.init_dftb_interp(para)
    para['scc'] = 'scc'  # nonscc, scc, xlbomd
    para['coor'] = t.tensor((
            [[6, 3.5390060395e-02, -1.7719925381e-03, -8.0449748784e-03],
             [1, -9.5395135880e-01,  5.7158148289e-01, -1.5887808800e-01],
             [1, -6.3309413195e-01, -9.2448824644e-01,  2.2396698594e-01],
             [1, 4.5421713591e-01,  5.9006392956e-01, 7.5088745356e-01],
             [1, 7.1141016483e-01, -2.1603724360e-01, -7.2022646666e-01]]),
            dtype=t.float64)
    para['atomNumber'] = para['coor'][:, 0]
    dftb_torch.Initialization(para)
    test_grad_compr.GenMLPara(para).get_spllabel()
    test_grad_compr.RunML(para).get_compr_specie()

    # build the ref data
    para['compr_ml'] = para['compr_init']
    slakot.SlaKo(para).genskf_interp_compr()
    test_grad_compr.RunCalc(para).idftb_torchspline()
    test_accuracy(para, 'CH4_scc_nonsym_C2.2_H2.2', './data',
                  q='CH4_scc_9compr.dat')


def scc_H(para):
    '''
    Test eigen values, charges of CH4 by using SCC DFTB;
    Before DFTB calculations, we will also test H0 and S;
    '''
    para['scc'] = 'scc'  # nonscc, scc, xlbomd
    para['convergenceType'], para['energy_tol'] = 'energy',  1e-6
    para['maxIter'] = 60
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
    para['convergenceType'], para['energy_tol'] = 'energy',  1e-6
    para['maxIter'] = 60
    para['coor'] = t.tensor((
            [[6, 0.0000000000, 0.0000000000, 0.0000000000]]), dtype=t.float64)
    para['atomNumber'] = para['coor'][:, 0]
    main(para)


def generate_compr():
    '''
    We use R = a * k ** n + b to generate series compression radius
    here we set k = np.array([1.1, 1.15, 1.2, 1.3])
    n equals to the number of compression radius grids
    and satisfy:
        a * k ** 1 + b = 2
        a * k ** 15 + b = 10
    '''
    k = np.array([1.1, 1.15, 1.2, 1.3, 1.5])
    for ik in range(len(k)):
        ii = k[ik]
        a = 8 / (ii ** 15 - ii)
        b = 2 - a * ii
        compr = np.zeros(15)
        for ir in range(15):
            compr[ir] = a * ii ** (ir + 1) + b
        print(compr)


def test_compr_para(para):
    '''
    test the best k value in generate_compr
    '''
    para['scclist'] = ['scc', 'nonscc']  # nonscc, scc, xlbomd
    para['Lml'] = True  # only perform DFTB part without ML
    para['Lperiodic'] = False
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
    para['Lrepulsive'] = False
    para['Lml_compr_global'] = False
    para['LreadSKFinterp'] = True
    para['Lonsite'] = False
    para['atomspecie_old'] = []
    para['n_dataset'] = 1
    para['coor'] = t.tensor((
            [[6, 3.5390060395e-02, -1.7719925381e-03, -8.0449748784e-03],
             [1, -9.5395135880e-01,  5.7158148289e-01, -1.5887808800e-01],
             [1, -6.3309413195e-01, -9.2448824644e-01,  2.2396698594e-01],
             [1, 4.5421713591e-01,  5.9006392956e-01, 7.5088745356e-01],
             [1, 7.1141016483e-01, -2.1603724360e-01, -7.2022646666e-01]]),
            dtype=t.float64)
    para['atomNumber'] = para['coor'][:, 0]
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
    para['compr_H'] = np.array([2.1, 2.1, 2.2, 2.2, 2.7, 2.4, 2.5, 2.5, 5,
                                2.7, 3, 5])
    para['compr_C'] = np.array([2.1, 2.3, 2.2, 2.7, 2.2, 2.4, 2.5, 5, 2.5,
                                2.7, 3, 5])
    para['CH4_nonscc_compr'] = t.tensor([
            [4.54222504, 0.90951905, 0.89506652, 0.84723615, 0.80595324],
            [4.51346052, 0.91145168, 0.90039668, 0.85559173, 0.81909940],
            [4.54877458, 0.90414736, 0.89171705, 0.84654232, 0.80881869],
            [4.48491188, 0.91035700, 0.90308465, 0.86536815, 0.83627831],
            [4.63626735, 0.87746000, 0.86587724, 0.82671359, 0.79368182],
            [4.55554297, 0.89585744, 0.88667872, 0.84686949, 0.81505137],
            [4.55692722, 0.89271487, 0.88479718, 0.84749998, 0.81806075],
            [4.39621741, 0.91806577, 0.91452962, 0.89408694, 0.87710025],
            [4.74736139, 0.83225687, 0.82908318, 0.80566861, 0.78562995],
            [4.55757414, 0.88787885, 0.88192651, 0.84911303, 0.82350747],
            [4.55634287, 0.88296256, 0.87903179, 0.85158529, 0.83007749],
            [4.56767973, 0.86576564, 0.86681774, 0.85528902, 0.84444786]],
            dtype=t.float64)

    para['CH4_scc_compr'] = t.tensor([
            [4.42661256, 0.93059198, 0.91873601, 0.87880395, 0.84525550],
            [4.40626883, 0.93237145, 0.92288854, 0.88463111, 0.85384007],
            [4.43328956, 0.92712097, 0.91655472, 0.87744897, 0.84558578],
            [4.38830761, 0.93181547, 0.92499370, 0.89058300, 0.86430023],
            [4.50593586, 0.90809171, 0.89766872, 0.85934482, 0.82895889],
            [4.44221632, 0.92141881, 0.91299468, 0.87604833, 0.84732186],
            [4.44522114, 0.91908073, 0.91151631, 0.87575851, 0.84842331],
            [4.33079794, 0.93562673, 0.93178550, 0.90948106, 0.89230877],
            [4.61624592, 0.87103181, 0.86624626, 0.83473415, 0.81174186],
            [4.44950900, 0.91518760, 0.90900459, 0.87561221, 0.85068660],
            [4.45366071, 0.91072798, 0.90602786, 0.87584618, 0.85373726],
            [4.48135560, 0.89327086, 0.89241721, 0.87342269, 0.85953364]],
            dtype=t.float64)
    assert para['CH4_scc_compr'].shape == para['CH4_nonscc_compr'].shape


def test_compr_para_10points(para):
    para['H_compr_grid_tol'] = t.tensor(([
            [2., 2.88, 3.77, 4.66, 5.55, 6.44, 7.33, 8.22, 9.11, 10.],
            [2., 2.58, 3.23, 3.95, 4.73, 5.59, 6.54, 7.58, 8.73, 10.],
            [2., 2.47, 3.02, 3.65, 4.37, 5.21, 6.17, 7.27, 8.54, 10.],
            [2., 2.38, 2.84, 3.40, 4.06, 4.86, 5.81, 6.96, 8.34, 10.],
            [2., 2.31, 2.69, 3.18, 3.78, 4.54, 5.49, 6.67, 8.15, 10.],
            [2., 2.24, 2.57, 2.99, 3.54, 4.25, 5.18, 6.39, 7.96, 10.],
            [2., 2.20, 2.47, 2.84, 3.33, 4.00, 4.90, 6.12, 7.77, 10.],
            [2., 2.16, 2.39, 2.70, 3.15, 3.78, 4.65, 5.88, 7.59, 10.],
            [2., 2.10, 2.26, 2.50, 2.86, 3.40, 4.22, 5.43, 7.26, 10.]]),
            dtype=t.float64)
    para['C_compr_grid_tol'] = t.tensor(([
            [2., 2.88, 3.77, 4.66, 5.55, 6.44, 7.33, 8.22, 9.11, 10.],
            [2., 2.58, 3.23, 3.95, 4.73, 5.59, 6.54, 7.58, 8.73, 10.],
            [2., 2.47, 3.02, 3.65, 4.37, 5.21, 6.17, 7.27, 8.54, 10.],
            [2., 2.38, 2.84, 3.40, 4.06, 4.86, 5.81, 6.96, 8.34, 10.],
            [2., 2.31, 2.69, 3.18, 3.78, 4.54, 5.49, 6.67, 8.15, 10.],
            [2., 2.24, 2.57, 2.99, 3.54, 4.25, 5.18, 6.39, 7.96, 10.],
            [2., 2.20, 2.47, 2.84, 3.33, 4.00, 4.90, 6.12, 7.77, 10.],
            [2., 2.16, 2.39, 2.70, 3.15, 3.78, 4.65, 5.88, 7.59, 10.],
            [2., 2.10, 2.26, 2.50, 2.86, 3.40, 4.22, 5.43, 7.26, 10.]]),
            dtype=t.float64)
    assert para['H_compr_grid_tol'].shape == para['C_compr_grid_tol'].shape

    # kk = ['00', '05', '10', '15', '20', '25', '30', '35', '40', '50']
    kk = ['00', '05', '10', '20', '30', '40', '50']
    dire_ = '/home/gz_fan/Downloads/test/work/nonuniform/test10'
    nkk = len(kk)
    nrr = para['CH4_scc_compr'].shape[0]

    qatomall = t.zeros((2, nkk, nrr, 5), dtype=t.float64)
    qdiff = t.zeros((2, nkk, nrr, 5), dtype=t.float64)
    qdiff2 = t.zeros((2, nkk, nrr), dtype=t.float64)

    with open(os.path.join('.data', 'test_compr_para_q.hsd'), 'w') as fpq:

        for iscc in range(len(para['scclist'])):
            para['scc'] = para['scclist'][iscc]

            for ik in range(nkk):
                para['H_compr_grid'] = para['H_compr_grid_tol'][ik]
                para['C_compr_grid'] = para['C_compr_grid_tol'][ik]
                para['dire_interpSK'] = os.path.join(dire_, kk[ik])

                for ir in range(nrr):
                    para['H_init_compr'] = para['compr_H'][ir]
                    para['C_init_compr'] = para['compr_C'][ir]

                    dftb_torch.Initialization(para)
                    test_grad_compr.GenMLPara(para).get_spllabel()
                    test_grad_compr.RunML(para).get_compr_specie()
                    # build the ref data
                    para['compr_ml'] = para['compr_init']
                    slakot.SlaKo(para).genskf_interp_compr()
                    test_grad_compr.RunCalc(para).idftb_torchspline()
                    qatomall[iscc, ik, ir, :] = para['qatomall']

                    if para['scc'] == 'scc':
                        qdiff[iscc, ik, ir, :] = \
                            para['CH4_scc_compr'][ir, :] - para['qatomall']
                        qdiff2[iscc, ik, ir] = \
                            sum(abs(qdiff[iscc, ik, ir, :]) /
                                abs(para['CH4_scc_compr'][ir, :])) / 5
                        print('1:', sum(abs(qdiff[iscc, ik, ir, :])) / 5)
                        print('2:', qdiff2[iscc, ik, ir])
                    elif para['scc'] == 'nonscc':
                        qdiff[iscc, ik, ir, :] = \
                            para['CH4_nonscc_compr'][ir, :] - para['qatomall']
                        qdiff2[iscc, ik, ir] = \
                            sum(abs(qdiff[iscc, ik, ir, :]) /
                                abs(para['CH4_scc_compr'][ir, :])) / 5
                        print('1:', sum(abs(qdiff[iscc, ik, ir, :])) / 5)
                        print('2:', qdiff2[iscc, ik, ir])
                    np.savetxt(fpq, para['qatomall'].numpy(),
                               fmt="%s", newline=" ")
                    fpq.write('\n')
                    np.savetxt(fpq, qdiff[iscc, ik, ir, :].numpy(),
                               fmt="%s", newline=" ")
                    fpq.write('\n')

    xx = np.linspace(1, nrr, nrr)
    yy = np.linspace(0, 0, nrr)
    plt.plot(xx, qdiff2[0, 0, :], color='r', linestyle='-', linewidth=2,
             label='para 1')
    plt.plot(xx, qdiff2[0, 1, :], color='b', linestyle='-', linewidth=2,
             label='para 2')
    plt.plot(xx, qdiff2[0, 2, :], color='y', linestyle='-', linewidth=2,
             label='para 3')
    plt.plot(xx, qdiff2[0, 3, :], color='c', linestyle='-', linewidth=2,
             label='para 4')
    plt.plot(xx, qdiff2[0, 4, :], color='g', linestyle='-', linewidth=2,
             label='para 5')
    plt.plot(xx, qdiff2[0, 5, :], color='m', linestyle='-', linewidth=2,
             label='para 6')
    # plt.plot(xx, qdiff2[0, 6, :], color='0.75', linestyle='-', linewidth=2,
    # label='para 7')
    plt.xlabel('different compression radius points')
    plt.ylabel('absolute charge difference (non-SCC)')
    plt.plot(xx, yy, color='k', linestyle='-', linewidth=70, alpha=.15)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.legend()
    plt.show()
    plt.plot(xx, qdiff2[1, 0, :], color='r', linestyle='--', linewidth=2,
             label='para 1')
    plt.plot(xx, qdiff2[1, 1, :], color='b', linestyle='--', linewidth=2,
             label='para 2')
    plt.plot(xx, qdiff2[1, 2, :], color='y', linestyle='--', linewidth=2,
             label='para 3')
    plt.plot(xx, qdiff2[1, 3, :], color='c', linestyle='--', linewidth=2,
             label='para 4')
    plt.plot(xx, qdiff2[1, 4, :], color='g', linestyle='--', linewidth=2,
             label='para 5')
    plt.plot(xx, qdiff2[1, 5, :], color='m', linestyle='--', linewidth=2,
             label='para 6')
    # plt.plot(xx, qdiff2[1, 6, :], color='0.5', linestyle='--', linewidth=2)
    plt.xlabel('different compression radius points')
    plt.ylabel('absolute charge difference (SCC)')
    plt.plot(xx, yy, color='k', linestyle='-', linewidth=70, alpha=.15)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.legend()
    plt.show()


def test_compr_para_15points(para):
    para['H_compr_grid_tol'] = t.tensor(([
            [2., 2.57, 3.14, 3.71, 4.28, 4.85, 5.42, 6.00, 6.57, 7.14,
             7.71, 8.28, 8.85, 9.42, 10.],
            [2., 2.28, 2.60, 2.94, 3.32, 3.74, 4.20, 4.71, 5.27, 5.88,
             6.55, 7.29, 8.11, 9.01, 10.],
            [2., 2.19, 2.42, 2.68, 2.98, 3.33, 3.72, 4.18, 4.71, 5.31,
             6.01, 6.80, 7.72, 8.78, 10.],
            [2., 2.15, 2.34, 2.56, 2.82, 3.12, 3.48, 3.91, 4.41, 5., 5.7,
             6.52, 7.49, 8.64, 10.00],
            [2., 2.13, 2.29, 2.49, 2.72, 3.00, 3.34, 3.74, 4.22, 4.81,
             5.50, 6.34, 7.34, 8.55, 10.],
            [2., 2.1, 2.23, 2.4, 2.6, 2.84, 3.14, 3.52, 3.97, 4.54, 5.23,
             6.08, 7.12, 8.41, 10.00],
            [2., 2.06, 2.14, 2.24, 2.38, 2.56, 2.79, 3.09, 3.49, 4.00,
             4.66, 5.52, 6.64, 8.10, 10.]]), dtype=t.float64)
    para['C_compr_grid_tol'] = t.tensor(([
            [2., 2.57, 3.14, 3.71, 4.28, 4.85, 5.42, 6.00, 6.57, 7.14,
             7.71, 8.28, 8.85, 9.42, 10.],
            [2., 2.28, 2.60, 2.94, 3.32, 3.74, 4.20, 4.71, 5.27, 5.88,
             6.55, 7.29, 8.11, 9.01, 10.],
            [2., 2.19, 2.42, 2.68, 2.98, 3.33, 3.72, 4.18, 4.71, 5.31,
             6.01, 6.80, 7.72, 8.78, 10.],
            [2., 2.15, 2.34, 2.56, 2.82, 3.12, 3.48, 3.91, 4.41, 5., 5.7,
             6.52, 7.49, 8.64, 10.00],
            [2., 2.13, 2.29, 2.49, 2.72, 3.00, 3.34, 3.74, 4.22, 4.81,
             5.50, 6.34, 7.34, 8.55, 10.],
            [2., 2.1, 2.23, 2.4, 2.6, 2.84, 3.14, 3.52, 3.97, 4.54, 5.23,
             6.08, 7.12, 8.41, 10.00],
            [2., 2.06, 2.14, 2.24, 2.38, 2.56, 2.79, 3.09, 3.49, 4.00,
             4.66, 5.52, 6.64, 8.10, 10.]]), dtype=t.float64)
    assert para['H_compr_grid_tol'].shape == para['C_compr_grid_tol'].shape

    kk = ['00', '10', '15', '18', '20', '23', '30']
    dire_ = '/home/gz_fan/Downloads/test/work/nonuniform/test15'
    nkk = len(kk)
    nrr = para['CH4_scc_compr'].shape[0]

    qatomall = t.zeros((2, nkk, nrr, 5), dtype=t.float64)
    qdiff = t.zeros((2, nkk, nrr, 5), dtype=t.float64)
    qdiff2 = t.zeros((2, nkk, nrr), dtype=t.float64)

    with open(os.path.join('.data', 'test_compr_para_q.hsd'), 'w') as fpq:

        for iscc in range(len(para['scclist'])):
            para['scc'] = para['scclist'][iscc]

            for ik in range(nkk):
                para['H_compr_grid'] = para['H_compr_grid_tol'][ik]
                para['C_compr_grid'] = para['C_compr_grid_tol'][ik]
                para['dire_interpSK'] = os.path.join(dire_, kk[ik])

                for ir in range(nrr):
                    para['H_init_compr'] = para['compr_H'][ir]
                    para['C_init_compr'] = para['compr_C'][ir]

                    dftb_torch.Initialization(para)
                    test_grad_compr.GenMLPara(para).get_spllabel()
                    test_grad_compr.RunML(para).get_compr_specie()
                    # build the ref data
                    para['compr_ml'] = para['compr_init']
                    slakot.SlaKo(para).genskf_interp_compr()
                    test_grad_compr.RunCalc(para).idftb_torchspline()
                    qatomall[iscc, ik, ir, :] = para['qatomall']

                    if para['scc'] == 'scc':
                        qdiff[iscc, ik, ir, :] = \
                            para['CH4_scc_compr'][ir, :] - para['qatomall']
                        qdiff2[iscc, ik, ir] = \
                            sum(abs(qdiff[iscc, ik, ir, :]) /
                                abs(para['CH4_scc_compr'][ir, :])) / 5
                        print('1:', sum(abs(qdiff[iscc, ik, ir, :])) / 5)
                        print('2:', qdiff2[iscc, ik, ir])
                    elif para['scc'] == 'nonscc':
                        qdiff[iscc, ik, ir, :] = \
                            para['CH4_nonscc_compr'][ir, :] - para['qatomall']
                        qdiff2[iscc, ik, ir] = \
                            sum(abs(qdiff[iscc, ik, ir, :]) /
                                abs(para['CH4_scc_compr'][ir, :])) / 5

                    np.savetxt(fpq, para['qatomall'].numpy(),
                               fmt="%s", newline=" ")
                    fpq.write('\n')
                    np.savetxt(fpq, qdiff[iscc, ik, ir, :].numpy(),
                               fmt="%s", newline=" ")
                    fpq.write('\n')

    xx = np.linspace(1, nrr, nrr)
    yy = np.linspace(0, 0, nrr)
    plt.plot(xx, qdiff2[0, 0, :], color='r', linestyle='-', linewidth=2,
             label='para 1')
    plt.plot(xx, qdiff2[0, 1, :], color='b', linestyle='-', linewidth=2,
             label='para 2')
    plt.plot(xx, qdiff2[0, 2, :], color='y', linestyle='-', linewidth=2,
             label='para 3')
    plt.plot(xx, qdiff2[0, 3, :], color='c', linestyle='-', linewidth=2,
             label='para 4')
    plt.plot(xx, qdiff2[0, 4, :], color='g', linestyle='-', linewidth=2,
             label='para 5')
    plt.plot(xx, qdiff2[0, 5, :], color='m', linestyle='-', linewidth=2,
             label='para 6')
    plt.plot(xx, qdiff2[0, 6, :], color='0.75', linestyle='-', linewidth=2,
             label='para 7')
    plt.legend()
    plt.xlabel('different compression radius pair points')
    plt.ylabel('absolute charge difference')
    plt.plot(xx, yy, color='k', linestyle='-', linewidth=70, alpha=.15)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.show()
    plt.plot(xx, qdiff2[1, 0, :], color='r', linestyle='--', linewidth=2,
             label='para 1')
    plt.plot(xx, qdiff2[1, 1, :], color='b', linestyle='--', linewidth=2,
             label='para 2')
    plt.plot(xx, qdiff2[1, 2, :], color='y', linestyle='--', linewidth=2,
             label='para 3')
    plt.plot(xx, qdiff2[1, 3, :], color='c', linestyle='--', linewidth=2,
             label='para 4')
    plt.plot(xx, qdiff2[1, 4, :], color='g', linestyle='--', linewidth=2,
             label='para 5')
    plt.plot(xx, qdiff2[1, 5, :], color='m', linestyle='--', linewidth=2,
             label='para 6')
    plt.plot(xx, qdiff2[1, 6, :], color='0.75', linestyle='--', linewidth=2,
             label='para 7')
    plt.legend()
    plt.xlabel('different compression radius pair points')
    plt.ylabel('absolute charge difference')
    plt.plot(xx, yy, color='k', linestyle='-', linewidth=70, alpha=.15)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.show()


def normal_test(para):
    """Normal test for DFTB."""
    testlist = ['scc_CH4', 'nonscc_CH4', 'scc_H2', 'nonscc_H2', 'scc_CO',
                'nonscc_CO', 'scc_CO2', 'nonscc_CO2', 'scc_C2H6',
                'nonscc_C2H6', 'scc_CH4_nonsym', 'nonscc_CH4_nonsym',
                'scc_C2H6O', 'nonscc_C2H6O']
    initpara.init_dftb(para)
    testlist = ['scc_CH4']
    if 'nonscc_CH4' in testlist:
        nonscc_CH4(para)
    if 'scc_CH4' in testlist:
        scc_CH4(para)
    if 'nonscc_H2' in testlist:
        nonscc_H2(para)
    if 'scc_H2' in testlist:
        scc_H2(para)
    if 'nonscc_CO' in testlist:
        nonscc_CO(para)
    if 'scc_CO' in testlist:
        scc_CO(para)
    if 'nonscc_CO2' in testlist:
        nonscc_CO2(para)
    if 'scc_CO2' in testlist:
        scc_CO2(para)
    if 'nonscc_C2H6' in testlist:
        nonscc_C2H6(para)
    if 'scc_C2H6' in testlist:
        scc_C2H6(para)
    if 'nonscc_CH4_nonsym' in testlist:
        nonscc_CH4_nonsym(para)
    if 'scc_CH4_nonsym' in testlist:
        scc_CH4_nonsym(para)
    if 'nonscc_C2H6O' in testlist:
        nonscc_C2H6O(para)
    if 'scc_C2H6O' in testlist:
        scc_C2H6O(para)


def single_test(para):
    """test for DFTB."""
    initpara.init_dftb(para)
    # nonscc_CH4(para)
    scc_CH4(para)


def compr_test(para):
    """Test DFTB with compression radius, but not for ML."""
    testlist_compr = ['scc_CH4', 'nonscc_CH4', 'nonscc_CH4_compr_nongrid',
                      'scc_CH4_compr_nongrid', 'scc_CO', 'nonscc_CO']
    if 'nonscc_CH4' in testlist_compr:
        nonscc_CH4_compr(para)
    if 'scc_CH4' in testlist_compr:
        scc_CH4_compr(para)
    if 'nonscc_CH4_compr_nongrid' in testlist_compr:
        nonscc_CH4_compr_nongrid(para)
    if 'scc_CH4_compr_nongrid' in testlist_compr:
        scc_CH4_compr_nongrid(para)

    # test_compr_para(para)  # common parameters for 10 or 15 points
    # test_compr_para_10points(para)
    # test_compr_para_15points(para)


if __name__ == '__main__':
    """Test function.

    test normal DFTB, DFTB with interpolation SKF
    """
    # set the precision = 15
    t.set_printoptions(precision=15)

    para = {}
    para["test_target"] = "single"

    if para["test_target"] == "single":
        single_test(para)
    if para["test_target"] == "normal":
        normal_test(para)
    elif para["test_target"] == "compr":
        compr_test(para)
