'''test dftbpy'''
from __future__ import absolute_import
import torch as t
import numpy as np
import sys
import os
sys.path.append(os.path.join('../'))
import dftbtorch.slakot as slakot
import dftbtorch.dftb_torch as dftb_torch
from dftb_torch import main
import test_grad_compr


def nonscc_CH4(para):
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
    para['grid0'] = 0.4
    para['Lml_skf'] = True
    para['Lrepulsive'] = True
    # para['direSK'] = '/home/gz_fan/Documents/ML/dftb/ml/test/slko'
    para['direSK'] = '../slko'
    para['qatom_xlbomd'] = t.Tensor([4.3, 0.9, 0.9, 0.9, 0.9])
    para['coor'] = t.Tensor([
            [6, 0.0000000000, 0.0000000000, 0.0000000000],
            [1, 0.6287614522, 0.6287614522, 0.6287614522],
            [1, -0.6287614522, -0.6287614522, 0.6287614522],
            [1, -0.6287614522, 0.6287614522, -0.6287614522],
            [1, 0.6287614522, -0.6287614522, -0.6287614522]])
    main(para)
    test_nonscc_CH4(para)


def test_nonscc_CH4(para):
    '''
    '''
    natom = para['natom']
    ind_nat = para['atomind'][natom]
    fphalf = open(os.path.join('./data', 'CH4_H0_half.dat'))
    fpall = open(os.path.join('./data', 'CH4_H0_all.dat'))
    fpSall = open(os.path.join('./data', 'CH4_S_all.dat'))
    dataH0half = np.zeros((ind_nat, ind_nat))
    dataH0all = np.zeros((ind_nat, ind_nat))
    overmatall = np.zeros((ind_nat, ind_nat))
    for iind in range(ind_nat):
        for jind in range(ind_nat):
            dataH0half[iind, jind] = \
                np.fromfile(fphalf, dtype=float, count=1, sep=' ')
            # print(np.fromfile(fpall, dtype=float, count=1, sep=' '))
            dataH0all[iind, jind] = \
                np.fromfile(fpall, dtype=float, count=1, sep=' ')
            overmatall[iind, jind] = \
                np.fromfile(fpSall, dtype=float, count=1, sep=' ')
    dataH0half = t.from_numpy(dataH0half)
    dataH0all = t.from_numpy(dataH0all)
    overmatall = t.from_numpy(overmatall)

    qatom = t.Tensor([4.459725033399517, 0.885068741650120, 0.885068741650120,
                      0.885068741650120, 0.885068741650120])

    if t.all(abs(dataH0all - para['hammat']) < 1e-4):
        print('Test: H0 correct')
    else:
        print('Test: H0 wrong')
        print('sum of difference', sum(abs(dataH0all - para['hammat'])))

    if t.all(abs(qatom - para['qatomall']) < 1e-4):
        print('Test: qatomall correct')
    else:
        print('Test: qatomall wrong')


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
    para['grid0'] = 0.4
    para['Lml_skf'] = True
    para['Lrepulsive'] = True
    para['direSK'] = '/home/gz_fan/Documents/ML/dftb/slko'
    para['qatom_xlbomd'] = t.Tensor([4.3, 0.9, 0.9, 0.9, 0.9])
    para['coor'] = t.Tensor([
            [6, 0.0000000000, 0.0000000000, 0.0000000000],
            [1, 0.6287614522, 0.6287614522, 0.6287614522],
            [1, -0.6287614522, -0.6287614522, 0.6287614522],
            [1, -0.6287614522, 0.6287614522, -0.6287614522],
            [1, 0.6287614522, -0.6287614522, -0.6287614522]])
    main(para)
    # test_scc_CH4(para)


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
    para['Lml_skf'] = True
    para['Lrepulsive'] = True
    para['direSK'] = '../slko'
    para['coor'] = t.Tensor([
            [6, 0.7601011, -0.0086139515, -0.004985126],
            [6, -0.75717264, 0.0053435215, 0.002526484],
            [1, 1.2256869, 0.93354744, -0.2479243],
            [1, 1.0585717, -0.10417827, 0.99811083],
            [1, 1.1498983, -0.8023914, -0.7629031],
            [1, -1.144929, 0.6961519, 0.6684843],
            [1, -1.1742622, -1.0236914, 0.38357222],
            [1, -1.1476712, 0.3453085, -1.017948]])
    main(para)


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
    # para['direSK'] = '/home/gz_fan/Documents/ML/dftb/slko'
    para['n_dataset'] = 1
    para['coor'] = t.Tensor([
            [6, 0.0000000000, 0.0000000000, 0.0000000000],
            [1, 0.6287614522, 0.6287614522, 0.6287614522],
            [1, -0.6287614522, -0.6287614522, 0.6287614522],
            [1, -0.6287614522, 0.6287614522, -0.6287614522],
            [1, 0.6287614522, -0.6287614522, -0.6287614522]])
    para['H_init_compr'] = 3.0
    para['C_init_compr'] = 3.0
    para['H_compr_grid'] = t.Tensor([2.00, 2.50, 3.00, 3.50, 4.00, 4.50,
                                     5.00, 5.50, 6.00])
    para['C_compr_grid'] = t.Tensor([2.00, 2.50, 3.00, 3.50, 4.00, 4.50,
                                     5.00, 5.50, 6.00])
    dftb_torch.Initialization(para)
    test_grad_compr.GenMLPara(para).get_spllabel()
    test_grad_compr.interpskf(para)
    test_grad_compr.RunML(para).get_compr_specie()

    # build the ref data
    para['compr_ml'] = para['compr_init']
    slakot.SlaKo(para).genskf_interp_compr()
    test_grad_compr.RunCalc(para).idftb_torchspline()


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
    para['coor'] = t.Tensor([
            [6, 0.0000000000, 0.0000000000, 0.0000000000],
            [1, 0.6287614522, 0.6287614522, 0.6287614522],
            [1, -0.6287614522, -0.6287614522, 0.6287614522],
            [1, -0.6287614522, 0.6287614522, -0.6287614522],
            [1, 0.6287614522, -0.6287614522, -0.6287614522]])
    para['H_init_compr'] = 3.34
    para['C_init_compr'] = 4.07
    para['H_compr_grid'] = t.Tensor([2.00, 2.50, 3.00, 3.50, 4.00, 4.50,
                                     5.00, 5.50, 6.00])
    para['C_compr_grid'] = t.Tensor([2.00, 2.50, 3.00, 3.50, 4.00, 4.50,
                                     5.00, 5.50, 6.00])
    dftb_torch.Initialization(para)
    test_grad_compr.GenMLPara(para).get_spllabel()
    test_grad_compr.interpskf(para)
    test_grad_compr.RunML(para).get_compr_specie()

    # build the ref data
    para['compr_ml'] = para['compr_init']
    slakot.SlaKo(para).genskf_interp_compr()
    test_grad_compr.RunCalc(para).idftb_torchspline()
    test_nonscc_CH4(para)


def nonscc_CO(para):
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
    para['symbols'] = ['C', 'H', 'H', 'H', 'H']
    para['HSsym'] = 'symall_chol'  # symhalf, symall, symall_chol
    para['dist_tailskf'] = 1.0
    para['ninterp'] = 8
    para['grid0'] = 0.4
    para['Lrepulsive'] = True
    para['direSK'] = '/home/gz_fan/Documents/ML/dftb/slko'
    para['qatom_xlbomd'] = t.Tensor([4.3, 0.9, 0.9, 0.9, 0.9])
    para['coor'] = t.Tensor([
            [6, 0.0000000000, 0.0000000000, 0.0000000000],
            [8, 0.6512511036458978, -0.6512511036458978, 0.6512511036458978]])
    main(para)
    test_nonsccCO(para)


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
    para['direSK'] = '/home/gz_fan/Documents/ML/dftb/slko'
    para['coor'] = t.Tensor([[1, 0.0000000000, 0.0000000000, 0.0000000000]])
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
    para['coor'] = t.Tensor([[6, 0.0000000000, 0.0000000000, 0.0000000000]])
    main(para)


def test_nonsccCO(para):
    qatom = t.Tensor([3.7336089389600122, 6.266391061039996])

    if t.all(abs(qatom - para['qatomall']) < 1e-4):
        print('Test: qatomall correct')
    else:
        print('Test: qatomall wrong')


if __name__ == '__main__':
    '''
    Attention:
        if read test H0, S, compression R of H: 3.34, C: 4.07
    '''
    t.set_printoptions(precision=10)
    para = {}
    para['LReadInput'] = False  # define parameters in python, not read input
    para['Lml_HS'] = False  # donot perform ML process
    para['scf'] = True
    # scc_CH4_compr(para)
    scc_C2H6(para)
    # nonscc_CH4_compr(para)
    # scc_CH4(para)
    # nonscc_CO(para)
    # nonscc_CH4(para)
