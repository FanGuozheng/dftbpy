'''test dftbpy'''

from dftb_torch import main
import torch as t


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
    para['direSK'] = '/home/gz_fan/Documents/ML/dftb/ml/test/slko'
    # para['direTest'] = '/home/gz_fan/Documents/ML/dftb/ml/test/data/test'
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
    The CH4 molecule coordination should be:
        [6, 0.0000000000, 0.0000000000, 0.0000000000]
        [1, 0.6287614522, 0.6287614522, 0.6287614522]
        [1, -0.6287614522, -0.6287614522, 0.6287614522]
        [1, -0.6287614522, 0.6287614522, -0.6287614522]
        [1, 0.6287614522, -0.6287614522, -0.6287614522]
    '''
    ham0_half = t.Tensor(
            [[-0.504891765480, 0.0000000000000, 0.0000000000000, 0.0000000000000, -0.3071154851893, -0.307115485189, -0.307115485189, -0.3071154851893],
             [0.0000000000000, -0.194355179918, 0.0000000000000, 0.0000000000000, -0.1479803805324, 0.1479803805324, 0.1479803805324, -0.1479803805324],
             [0.0000000000000, 0.0000000000000, -0.194355179918, 0.0000000000000, -0.1479803805324, 0.1479803805324, -0.147980380532, 0.1479803805324],
             [0.0000000000000, 0.0000000000000, 0.0000000000000, -0.194355179918, -0.1479803805324, -0.147980380532, 0.1479803805324, 0.1479803805324],
             [0.0000000000000, 0.0000000000000, 0.0000000000000, 0.0000000000000, -0.2386005440483, -0.082562905162, -0.082562905162, -0.0825629051626],
             [0.0000000000000, 0.0000000000000, 0.0000000000000, 0.0000000000000, 0.00000000000000, -0.238600544048, -0.082562905162, -0.0825629051626],
             [0.0000000000000, 0.0000000000000, 0.0000000000000, 0.0000000000000, 0.00000000000000, 0.0000000000000, -0.238600544048, -0.0825629051626],
             [0.0000000000000, 0.0000000000000, 0.0000000000000, 0.0000000000000, 0.00000000000000, 0.0000000000000, 0.0000000000000, -0.2386005440483]])

    ham0_all = t.Tensor(
            [[-0.504891765480, 0.0000000000000, 0.0000000000000, 0.0000000000000, -0.33807950250023, -0.3380795025002, -0.3380795025002, -0.3380795025002],
             [0.0000000000000, -0.194355179918, 0.0000000000000, 0.0000000000000, -0.15898409162282387, 0.15898409162282387, 0.15898409162282387, -0.15898409162282387],
             [0.0000000000000, 0.0000000000000, -0.194355179918, 0.0000000000000, -0.15898409162282387, 0.15898409162282387, -0.15898409162282387, 0.15898409162282387],
             [0.0000000000000, 0.0000000000000, 0.0000000000000, -0.194355179918, -0.15898409162282387, -0.15898409162282387, 0.15898409162282387, 0.15898409162282387],
             [-0.3380795025002, -0.15898409162282387, -0.15898409162282387, -0.15898409162282387, -0.2386005440483, -0.08545111268100615, -0.08545111268100615, -0.08545111268100615],
             [-0.3380795025002, 0.15898409162282387, 0.15898409162282387, -0.15898409162282387, -0.08545111268100615, -0.238600544048, -0.08545111268100615, -0.08545111268100615],
             [-0.3380795025002, 0.15898409162282387, -0.15898409162282387, 0.15898409162282387, -0.08545111268100615, -0.08545111268100615, -0.238600544048, -0.08545111268100615],
             [-0.3380795025002, -0.15898409162282387, 0.15898409162282387, 0.15898409162282387, -0.08545111268100615, -0.08545111268100615, -0.08545111268100615, -0.2386005440483]])

    ham0_all2 = t.Tensor(
            [[-0.504891765480, 0.0000000000000, 0.0000000000000, 0.0000000000000, -0.3071154851893, -0.307115485189, -0.307115485189, -0.3071154851893],
             [0.0000000000000, -0.194355179918, 0.0000000000000, 0.0000000000000, -0.1479803805324, 0.1479803805324, 0.1479803805324, -0.1479803805324],
             [0.0000000000000, 0.0000000000000, -0.194355179918, 0.0000000000000, -0.1479803805324, 0.1479803805324, -0.147980380532, 0.1479803805324],
             [0.0000000000000, 0.0000000000000, 0.0000000000000, -0.194355179918, -0.1479803805324, -0.147980380532, 0.1479803805324, 0.1479803805324],
             [-0.307115485189, -0.1479803805324, -0.1479803805324, -0.1479803805324, -0.2386005440483, -0.082562905162, -0.082562905162, -0.0825629051626],
             [-0.307115485189, 0.1479803805324, 0.1479803805324, -0.1479803805324, -0.082562905162, -0.238600544048, -0.082562905162, -0.0825629051626],
             [-0.307115485189, 0.1479803805324, -0.1479803805324, 0.1479803805324, -0.082562905162, -0.082562905162, -0.238600544048, -0.0825629051626],
             [-0.307115485189, -0.1479803805324, 0.1479803805324, 0.1479803805324, -0.082562905162, -0.082562905162, -0.082562905162, -0.2386005440483]])

    overmat = t.Tensor(
            [[1.000000000000, 0.00000000000, 0.000000000000, 0.000000000000, 0.444545870117, 0.4445458701170, 0.4445458701170, 0.4445458701170],
             [0.000000000000, 1.00000000000, 0.000000000000, 0.000000000000, 0.280401201432, -0.280401201432, -0.280401201432, 0.2804012014322],
             [0.000000000000, 0.00000000000, 1.000000000000, 0.000000000000, 0.280401201432, -0.280401201432, 0.2804012014322, -0.280401201432],
             [0.000000000000, 0.00000000000, 0.000000000000, 1.000000000000, 0.280401201432, 0.280401201432, -0.2804012014322, -0.280401201432],
             [0.444545870117, 0.280401201432, 0.280401201432, 0.280401201432, 1.000000000000, 0.1373426730235, 0.1373426730235, 0.1373426730235],
             [0.444545870117, -0.280401201432, -0.280401201432, 0.280401201432, 0.1373426730235, 1.000000000000, 0.1373426730235, 0.1373426730235],
             [0.444545870117, -0.280401201432, 0.280401201432, -0.280401201432, 0.1373426730235, 0.1373426730235, 1.00000000000000, 0.1373426730235],
             [0.444545870117, 0.280401201432, -0.280401201432, -0.280401201432, 0.1373426730235, 0.1373426730235, 0.1373426730235, 1.00000000000000]])

    qatom = t.Tensor([4.459725033399517, 0.885068741650120, 0.885068741650120,
                      0.885068741650120, 0.885068741650120])

    if t.all(abs(ham0_all - para['hammat']) < 1e-4):
        print('Test: H0 correct')
    else:
        print('Test: H0 wrong')
        print('sum of difference', sum(abs(ham0_all - para['hammat'])))

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
    Required:
        .skf: compression R of H: 3.34, C: 4.07
    '''
    t.set_printoptions(precision=15)
    para = {}
    para['LReadInput'] = False  # define parameters in python, not read input
    para['Lml_HS'] = False  # donot perform ML process
    para['scf'] = True
    scc_C(para)
