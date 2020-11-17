'''Test dftbpy precision with DFTB+.'''
import unittest
import torch as t
from dftbtorch.dftbcalculator import DFTBCalculator


class DFTBPrecision(unittest.TestCase):
    """Reference is from DFTB+ mio."""

    def template_nonscc(self):
        """Template for Non-SCC-DFTB."""
        parameter = {}
        parameter['scc'] = 'nonscc'  # nonscc, scc, xlbomd
        parameter['directorySK'] = '../../slko/test/'
        return parameter

    def template_scc(self):
        """Template for SCC-DFTB."""
        parameter = {}
        parameter['scc'] = 'scc'  # nonscc, scc, xlbomd
        parameter['directorySK'] = '../../slko/test/'
        return parameter

    def test_scc_H(self):
        """Test eigen values, charges of single H by using SCC DFTB."""
        dataset = {
            'positions': t.tensor([[0.000000000, 0.000000000, 0.000000000]]),
            'numbers': [[1]]}
        parameter = self.template_scc()
        result = DFTBCalculator(parameter, dataset)
        self.assertAlmostEqual(result.parameter['charge'].squeeze(), 1., 1E-14)

    def test_scc_C(self):
        """Test eigen values, charges of single C by using SCC DFTB."""
        dataset = {
            'positions': t.tensor([[0.000000000, 0.000000000, 0.000000000]]),
            'numbers': [[6]]}
        parameter = self.template_scc()
        result = DFTBCalculator(parameter, dataset)
        self.assertAlmostEqual(result.parameter['charge'].squeeze(), 4., 1E-14)

    def test_CH4_nonscc(self):
        """Test charge of CH4 with non-scc DFTB."""
        dataset = {'positions': t.tensor([
            [0.0000000000, 0.0000000000, 0.0000000000],
            [0.6287614522, 0.6287614522, 0.6287614522],
            [-0.6287614522, -0.6287614522, 0.6287614522],
            [-0.6287614522, 0.6287614522, -0.6287614522],
            [0.6287614522, -0.6287614522, -0.6287614522]]),
            'numbers': [[6, 1, 1, 1, 1]]}
        parameter = self.template_nonscc()
        result = DFTBCalculator(parameter, dataset)
        refq = [4.4496774784067616, 0.88758063039831014, 0.88758063039831003,
                0.88758063039830970, 0.88758063039831003]
        [self.assertAlmostEqual(i, j, delta=1E-14)
         for i, j in zip(result.parameter['charge'].squeeze(), refq)]

    def test_scc_CH4(self):
        """Test charge of CH4 with scc DFTB."""
        dataset = {}
        dataset['positions'] = t.tensor([
                [0.0000000000, 0.0000000000, 0.0000000000],
                [0.6287614522, 0.6287614522, 0.6287614522],
                [-0.6287614522, -0.6287614522, 0.6287614522],
                [-0.6287614522, 0.6287614522, -0.6287614522],
                [0.6287614522, -0.6287614522, -0.6287614522]])
        dataset['numbers'] = [[6, 1, 1, 1, 1]]
        parameter = self.template_scc()
        result = DFTBCalculator(parameter, dataset)
        refq = [4.3646063221278348, 0.9088484194680416, 0.9088484194680417,
                0.9088484194680415, 0.9088484194680422]
        refTS = [9.79705420433358, 2.57029836887912, 2.57029836887912,
                 2.57029836887912, 2.57029836887912]
        refMBD = [10.5834157921756, 1.82998716394802, 1.82998716394802,
                  1.82998716394802, 1.82998716394802]
        # highest test precision here is 1E-10
        [self.assertAlmostEqual(i, j, delta=1E-10)
         for i, j in zip(result.parameter['charge'].squeeze(), refq)]

    def test_ase(self):
        """Test DFTB with ASE input."""
        from ase import Atoms
        h2 = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.7]])
        dataset = {}
        dataset['positions'] = t.tensor(h2.get_positions())
        dataset['numbers'] = t.tensor(h2.get_atomic_numbers())
        parameter = {}
        parameter['directorySK'] = '../../slko/test/'
        DFTBCalculator(parameter, dataset)

    def test_nonscc_CH4_nonsym(self):
        """Test non-symmetric CH4 with non-scc DFTB."""
        dataset = {}
        dataset['positions'] = t.tensor(
            [[3.5390060395e-02, -1.7719925381e-03, -8.0449748784e-03],
             [-9.5395135880e-01,  5.7158148289e-01, -1.5887808800e-01],
             [-6.3309413195e-01, -9.2448824644e-01,  2.2396698594e-01],
             [4.5421713591e-01,  5.9006392956e-01, 7.5088745356e-01],
             [7.1141016483e-01, -2.1603724360e-01, -7.2022646666e-01]])
        dataset['numbers'] = [[6, 1, 1, 1, 1]]
        parameter = self.template_nonscc()
        refq = t.tensor([4.4997361516795538, 0.9005492428500024,
                         0.8953750140152250, 0.8641715463062267,
                         0.8401680451489958])
        result = DFTBCalculator(parameter, dataset)
        [self.assertAlmostEqual(i, j, delta=1E-14)
         for i, j in zip(result.parameter['charge'].squeeze(), refq)]

    def test_scc_CH4_nonsym(self):
        """Test non-symmetric CH4 with scc DFTB."""
        dataset = {}
        dataset['positions'] = t.tensor(
            [[3.5390060395e-02, -1.7719925381e-03, -8.0449748784e-03],
             [-9.5395135880e-01,  5.7158148289e-01, -1.5887808800e-01],
             [-6.3309413195e-01, -9.2448824644e-01,  2.2396698594e-01],
             [4.5421713591e-01,  5.9006392956e-01, 7.5088745356e-01],
             [7.1141016483e-01, -2.1603724360e-01, -7.2022646666e-01]])
        dataset['numbers'] = [[6, 1, 1, 1, 1]]
        parameter = self.template_scc()
        result = DFTBCalculator(parameter, dataset)
        refq = t.tensor([4.4046586991616872, 0.9243139211840105,
                         0.9189034167482688, 0.8876746127630650,
                         0.8644493501429688])
        refpolts = t.tensor([9.93812348342835, 2.76774226013437,
                             2.73426821500725, 2.45225760746137,
                             2.29442053432681])
        refpolmbd = t.tensor([10.6544331300661, 2.13683704440973,
                              2.15230148694062, 1.63880440230659,
                              1.42140268339990])
        # highest test precision here is 1E-7
        [self.assertAlmostEqual(i, j, delta=1E-7)
         for i, j in zip(result.parameter['charge'].squeeze(), refq)]

    def test_nonscc_H2(self):
        """Test eigen values, charges of H2  by using Non-SCC DFTB."""
        dataset = {}
        dataset['positions'] = t.tensor(
            [[0.0000000000, 0.0000000000, 0.0000000000],
             [0.5000000000, 0.5000000000, 0.5000000000]])
        dataset['numbers'] = [[1, 1]]
        parameter = self.template_nonscc()
        result = DFTBCalculator(parameter, dataset)
        refq = t.tensor([1.00000000, 1.00000000])
        [self.assertAlmostEqual(i, j, delta=1E-7)
         for i, j in zip(result.parameter['charge'].squeeze(), refq)]

    def test_scc_H2(self):
        """Test eigen values, charges of H2 by using SCC DFTB."""
        dataset = {}
        dataset['positions'] = t.tensor(
            [[0.0000000000, 0.0000000000, 0.0000000000],
             [0.5000000000, 0.5000000000, 0.5000000000]])
        dataset['numbers'] = [[1, 1]]
        parameter = self.template_scc()
        result = DFTBCalculator(parameter, dataset)
        refq = t.tensor([1.00000000, 1.00000000])
        refpolts = t.tensor([2.96106578790466, 2.96106578790466])
        refpolmbd = t.tensor([2.66188170934323, 2.66188170934323])
        [self.assertAlmostEqual(i, j, delta=1E-14)
         for i, j in zip(result.parameter['charge'].squeeze(), refq)]

    def test_nonscc_CO(self):
        """Test eigen values, charges of CO by using Non-SCC DFTB."""
        dataset = {}
        dataset['positions'] = t.tensor(
            [[0.0000000000, 0.0000000000, 0.0000000000],
             [0.6512511036458978, -0.6512511036458978, 0.6512511036458978]])
        dataset['numbers'] = [[6, 8]]
        parameter = self.template_nonscc()
        result = DFTBCalculator(parameter, dataset)
        refq = t.tensor([3.7284970028511140, 6.2715029971488887])
        [self.assertAlmostEqual(i, j, delta=1E-14)
         for i, j in zip(result.parameter['charge'].squeeze(), refq)]

    def test_scc_CO(self):
        """Test eigen values, charges of CO by using SCC DFTB."""
        dataset = {}
        dataset['positions'] = t.tensor(
            [[0.0000000000, 0.0000000000, 0.0000000000],
             [0.6512511036458978, -0.6512511036458978, 0.6512511036458978]])
        dataset['numbers'] = [[6, 8]]
        parameter = self.template_scc()
        result = DFTBCalculator(parameter, dataset)
        refq = t.tensor([3.8826653985894914, 6.1173346014105121])
        refpolts = t.tensor([10.4472195378344, 5.03433839483291])
        refpolmbd = t.tensor([10.0464154654307, 3.75102069485937])
        [self.assertAlmostEqual(i, j, delta=1E-8)
         for i, j in zip(result.parameter['charge'].squeeze(), refq)]

    def test_nonscc_CO2(self):
        """Test eigen values, charges of CO2 by using Non-SCC DFTB."""
        dataset = {}
        dataset['positions'] = t.tensor(
            [[-2.0357279573e-03, -1.7878314480e-02, 1.1467019320e+00],
             [5.4268823005e-03,  4.7660354525e-02, 7.7558560297e-03],
             [-2.0357279573e-03, -1.7878314480e-02, -1.1525206566e+00]])
        dataset['numbers'] = [[8, 6, 8]]
        parameter = self.template_nonscc()
        result = DFTBCalculator(parameter, dataset)
        refq = t.tensor([6.6135557981748079, 2.7109071259517026, 6.6755370758734962])
        [self.assertAlmostEqual(i, j, delta=1E-14)
         for i, j in zip(result.parameter['charge'].squeeze(), refq)]

    def test_scc_CO2(self):
        """Test eigen values, charges of CO2 by using SCC DFTB."""
        dataset = {}
        dataset['positions'] = t.tensor(
            [[-2.0357279573e-03, -1.7878314480e-02, 1.1467019320e+00],
             [5.4268823005e-03,  4.7660354525e-02, 7.7558560297e-03],
             [-2.0357279573e-03, -1.7878314480e-02, -1.1525206566e+00]])
        dataset['numbers'] = [[8, 6, 8]]
        parameter = self.template_scc()
        result = DFTBCalculator(parameter, dataset)
        refq = t.tensor([6.4270792466227178, 3.1244580919585441, 6.4484626614187430])
        refpolts = t.tensor(
            [5.29721592768678, 7.83915424274904, 5.32206102771691])
        refpolmbd = t.tensor([5.33032172678994, 7.09741229651905,
                              5.41673616918169])
        [self.assertAlmostEqual(i, j, delta=1E-7)
         for i, j in zip(result.parameter['charge'].squeeze(), refq)]

    def test_nonscc_C2H6(self):
        """Test eigen values, charges of C2H6 by using SCC DFTB."""
        dataset = {}
        dataset['positions'] = t.tensor(
            [[7.8179776669e-01,  1.5335133066e-03, 2.6934888214e-02],
             [-7.9243135452e-01, -3.9727156982e-03, -1.3786645606e-02],
             [1.1178507805e+00,  9.4269967079e-01, -2.2845230997e-01],
             [1.3574218750e+00, -7.3365643620e-02,  8.7523090839e-01],
             [1.0803720951e+00, -8.7363147736e-01, -8.4418308735e-01],
             [-1.2459375858e+00,  7.0729362965e-01, 6.3562983274e-01],
             [-1.1666057110e+00, -1.0699002743e+00, 5.0889712572e-01],
             [-1.0138797760e+00,  3.6464625597e-01, -1.0678678751e+00]])
        dataset['numbers'] = [[6, 6, 1, 1, 1, 1, 1, 1]]
        parameter = self.template_nonscc()
        result = DFTBCalculator(parameter, dataset)
        refq = t.tensor(
            [4.3325292970327069, 4.2829914231660391, 0.8846001605884933,
             0.8603002498060950, 0.9483650364409464, 0.8938554239478647,
             0.9200462807151628, 0.8773121283026949])
        [self.assertAlmostEqual(i, j, delta=1E-14)
         for i, j in zip(result.parameter['charge'].squeeze(), refq)]

    def test_scc_C2H6(self):
        """Test eigenvalue, charges of C2H6 by using SCC DFTB."""
        dataset = {}
        dataset['positions'] = t.tensor(
            [[7.8179776669e-01,  1.5335133066e-03, 2.6934888214e-02],
             [-7.9243135452e-01, -3.9727156982e-03, -1.3786645606e-02],
             [1.1178507805e+00,  9.4269967079e-01, -2.2845230997e-01],
             [1.3574218750e+00, -7.3365643620e-02,  8.7523090839e-01],
             [1.0803720951e+00, -8.7363147736e-01, -8.4418308735e-01],
             [-1.2459375858e+00,  7.0729362965e-01, 6.3562983274e-01],
             [-1.1666057110e+00, -1.0699002743e+00, 5.0889712572e-01],
             [-1.0138797760e+00,  3.6464625597e-01, -1.0678678751e+00]])
        dataset['numbers'] = [[6, 6, 1, 1, 1, 1, 1, 1]]
        parameter = self.template_scc()
        result = DFTBCalculator(parameter, dataset)
        refq = t.tensor(
            [4.2412486605036905, 4.1940082696488874, 0.9172216094361348,
             0.8963994303673891, 0.9656809937193864, 0.9264392363000882,
             0.9462410703162227, 0.9127607297082045])
        refpolts = t.tensor(
            [9.74943687855973, 9.71254658994863, 2.60683883074831,
             2.51041218529889, 3.03792177278874, 2.65582313391292,
             2.93939838062613, 2.66016640041776])
        refpolmbd = t.tensor(
            [10.7462967440674, 10.9812609031112, 1.49177182425342,
             1.58170558886044, 2.48102887608234, 1.63630423450018,
             2.37498195132055, 1.85438880204967])
        [self.assertAlmostEqual(i, j, delta=1E-6)
         for i, j in zip(result.parameter['charge'].squeeze(), refq)]

    def test_nonscc_C2H6O(self):
        """Test eigen values, charges of C2H6O by using non-SCC DFTB."""
        dataset = {}
        dataset['positions'] = t.tensor(
            [[-1.1924011707, -0.2497887760, -0.0266653895],
             [0.1042943373,  0.5966255069,  0.0842601359],
             [1.2215454578, -0.2753631771, -0.1334060133],
             [-2.1476652622,  0.2638695240, -0.1319026351],
             [-1.4004095793, -0.8033137321,  0.8916190267],
             [-1.1552665234, -0.8966175318, -0.9011813998],
             [0.0644546151,  1.2907464504, -0.7243113518],
             [0.1250893772,  1.1639704704,  1.0000016689],
             [1.1436977386, -0.9601760507,  0.5842682123]])
        dataset['numbers'] = [[6, 6, 8, 1, 1, 1, 1, 1, 1]]
        parameter = self.template_nonscc()
        result = DFTBCalculator(parameter, dataset)
        refq = t.tensor(
            [4.3675430061371792, 3.7826107550854298, 6.7603868320346310,
             0.8969363587360036, 0.9021808864900093, 0.8933849747787559,
             0.9216715792067960, 0.9061965403164295, 0.5690890672147806])
        [self.assertAlmostEqual(i, j, delta=1E-14)
         for i, j in zip(result.parameter['charge'].squeeze(), refq)]

    def test_scc_C2H6O(self):
        """Test eigen values, charges of C2H6O by using SCC DFTB."""
        dataset = {}
        dataset['positions'] = t.tensor(
            [[-1.1924011707, -0.2497887760, -0.0266653895],
             [0.1042943373,  0.5966255069,  0.0842601359],
             [1.2215454578, -0.2753631771, -0.1334060133],
             [-2.1476652622,  0.2638695240, -0.1319026351],
             [-1.4004095793, -0.8033137321,  0.8916190267],
             [-1.1552665234, -0.8966175318, -0.9011813998],
             [0.0644546151,  1.2907464504, -0.7243113518],
             [0.1250893772,  1.1639704704,  1.0000016689],
             [1.1436977386, -0.9601760507,  0.5842682123]])
        dataset['numbers'] = [[6, 6, 8, 1, 1, 1, 1, 1, 1]]
        parameter = self.template_scc()
        result = DFTBCalculator(parameter, dataset)
        refq = t.tensor(
            [4.3045137517383676, 3.8291018519172604, 6.5028095628105218,
             0.9296529355508590, 0.9342688752419560, 0.9036781748891559,
             0.9481281985300407, 0.9659838956731005, 0.6818627536487445])
        refpolts = t.tensor(
            [9.84660891982198, 8.98857943866821, 5.35847589393927,
             2.70480456707182, 2.72124081467918, 2.56503906488386,
             2.77880305439828, 2.88633663053240, 1.76563961688359])
        refpolmbd = t.tensor(
            [10.7704481153938, 10.4206620417722, 5.17107415784681,
             1.87077049923922, 1.75948649387994, 1.62631948159624,
             1.72253860663783, 1.87086245531241, 1.29692091396141])
        [self.assertAlmostEqual(i, j, delta=1E-6)
         for i, j in zip(result.parameter['charge'].squeeze(), refq)]


if __name__ == "__main__":
    """Test precision compare with DFTB+."""
    # set the print precision
    t.set_printoptions(precision=15)

    # set the data precision
    t.set_default_dtype(d=t.float64)

    # run unittest
    unittest.main()
