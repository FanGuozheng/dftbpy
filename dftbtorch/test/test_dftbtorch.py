"""Test DFTB torch code."""
import numpy as np
import torch as t
import unittest
import logging
from dftbtorch.dftb_torch import SCF, Initialization
import dftbtorch.init_parameter as initpara
import dftbtorch.parameters as parameters
from dftbtorch.dftb_torch import dftb

para = {}
para['eigenmethod'] = 'lowdin'  # lowdin, cholesky

# fix random seed
t.manual_seed(1000)

# initialize DFTB
initpara.init_dftb(para)

# get the constant parameters for DFTB
parameters.dftb_parameter(para)


class DFTBTorchTest(unittest.TestCase):

    def construct_ab(self, para):
        """construct H, S by random value."""
        A = t.randn(para['natom'], para['natom'])
        B = t.randn(para['natom'], para['natom'])

        # make sure A, B are symmetric, positive
        para['hammat'] = (A @ A.T).unsqueeze(0)
        para['overmat'] = (B @ B.T).unsqueeze(0)

    def construct_ab_batch(self, size=5):
        """construct H, S by random value."""
        A, B = [], []
        for i in range(size):
            iA = t.randn(para['natom'], para['natom'])
            iB = t.randn(para['natom'], para['natom'])

            # make sure A, B are symmetric, positive
            A.append(iA @ iA.T)
            B.append(iB @ iB.T)

        para['hammat'] = t.stack(A)
        para['overmat'] = t.stack(B)

    def test_nonscc_from_hs(self):
        """Test DFTB calculation from H, S."""
        print('=' * 50)
        print('Test SCC-DFTB for single system from H, S.')
        para['natom'] = 3
        para['atomnameall'] = [['H'] * para['natom']]
        para['atomind2'] = [para['natom']]

        # get input H, S
        self.construct_ab(para)

        # define some input parameter for DFTB
        para['Lbatch'] = False
        para['nbatch'] = 1
        para['distance'] = t.ones(para['natom'], para['natom']).unsqueeze(0)
        para['atomind'] = [np.linspace(
            0, para['natom'], para['natom'] + 1, dtype=int).tolist()]
        para['natom'] = [para['natom']]

        # DFTB claculation
        Initialization(para, Lreadskf=False)
        SCF(para).scf_npe_nscc()

    def test_nonscc(self):
        """Test non-SCC-DFTB for single system."""
        print('=' * 50)
        print('Test non-SCC-DFTB for single system.')

        # define some input parameter for DFTB
        para['direSK'] = '../../slko/test'
        para['scc'] = 'nonscc'
        para['coor'] = t.tensor([[1, 0., 0., 0.],
                                 [1, 1., 0., 0.],
                                 [1, 0., 1., 0.]])

        # DFTB claculation
        main(para)

    def test_nonscc_batch(self):
        """Test non-SCC-DFTB for multi system."""
        print('=' * 50)
        print('Test non-SCC-DFTB for multi system.')

        # get input H, S
        para['nbatch'] = 5
        para['natom'] = 3
        para['atomnameall'] = [['H'] * para['natom']] * para['nbatch']
        self.construct_ab_batch(size=para['nbatch'])

        # define some input parameter for DFTB
        para['distance'] = t.ones(para['natom'], para['natom']).expand(
            [para['nbatch'], para['natom'], para['natom']])
        para['Lbatch'] = True

        para['atomind'] = [np.linspace(0, para['natom'], para['natom'] + 1,
                                       dtype=int).tolist()] * para['nbatch']
        para['natom'] = [para['natom']] * para['nbatch']

        # DFTB claculation
        SCF(para).scf_npe_nscc()

    def test_scc_from_hs(self):
        """Test DFTB calculation from H, S."""
        print('=' * 50)
        print('Test SCC-DFTB for single system from H, S.')
        para['natom'] = 3
        para['atomnameall'] = [['H'] * para['natom']]
        para['atomind2'] = [para['natom']]

        # get input H, S
        self.construct_ab(para)

        # define some input parameter for DFTB
        para['Lbatch'] = False
        para['nbatch'] = 1
        para['distance'] = t.ones(para['natom'], para['natom']).unsqueeze(0)
        para['atomind'] = [np.linspace(
            0, para['natom'], para['natom'] + 1, dtype=int).tolist()]
        para['natom'] = [para['natom']]

        # DFTB claculation
        Initialization(para, Lreadskf=False)
        SCF(para).scf_npe_scc()

    def test_scc(self):
        """Test SCC-DFTB for single system."""
        print('=' * 50)
        print('Test SCC-DFTB for single system.')

        # define some input parameter for DFTB
        para['direSK'] = '../../slko/test'
        para['scc'] = 'scc'
        para['coor'] = t.tensor([[1, 0., 0., 0.],
                                 [1, 1., 0., 0.],
                                 [1, 0., 1., 0.]])

        # DFTB claculation
        main(para)

    def test_scc_batch(self):
        """Test SCC-DFTB for multi system."""
        print('=' * 50)
        print('Test non-SCC-DFTB for multi system.')

        # get input H, S
        para['nbatch'] = 5
        para['natom'] = 3
        para['atomnameall'] = [['H'] * para['natom']] * para['nbatch']
        self.construct_ab_batch(size=para['nbatch'])

        # define some input parameter for DFTB
        para['distance'] = t.ones(para['natom'], para['natom']).expand(
            [para['nbatch'], para['natom'], para['natom']])
        para['Lbatch'] = True

        para['atomind'] = [np.linspace(0, para['natom'], para['natom'] + 1,
                                       dtype=int).tolist()] * para['nbatch']
        para['natom'] = [para['natom']] * para['nbatch']

        # DFTB claculation
        SCF(para).scf_npe_scc()


if __name__ == "__main__":
    # set the data type precision
    t.set_default_dtype(d=t.float64)
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

    # run unittest
    unittest.main()
