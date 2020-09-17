"""Test DFTB torch code."""
import torch as t
import unittest
import logging
from dftbtorch.dftb_torch import SCF
import dftbtorch.init_parameter as initpara
import dftbtorch.parameters as parameters

para = {}
para['eigenmethod'] = 'lowdin'  # lowdin, cholesky

# fix random seed
t.manual_seed(1000)

# initialize DFTB
initpara.init_dftb_interp(para)

# get the constant parameters for DFTB
parameters.dftb_parameter(para)

para['natom'] = 2
para['atomnameall'] = ['H', 'H']
para['atomind2'] = [2]


class DFTBTorchTest(unittest.TestCase):

    def construct_ab(self, para):
        """construct H, S by random value."""
        A = t.randn(2, 2)
        B = t.randn(2, 2)

        # make sure A, B are symmetric, positive
        para['hammat'] = A @ A.T
        para['overmat'] = B @ B.T

    def construct_ab_batch(self, size=2):
        """construct H, S by random value."""
        A, B = [], []
        for i in range(size):
            iA = t.randn(2, 2)
            iB = t.randn(2, 2)

            # make sure A, B are symmetric, positive
            A.append(iA @ iA.T)
            B.append(iB @ iB.T)

        para['hammat'] = t.stack(A)
        para['overmat'] = t.stack(B)

    def test_nonscc(self):
        """Test non-SCC-DFTB for single system."""
        print('*' * 50)
        print('Test non-SCC-DFTB for single system.')
        print('*' * 50)

        # get input H, S
        self.construct_ab(para)

        # define some input parameter for DFTB
        nat = para['natom']
        para['distance'] = t.ones(nat, nat)
        para['atomind'] = [0, 1, 2]

        # DFTB claculation
        SCF(para).scf_npe_nscc()

    def test_nonscc_batch(self):
        """Test non-SCC-DFTB for multi system."""
        print('*' * 50)
        print('Test non-SCC-DFTB for multi system.')
        print('*' * 50)

        # get input H, S
        nbatch = 5
        nat = para['natom']
        self.construct_ab_batch(size=nbatch)

        # define some input parameter for DFTB
        para['distance'] = t.ones(nat, nat).expand([nbatch, nat, nat])
        para['atomind'] = [0, 1, 2]
        para['atomind'] = [para['atomind']] * nbatch

        # DFTB claculation
        SCF(para).scf_npe_nscc()

    def test_scc(self):
        """Test SCC-DFTB for single system."""
        print('*' * 50)
        print('Test SCC-DFTB for single system.')
        print('*' * 50)

        # get input H, S
        self.construct_ab(para)

        # define some input parameter for DFTB
        nat = para['natom']
        para['distance'] = t.ones(nat, nat)
        para['atomind'] = [0, 1, 2]

        # DFTB claculation
        SCF(para).scf_npe_scc()

    def test_scc_batch(self):
        """Test SCC-DFTB for multi system."""
        print('*' * 50)
        print('Test SCC-DFTB for multi system.')
        print('*' * 50)

        # get input H, S
        nbatch = 5
        nat = para['natom']
        self.construct_ab_batch(size=nbatch)

        # define some input parameter for DFTB
        para['distance'] = t.ones(nat, nat).expand([nbatch, nat, nat])
        para['atomind'] = [0, 1, 2]
        para['atomind'] = [para['atomind']] * nbatch

        # DFTB claculation
        SCF(para).scf_npe_scc()


if __name__ == "__main__":
    # set the data type precision
    t.set_default_dtype(d=t.float64)
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

    # run unittest
    unittest.main()
