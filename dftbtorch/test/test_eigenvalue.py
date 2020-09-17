"""Test general eigenvalue problem."""
import torch as t
import numpy as np
import unittest
import logging
import scipy.linalg as linalg
from dftbtorch.matht import EigenSolver


para = {}
para['eigenmethod'] = 'lowdin'  # lowdin, cholesky
tol4 = 1E-3
t.manual_seed(1000)


class EigenValueTest(unittest.TestCase):

    def construct_ab(self):
        A = t.randn(3, 3)
        B = t.randn(3, 3)

        # make sure A, B are symmetric, positive
        A = A @ A.T
        B = B @ B.T
        return A, B

    def construct_ab_batch(self, size=2):
        A, B = [], []
        for i in range(size):
            iA = t.randn(3, 3)
            iB = t.randn(3, 3)

            # make sure A, B are symmetric, positive
            A.append(iA @ iA.T)
            B.append(iB @ iB.T)
        return t.stack(A), t.stack(B)

    def linalg_eigh(self, A, B):
        """Get general eigenvalue from scipy."""
        epsilon, vector = linalg.eigh(A, B)
        return epsilon, vector

    def test_cholesky(self):
        """Test general eigenvalue from cholesky decomposition."""
        print('*' * 50,
              '\n Test general eigenvalue with cholesky decomposition. \n',
              '*' * 50)
        A, B = self.construct_ab()
        refval, refeig = self.linalg_eigh(A, B)
        eigval, eigm_ab = EigenSolver(para['eigenmethod']).eigen(A, B)

        # compare each element of eigenvalue from scipy and cholesky
        [self.assertAlmostEqual(i, j, delta=tol4) for i, j in zip(eigval, refval)]

    def test_cholesky_batch(self):
        """Test general eigenvalue from cholesky decomposition."""
        print('*' * 50,
              '\n Test general eigenvalue with cholesky decomposition for',
              'multi matrices. \n', '*' * 50)
        A, B = self.construct_ab_batch(size=5)
        # eigval, eigm_ab = cholesky(A, B)
        eigval, eigm_ab = EigenSolver(para['eigenmethod']).eigen(A, B)
        for i in range(5):
            refval, refeig = self.linalg_eigh(A[i].numpy(), B[i].numpy())
            # eigval, eigm_ab = EigenSolver(para).eigen(A[i], B[i])

            # compare each element of eigenvalue from scipy and cholesky
            [self.assertAlmostEqual(i, j, delta=tol4) for i, j in zip(eigval[i], refval)]

    def test_cholesky_grad(self):
        """Test general eigenvalue from cholesky decomposition with gradient."""
        A, B = self.construct_ab()
        refval, refeig = self.linalg_eigh(A, B)
        eigval, eigm_ab = EigenSolver(para['eigenmethod']).eigen(A, B)

        # compare each element of eigenvalue from scipy and cholesky
        [self.assertAlmostEqual(i, j, delta=tol4) for i, j in zip(eigval, refval)]


if __name__ == "__main__":
    # set the data type precision
    t.set_default_dtype(d=t.float64)
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

    # run unittest
    unittest.main()
