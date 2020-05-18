#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from torch.autograd import Variable

mat = torch.randn(4, 4, dtype=torch.float64)
print('mat', mat.transpose(-1, -2), '\n', mat.transpose(-1, -2).div_(5).add_(torch.eye(4, dtype=torch.float64)))
mat = (mat @ mat.transpose(-1, -2)).div_(5).add_(torch.eye(4, dtype=torch.float64))
print('mat', mat)
mat = mat.detach().clone().requires_grad_(True)
mat_clone = mat.detach().clone().requires_grad_(True)

# Way 1
inv_mat1 = mat_clone.inverse()  # A^{-1} = A^{-1}

# Way 2
chol_mat = mat.cholesky()
chol_inv_mat = chol_mat.inverse().transpose(-2, -1)
inv_mat2 = chol_inv_mat @ chol_inv_mat.transpose(-2, -1)  # A^{-1} = L^{-T}L^{-1}

# True
print('Are these both A^{-1}?', bool(torch.norm(inv_mat1 - inv_mat2) < 1e-8))

inv_mat1.trace().backward()
inv_mat2.trace().backward()

print('Way 1\n', mat_clone.grad)
print('Way 2\n', mat.grad)  # :-(

corrected_deriv = mat.grad.clone() / 2
corrected_deriv = corrected_deriv.tril() + corrected_deriv.tril().t()
print('Corrected derivative\n', corrected_deriv)  # Simple correction to derivative works.

# True
print('Is the corrected derivative correct?', bool(torch.norm(corrected_deriv - mat_clone.grad) < 1e-8))

'''
mat = torch.tensor(
        [[3.0, 1.0, 0.5, 0.0],
         [1.0, 3.0, -1., 1.0],
         [0.5, -1., 3.0, 2.0],
         [0.0, 2.0, 2.0, 3.0]])

mat2 = tensor(
        [[-0.5049,  0.0000,  0.0000,  0.0000, -0.4079, -0.4079, -0.4079, -0.4079],
         [ 0.0000, -0.1944,  0.0000,  0.0000, -0.1815,  0.1815,  0.1815, -0.1815],
         [ 0.0000,  0.0000, -0.1944,  0.0000, -0.1815,  0.1815, -0.1815,  0.1815],
         [ 0.0000,  0.0000,  0.0000, -0.1944, -0.1815, -0.1815,  0.1815,  0.1815],
         [-0.4079, -0.1815, -0.1815, -0.1815, -0.2386, -0.1481, -0.1481, -0.1481],
         [-0.4079,  0.1815,  0.1815, -0.1815, -0.1481, -0.2386, -0.1481, -0.1481],
         [-0.4079,  0.1815, -0.1815,  0.1815, -0.1481, -0.1481, -0.2386, -0.1481],
         [-0.4079, -0.1815,  0.1815,  0.1815, -0.1481, -0.1481, -0.1481, -0.2386]])
mat_v = Variable(mat2, requires_grad=True)
mat_chol = torch.cholesky(mat_v)
mat_inv = torch.inverse(mat_chol)
'''
