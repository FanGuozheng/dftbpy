#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import numpy as np
import scipy.linalg as linalg


a = np.array([[1, 0.2], [0., 1.]])
b = np.array([[1, 0.5], [0.5, 1]])
# b = np.dot(a, a.transpose())
eigval, eigvec = linalg.eigh(a, b, lower=False, overwrite_a=True,
                             overwrite_b=True)
print('eigval_linalg', eigval)
print('eigvec_linalg', eigvec, '\n\n')
'''binv = torch.inverse(torch.from_numpy(b))
w = torch.from_numpy(np.random.rand(2, 1))
print('w', w)
aw = torch.mm(binv, w)
print(torch.mm(aw, bt))
ab = torch.mm(binv, at)'''
at = torch.from_numpy(a)
bt = torch.from_numpy(b)
l = torch.cholesky(bt)
print('torch.mm', torch.mm(l, l.t()))
print('bt', bt, 'at', at, 'l', l, '')
C1 = torch.mm(torch.inverse(l), at)
ltinv = torch.inverse(l.t())
C2 = torch.mm(C1, ltinv)
eigval, eigvec = torch.symeig(C2, eigenvectors=True, upper=True)
X = torch.mm(torch.inverse(l.t()), eigvec)
print('eigval2', eigval)
print('eigvec2', X, '\n\n')
binv = torch.inverse(bt)
print('inverse * matrix', torch.mm(binv, bt))
print('bt', bt)
print('binv', binv)
ab = torch.mm(binv, at)
eigval, eigvec = torch.symeig(ab, eigenvectors=True, upper=True)
print('eigval3', eigval)
print('eigvec3', eigvec, '\n')


'''oldovermat.inverse()
tensor([[ 1.0000,  0.0000,  0.0000,  0.0000, -0.4291, -0.3806, -0.3376, -0.2995],
        [ 0.0000,  1.0000,  0.0000,  0.0000,  0.2582, -0.2874, -0.2549,  0.2903],
        [ 0.0000,  0.0000,  1.0000,  0.0000,  0.2582, -0.2874,  0.2615, -0.2845],
        [ 0.0000,  0.0000,  0.0000,  1.0000,  0.2582,  0.2291, -0.3133, -0.2779],
        [ 0.0000,  0.0000,  0.0000,  0.0000,  1.0000, -0.1129, -0.1002, -0.0889],
        [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  1.0000, -0.1129, -0.1002],
        [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  1.0000, -0.1129],
        [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  1.0000]])
torch.cholesky_inverse(oldovermat)
tensor([[1., 0., 0., 0., 0., 0., 0., -0.],
        [0., 1., 0., 0., 0., 0., 0., -0.],
        [0., 0., 1., 0., 0., 0., 0., -0.],
        [0., 0., 0., 1., 0., 0., 0., -0.],
        [0., 0., 0., 0., 1., 0., 0., -0.],
        [0., 0., 0., 0., 0., 1., 0., -0.],
        [0., 0., 0., 0., 0., 0., 1., -0.],
        [-0., -0., -0., -0., -0., -0., -0., 1.]])
'''