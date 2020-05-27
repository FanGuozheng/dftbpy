#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch as t
from torch.autograd import Variable


# test back propagation with neural network
class TwoLayerNet(t.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = t.nn.Linear(D_in, H)
        self.linear2 = t.nn.Linear(H, D_out)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


x0 = t.Tensor([[2.3, 1, 1],
               [3.5, 1, 6],
               [3.7, 1, 6],
               [3.9, 1, 6],
               [4.1, 1, 6],
               [4.2, 1, 6],
               [4.4, 1, 6],
               [4.7, 1, 6],
               [4.9, 6, 6],
               [6.3, 6, 6]])

h0 = t.Tensor([[[2.3, 0.],
                [0., 2.4]],
               [[2.5, 0.],
                [0., 3.1]],
               [[5.4, 0.],
                [0., 4.1]],
               [[4.4, 0.],
                [0., 4.3]],
               [[5.7, 0.],
                [0., 5.1]],
               [[5.9, 0.],
                [0., 7.2]],
               [[6.8, 0.],
                [0., 7.1]],
               [[6.4, 0.],
                [0., 6.9]],
               [[7.9, 0.],
               [0., 7.6]],
               [[11.3, 0.],
               [0., 9.9]]])
y0 = t.Tensor([[2.3],
               [3.5],
               [3.7],
               [3.9],
               [4.1],
               [4.2],
               [4.4],
               [4.7],
               [4.9],
               [6.3]])


def test(matrices):
    eigval, YY = t.symeig(matrices, eigenvectors=True, upper=True)
    return eigval[0]


def test2(in_):
    # in_ = t.mm(in_, in_.t())
    eigval, eigvec = t.symeig(in_, eigenvectors=True)
    out_ = t.sin(in_[0]) * t.exp(in_[0]) + t.sin(in_[1]) * t.exp(in_[1]) +\
        t.sin(in_[4]) * t.exp(in_[4])
    return eigval


if __name__ == '__main__':
    '''D_in, H_, D_out, N = 3, 10, 1, 10
    model = TwoLayerNet(D_in, H_, D_out)
    test_ = t.empty(N)
    for ii in range(0, N):
        h = h0[ii, :, :]
        x = x0[ii, :]
        h[0, 0] = model(x)
    A = Variable(h, requires_grad=True)
    b = Variable(t.zeros(10), requires_grad=True)
    ref = y0[ii]
    A2 = t.mm(A, A.t())/2
    A_ch = t.cholesky(A2)
    # optimizer = t.optim.SGD([A, b], lr=1e-1)
    print('-' * 100)
    print('A', A, 'A2', A2, 'A_ch', A_ch)
    optimizer = t.optim.SGD([A, b], lr=1e-3)
    for tt in range(0, 50):
        pred = test(A_ch) + b[ii]
        test_[ii] = pred
        criterion = t.nn.MSELoss(reduction='sum')
        loss = criterion(test_[ii], ref)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        if tt % 10 == 9:
            print(' ' * 20, '-' * 60)
            print('y_pred', pred, 'y_ref', ref)
            print(ii, tt, loss, A, b[ii], '\n', pred, ref)'''
    input_ = t.zeros(10, 10)
    for i in range(0, 10):
        input_[i, i] = 2
    test_multi = Variable(input_ * 5e-5, requires_grad=True)
    ref = test2(input_ * 1e-5)
    optimizer = t.optim.SGD([test_multi], lr=1e-1)
    for tt in range(0, 5):
        pred = test2(test_multi)
        criterion = t.nn.MSELoss(reduction='sum')
        loss = criterion(pred[:8], ref[:8])
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        if tt % 3 == 0:
            print(' ' * 20, '-' * 60)
            # print('y_pred', pred, 'y_ref', ref)
            print(tt, loss, test_multi.grad)
