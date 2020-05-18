#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch as t
import tensorflow as tf


def test_mat():
    ref = t.tensor([[1.000000000000, 0.767505228519],
                    [0.767505228519, 1.000000000000]])
    test_arr = t.tensor(
            [[1.000000000000, 0.822218239307],
             [0.822218239307, 1.000000000000]]).requires_grad_(True)
    optimizer = t.optim.SGD([test_arr], lr=1)
    criterion = t.nn.MSELoss(reduction='sum')

    print('-' * 40, 'test 2 * 2 tensor', '-' * 40)
    print('ref \n', ref)
    print('original data \n', test_arr)
    for ii in range(0, 1):
        loss = criterion(test_arr, ref)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        print('optmized data \n', test_arr)
        print('test_arr.grad \n', test_arr.grad)
        optimizer.step()


def test_inv():
    ref = t.tensor([[1.000000000000, 0.767505228519],
                    [0.767505228519, 1.000000000000]])
    test_arr = t.tensor(
            [[1.000000000000, 0.822218239307],
             [0.822218239307, 1.000000000000]]).requires_grad_(True)
    optimizer = t.optim.SGD([test_arr], lr=5e-1)
    criterion = t.nn.MSELoss(reduction='sum')
    # test_chol, ref_chol = t.cholesky(test_arr), t.cholesky(ref)
    test_inv, ref_inv = test_arr.inverse(), ref.inverse()
    print('-' * 80, '\n test inverse')
    print('ref_inv \n', ref_inv)
    print('original inv data \n', test_inv)
    for ii in range(0, 3):
        loss = criterion(test_inv, ref_inv)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        print('optmized data \n', test_arr)
        print('optmized chol data \n', test_inv)
        print('test_arr.grad \n', test_arr.grad)
        optimizer.step()


def test_symeig():
    ref = t.tensor([[1.000000000000, 0.767505228519, 0.767505228519],
                    [0.767505228519, 1.000000000000, 0.467505228519],
                    [0.767505228519, 0.467505228519, 1.000000000000]])
    init = t.Tensor([0.822218239307, 0.822218239307, 0.567505228519]).requires_grad_(True)
    test_arr = t.zeros(3, 3)
    test_arr[0, 1], test_arr[0, 2] = init[0], init[0]
    test_arr[1, 0], test_arr[2, 0] = init[1], init[1]
    test_arr[1, 2], test_arr[2, 1] = init[2], init[2]
    test_arr = test_arr + t.eye(3, 3)
    optimizer = t.optim.SGD([init], lr=5)
    criterion = t.nn.MSELoss(reduction='sum')
    test_val, test_vec = t.symeig(test_arr, eigenvectors=True)
    ref_val, ref_vec = t.symeig(ref, eigenvectors=True)
    print('-' * 80, '\n test symeig')
    # print('ref_chol \n', ref_chol)
    # print('original chol data \n', test_chol)
    loss = criterion(test_val, ref_val)
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    # print('optmized chol data \n', test_arr)
    print('init.grad: ', init.grad)
    optimizer.step()


def test_chol():
    ref = t.tensor([[1.000000000000, 0.767505228519],
                    [0.767505228519, 1.000000000000]])
    test_arr = t.tensor(
            [[1.000000000000, 0.822218239307],
             [0.822218239307, 1.000000000000]]).requires_grad_(True)
    optimizer = t.optim.SGD([test_arr], lr=5e-1)
    criterion = t.nn.MSELoss(reduction='sum')
    # test_chol, ref_chol = t.cholesky(test_arr), t.cholesky(ref)
    test_chol, ref_chol = test_arr.cholesky(), ref.cholesky()
    print('-' * 80, '\n test cholesky')
    print('ref_chol \n', ref_chol)
    print('original chol data \n', test_chol)
    for ii in range(0, 3):
        loss = criterion(test_chol, ref_chol)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        print('optmized data \n', test_arr)
        print('optmized chol data \n', test_chol)
        print('test_arr.grad \n', test_arr.grad)
        optimizer.step()


def test_chol_tf():
    ref = tf.constant(
            [[1.000000000000, 0.767505228519],
             [0.767505228519, 1.000000000000]])
    test_arr = tf.Variable(tf.constant(
            [[1.000000000000, 0.822218239307],
             [0.822218239307, 1.000000000000]]))
    test_chol, ref_chol = tf.cholesky(test_arr), tf.cholesky(ref)

    loss = tf.reduce_sum(tf.square(ref_chol - test_chol))
    train = tf.train.AdadeltaOptimizer(100).minimize(loss)
    print('-' * 80, '\n test cholesky tensorflow')
    for i in range(0, 3):
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            sess.run(train)
            print('opt data \n', sess.run(test_arr))
            print('opt arr \n', sess.run(tf.cholesky(test_arr)))
            print('ref chol  \n', sess.run(tf.cholesky(ref)))
            print('opt gradient \n',
                  sess.run(tf.gradients(loss, test_arr)))
            print('opt gradient \n',
                  sess.run(tf.gradients(test_chol, test_arr)))


def test_chol2():
    ref = t.tensor([[1.000000000000, 0.767505228519],
                    [0.767505228519, 1.000000000000]])
    test_off = t.tensor(
            [[0.000000000000, 0.000000000000],
             [0.822218239307, 0.000000000000]]).requires_grad_(True)
    diag = t.tensor([1.00000000000000, 1.00000000000000], requires_grad=True)
    test_arr = test_off + test_off.t() + diag.diag()
    optimizer = t.optim.SGD([test_off, diag], lr=5)
    criterion = t.nn.MSELoss(reduction='sum')
    # test_chol, ref_chol = t.cholesky(test_arr), t.cholesky(ref)
    test_chol, ref_chol = test_arr.cholesky(), ref.cholesky()
    print('-' * 80, '\n test cholesky2')
    print('ref_chol \n', ref_chol)
    print('original chol data \n', test_chol)
    for ii in range(0, 3):
        loss = criterion(test_chol, ref_chol)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        print('optmized data \n', test_off, diag)
        print('optmized chol data \n', test_chol)
        print('test_off.grad, diag.grid \n', test_off.grad, diag.grad)
        optimizer.step()


def test_chol3():
    ref = t.tensor([[1.000000000000, 0.767505228519, 0.767505228519],
                    [0.767505228519, 1.000000000000, 0.467505228519],
                    [0.767505228519, 0.467505228519, 1.000000000000]])
    init = t.Tensor([0.822218239307, 0.822218239307, 0.567505228519]).requires_grad_(True)
    test_arr = t.zeros(3, 3)
    test_arr[0, 1], test_arr[0, 2] = init[0], init[0]
    test_arr[1, 0], test_arr[2, 0] = init[1], init[1]
    test_arr[1, 2], test_arr[2, 1] = init[2], init[2]
    test_arr = test_arr + t.eye(3, 3)
    optimizer = t.optim.SGD([init], lr=5)
    criterion = t.nn.MSELoss(reduction='sum')
    test_chol, ref_chol = t.cholesky(test_arr), t.cholesky(ref)
    test_inv, ref_inv = test_chol.inverse(), ref_chol.inverse()
    print('-' * 80, '\n test cholesky2')
    print('ref_chol \n', ref_chol)
    print('original chol data \n', test_chol)
    for ii in range(0, 3):
        loss = criterion(test_chol, ref_chol)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        print('optmized chol data \n', test_arr)
        print('init.grad, diag.grid \n', init.grad)
        optimizer.step()


def test_chol4():
    ref = t.tensor([[1.000000000000, 0.767505228519, 0.767505228519],
                    [0.767505228519, 1.000000000000, 0.467505228519],
                    [0.767505228519, 0.467505228519, 1.000000000000]])
    init1 = t.Tensor([0.822218239307, 0.822218239307, 0.567505228519]).requires_grad_(True)
    init2 = t.Tensor([0.822218239307, 0.822218239307, 0.567505228519]).requires_grad_(True)
    test_arr1, test_arr2 = t.eye(3), t.eye(3)

    # way 1 to build test tensor
    test_arr1[0, 1] = test_arr1[1, 0] = init1[0]
    test_arr1[0, 2] = test_arr1[2, 0] = init1[1]
    test_arr1[1, 2] = test_arr1[2, 1] = init1[2]

    # way 2 to build test tensor
    test_arr2[0, 1:], test_arr2[1:, 0] = init2[0], init2[1]
    test_arr2[1, 2], test_arr2[2, 1] = init2[2], init2[2]
    test_arr2 = test_arr2 + t.eye(3, 3)

    criterion = t.nn.MSELoss(reduction='sum')
    print('*' * 40, 'cholesky 4', '*' * 40)
    print('-' * 80, '\n way 1 test cholesky: gradient')
    test_chol1, ref_chol = t.cholesky(test_arr1), t.cholesky(ref)
    loss = criterion(test_chol1, ref_chol)
    loss.backward(retain_graph=True)
    print(test_arr1, '\n', init1.grad)
    if init1.grad[0] == init1.grad[1]:
        print('same gradient')

    print('-' * 80, '\n way 2 test cholesky: gradient')
    test_chol2 = t.cholesky(test_arr2)
    loss = criterion(test_chol2, ref_chol)
    loss.backward(retain_graph=True)
    print(init2.grad)
    if init2.grad[0] == init2.grad[1]:
        print('same gradient')


def test_chol5():
    ref = t.tensor([[1.000000000000, 0.767505228519, 0.767505228519],
                    [0.767505228519, 1.000000000000, 0.467505228519],
                    [0.767505228519, 0.467505228519, 1.000000000000]])
    init1 = t.Tensor([0.822218239307, 0.822218239307, 0.567505228519]).requires_grad_(True)
    init2 = t.Tensor([0.822218239307, 0.822218239307, 0.567505228519]).requires_grad_(True)
    test_arr1, test_arr2 = t.zeros(3, 3), t.zeros(3, 3)

    # way 1 to build test tensor
    test_arr1[0, 1], test_arr1[1, 0] = init1[0], init1[0]
    test_arr1[2, 0], test_arr1[0, 2] = init1[1], init1[1]
    test_arr1[1, 2], test_arr1[2, 1] = init1[2], init1[2]
    test_arr1 = test_arr1 + t.eye(3, 3)

    # way 2 to build test tensor
    test_arr2[0, 1:], test_arr2[1:, 0] = init2[0], init2[1]
    test_arr2[1, 2], test_arr2[2, 1] = init2[2], init2[2]
    test_arr2 = test_arr2 + t.eye(3, 3)

    criterion = t.nn.MSELoss(reduction='sum')
    print('*' * 40, 'cholesky 5', '*' * 40)
    print('-' * 80, '\n way 1 test cholesky: gradient')
    test_chol1, ref_chol = matrix_cholesky(test_arr1), matrix_cholesky(ref)
    loss = criterion(test_chol1, ref_chol)
    loss.backward(retain_graph=True)
    print(test_arr1, '\n', init1.grad)
    if init1.grad[0] == init1.grad[1]:
        print('same gradient')

    print('-' * 80, '\n way 2 test cholesky: gradient')
    test_chol2 = matrix_cholesky(test_arr2)
    loss = criterion(test_chol2, ref_chol)
    loss.backward(retain_graph=True)
    print(init2.grad)
    if init2.grad[0] == init2.grad[1]:
        print('same gradient')


def test_chol6():
    ref = t.tensor([[1.000000000000, 0.767505228519, 0.767505228519],
                    [0.767505228519, 1.000000000000, 0.467505228519],
                    [0.767505228519, 0.467505228519, 1.000000000000]])
    init1 = t.Tensor([0.822218239307, 0.822218239307, 0.567505228519]).requires_grad_(True)
    init2 = t.Tensor([0.822218239307, 0.822218239307, 0.567505228519]).requires_grad_(True)
    test_arr1, test_arr2 = t.zeros(3, 3), t.zeros(3, 3)

    # way 1 to build test tensor
    test_arr1[0, 1] = test_arr1[1, 0] = init1[0] + init1[1]
    test_arr1[2, 0] = test_arr1[0, 2] = init1[1] + init1[0]
    test_arr1[1, 2] = test_arr1[2, 1] = init1[2]
    test_arr1 = test_arr1 + t.eye(3)

    # way 2 to build test tensor
    test_arr2[0, 1:], test_arr2[1:, 0] = init2[0], init2[1]
    test_arr2[1, 2], test_arr2[2, 1] = init2[2], init2[2]
    test_arr2 = test_arr2 + t.eye(3)

    criterion = t.nn.MSELoss(reduction='sum')
    print('*' * 40, 'cholesky 6', '*' * 40)
    print('-' * 80, '\n way 1 test cholesky: gradient')
    u1, sigma1, v1 = t.svd(test_arr1)
    ur, sigmar, vr = t.svd(ref)
    loss = criterion(u1, ur)
    loss.backward(retain_graph=True)
    print(test_arr1, '\n', init1.grad)
    if init1.grad[0] == init1.grad[1]:
        print('same gradient')

    print('-' * 80, '\n way 2 test cholesky: gradient')
    u2, sigma2, v2 =  t.svd(test_arr2)
    loss = criterion(u2, ur)
    loss.backward(retain_graph=True)
    print(init2.grad)
    if init2.grad[0] == init2.grad[1]:
        print('same gradient')


def chol2(A):
    L = t.zeros(3, 3)
    L[0, 0] = t.sqrt(A[0, 0])
    L[1, 0] = A[1, 0] / t.sqrt(A[0, 0])
    L[1, 1] = t.sqrt(A[1, 1] - (A[1, 0] / t.sqrt(A[0, 0])) ** 2)
    return L


def chol3(A):
    L = t.zeros(3, 3)
    L[0, 0] = t.sqrt(A[0, 0])
    L[1, 0] = A[1, 0] / t.sqrt(A[0, 0])
    L[1, 1] = t.sqrt(A[1, 1] - (A[1, 0] / t.sqrt(A[0, 0])) ** 2)
    # L[1, 1] = t.sqrt(A[1, 1] - L[1, 0] ** 2)
    L[2, 0] = A[2, 0] / t.sqrt(A[0, 0])
    L[2, 1] = (A[2, 1] - (A[2, 0] / t.sqrt(A[0, 0])) *
        (A[1, 0] / t.sqrt(A[0, 0]))) / (t.sqrt(A[1, 1] - (A[1, 0] / t.sqrt(A[0, 0])) ** 2))
    # L[2, 1] = (A[2, 1] - L[2, 0] * L[1, 0]) / L[1, 1]
    L[2, 2] = t.sqrt(A[2, 2] - (A[2, 0] / t.sqrt(A[0, 0])) ** 2 - ((A[2, 1] - (A[2, 0] / t.sqrt(A[0, 0])) *
        (A[1, 0] / t.sqrt(A[0, 0]))) / (t.sqrt(A[1, 1] - (A[1, 0] / t.sqrt(A[0, 0])) ** 2))) ** 2)
    # L[2, 2] = t.sqrt(A[2, 2] - L[2, 0] ** 2 - L[2, 1] ** 2)
    return L


def matrix_cholesky(A):
    L = t.zeros_like(A)

    for i in range(A.shape[-1]):
        for j in range(i+1):
            s = 0.0
            for k in range(j):
                s = s + L[i, k].clone() * L[j, k].clone()

            L[i, j] = t.sqrt(A[i, i] - s) if (i == j) else \
                      (1.0 / L[j, j].clone() * (A[i, j] - s))
    return L


def test_eig():
    ref = t.tensor([[1.000000000000, 0.767505228519],
                    [0.767505228519, 1.000000000000]])
    test_arr = t.tensor(
            [[1.000000000000, 0.822218239307],
             [0.822218239307, 1.000000000000]]).requires_grad_(True)
    optimizer = t.optim.SGD([test_arr], lr=5e-1)
    criterion = t.nn.MSELoss(reduction='sum')
    eigval_ref, eigvec_ref = t.symeig(ref, eigenvectors=True)
    eigval_test, eigvec_test = t.symeig(test_arr, eigenvectors=True)
    print('-' * 80, '\n test eigen solver')
    print('ref_eigval \n', eigval_ref)
    print('original chol data \n', eigval_test)
    for ii in range(0, 3):
        loss = criterion(eigval_test, eigval_ref)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        print('optmized data \n', test_arr)
        print('optmized eigval data \n', eigval_test)
        print('test_arr.grad \n', test_arr.grad)
        optimizer.step()


def _cholesky(matrixa, matrixb):
    '''
    cholesky decomposition of B: B = LL^{T}
        AX = (lambda)BX ==> (L^{-1}AL^{-T})(L^{T}X) = (lambda)(L^{T}X)
    matrix_a: Fock operator
    matrix_b: overm'''

    chol_l = t.cholesky(matrixb)
    # self.para['eigval'] = t.inverse(matrixb)
    linv_a = t.mm(t.inverse(chol_l), matrixa)
    l_invtran = t.inverse(chol_l.t())
    linv_a_linvtran = t.mm(linv_a, l_invtran)
    eigval, eigm = t.symeig(linv_a_linvtran, eigenvectors=True)
    eigm_ab = t.mm(l_invtran, eigm)


def lowdin(matrixa, matrixb):
    lam_b, l_b = t.symeig(matrixb, eigenvectors=True)
    # lam_sqrt_inv = t.sqrt(1 / lam_b)
    l_b = t.mm(l_b, l_b.t())
    lam_sqrt_inv = t.inverse(lam_b.diag())
    S_sym = t.mm(l_b, t.mm(lam_sqrt_inv, l_b.t()))
    SHS = t.mm(S_sym, t.mm(matrixa, S_sym))
    eigval, eigvec_ = t.symeig(SHS, eigenvectors=True)
    eigvec = t.mm(S_sym, eigvec_)
    print('return: ', S_sym)
    return eigval


def lowdin_svd(matrixa, matrixb):
    '''
    SVD decomposition of B: B = USV_{T}
        S_{-1/2} = US_{-1/2}V_{T}
        AX = (lambda)BX ==>
        (S_{-1/2}AS_{-1/2})(S_{1/2}X) = (lambda)(S_{1/2}X)
    matrix_a: Fock operator
    matrix_b: overlap matrix
    '''
    ub, sb, vb = t.svd(matrixb)
    sb_sqrt_inv = t.sqrt(1 / sb)
    S_sym = t.mm(ub, t.mm(sb_sqrt_inv.diag(), vb.t()))
    SHS = t.mm(S_sym, t.mm(matrixa, S_sym))
    eigval, eigvec_ = t.symeig(SHS, eigenvectors=True)
    eigvec = t.mm(S_sym, eigvec_)
    print('eigval, eigvec', eigval, eigvec)
    # eigval3, eigvec_ = t.symeig(lam_b_2d, eigenvectors=True)
    return eigval


def symmetric_tensor(values):
    s = t.eye(3)
    s[0, 1] = s[1, 0] = values[0]
    s[0, 2] = s[2, 0] = values[1]
    s[2, 1] = s[1, 2] = values[2]
    return s


def generalized_eigvals(A, B):
    L = t.cholesky(B)
    A1, LU_A = t.solve(A, L)
    # print('A1', A1, '\n', L.matmul(A1) - A)
    A2, LU_A1 = t.solve(A1.t(), L)
    A3 = A2.t()
    eigvals, _ = t.symeig(A3, eigenvectors=True)
    return eigvals


def test():
    print('*' * 40, 'test', '*' * 40)
    lossfunc = t.nn.MSELoss(reduction='mean')
    delta = 0.02

    A = symmetric_tensor([0.75, 0.75, 0.75])
    params0 = t.tensor([0.2, 0.2, 0.2])
    B0 = symmetric_tensor(params0)
    # print('A', A, 'B0', B0)
    eigvals0 = lowdin_svd(A, B0)
    print("REF eigvals: \n", eigvals0)

    '''for ii in range(3):
        params = t.tensor([0.1, 0.1, 0.1])
        params[ii] += delta
        B = symmetric_tensor(params)
        eigvals = generalized_eigvals(A, B)
        print("EIGVALS: ", eigvals)
        print("LOSSFUNC: ", lossfunc(eigvals, eigvals0))'''

    params = t.tensor([0.1, 0.1, 0.1]).requires_grad_(True)
    B = symmetric_tensor(params)
    eigvals = lowdin_svd(A, B)
    print("EIGVALS: \n", eigvals)
    lossval = lossfunc(eigvals, eigvals0)
    # lossval = lossfunc(t.cholesky(B), t.cholesky(B0))
    print("LOSSVAL: ", lossval)
    lossval.backward(retain_graph=True)
    print("LOSSVAL GRAD:", params.grad)


def test_chol_symeig():
    print('*' * 40, 'test chol and symeig', '*' * 40)
    mat = t.randn(4, 4, dtype=t.float64)
    mat = (mat @ mat.transpose(-1, -2)).div_(2).add_(t.eye(4, dtype=t.float64))
    mat = mat.detach().clone().requires_grad_(True)
    mat_clone = mat.detach().clone().requires_grad_(True)
    # Way 1
    chol_mat = mat.cholesky()
    logdet1 = 2 * chol_mat.diagonal().log().sum()

    # Way 2
    w, _ = mat_clone.symeig(eigenvectors=True)
    logdet2 = w.log().sum()
    print('Are these both log(det(A))?', bool(logdet1 - logdet2 < 1e-8))  # T
    logdet1.backward()
    logdet2.backward()
    inv_mat = mat.inverse()  # the analytical solution
    print('chol_mat.diagonal()', chol_mat.diagonal(), '\n', w)
    print(mat.inverse(), '\n', mat.grad)
    print('Does Way 1 yield A^{-1}?', bool(t.norm(mat.grad - inv_mat) < 1e-8))
    print('Does Way 2 yield A^{-1}?',
          bool(t.norm(mat_clone.grad - inv_mat) < 1e-8))


if __name__ == '__main__':
    '''
    '''
    t.set_printoptions(precision=12)
    t.autograd.set_detect_anomaly(True)
    # test_mat()
    # test_inv()
    # test_chol3()
    # test_chol4()
    # test_chol5()
    # test_chol6()
    test()
    # test_chol_symeig()
    # test_symeig()
    # test_chol_tf()
    # test_eig()
