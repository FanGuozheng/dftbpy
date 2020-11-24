import os
import h5py
import torch as t
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from dftbtorch.interpolator import PolySpline, BicubInterpVec
from dftbtorch.geninterpskf import SkInterpolator
from dftbtorch.sk import SKTran, GetSKTable, GetSK_
from dftbtorch.matht import BicubInterp
ATOMNUM = {'H': 1, 'C': 6, 'N': 7, 'O': 8}
ATOMNAME = {1: 'H', 6: 'C', 7: 'N', 8: 'O'}


def test_bicub_interp():
    """Test Bicubic interpolation."""
    bicinterp = BicubInterp()
    xmesh = t.Tensor([1.5, 1.6, 1.9, 2.4, 3.0, 3.7, 4.5])
    ymesh = t.Tensor([1.5, 1.6, 1.9, 2.4, 3.0, 3.7, 4.5])
    xmesh_ = t.Tensor([1.5, 2., 2.5, 3.0, 3.5, 4.0, 4.5])
    ymesh_ = t.Tensor([1.5, 2., 2.5, 3.0, 3.5, 4.0, 4.5])
    zmesh = t.Tensor([[.4, .45, .51, .57, .64, .72, .73],
                      [.45, .51, .58, .64, .73, .83, .85],
                      [.51, .58, .64, .73, .83, .94, .96],
                      [.57, .64, .73, .84, .97, 1.12, 1.14],
                      [.64, .72, .83, .97, 1.16, 1.38, 1.41],
                      [.72, .83, .94, 1.12, 1.38, 1.68, 1.71],
                      [.73, .85, .96, 1.14, 1.41, 1.71, 1.74]])
    xuniform = t.linspace(1.5, 4.49, 31)
    yuniform = t.linspace(1.5, 4.49, 31)
    x, y = np.meshgrid(xmesh, ymesh)
    xnew, ynew = np.meshgrid(xuniform, yuniform)
    x_, y_ = np.meshgrid(xmesh_, ymesh_)
    nx = len(xuniform)
    znew = t.empty(nx, nx)
    znew_ = t.empty(nx, nx)
    for ix in range(0, nx):
        for jy in range(0, nx):
            znew[ix, jy] = bicinterp.bicubic_2d(xmesh, ymesh, zmesh,
                                           xuniform[ix], yuniform[jy])
            znew_[ix, jy] = bicinterp.bicubic_2d(xmesh_, ymesh_, zmesh,
                                            xuniform[ix], yuniform[jy])
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.pcolormesh(x, y, zmesh)
    plt.title('original data', fontsize=15)
    plt.subplot(1, 2, 2)
    plt.pcolormesh(xnew, ynew, znew)
    plt.title('grid difference not same', fontsize=15)
    plt.show()
    plt.subplot(1, 2, 1)
    plt.pcolormesh(x_, y_, zmesh)
    plt.title('original data', fontsize=15)
    plt.subplot(1, 2, 2)
    plt.pcolormesh(xnew, ynew, znew_)
    plt.title('grid difference same', fontsize=15)
    plt.show()

def test_bicub_interp_ml():
    """Test gradients of bicubic method."""
    bicinterp = BicubInterp()
    xmesh = t.Tensor([1.9, 2., 2.3, 2.7, 3.3, 4.0, 4.1])
    ymesh = t.Tensor([1.9, 2., 2.3, 2.7, 3.3, 4.0, 4.1])
    # xin = t.Tensor([2, 3]).clone().detach().requires_grad_()
    t.enable_grad()
    xin = Variable(t.Tensor([4.05, 3]), requires_grad=True)
    zmesh = t.Tensor([[.4, .45, .51, .57, .64, .72, .73],
                      [.45, .51, .58, .64, .73, .83, .85],
                      [.51, .58, .64, .73, .83, .94, .96],
                      [.57, .64, .73, .84, .97, 1.12, 1.14],
                      [.64, .72, .83, .97, 1.16, 1.38, 1.41],
                      [.72, .83, .94, 1.12, 1.38, 1.68, 1.71],
                      [.73, .85, .96, 1.14, 1.41, 1.71, 1.74]])
    yref = t.Tensor([1.1])
    for istep in range(0, 10):
        optimizer = t.optim.SGD([xin, xmesh, ymesh], lr=1e-1)
        ypred = bicinterp.bicubic_2d(xmesh, ymesh, zmesh, xin[0], xin[1])
        criterion = t.nn.MSELoss(reduction='sum')
        loss = criterion(ypred, yref)
        optimizer.zero_grad()
        print('ypred',  ypred, 'xin', xin)
        loss.backward(retain_graph=True)
        optimizer.step()
        print('loss', loss)


def read_skf_hdf5(para, skf, dataset, ml):
    t.set_default_dtype(t.float64)
    para['datasetSK'] = '../slko/hdf/skf.hdf5'
    skf['ReadSKType'] = 'compressionRadii'
    dataset['LSKFinterpolation'] = True

    # ninterp = skf['sizeInterpolationPoints']
    skf['hs_compr_all'] = []
    # index of row, column of distance matrix, no digonal
    # ind = t.triu_indices(distance.shape[0], distance.shape[0], 1)
    # dist_1d = distance[ind[0], ind[1]]
    # get the skf with hdf type
    hdfsk = para['datasetSK']
    if not os.path.isfile(hdfsk):
        raise FileExistsError('dataset %s do not exist' % hdfsk)
    # read all skf according to atom number (species) and indices and add
    # these skf to a list, attention: hdf only store numpy type data
    # with h5py.File(hdfsk, 'r') as f:
    #     # get the grid sidtance, which should be the same
    #     grid_dist = f['globalgroup'].attrs['grid_dist']

    # get integrals with ninterp (normally 8) line for interpolation
    with h5py.File(hdfsk, 'r') as f:
        yyhh = f['HH' + '/hs_all_rall'][:, :, 10, :]
        yyhc = f['HC' + '/hs_all_rall'][:, :, 10, :]
        yych = f['CH' + '/hs_all_rall'][:, :, 10, :]
        yycc = f['CC' + '/hs_all_rall'][:, :, 10, :]
    # get the distances corresponding to the integrals
    # xx = [[(t.arange(ninterp) + indd[i, j] - ninterp) * grid_dist
    #        for j in range(len(distance))] for i in range(len(distance))]
    # skf['hs_compr_all'] = t.stack([t.stack([math.poly_check(
    #     xx[i][j], t.from_numpy(yy[i][j]).type(para['precision']), distance[i, j], i==j)
    #     for j in range(natom)]) for i in range(natom)])
    bicubic = BicubInterpVec(para, ml)
    zmesh = t.from_numpy(np.stack([np.stack([yyhh, yyhc]), np.stack([yych, yycc])]))
    mesh = t.tensor([[1., 1.5, 2., 2.5, 3., 3.5, 4., 5., 6., 8., 10.],
                     [1., 1.5, 2., 2.5, 3., 3.5, 4., 5., 6., 8., 10.]])
    for ii, compr in enumerate([t.tensor([2.2, 2.2]), t.tensor([2.5, 2.5])]):
        hs_ij = bicubic.bicubic_2d(mesh, zmesh, compr, compr)
        H_H_22_10 = t.tensor([0.000000000000E+00, 0.000000000000E+00, 0.000000000000E+00, 0.000000000000E+00,
                              0.000000000000E+00, 0.000000000000E+00, 0.000000000000E+00, 0.000000000000E+00,
                              0.000000000000E+00, -1.719871352161E-01, 0.000000000000E+00, 0.000000000000E+00,
                              0.000000000000E+00, 0.000000000000E+00, 0.000000000000E+00, 0.000000000000E+00,
                              0.000000000000E+00, 0.000000000000E+00, 0.000000000000E+00, 2.739327486239E-01])
        C_H_22_10 = t.tensor([0.000000000000E+00, 0.000000000000E+00, 0.000000000000E+00, 0.000000000000E+00,
                              0.000000000000E+00, 0.000000000000E+00, 0.000000000000E+00, 0.000000000000E+00,
                              0.000000000000E+00, -2.145555344059E-01, 0.000000000000E+00, 0.000000000000E+00,
                              0.000000000000E+00, 0.000000000000E+00, 0.000000000000E+00, 0.000000000000E+00,
                              0.000000000000E+00, 0.000000000000E+00, 0.000000000000E+00, 3.090342767881E-01])
        C_C_22_10 = t.tensor([0.000000000000E+00, 0.000000000000E+00, 0.000000000000E+00, 0.000000000000E+00,
                              0.000000000000E+00, 1.553059957276E-01, -1.254378770489E-01, 0.000000000000E+00,
                              2.141868282553E-01, -2.520364268543E-01, 0.000000000000E+00, 0.000000000000E+00,
                              0.000000000000E+00, 0.000000000000E+00, 0.000000000000E+00, -3.911118200884E-01,
                              1.981899126309E-01, 0.000000000000E+00, -3.810361403951E-01, 3.410149885712E-01])
        H_H_25_10 = t.tensor([0.000000000000E+00, 0.000000000000E+00, 0.000000000000E+00, 0.000000000000E+00,
                              0.000000000000E+00, 0.000000000000E+00, 0.000000000000E+00, 0.000000000000E+00,
                              0.000000000000E+00, -1.834360638907E-01, 0.000000000000E+00, 0.000000000000E+00,
                              0.000000000000E+00, 0.000000000000E+00, 0.000000000000E+00, 0.000000000000E+00,
                              0.000000000000E+00, 0.000000000000E+00, 0.000000000000E+00, 3.182141942151E-01])
        H_C_25_10 = t.tensor([0.000000000000E+00, 0.000000000000E+00, 0.000000000000E+00, 0.000000000000E+00,
                              0.000000000000E+00, 0.000000000000E+00, 0.000000000000E+00, 0.000000000000E+00,
                              2.071400043802E-01, -2.368709160456E-01, 0.000000000000E+00, 0.000000000000E+00,
                              0.000000000000E+00, 0.000000000000E+00, 0.000000000000E+00, 0.000000000000E+00,
                              0.000000000000E+00, 0.000000000000E+00, -4.002627595767E-01, 3.494647060923E-01])
        C_C_25_10 = t.tensor([0.000000000000E+00, 0.000000000000E+00, 0.000000000000E+00, 0.000000000000E+00,
                              0.000000000000E+00, 1.790965185832E-01, -1.409264246697E-01, 0.000000000000E+00,
                              2.470704176958E-01, -2.870447990848E-01, 0.000000000000E+00, 0.000000000000E+00,
                              0.000000000000E+00, 0.000000000000E+00, 0.000000000000E+00, -3.821901376240E-01,
                              2.341676047591E-01, 0.000000000000E+00, -4.081171181630E-01, 3.770452369543E-01])

        if ii == 0:
            print(hs_ij[0, 0] - H_H_22_10, '\n', hs_ij[1, 0] - C_H_22_10, '\n', hs_ij[1, 1] - C_C_22_10, '\n')
        elif ii == 1:
            print(hs_ij[0, 0] - H_H_25_10, '\n', hs_ij[0, 1] - H_C_25_10, '\n', hs_ij[1, 1] - C_C_25_10, '\n')
            # print(zmesh[1, 1, 2:4, 2:4])


read_skf_hdf5({}, {}, {}, {})


def test_bicubvec_interp():
    interp = BicubInterpVec({}, {})
    xmesh = t.tensor([[1, 2, 3], [2, 3, 4]])
    zmesh = t.randn(2, 2, 3, 3, 20)
    xi = t.tensor([1.4, 2.5])
    result = interp.bicubic_2d(xmesh, zmesh, xi, xi)
    print('result', result, result.shape)

if __name__ == '__main__':
    para = {}
    para['task'] = 'bicubic_vec'
    if para['task'] == 'bicub_interp':
        test_bicub_interp()
    elif para['task'] == 'bicub_interp_ml':
        test_bicub_interp_ml()
    elif para['task'] == 'bicubic_vec':
        pass # test_bicubvec_interp()
