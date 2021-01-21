import os
import h5py
import torch as t
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from dftbtorch.interpolator import PolySpline, BicubInterpVec
from dftbtorch.sk import SKTran, GetSKTable, GetSK_
from IO.save import Save1D, Save2D
from dftbtorch.matht import BicubInterp
from dftbtorch.dftbcalculator import DFTBCalculator
ATOMNUM = {'H': 1, 'C': 6, 'N': 7, 'O': 8}
ATOMNAME = {1: 'H', 6: 'C', 7: 'N', 8: 'O'}
t.set_default_dtype(t.float64)


def test_bicub_interp():
    """Test Bicubic interpolation."""
    nx = 50
    new = t.linspace(1.51, 4.49, nx)
    xnew, ynew = np.meshgrid(new, new)
    zmesh = t.tensor([
        [.4, .45, .51, .57, .64, .72, .73], [.45, .51, .58, .64, .73, .83, .85],
        [.51, .58, .64, .73, .83, .94, .96], [.57, .64, .73, .84, .97, 1.12, 1.14],
        [.64, .72, .83, .97, 1.16, 1.28, 1.31], [.72, .83, .94, 1.12, 1.38, 1.48, 1.51],
        [.73, .85, .96, 1.14, 1.41, 1.51, 1.57]])
    zmesh_ = t.unsqueeze(t.stack([t.stack([zmesh, zmesh]), t.stack([zmesh, zmesh])]), -1)

    bicinterp = BicubInterp()
    # non-uniformed grid
    x_non = y_non = t.tensor([1.5, 1.6, 1.9, 2.4, 3.0, 3.7, 4.5])
    x_non_ = t.tensor([[1.5, 1.6, 1.9, 2.4, 3.0, 3.7, 4.5],
                       [1.5, 1.6, 1.9, 2.4, 3.0, 3.7, 4.5]])
    
    # uniformed grid
    x_uni = y_uni= t.tensor([1.5, 2., 2.5, 3.0, 3.5, 4.0, 4.5])
    x_uni_ = t.tensor([[1.5, 2., 2.5, 3.0, 3.5, 4.0, 4.5],
                       [1.5, 2., 2.5, 3.0, 3.5, 4.0, 4.5]])
    znon, zuni = t.zeros(nx, nx), t.zeros(nx, nx)
    bicubic = BicubInterpVec({}, {})
    bicubic.bicubic_2d(x_non_, zmesh_, t.tensor([2., 3.,]), t.tensor([2., 3.,]))

    for ii, ix in enumerate(t.unsqueeze(new, -1)):
        for jj, jy in enumerate(t.unsqueeze(new, -1)):
            compr = t.tensor([ix, jy])
            znon[ii, jj] = bicubic.bicubic_2d(x_non_, zmesh_, compr, compr).squeeze()[0, 1]
            zuni[ii, jj] = bicubic.bicubic_2d(x_uni_, zmesh_, compr, compr).squeeze()[0, 1]
            
    f_non = interpolate.interp2d(x_non, y_non, zmesh, kind='cubic')
    znew_scipy_non = f_non(new, new)
    f_uni = interpolate.interp2d(x_uni, y_uni, zmesh, kind='cubic')
    znew_scipy_uni = f_uni(new, new)

    plt.figure()
    fig, axs = plt.subplots(2, 2)  # sharex=True, sharey=True
    pc1 = axs[0, 0].pcolormesh(x_non, y_non, zmesh)
    axs[0, 0].set_title('original data')
    fig.colorbar(pc1, ax=axs[0, 0], shrink=0.6)
    pc2 = axs[0, 1].pcolormesh(xnew, ynew, znon)
    axs[0, 1].set_title('bicubic interpolation')
    fig.colorbar(pc2, ax=axs[0, 1], shrink=0.6)
    pc3 = axs[1, 0].pcolormesh(xnew, ynew, znew_scipy_non)
    fig.colorbar(pc3, ax=axs[1, 0], shrink=0.6)
    axs[1, 0].set_title('numpy interpolation')
    pc4 = axs[1, 1].pcolormesh(xnew, ynew, t.from_numpy(znew_scipy_non) - znon)
    axs[1, 1].set_title('interpolation difference')
    fig.colorbar(pc4, ax=axs[1, 1], shrink=0.6)
    plt.show()

    plt.figure()
    fig, axs = plt.subplots(2, 2)  # sharex=True, sharey=True
    pc1 = axs[0, 0].pcolormesh(x_uni, y_uni, zmesh)
    axs[0, 0].set_title('original data')
    fig.colorbar(pc1, ax=axs[0, 0], shrink=0.6)
    pc2 = axs[0, 1].pcolormesh(xnew, ynew, zuni)
    axs[0, 1].set_title('bicubic interpolation')
    fig.colorbar(pc2, ax=axs[0, 1], shrink=0.6)
    pc3 = axs[1, 0].pcolormesh(xnew, ynew, znew_scipy_uni)
    fig.colorbar(pc3, ax=axs[1, 0], shrink=0.6)
    axs[1, 0].set_title('numpy interpolation')
    pc4 = axs[1, 1].pcolormesh(xnew, ynew, t.from_numpy(znew_scipy_uni) - zuni)
    axs[1, 1].set_title('interpolation difference')
    fig.colorbar(pc4, ax=axs[1, 1], shrink=0.6)
    plt.show()


def test_skf(para, skf, dataset, ml):
    para['datasetSK'] = '../slko/hdf/skf.hdf5'
    skf['ReadSKType'] = 'compressionRadii'
    dataset['LSKFinterpolation'] = True
    # ninterp = skf['sizeInterpolationPoints']
    hdfsk = para['datasetSK']
    if not os.path.isfile(hdfsk):
        raise FileExistsError('dataset %s do not exist' % hdfsk)
    with h5py.File(hdfsk, 'r') as f:
        yyhh = f['HH' + '/hs_all_rall'][:, :, 10, :]
        yyhc = f['HC' + '/hs_all_rall'][:, :, 10, :]
        yych = f['CH' + '/hs_all_rall'][:, :, 10, :]
        yycc = f['CC' + '/hs_all_rall'][:, :, 10, :]
    bicubic = BicubInterpVec(para, ml)
    zmesh = t.from_numpy(np.stack([np.stack([yyhh, yyhc]), np.stack([yych, yycc])]))
    mesh = t.tensor([[1., 1.5, 2., 2.5, 3., 3.5, 4., 5., 6., 8., 10.],
                     [1., 1.5, 2., 2.5, 3., 3.5, 4., 5., 6., 8., 10.]])
    for ii, compr in enumerate([t.tensor([2.25, 2.25]), t.tensor([2.75, 2.75])]):
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
            print('H-H.skf, 10th line', hs_ij[0, 0] - H_H_22_10)
            print('C-H.skf, 10th line', hs_ij[1, 0] - C_H_22_10)
            print('C-C.skf, 10th line', hs_ij[1, 1] - C_C_22_10)
        elif ii == 1:
            print(hs_ij[0, 0] - H_H_25_10, '\n', hs_ij[0, 1] - H_C_25_10, '\n', hs_ij[1, 1] - C_C_25_10, '\n')
            # print(zmesh[1, 1, 2:4, 2:4])

# Hamiltonian data
# H means Hamiltonian table, s, p is orbital, 12 means 1.2 angstrom
Hpp1_12 = t.tensor(
    [-3.742602360128E-01, -4.449762775791E-01, -4.932742066579E-01,
     -5.264237972467E-01, -5.493518120981E-01, -5.656074700325E-01])
Hsp0_12 = t.tensor(
    [1.422770383839E-01, 2.220799798588E-01, 2.726999859220E-01,
     3.062901605651E-01, 3.294371779521E-01, 3.459327480438E-01])
Hss0_12 = t.tensor(
    [-3.231289233209E-01, -4.438323410893E-01, -5.239208519073E-01,
     -5.790245117931E-01, -6.181518110397E-01, -6.466135144784E-01])
Hpp1_24 = t.tensor(
    [-1.040054565948E-01, -1.245271180944E-01, -1.431125379623E-01,
     -1.585562821064E-01, -1.708458454207E-01, -1.805029788286E-01])
Hsp0_24 = t.tensor(
    [2.054470817742E-01, 2.493214414940E-01, 2.863199372436E-01,
     3.154242499608E-01, 3.377491571099E-01, 3.548049683591E-01])
Hss0_24 = t.tensor(
    [-2.283267780320E-01, -2.745945775683E-01, -3.128174377176E-01,
     -3.426404671777E-01, -3.654339344317E-01, -3.827261393906E-01])


def test_interp_compr(para, skf, dataset, ml):
    skf['ReadSKType'] = 'compressionRadii'
    dataset['LSKFinterpolation'] = True
    ngrid = 2  # choose gird distance, 0==>0.02, 1==>0.05, 2==>0.1, 3==>0.2
    nline1 = [59, 23, 11, 5, 3, 1]
    nline2 = [119, 47, 23, 11, 7, 3]
    compr_all = t.tensor([[2.25, 2.25], [2.75, 2.75], [3.25, 3.25],
                          [3.75, 3.75], [4.25, 4.25], [4.75, 4.75]])
    datasetskf = ['../slko/hdf/skf_002.hdf5', '../slko/hdf/skf_005.hdf5',
                  '../slko/hdf/skf_01.hdf5', '../slko/hdf/skf_02.hdf5',
                  '../slko/hdf/skf_03.hdf5', '../slko/hdf/skf_06.hdf5']
    mesh = t.tensor([[1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5., 5.5, 6.],
                     [1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5., 5.5, 6.]])

    # dim1: compr point, dim2: ss, sp, pp, dim3, 2 distance, dim4, method
    diff = t.zeros(6, 3, 2, 2)
    with h5py.File(datasetskf[ngrid], 'r') as f:
        yycc1 = t.from_numpy(f['CC' + '/hs_all_rall'][:, :, nline1[ngrid], :])
        yycc2 = t.from_numpy(f['CC' + '/hs_all_rall'][:, :, nline2[ngrid], :])
    bicubic = BicubInterpVec(para, ml)
    zmesh1 = t.stack([t.stack([yycc1, yycc1]), t.stack([yycc1, yycc1])])
    zmesh2 = t.stack([t.stack([yycc2, yycc2]), t.stack([yycc2, yycc2])])
    print("yycc2, yycc2", yycc2[1:-2, 1:-2, 6])
    # Save2D(yycc2.detach().cpu().numpy(),
    #        name='CC2_4.dat', dire='.', ty='a')
    for jj, icompr in enumerate(compr_all):
        hs_ij1 = bicubic.bicubic_2d(mesh, zmesh1, icompr, icompr)[0, 0]
        hs_ij2 = bicubic.bicubic_2d(mesh, zmesh2, icompr, icompr)[0, 0]
        # numpy interpolation
        # print(mesh.shape, zmesh1.shape)
        f_HH = interpolate.interp2d(mesh[0], mesh[1], zmesh1[0, 0, :, :, 6], kind='cubic')
        H_pp1 = f_HH(icompr[0], icompr[1])
        f_HH = interpolate.interp2d(mesh[0], mesh[1], zmesh1[0, 0, :, :, 8], kind='cubic')
        H_sp1 = f_HH(icompr[0], icompr[1])
        f_HH = interpolate.interp2d(mesh[0], mesh[1], zmesh1[0, 0, :, :, 9], kind='cubic')
        H_ss1 = f_HH(icompr[0], icompr[1])
        f_HH = interpolate.interp2d(mesh[0], mesh[1], zmesh2[0, 0, :, :, 6], kind='cubic')
        H_pp2 = f_HH(icompr[0], icompr[1])
        f_HH = interpolate.interp2d(mesh[0], mesh[1], zmesh2[0, 0, :, :, 8], kind='cubic')
        H_sp2 = f_HH(icompr[0], icompr[1])
        f_HH = interpolate.interp2d(mesh[0], mesh[1], zmesh2[0, 0, :, :, 9], kind='cubic')
        H_ss2 = f_HH(icompr[0], icompr[1])
        diff[jj, :, 0, 0] = t.tensor([hs_ij1[9] - Hss0_12[jj],
                                      hs_ij1[8] - Hsp0_12[jj], hs_ij1[6] - Hpp1_12[jj]])
        diff[jj, :, 0, 1] = t.tensor([t.from_numpy(H_ss1) - Hss0_12[jj],
                                      t.from_numpy(H_sp1) - Hsp0_12[jj], t.from_numpy(H_pp1) - Hpp1_12[jj]])
        diff[jj, :, 1, 0] = t.tensor([hs_ij2[9] - Hss0_24[jj],
                                      hs_ij2[8] - Hsp0_24[jj], hs_ij2[6] - Hpp1_24[jj]])
        diff[jj, :, 1, 1] = t.tensor([t.from_numpy(H_ss2) - Hss0_24[jj],
                                      t.from_numpy(H_sp2) - Hsp0_24[jj], t.from_numpy(H_pp2) - Hpp1_24[jj]])
        # print(diff[jj, :, 1, 0], hs_ij2[9], Hss0_24[jj],
        #       hs_ij2[8], Hsp0_24[jj], hs_ij2[6], Hpp1_24[jj])
    lab0 = ['ss0, distance: 1.2', 'sp0, distance: 1.2', 'pp1, distance: 1.2']
    lab1 = ['ss0, distance: 2.4', 'sp0, distance: 2.4', 'pp1, distance: 2.4']
    compr = [2.25, 2.75, 3.25, 3.75, 4.5, 5.5]
    for i in range(len(lab0)):
        plt.plot(compr, diff[:, i, 0, 0], label=lab0[i])
        plt.plot(compr, diff[:, i, 1, 0], label=lab1[i], linestyle=':')
        plt.legend()
    plt.ylabel('Hamiltonian Error')
    plt.xlabel('Compression Radii (Bohr)')
    plt.savefig('Hamiltonian_compr.png', dpi=300)
    plt.show()


def test_interp_grid(para, skf, dataset, ml):
    grid = t.tensor([0.02, 0.05, 0.1, 0.2, 0.3])
    distance = [t.tensor([[0., 0., 0.], [0.5, 0.5, 0.5]]),
                t.tensor([[0., 0., 0.], [0.7, 0.7, 0.7]]),
                t.tensor([[0., 0., 0.], [0.9, 0.9, 0.9]])]
    para['scc'] = 'nonscc'  # nonscc, scc, xlbomd
    pathsk = ['../slko/test/grid0.02', '../slko/test/grid0.05',
              '../slko/test/grid0.1', '../slko/test/grid0.2',
              '../slko/test/grid0.3']
    # dim0: grid, dim1: distance, dim2, ss0, sp0, pp1
    h = t.zeros(5, 5, 2)
    for ii, _ in enumerate(pathsk):
        para['directorySK'] = pathsk[ii]
        for jj, jdist in enumerate(distance):
            dataset = {'positions': distance[jj], 'numbers': [[6, 6]]}
            print(ii, jj)
            dataset['nfile'] = 1
            result = DFTBCalculator(para, dataset, skf)
            skf = result.skf
            h[ii, jj, 0], h[ii, jj, 1] = skf['hammat'][0, 4], skf['hammat'][0, 5]

    # dist = np.sqrt(np.array([0.5, 0.6, 0.7, 0.8, 0.9]) ** 2 * 3) / .529177249
    dist = [1.64, 2.29, 2.95]
    for jj, jdist in enumerate(distance):
        plt.plot(grid[1:], h[1:, jj, 0] - h[0, jj, 0], label='distance: '+str(dist[jj])+', ss0')
        plt.plot(grid[1:], h[1:, jj, 1] - h[0, jj, 1], label='distance: '+str(dist[jj])+', sp0')
        plt.legend(loc='upper left')
    plt.xlabel('grid distance (Bohr)')
    plt.ylabel('Hamiltonian Error')
    plt.savefig('Hamiltonian_grid.png', dpi=300)
    plt.show()

def test_bicubvec_interp():
    interp = BicubInterpVec({}, {})
    xmesh = t.tensor([[1, 2, 3], [2, 3, 4]])
    zmesh = t.randn(2, 2, 3, 3, 20)
    xi = t.tensor([1.4, 2.5])
    result = interp.bicubic_2d(xmesh, zmesh, xi, xi)
    print('result', result, result.shape)


if __name__ == '__main__':
    t.set_default_dtype(t.float64)
    t.set_printoptions(15)
    task = 'compr'
    if task == 'bicub_interp':
        test_bicub_interp()
    elif task == 'bicubic_vec':
        test_skf({}, {}, {}, {})
    elif task == 'compr':
        test_interp_compr({}, {}, {}, {})
    elif task == 'grid':
        test_interp_grid({}, {}, {}, {})
