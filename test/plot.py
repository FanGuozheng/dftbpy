'''this file is for plot different data'''
import numpy as np
import os
import matplotlib.pyplot as plt


def plot_main(para):
    if 'hstable' not in para['target']:
        plot_eig(para)
    # plot_ham(para)
    if 'dipole' in para['target']:
        plot_dip(para)
    if 'energy' in para['target']:
        plot_energy(para)
    if para['Lml_HS']:
        plot_spl(para)
    elif para['Lml_skf']:
        plot_compr(para)


def plot_eig(para):
    '''
    the directory will store data is fixed, in .data
    '''
    nfile = int(para['n_dataset'][0])
    if para['ref'] == 'aims':
        eigref = '.data/HLaims.dat'
    elif para['ref'] == 'dftbplus':
        eigref = '.data/HLdftbplus.dat'
    elif para['ref'] == 'dftb':
        eigref = '.data/HLref.dat'
    fpopt, fpref = open('.data/HLbp.dat', 'r'), open(eigref, 'r')
    nsteps = int(para['mlsteps'] / para['save_steps'])
    datref = np.zeros((nfile, 2))
    datopt = np.zeros((nfile, nsteps, 2))

    print('plot HOMO-LUMO values')
    for ifile in range(0, nfile):
        datref[ifile, :] = np.fromfile(fpref, dtype=float, count=2, sep=' ')
        for istep in range(0, nsteps):
            datopt[ifile, istep, :] = np.fromfile(
                   fpopt, dtype=float, count=2, sep=' ')

    icount = 1
    for ifile in range(0, nfile):
        for istep in range(0, nsteps):
            plt.plot(icount, datref[ifile, 0] - datopt[ifile, istep, 0], 'xr')
            plt.plot(icount, datref[ifile, 1] - datopt[ifile, istep, 1], 'ob')
            icount += 1
    plt.show()


def plot_dip(para):
    if para['ref'] == 'aims':
        dipref = '.data/dipaims.dat'
    elif para['ref'] == 'dftbplus':
        dipref = '.data/dipdftbplus.dat'
    elif para['ref'] == 'dftb':
        dipref = '.data/dipref.dat'
    nfile = int(para['n_dataset'][0])
    fpopt, fpref = open('.data/dipbp.dat', 'r'), open(dipref, 'r')
    nsteps = int(para['mlsteps'] / para['save_steps'])
    dref = np.zeros((nfile, 3))
    dopt = np.zeros((nfile, nsteps, 3))

    print('plot dipole values')
    for ifile in range(0, nfile):
        dref[ifile, :] = np.fromfile(fpref, dtype=float, count=3, sep=' ')
        for istep in range(0, nsteps):
            dopt[ifile, istep, :] = np.fromfile(
                    fpopt, dtype=float, count=3, sep=' ')

    icount = 1
    for ifile in range(0, nfile):
        for istep in range(0, nsteps):
            plt.plot(icount, abs(dref[ifile, 0] - dopt[ifile, istep, 0]), 'xr')
            plt.plot(icount, abs(dref[ifile, 1] - dopt[ifile, istep, 1]), 'ob')
            plt.plot(icount, abs(dref[ifile, 2] - dopt[ifile, istep, 2]), 'vy')
            icount += 1
    plt.xlabel('steps * molecules')
    plt.ylabel('dipole absolute difference')
    plt.show()


def plot_energy(para):
    if para['ref'] == 'aims':
        enerref = '.data/energyaims.dat'
    elif para['ref'] == 'dftbplus':
        enerref = '.data/energydftbplus.dat'
    elif para['ref'] == 'dftb':
        enerref = '.data/energyref.dat'
    nfile = int(para['n_dataset'][0])
    fpopt, fpref = open('energybp.dat', 'r'), open(enerref, 'r')
    nsteps = int(para['mlsteps'] / para['save_steps'])
    dataref = np.zeros((nfile))
    dataopt = np.zeros((nfile, nsteps))

    print('plot energy values')
    for ifile in range(0, nfile):
        dataref[ifile] = np.fromfile(fpref, dtype=float, count=1, sep=' ')
        for istep in range(0, nsteps):
            dataopt[ifile, istep] = np.fromfile(
                    fpopt, dtype=float, count=1, sep=' ')

    icount = 1
    for ifile in range(0, nfile):
        for istep in range(0, nsteps):
            plt.plot(istep, abs(dataref[ifile] - dataopt[ifile, istep]), 'xr')
            icount += 1
    plt.xlabel('steps * molecules')
    plt.ylabel('energy absolute difference')
    plt.show()


def plot_eig_(para, dire, ty):
    fpopt = open(os.path.join(dire, 'eigbp.dat'), 'r')
    fpref = open(os.path.join(dire, 'eigref.dat'), 'r')
    nfile = int(para['n_dataset'][0])
    nsteps = int(para['mlsteps'] / para['save_steps'])
    dataref = np.zeros((nfile, 2))
    dataopt = np.zeros((nfile, nsteps, 2))

    print('plot eigen values')
    for ifile in range(0, nfile):
        dataref[ifile, :] = np.fromfile(fpref, dtype=float, count=2, sep=' ')
        for istep in range(0, nsteps):
            dataopt[ifile, istep, :] = np.fromfile(
                    fpopt, dtype=float, count=2, sep=' ')

    icount = 0
    for ifile in range(0, nfile):
        if ty == 'all':
            plt.ylabel('LUMO/HOMO difference (eV)')
            for istep in range(0, nsteps):
                p1, = plt.plot(icount, abs(dataref[ifile, 0] -
                               dataopt[ifile, istep, 0]), 'xr')
                p2, = plt.plot(icount, abs(dataref[ifile, 1] -
                               dataopt[ifile, istep, 1]), 'ob')
            icount += 1
            plt.legend([p1, p2], ['HOMO', 'LUMO'])
        elif ty == 'be_end':
            plt.ylabel('LUMO/HOMO difference (eV)')
            p1, = plt.plot(icount, abs(dataref[ifile, 0] -
                           dataopt[ifile, 0, 0]), 'xr')
            p2, = plt.plot(icount, abs(dataref[ifile, 0] -
                           dataopt[ifile, nsteps - 1, 0]), 'or')
            p3, = plt.plot(icount, abs(dataref[ifile, 1] -
                           dataopt[ifile, 0, 1]), 'xb')
            p4, = plt.plot(icount, abs(dataref[ifile, 1] -
                           dataopt[ifile, nsteps - 1, 1]), 'ob')
            icount += 1
            plt.legend([p1, p2, p3, p4], ['HOMO-init', 'HOMO-opt', 'LUMO-init',
                       'LUMO-opt'])
    plt.xlabel('steps * molecules')
    plt.show()


def plot_gap_(para, dire, ty):
    fpopt = open(os.path.join(dire, 'eigbp.dat'), 'r')
    fpref = open(os.path.join(dire, 'eigref.dat'), 'r')
    nfile = int(para['n_dataset'][0])
    nsteps = int(para['mlsteps'] / para['save_steps'])
    dataref = np.zeros((nfile, 2))
    dataopt = np.zeros((nfile, nsteps, 2))

    print('plot eigen values')
    for ifile in range(0, nfile):
        dataref[ifile, :] = np.fromfile(fpref, dtype=float, count=2, sep=' ')
        for istep in range(0, nsteps):
            dataopt[ifile, istep, :] = np.fromfile(
                    fpopt, dtype=float, count=2, sep=' ')

    icount = 0
    for ifile in range(0, nfile):
        if ty == 'all':
            plt.ylabel('gap difference (eV)')
            for istep in range(0, nsteps):
                gapref = abs(dataref[ifile, 0] - dataref[ifile, 1])
                gap = abs(dataopt[ifile, istep, 0] - dataopt[ifile, istep, 1])
                plt.plot(icount, gap - gapref, 'xr')
                icount += 1
            plt.legend('gap')
        elif ty == 'be_end':
            plt.ylabel('gap difference (eV)')
            gapref = abs(dataref[ifile, 0] - dataref[ifile, 1])
            gap_init = abs(dataopt[ifile, 0, 0] - dataopt[ifile, 0, 1])
            gap_end = abs(dataopt[ifile, nsteps - 1, 0] -
                          dataopt[ifile, nsteps - 1, 1])
            p1, = plt.plot(icount, abs(gapref - gap_init), 'xr')
            p2, = plt.plot(icount, abs(gapref - gap_end), 'or')
            icount += 1
            plt.legend([p1, p2], ['gap-init', 'gap-opt'])
    plt.xlabel('steps * molecules')
    plt.show()


def plot_dip_(para, dire, ty):
    fpopt = open(os.path.join(dire, 'dipbp.dat'), 'r')
    fpref = open(os.path.join(dire, 'dipref.dat'), 'r')
    nfile = int(para['n_dataset'][0])
    nsteps = int(para['mlsteps'] / para['save_steps'])
    dataref = np.zeros((nfile, 3))
    dataopt = np.zeros((nfile, nsteps, 3))

    print('plot eigen values')
    for ifile in range(0, nfile):
        dataref[ifile, :] = np.fromfile(fpref, dtype=float, count=3, sep=' ')
        for istep in range(0, nsteps):
            dataopt[ifile, istep, :] = np.fromfile(
                    fpopt, dtype=float, count=3, sep=' ')

    icount = 0
    for ifile in range(0, nfile):
        if ty == 'all':
            plt.ylabel('dipole difference')
            for istep in range(0, nsteps):
                plt.plot(icount, abs(dataopt[ifile, istep, 0] -
                                     dataref[ifile, 0]), 'xr')
                plt.plot(icount, abs(dataopt[ifile, istep, 1] -
                                     dataref[ifile, 1]), 'xr')
                plt.plot(icount, abs(dataopt[ifile, istep, 2] -
                                     dataref[ifile, 2]), 'xr')
                icount += 1
            plt.legend('dipole')
        elif ty == 'be_end':
            plt.ylabel('dipole difference')
            p1, = plt.plot(icount, abs(dataopt[ifile, 0, 0] -
                                       dataref[ifile, 0]), 'xr')
            p1, = plt.plot(icount, abs(dataopt[ifile, 0, 1] -
                                       dataref[ifile, 1]), 'xr')
            p1, = plt.plot(icount, abs(dataopt[ifile, 0, 2] -
                                       dataref[ifile, 2]), 'xr')
            p2, = plt.plot(icount, abs(dataopt[ifile, nsteps - 1, 0] -
                                       dataref[ifile, 0]), 'ob')
            p2, = plt.plot(icount, abs(dataopt[ifile, nsteps - 1, 1] -
                                       dataref[ifile, 1]), 'ob')
            p2, = plt.plot(icount, abs(dataopt[ifile, nsteps - 1, 2] -
                                       dataref[ifile, 2]), 'ob')
            icount += 1
            plt.legend([p1, p2], ['dipole-init', 'dipole-opt'])
    plt.xlabel('steps * molecules')
    plt.show()
    for ifile in range(0, nfile):
        plt.ylabel('dipole difference')
        p1, = plt.plot(dataref[ifile, 0], dataopt[ifile, 0, 0], 'xr')
        p1, = plt.plot(dataref[ifile, 1], dataopt[ifile, 0, 1], 'xr')
        p1, = plt.plot(dataref[ifile, 2], dataopt[ifile, 0, 2], 'xr')
        p2, = plt.plot(dataref[ifile, 0], dataopt[ifile, nsteps - 1, 0], 'ob')
        p2, = plt.plot(dataref[ifile, 1], dataopt[ifile, nsteps - 1, 1], 'ob')
        p2, = plt.plot(dataref[ifile, 2], dataopt[ifile, nsteps - 1, 2], 'ob')
        plt.legend([p1, p2], ['dipole-init', 'dipole-opt'])
    plt.xlabel('reference dipole')
    plt.show()


def plot_dip_pred(para, dire,  aims=None, dftbplus=None):
    '''plot dipole with various results'''
    nfile = int(para['n_dataset'][0])
    ntest = int(para['n_test'][0])
    nsteps = int(para['mlsteps'] / para['save_steps'])

    if aims is not None:
        fpref = open(os.path.join(dire, 'dipaims.dat'), 'r')
    if dftbplus is not None:
        fpdftbplus = open(os.path.join(dire, 'dipdftbplus.dat'), 'r')
    fppred = open(os.path.join(dire, 'dippred.dat'), 'r')
    fpinit = open(os.path.join(dire, 'dipbp.dat'), 'r')
    dinit = np.zeros((nfile, 3), dtype=float)
    diff_init = np.zeros((nfile), dtype=float)

    dref = np.zeros((ntest, 3), dtype=float)
    dpred = np.zeros((ntest, 3), dtype=float)
    ddftbplus = np.zeros((ntest, 3), dtype=float)
    diff_pred = np.zeros((ntest), dtype=float)
    diff_dftbplus = np.zeros((ntest), dtype=float)

    print('plot dipole values')
    for ifile in range(nfile):
        dinit_ = np.fromfile(fpinit, dtype=float, count=3*nsteps, sep=' ')
        dinit[ifile, :] = dinit_[:3]

    for ifile in range(ntest):
        dref[ifile, :] = np.fromfile(fpref, dtype=float, count=3, sep=' ')
        dpred[ifile, :] = np.fromfile(fppred, dtype=float, count=3, sep=' ')
        ddftbplus[ifile, :] = np.fromfile(
                fpdftbplus, dtype=float, count=3, sep=' ')

    min_ = min(nfile, ntest)
    for ii in range(min_):
        diff_init[ii] = sum(abs(dref[ii, :] - dinit[ii, :]))
        p1, = plt.plot(dref[ii, :], dinit[ii, :], 'xr')

    for ii in range(ntest):
        diff_pred[ii] = sum(abs(dref[ii, :] - dpred[ii, :]))
        p2, = plt.plot(dref[ii, :], dpred[ii, :], 'ob')

    if aims is not None:
        for ii in range(ntest):
            diff_dftbplus[ii] = sum(abs(dref[ii, :] - ddftbplus[ii, :]))
            p3, = plt.plot(dref[ii, :], ddftbplus[ii, :], '*y')

    if aims is not None:
        plt.legend([p1, p2, p3],
                   ['dipole-init', 'dipole-pred', 'dipole-DFTB+'])
    else:
        # p1, = plt.plot(dref[: min_], dinit[: min_], 'xr')
        # p2, = plt.plot(dref, dpred, 'ob')
        plt.legend([p1, p2], ['dipole-init', 'dipole-pred'])
    plt.xlabel('reference dipole')
    plt.ylabel('dipole with initial r vs. predict r')

    minref, maxref = np.min(dref), np.max(dref)
    refrange = np.linspace(minref, maxref)
    plt.plot(refrange, refrange, 'k')
    plt.show()

    p1 = plt.plot(np.arange(1, min_ + 1, 1), diff_init[: min_], marker='x',
                  color='r', label='dipole-init')
    p2 = plt.plot(np.arange(1, ntest + 1, 1), diff_pred, marker='o',
                  color='b', label='dipole-pred')
    if aims is not None:
        p3 = plt.plot(np.arange(1, ntest + 1, 1), diff_dftbplus, marker='*',
                      color='y', label='dipole-dftbplus')
        print('(prediction -reference) / (DFTB+ - reference):',
              sum(diff_pred) / sum(diff_dftbplus))
    plt.xlabel('molecule number')
    plt.ylabel('dipole difference between DFTB and aims')
    plt.legend()
    print('(prediction -reference) / (initial - reference):',
          (sum(diff_pred) / (ntest + 1) / (sum(diff_init) / (min_ + 1))))
    plt.show()


def plot_pol_pred(para, dire,  aims=None, dftbplus=None):
    '''plot dipole with various results'''
    nfile = int(para['n_dataset'][0])
    ntest = int(para['n_test'][0])
    nsteps = int(para['mlsteps'] / para['save_steps'])
    natommax = para['natommax']

    if aims is not None:
        fpref = open(os.path.join(dire, 'polaims.dat'), 'r')
    if dftbplus is not None:
        fpdftbplus = open(os.path.join(dire, 'poldftbplus.dat'), 'r')
    fppred = open(os.path.join(dire, 'polpred.dat'), 'r')
    fpinit = open(os.path.join(dire, 'pol.dat'), 'r')
    dinit = np.zeros((nfile, natommax), dtype=float)
    diff_init = np.zeros((nfile), dtype=float)

    dref = np.zeros((ntest, natommax), dtype=float)
    dpred = np.zeros((ntest, natommax), dtype=float)
    ddftbplus = np.zeros((ntest, natommax), dtype=float)
    diff_pred = np.zeros((ntest), dtype=float)
    diff_dftbplus = np.zeros((ntest), dtype=float)

    print('plot polarizability values')
    for ifile in range(nfile):
        infile = int(para['natomall'][ifile])
        dinit_ = np.fromfile(fpinit, dtype=float, count=infile*nsteps, sep=' ')
        dinit[ifile, :infile] = dinit_[:infile]

    for ifile in range(ntest):
        dref[ifile, :infile] = np.fromfile(fpref, dtype=float, count=infile, sep=' ')
        dpred[ifile, :infile] = np.fromfile(fppred, dtype=float, count=infile, sep=' ')
        ddftbplus[ifile, :infile] = np.fromfile(
                fpdftbplus, dtype=float, count=infile, sep=' ')

    min_ = min(nfile, ntest)
    for ii in range(min_):
        diff_init[ii] = sum(abs(dref[ii, :] - dinit[ii, :]))
        p1, = plt.plot(dref[ii, :], dinit[ii, :], 'xr')

    for ii in range(ntest):
        diff_pred[ii] = sum(abs(dref[ii, :] - dpred[ii, :]))
        p2, = plt.plot(dref[ii, :], dpred[ii, :], 'ob')

    if aims is not None:
        for ii in range(ntest):
            diff_dftbplus[ii] = sum(abs(dref[ii, :] - ddftbplus[ii, :]))
            p3, = plt.plot(dref[ii, :], ddftbplus[ii, :], '*y')

    if aims is not None:
        plt.legend([p1, p2, p3],
                   ['polarizability-init', 'polarizability-pred', 'polarizability-DFTB+'])
    else:
        # p1, = plt.plot(dref[: min_], dinit[: min_], 'xr')
        # p2, = plt.plot(dref, dpred, 'ob')
        plt.legend([p1, p2], ['polarizability-init', 'polarizability-pred'])
    plt.xlabel('reference polarizability')
    plt.ylabel('polarizability with initial r vs. predict r')

    minref, maxref = np.min(dref), np.max(dref)
    refrange = np.linspace(minref, maxref)
    plt.plot(refrange, refrange, 'k')
    plt.show()

    p1 = plt.plot(np.arange(1, min_ + 1, 1), diff_init[: min_], marker='x',
                  color='r', label='polarizability-init')
    p2 = plt.plot(np.arange(1, ntest + 1, 1), diff_pred, marker='o',
                  color='b', label='polarizability-pred')
    if aims is not None:
        p3 = plt.plot(np.arange(1, ntest + 1, 1), diff_dftbplus, marker='*',
                      color='y', label='polarizability-dftbplus')
        print('(prediction -reference) / (DFTB+ - reference):',
              sum(diff_pred) / sum(diff_dftbplus))
    plt.xlabel('molecule number')
    plt.ylabel('polarizability difference between DFTB and aims')
    plt.legend()
    print('(prediction -reference) / (initial - reference):',
          (sum(diff_pred) / (ntest + 1) / (sum(diff_init) / (min_ + 1))))
    plt.show()


def plot_compr(para):
    nfile = int(para['n_dataset'][0])
    fpr = open('.data/comprbp.dat', 'r')
    nsteps = int(para['mlsteps'] / para['save_steps'])
    max_natom = 10
    datafpr = np.zeros((nfile, nsteps, max_natom))
    icount = 0
    print('plot compression R')
    for ifile in range(0, nfile):
        natom = para['coorall'][ifile].shape[0]
        for istep in range(0, nsteps):
            datafpr[ifile, istep, :natom] = np.fromfile(
                    fpr, dtype=float, count=natom, sep=' ')

    plt.ylabel('compression radius of each atom')
    plt.xlabel('steps * molecule')
    for ifile in range(0, nfile):
        for istep in range(0, nsteps):
            natom = para['coorall'][ifile].shape[0]
            for iatom in range(0, natom):
                if iatom == 0:
                    p1, = plt.plot(icount, datafpr[ifile, istep, iatom], 'xb')
                else:
                    p2, = plt.plot(icount, datafpr[ifile, istep, iatom], 'or')
            icount += 1
    plt.legend([p1, p2], ['C during optmization', 'H during optmization'])
    plt.show()


def plot_compr_(para, dire, ty):
    fpr = open(os.path.join(dire, 'compr.dat'), 'r')
    nfile = int(para['n_dataset'][0])
    nsteps = int(para['mlsteps'] / para['save_steps'])
    max_natom = 10
    datafpr = np.zeros((nfile, nsteps, max_natom))
    icount = 0
    natom = para['natom']
    print('plot compression R')
    for ifile in range(0, nfile):
        for istep in range(0, nsteps):
            datafpr[ifile, istep, :natom] = np.fromfile(
                    fpr, dtype=float, count=natom, sep=' ')

    for ifile in range(0, nfile):
        if ty == 'all':
            plt.ylabel('compression radius of each atom')
            for istep in range(0, nsteps):
                natom = 5  # need revise !!!
                for iatom in range(0, natom):
                    if iatom == 0:
                        p1, = plt.plot(icount, datafpr[ifile, istep, iatom],
                                       'xr')
                    else:
                        p2, = plt.plot(icount, datafpr[ifile, istep, iatom],
                                       'ob')
                icount += 1
            plt.legend([p1, p2], ['compression R of C', 'compression R of H'])
        elif ty == 'be_end':
            plt.ylabel('initial and optimized compression radius of each atom')
            natom = 5  # need revise !!!
            for iatom in range(0, natom):
                if iatom == 0:
                    p1, = plt.plot(icount, datafpr[ifile, 0, iatom], 'rx')
                    p2, = plt.plot(icount, datafpr[ifile, nsteps - 1, iatom],
                                   'ro')
                else:
                    p3, = plt.plot(icount, datafpr[ifile, 0, iatom], 'bx')
                    p4, = plt.plot(icount, datafpr[ifile, nsteps - 1, iatom],
                                   'bo')
            icount += 1
            plt.legend([p1, p2, p3, p4],
                       ['initial C', 'opt C', 'initial H', 'opt H'])
    plt.xlabel('steps * molecules')
    plt.show()


def plot_qatom_compare():
    qatom_sim = \
        np.array([[4.515630245208740234e+00, 8.628985285758972168e-01,
                   8.742979168891906738e-01, 8.644992113113403320e-01,
                   8.826719522476196289e-01,],
                  [4.489239215850830078e+00, 8.700770139694213867e-01,
                   8.799759745597839355e-01, 8.715157508850097656e-01,
                   8.891907930374145508e-01],
                  [4.469466209411621094e+00, 8.754418492317199707e-01,
                   8.842838406562805176e-01, 8.767709732055664062e-01,
                   8.940382003784179688e-01], 
                  [4.454654216766357422e+00, 8.794505000114440918e-01,
                   8.875507116317749023e-01, 8.807054162025451660e-01,
                   8.976424336433410645e-01],
               [4.443555831909179688e+00, 8.824442625045776367e-01,
                8.900250196456909180e-01, 8.836497664451599121e-01,
                       9.003223180770874023e-01],
                      [4.435251235961914062e+00, 8.846810460090637207e-01,
                       8.918996453285217285e-01, 8.858537673950195312e-01,
                       9.023141860961914062e-01],
                      [4.429032802581787109e+00, 8.863517642021179199e-01,
                       8.933195471763610840e-01, 8.875028491020202637e-01,
                       9.037961959838867188e-01],
                      [4.424376010894775391e+00, 8.875995874404907227e-01,
                       8.943922519683837891e-01, 8.887349963188171387e-01,
                       9.048963785171508789e-01],
                      [4.420890808105468750e+00, 8.885324597358703613e-01,
                       8.952043056488037109e-01, 8.896572589874267578e-01,
                       9.057159423828125000e-01]])
    qatom_and = \
        np.array([[4.515630245208740234e+00, 8.628985285758972168e-01,
                   8.742979168891906738e-01, 8.644992113113403320e-01,
                   8.826719522476196289e-01],
                  [4.489239215850830078e+00, 8.700770139694213867e-01,
                   8.799759745597839355e-01, 8.715157508850097656e-01,
                   8.891907930374145508e-01],
                  [4.410413742065429688e+00, 8.914681673049926758e-01,
                   8.971409797668457031e-01, 8.924694657325744629e-01,
                   9.085090756416320801e-01],
                  [4.410467147827148438e+00, 8.914093971252441406e-01,
                   8.972904682159423828e-01, 8.924435377120971680e-01,
                   9.083911180496215820e-01]])

    qatom_dftb = np.array([4.41797511, 0.89550622, 0.89550622, 0.89550622,
                           0.89550622])
    len_sim = np.shape(qatom_sim)[0]
    len_and = np.shape(qatom_and)[0]
    sim_x = np.linspace(0, len_sim, len_sim)
    and_x = np.linspace(0, len_and, len_and)
    for ii in range(0, len_sim):
        for jj in range(0, 5):
            plt.plot(sim_x[ii], qatom_sim[ii, 1], 'xr')
    for ii in range(0, len_and):
        for jj in range(0, 5):
            plt.plot(and_x[ii], qatom_and[ii, 1], 'oy')
    for ii in range(0, len_sim):
        for jj in range(0, 5):
            plt.plot(sim_x[ii], qatom_dftb[1], 'vb')


def plot_compr_env(para, dire='.'):
    nsteps = para['mlsteps']
    nfile = int(para['n_dataset'][0])
    save_steps = para['save_steps']
    nsteps_ = int(nsteps / save_steps)
    fpcompr = open(os.path.join(dire, 'compr.dat'), 'r')
    fprad = open(os.path.join(dire, 'env_rad.dat'), 'r')
    fpang = open(os.path.join(dire, 'env_ang.dat'), 'r')
    print('plot compression radius values')
    compr = np.zeros((nfile, 5))
    rad = np.zeros((nfile, 5))
    ang = np.zeros((nfile, 5))
    for ifile in range(0, nfile):
        natom = 5
        datafpcompr = np.fromfile(fpcompr, dtype=float, count=natom*nsteps_,
                                  sep=' ')
        datafprad = np.fromfile(fprad, dtype=float, count=natom, sep=' ')
        datafpang = np.fromfile(fpang, dtype=float, count=natom,  sep=' ')
        compr[ifile, :] = datafpcompr[natom * (nsteps_ - 1):natom * nsteps_]
        # print(compr[ifile, :], datafprad)
        rad[ifile, :] = datafprad[:] #+ datafpang[:] * 20
        ang[ifile, :] = datafpang[:]
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(rad[:, 0], compr[:, 0], 'rx')
    plt.title('C', fontsize=15)
    plt.xlabel('radial environmental parameter')
    plt.ylabel('compression radius')
    plt.subplot(1, 2, 2)
    plt.plot(rad[:, 1:], compr[:, 1:], 'bo')
    plt.title('H', fontsize=15)
    plt.xlabel('radial environmental parameter')
    plt.show()

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(ang[:, 0], compr[:, 0], 'rx')
    plt.subplot(1, 2, 2)
    plt.plot(ang[:, 1:], compr[:, 1:], 'bo')
    plt.show()


def plot_spl(para):
    fpspl = open('spl.dat', 'r')
    fpsplref = open('splref.dat', 'r')
    row = 7
    col = 10
    nfile = int(para['n_dataset'][0])
    print('plot spline parameters')
    nsteps = int(para['mlsteps'] / para['save_steps'])
    icount = 0
    for ifile in range(nfile):
        splref = np.fromfile(fpsplref, dtype=float,
                             count=row*col, sep=' ')
        splupdate = np.fromfile(fpspl, dtype=float, count=row*col*nsteps,
                                sep=' ')
        for istep in range(nsteps):
            diff = splupdate[istep*row*col:(istep+1)*row*col] - splref[:]
            xx = np.linspace(icount*row*col+1, (icount+1)*row*col, row*col)
            plt.plot(xx, abs(diff))
            icount += 1
    plt.show()


def plot_ham(para):
    print('plot ham from spline interpolation')
    nfile = int(para['n_dataset'][0])
    nsteps = int(para['mlsteps'] / para['save_steps'])
    fphamupdate = open('ham.dat', 'r')
    fphamref = open('hamref.dat', 'r')
    istep = 0
    for ifile in range(0, nfile):
        hamref = np.fromfile(fphamref, dtype=float, count=36, sep=' ')
        for i in range(0, nsteps):
            hamupdate = np.fromfile(fphamupdate, dtype=float, count=36,
                                    sep=' ')
            plt.plot(istep, hamref[31], 'x', color='r')
            plt.plot(istep, hamupdate[31], 'o', color='b')
            istep += 1
    plt.show()


if __name__ == '__main__':
    para = {}
    dire = '.'
    para['n_dataset'] = ['50']
    # nsteps = 30
    para['mlsteps'] = 30
    para['save_steps'] = 5
    para['natom'] = 5
    para['qatomlist'] = ['qatom_dip_sim.dat', 'qatom_dip_and.dat']
    # plot_gap_(para, dire, 'all')
    # plot_dip_(para, dire, 'be_end')
    # plot_qatom_compare(para, dire)
    plot_compr_(para, dire, 'be_end')
    plot_dip_pred(para, dire)
    para['row'] = 7
    para['col'] = 10
    plot_compr_env(para)
