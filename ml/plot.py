'''this file is for plot different data'''
import numpy as np
import os
import matplotlib.pyplot as plt


def plot_main(outpara):
    if 'hstable' not in outpara['target']:
        plot_eig(outpara)
    # plot_ham(outpara)
    if 'dipole' in outpara['target']:
        plot_dip(outpara)
    if outpara['Lml_HS']:
        plot_spl(outpara)
    elif outpara['Lml_skf']:
        plot_compr(outpara)


def plot_eig(outpara):
    nfile = int(outpara['n_dataset'][0])
    fpopt, fpref = open('eigbp.dat', 'r'), open('eigref.dat', 'r')
    nsteps = int(outpara['mlsteps'] / outpara['save_steps'])
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
        for istep in range(0, nsteps):
            plt.plot(istep, dataref[ifile, 0] - dataopt[ifile, istep, 0], 'xr')
            plt.plot(istep, dataref[ifile, 1] - dataopt[ifile, istep, 1], 'ob')
            icount += 1
    plt.show()


def plot_dip(outpara):
    nfile = int(outpara['n_dataset'][0])
    fpopt, fpref = open('dipbp.dat', 'r'), open('dipref.dat', 'r')
    nsteps = int(outpara['mlsteps'] / outpara['save_steps'])
    dataref = np.zeros((nfile, 3))
    dataopt = np.zeros((nfile, nsteps, 3))

    print('plot dipole values')
    for ifile in range(0, nfile):
        dataref[ifile, :] = np.fromfile(fpref, dtype=float, count=3, sep=' ')
        for istep in range(0, nsteps):
            dataopt[ifile, istep, :] = np.fromfile(
                    fpopt, dtype=float, count=3, sep=' ')

    icount = 0
    for ifile in range(0, nfile):
        for istep in range(0, nsteps):
            plt.plot(istep, abs(dataref[ifile, 0] - dataopt[ifile, istep, 0]), 'xr')
            plt.plot(istep, abs(dataref[ifile, 1] - dataopt[ifile, istep, 1]), 'ob')
            plt.plot(istep, abs(dataref[ifile, 2] - dataopt[ifile, istep, 2]), 'vy')
            icount += 1
    plt.show()


def plot_eig_(outpara, dire, ty):
    fpopt = open(os.path.join(dire, 'eigbp.dat'), 'r')
    fpref = open(os.path.join(dire, 'eigref.dat'), 'r')
    nfile = int(outpara['n_dataset'][0])
    nsteps = int(outpara['mlsteps'] / outpara['save_steps'])
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


def plot_gap_(outpara, dire, ty):
    fpopt = open(os.path.join(dire, 'eigbp.dat'), 'r')
    fpref = open(os.path.join(dire, 'eigref.dat'), 'r')
    nfile = int(outpara['n_dataset'][0])
    nsteps = int(outpara['mlsteps'] / outpara['save_steps'])
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


def plot_dip_(outpara, dire, ty):
    fpopt = open(os.path.join(dire, 'dipbp.dat'), 'r')
    fpref = open(os.path.join(dire, 'dipref.dat'), 'r')
    nfile = int(outpara['n_dataset'][0])
    nsteps = int(outpara['mlsteps'] / outpara['save_steps'])
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


def plot_dip_pred(outpara, dire):
    fpinit = open(os.path.join(dire, 'dipbp.dat'), 'r')
    fpref = open(os.path.join(dire, 'dipref.dat'), 'r')
    fppred = open(os.path.join(dire, 'dippred.dat'), 'r')
    fpdftbplus = open(os.path.join(dire, 'dipdftbplus.dat'), 'r')
    nfile = int(outpara['n_dataset'][0])
    nsteps = int(outpara['mlsteps'] / outpara['save_steps'])
    dataref = np.zeros((nfile, 3))
    datainit = np.zeros((nfile, 3))
    datapred = np.zeros((nfile, 3))
    datadftbplus = np.zeros((nfile, 3))
    diff_init = np.zeros(nfile)
    diff_pred = np.zeros(nfile)
    diff_dftbplus = np.zeros(nfile)

    print('plot eigen values')
    for ifile in range(0, nfile):
        dataref[ifile, :] = np.fromfile(fpref, dtype=float, count=3, sep=' ')
        datainit[ifile, :] = np.fromfile(fpinit, dtype=float, count=3, sep=' ')
        datapred[ifile, :] = np.fromfile(fppred, dtype=float, count=3, sep=' ')
        datadftbplus[ifile, :] = np.fromfile(fpdftbplus, dtype=float, count=3, sep=' ')
        diff_init[ifile] = sum(abs(dataref[ifile, :] - datainit[ifile, :]))
        diff_pred[ifile] = sum(abs(dataref[ifile, :] - datapred[ifile, :]))
        diff_dftbplus[ifile] = sum(abs(dataref[ifile, :] - datadftbplus[ifile, :]))
    for ifile in range(0, nfile):
        plt.ylabel('dipole with initial r vs. predict r')
        p1, = plt.plot(dataref[ifile, 0], datainit[ifile, 0], 'xr')
        p1, = plt.plot(dataref[ifile, 1], datainit[ifile, 1], 'xr')
        p1, = plt.plot(dataref[ifile, 2], datainit[ifile, 2], 'xr')
        p2, = plt.plot(dataref[ifile, 0], datapred[ifile, 0], 'ob')
        p2, = plt.plot(dataref[ifile, 1], datapred[ifile, 1], 'ob')
        p2, = plt.plot(dataref[ifile, 2], datapred[ifile, 2], 'ob')
        p3, = plt.plot(dataref[ifile, 0], datadftbplus[ifile, 0], '*y')
        p3, = plt.plot(dataref[ifile, 1], datadftbplus[ifile, 1], '*y')
        p3, = plt.plot(dataref[ifile, 2], datadftbplus[ifile, 2], '*y')
        plt.legend([p1, p2, p3], ['dipole-init', 'dipole-pred', 'dipole-dftbplus'])
    plt.xlabel('reference dipole')
    plt.plot(dataref, dataref, 'k')
    plt.show()
    plt.plot(np.arange(1, nfile + 1, 1), diff_init, 'xr')
    plt.plot(np.arange(1, nfile + 1, 1), diff_pred, 'ob')
    plt.plot(np.arange(1, nfile + 1, 1), diff_dftbplus, 'ob')
    print(sum(diff_pred) / sum(diff_init), sum(diff_pred) / sum(diff_dftbplus))
    plt.show()


def plot_compr(outpara):
    nfile = int(outpara['n_dataset'][0])
    fpr = open('compr.dat', 'r')
    nsteps = int(outpara['mlsteps'] / outpara['save_steps'])
    max_natom = 10
    datafpr = np.zeros((nfile, nsteps, max_natom))
    icount = 0
    print('plot compression R')
    for ifile in range(0, nfile):
        natom = outpara['coorall'][ifile].shape[0]
        for istep in range(0, nsteps):
            datafpr[ifile, istep, :natom] = np.fromfile(
                    fpr, dtype=float, count=natom, sep=' ')

    plt.ylabel('compression radius of each atom')
    plt.xlabel('steps * molecule')
    for ifile in range(0, nfile):
        for istep in range(0, nsteps):
            natom = outpara['coorall'][ifile].shape[0]
            for iatom in range(0, natom):
                if iatom == 0:
                    p1, = plt.plot(icount, datafpr[ifile, istep, iatom], 'xb')
                else:
                    p2, = plt.plot(icount, datafpr[ifile, istep, iatom], 'or')
            icount += 1
    plt.legend([p1, p2], ['C during optmization', 'H during optmization'])
    plt.show()


def plot_compr_(outpara, dire, ty):
    fpr = open(os.path.join(dire, 'compr.dat'), 'r')
    nfile = int(outpara['n_dataset'][0])
    nsteps = int(outpara['mlsteps'] / outpara['save_steps'])
    max_natom = 10
    datafpr = np.zeros((nfile, nsteps, max_natom))
    icount = 0
    print('plot compression R')
    for ifile in range(0, nfile):
        natom = 5
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


def plot_ham(outpara):
    print('plot ham from spline interpolation')
    nfile = int(outpara['n_dataset'][0])
    nsteps = int(outpara['mlsteps'] / outpara['save_steps'])
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
    dire = '/home/gz_fan/Documents/ML/dftb/ml'
    para['n_dataset'] = ['100']
    nsteps = 20
    para['mlsteps'] = 50
    para['save_steps'] = 10
    para['qatomlist'] = ['qatom_dip_sim.dat', 'qatom_dip_and.dat']
    # plot_gap_(para, dire, 'all')
    # plot_dip_(para, dire, 'be_end')
    # plot_qatom_compare(para, dire)
    plot_compr_(para, dire, 'be_end')
    plot_dip_pred(para, dire)
    para['row'] = 7
    para['col'] = 10
    plot_compr_env(para)
