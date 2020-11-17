"""Plot with different options."""
import numpy as np
import os
import matplotlib.pyplot as plt
import dftbtorch.initparams as initpara
from IO.dataloader import LoadData


def plot_ml(para, ml):
    """Plot for DFTB-ML optimization."""
    # read_nstep(para)  # read how many steps have been saved each molecule
    # plot_humolumo(para)

    # plot dipole
    '''if para['Ldipole']:
        plot_dip(para, ml)'''

    # plot polarizability
    if para['LMBD']:
        plot_pol(para, ml)

    # plot_energy(para)
    # plot_charge(para)
    if para['task'] in ('mlCompressionR', 'mlIntegral'):
        plot_loss(para, ml)


def plot_ml_feature(para, ml):
    """Plot for DFTB-ML optimization."""
    plot_humolumo_f(para, ml)
    plot_dip_f(para, ml)
    plot_pol_f(para, ml)
    plot_energy_f(para, ml)
    plot_loss_f(para, ml)
    plot_compr_f(para, ml)


def read_nstep(para):
    """Read how many saved data each molecule."""
    dire = para['dire_data']
    fpstep = open(dire + '/nsave.dat', 'r')
    print('read how many saved data each molecule')
    nstep = np.fromfile(fpstep, dtype=float, count=-1, sep=' ')
    para['nsteps'] = nstep


def plot_humolumo(para):
    """Plot eigenvalues [ntrain, nsteps, 2] in .data."""
    ntrain = para['ntrain']
    dire = para['dire_data']
    if para['ref'] == 'aims':
        eigref = dire + '/HLaims.dat'
    elif para['ref'] == 'dftbplus' or para['ref'] == 'dftbase':
        eigref = dire + '/HLdftbplus.dat'
    elif para['ref'] == 'dftb':
        eigref = dire + '/HLdftb.dat'
    fpopt, fpref = open(dire + '/HLbp.dat', 'r'), open(eigref, 'r')
    nstep = para['nsteps']  # save steps
    nstepmax = int(nstep.max())
    datref = np.zeros((ntrain, 2), dtype=float)
    gapref = np.zeros((ntrain), dtype=float)
    datopt = np.zeros((ntrain, nstepmax, 2), dtype=float)
    gapopt = np.zeros((ntrain, nstepmax), dtype=float)
    print('plot HOMO-LUMO values')
    for ifile in range(ntrain):
        nstep_ = int(nstep[ifile])
        datref[ifile] = np.fromfile(fpref, dtype=float, count=2, sep=' ')
        gapref[ifile] = abs(datref[ifile][1] - datref[ifile][0])
        datopt[ifile, :nstep_, :] = np.fromfile(
            fpopt, dtype=float, count=2*nstep_, sep=' ').reshape(nstep_, 2)
        gapopt[ifile, :] = abs(datopt[ifile, :, 1] - datopt[ifile, :, 0])
    icount = 1
    for ifile in range(ntrain):
        count = np.linspace(icount, icount + nstep_, nstep_)
        p1, = plt.plot(count, abs(datref[ifile, 0] - datopt[ifile, :, 0]), 'r')
        p2, = plt.plot(count, abs(datref[ifile, 1] - datopt[ifile, :, 1]), 'b')
        icount += nstep_

    plt.xlabel('steps * molecules')
    plt.ylabel('eigenvalue absolute difference')
    plt.legend([p1, p2], ['HOMO-diff', 'LUMO-diff'])
    plt.show()
    icount = 1
    for ifile in range(ntrain):
        count = np.linspace(icount, icount + nstep_ - 1, nstep_)
        p1, = plt.plot(count, abs(gapref[ifile] - gapopt[ifile, :]), 'r')
        icount += nstep_
    plt.xlabel('steps * molecules')
    plt.ylabel('gap absolute difference')
    plt.legend([p1], ['gap-diff'])
    plt.show()


def plot_humolumo_f(para):
    """Plot eigenvalues [ntrain, nsteps, 2] in .data."""
    ntrain = para['nfile']
    dire = para['dire_data']
    if para['ref'] == 'aims':
        eigref = dire + '/HLaims.dat'
    elif para['ref'] == 'dftbplus':
        eigref = dire + '/HLdftbplus.dat'
    elif para['ref'] == 'dftb':
        eigref = dire + '/HLdftb.dat'
    fpopt, fpref = open(dire + '/HLbp.dat', 'r'), open(eigref, 'r')
    nstep = int(para['mlsteps'] / para['save_steps'])
    datref = np.zeros((ntrain, 2), dtype=float)
    gapref = np.zeros((ntrain), dtype=float)
    datopt = np.zeros((ntrain, nstep, 2), dtype=float)
    gapopt = np.zeros((ntrain, nstep), dtype=float)

    print('plot HOMO-LUMO values')
    for ifile in range(ntrain):
        datref[ifile] = np.fromfile(fpref, dtype=float, count=2, sep=' ')
        gapref[ifile] = abs(datref[ifile][1] - datref[ifile][0])
    for istep in range(nstep):
        datopt[:, istep, :] = np.fromfile(
            fpopt, dtype=float, count=2*ntrain, sep=' ').reshape(ntrain, 2)
        gapopt[:, istep] = abs(datopt[:, istep, 1] - datopt[:, istep, 0])
    icount = 1
    for istep in range(nstep):
        count = np.linspace(icount, icount + ntrain - 1, ntrain)
        p1, = plt.plot(count, abs(datref[:, 0] - datopt[:, istep, 0]), 'r')
        p2, = plt.plot(count, abs(datref[:, 1] - datopt[:, istep, 1]), 'b')
        icount += ntrain
    plt.xlabel('steps * molecules')
    plt.ylabel('eigenvalue absolute difference')
    plt.legend([p1, p2], ['HOMO-diff', 'LUMO-diff'])
    plt.show()
    icount = 1
    for istep in range(nstep):
        count = np.linspace(icount, icount + ntrain - 1, ntrain)
        p1, = plt.plot(count, abs(gapref[:] - gapopt[:, istep]), 'r')
        icount += ntrain
    plt.xlabel('steps * molecules')
    plt.ylabel('gap absolute difference')
    plt.legend([p1], ['gap-diff'])
    plt.show()


def plot_homolumo_pred_weight(para, dire, ref=None, dftbplus=None):
    """Plot HUMO-LUMO, gap with various results.

    This is for test, will plot, DFTB+, DFT, DFTB-pred dipoles together.
    """
    npred = para['npred']
    print('plot HOMO-LUMO values')

    fppred = open(os.path.join(dire, 'HLpred.dat'), 'r')
    if ref == 'aims':
        fpref = open(os.path.join(dire, 'HLaims.dat'), 'r')
    if dftbplus is not None:
        fpdftbplus = open(os.path.join(dire, 'HLdftbplus.dat'), 'r')

    humolumo = np.fromfile(
        fppred, dtype=float, count=npred*2, sep=' ').reshape(npred, 2)
    if ref == 'aims':
        dref = np.fromfile(
            fpref, dtype=float, count=npred*2, sep=' ').reshape(npred, 2)
    if dftbplus is not None:
        ddftbplus = np.fromfile(
            fpdftbplus, dtype=float, count=npred*2, sep=' ').reshape(npred, 2)

    if ref == 'aims':
        diff_pred = abs(dref - humolumo)
        p1, = plt.plot(dref[:, 0], humolumo[:, 0], 'or')
        p2, = plt.plot(dref[:, 1], humolumo[:, 1], 'ob')
        diff_dftbplus = abs(dref - ddftbplus)
        p3, = plt.plot(dref[:, 0], ddftbplus[:, 0], '*c')
        p4, = plt.plot(dref[:, 1], ddftbplus[:, 1], '*y')
        plt.legend([p3, p4], ['HOMO-pred', 'HOMO-DFTB+'])
        plt.legend([p3, p4], ['LUMO-pred', 'LUMO-DFTB+'])
        plt.xlabel('reference HOMO-LUMO')
        plt.ylabel('HOMO-LUMO with initial r vs. predict r')
        minref, maxref = np.min(dref), np.max(dref)
        refrange = np.linspace(minref, maxref)
        plt.plot(refrange, refrange, 'k')
        plt.show()
    else:
        for ii in range(npred):
            p2, = plt.plot(ddftbplus, humolumo, 'ob')
            plt.legend([p2], ['HOMO-LUMO-pred'])
        plt.xlabel('reference HUMO-LUMO')
        plt.ylabel('HOMO-LUMO with initial r vs. predict r')
        plt.show()
    if ref == 'aims':
        gapref = abs(dref[:, 0] - dref[:, 1])
        gappred = abs(humolumo[:, 0] - humolumo[:, 1])
        gapdftbplus = abs(ddftbplus[:, 0] - ddftbplus[:, 1])
        diff_gap_pred = abs(gapref - gappred)
        p3, = plt.plot(gapref, gappred, 'ob')
        diff_gap_dftbplus = abs(gapref - gapdftbplus)
        p4, = plt.plot(gapref, gapdftbplus, '*y')
        plt.legend([p3, p4], ['gap-pred', 'gap-DFTB+'])
        plt.xlabel('reference gap')
        plt.ylabel('gap with initial r vs. predict r')
        minref, maxref = np.min(gapref), np.max(gapref)
        refrange = np.linspace(minref, maxref)
        plt.plot(refrange, refrange, 'k')
        plt.show()

    plt.plot(np.linspace(1, npred, npred), diff_pred[:, 0], marker='o',
             color='r', label='difference-pred-HOMO')
    plt.plot(np.linspace(1, npred, npred), diff_pred[:, 1], marker='x',
             color='b', label='difference-pred-LUMO')
    if dftbplus is not None:
        plt.plot(np.linspace(1, npred, npred), diff_dftbplus[:, 0], marker='*',
                 color='y', label='difference-dftbplus-HOMO')
        plt.plot(np.linspace(1, npred, npred), diff_dftbplus[:, 1], marker='*',
                 color='c', label='difference-dftbplus-LUMO')
        print('(prediction -reference) / (DFTB+ - reference):',
              sum(sum(diff_pred)) / sum(sum(diff_dftbplus)))
    plt.xlabel('molecule number')
    plt.ylabel('HOMO-LUMO difference between DFTB and aims')
    plt.legend()
    plt.show()

    plt.plot(np.linspace(1, npred, npred), diff_gap_pred, marker='o',
             color='r', label='difference-pred-gap')
    if dftbplus is not None:
        plt.plot(np.linspace(1, npred, npred), diff_gap_dftbplus,
                 marker='*', color='y', label='difference-dftbplus-HOMO')
        print('(prediction -reference) / (DFTB+ - reference):',
              sum(diff_gap_pred) / sum(diff_gap_dftbplus))
    plt.xlabel('molecule number')
    plt.ylabel('gap difference between DFTB and aims')
    plt.legend()
    plt.show()


def plot_dip(para, ml):
    """Plot dipole [nfile, nsteps, 3] in .data."""
    dire = para['dire_data']
    if ml['ref'] == 'aims':
        dipref = dire + '/dipaims.dat'
    elif ml['ref'] == 'dftbplus':
        dipref = dire + '/dipdftbplus.dat'
    elif ml['ref'] == 'dftb':
        dipref = dire + '/dipdftb.dat'
    elif ml['ref'] == 'hdf':
        dipref = dire + '/diphdf.dat'
    nfile = para['nfile']
    fpopt, fpref = open(dire + '/dipbp.dat', 'r'), open(dipref, 'r')
    nstep = para['nsteps']
    nstepmax = int(nstep.max())
    dref = np.zeros((nfile, 3), dtype=float)
    dopt = np.zeros((nfile, nstepmax, 3), dtype=float)

    print('plot dipole values')
    for ifile in range(nfile):
        nstep_ = int(nstep[ifile])
        dref[ifile, :] = np.fromfile(fpref, dtype=float, count=3, sep=' ')
        dopt[ifile, :nstep_, :] = np.fromfile(
            fpopt, dtype=float, count=3*nstep_, sep=' ').reshape(nstep_, 3)
    icount = 1
    for ifile in range(nfile):
        nstep_ = int(nstep[ifile])
        count = np.linspace(icount, icount + nstep_ - 1, nstep_)
        p1, = plt.plot(count, abs(dref[ifile, 0] - dopt[ifile, :nstep_, 0]))
        p2, = plt.plot(count, abs(dref[ifile, 1] - dopt[ifile, :nstep_, 1]))
        p3, = plt.plot(count, abs(dref[ifile, 2] - dopt[ifile, :nstep_, 2]))
        icount += nstep_
    plt.xlabel('steps * molecules')
    plt.ylabel('dipole absolute difference')
    plt.legend([p1, p2, p3],
               ['diff_dipole_x', 'diff_dipole_y', 'diff_dipole_z'])
    plt.show()


def plot_dip_f(para):
    """Plot dipole [ntrain, nsteps, 3] in .data."""
    if para['ref'] == 'aims':
        dipref = '.data/dipaims.dat'
    elif para['ref'] == 'dftbplus':
        dipref = '.data/dipdftbplus.dat'
    elif para['ref'] == 'dftb':
        dipref = '.data/dipdftb.dat'
    ntrain = para['nfile']
    fpopt, fpref = open('.data/dipbp.dat', 'r'), open(dipref, 'r')
    nstep = int(para['mlsteps'] / para['save_steps'])
    dref = np.zeros((ntrain, 3), dtype=float)
    dopt = np.zeros((ntrain, nstep, 3), dtype=float)

    print('plot dipole values')
    for ifile in range(ntrain):
        dref[ifile, :] = np.fromfile(fpref, dtype=float, count=3, sep=' ')
    for istep in range(nstep):
        dopt[:, istep, :] = np.fromfile(
            fpopt, dtype=float, count=3*ntrain, sep=' ').reshape(ntrain, 3)
    icount = 1
    for istep in range(nstep):
        count = np.linspace(icount, icount + ntrain - 1, ntrain)
        p1, = plt.plot(count, abs(dref[:, 0] - dopt[:, istep, 0]))
        p2, = plt.plot(count, abs(dref[:, 1] - dopt[:, istep, 1]))
        p3, = plt.plot(count, abs(dref[:, 2] - dopt[:, istep, 2]))
        icount += ntrain
    plt.xlabel('steps * molecules')
    plt.ylabel('dipole absolute difference')
    plt.legend([p1, p2, p3],
               ['diff_dipole_x', 'diff_dipole_y', 'diff_dipole_z'])
    plt.show()


def plot_charge(para):
    """Plot polarizability [ntrain, nsteps, natom] in .data."""
    dire = para['dire_data']
    if para['ref'] == 'aims':
        polref = dire + '/*.dat'
    elif para['ref'] == 'dftbplus' or para['ref'] == 'dftbase':
        polref = dire + '/refqatom.dat'
    elif para['ref'] == 'dftb':
        polref = dire + '/*.dat'
    ntrain = para['ntrain']
    fpopt, fpref = open(dire + '/qatombp.dat', 'r'), open(polref, 'r')
    nstep = para['nsteps']
    nstepmax = int(nstep.max())
    natommax = int(max(para['natomall']))
    dref = np.zeros((ntrain, natommax), dtype=float)
    dopt = np.zeros((ntrain, nstepmax, natommax), dtype=float)

    print('plot charge values')
    for ifile in range(ntrain):
        nstep_ = int(nstep[ifile])
        nat = int(para['natomall'][ifile])
        dref[ifile, :nat] = np.fromfile(fpref, dtype=float, count=nat, sep=' ')
        dopt[ifile, :nstep_, :nat] = np.fromfile(
            fpopt, dtype=float, count=nat*nstep_, sep=' ').reshape(nstep_, nat)
    icount = 1
    print("charge", dref, '\n', dopt)

    for ifile in range(ntrain):
        nstep_ = int(nstep[ifile])
        for ist in range(nstep_):
            plt.plot(icount, sum(abs(dref[ifile, :] - dopt[ifile, ist, :])), 'x')
            print("charge", dref[ifile, :] - dopt[ifile, ist, :])
            icount += 1
    plt.xlabel('steps * molecules')
    plt.ylabel('sum of charge absolute difference')
    plt.legend()
    plt.show()


def plot_pol(para):
    """Plot polarizability [ntrain, nsteps, natom] in .data."""
    dire = para['dire_data']
    if para['ref'] == 'aims':
        polref = dire + '/polaims.dat'
    elif para['ref'] == 'dftbplus':
        polref = dire + '/poldftbplus.dat'
    elif para['ref'] == 'dftb':
        polref = dire + '/poldftb.dat'
    ntrain = para['ntrain']
    fpopt, fpref = open(dire + '/polbp.dat', 'r'), open(polref, 'r')
    nstep = para['nsteps']
    nstepmax = int(nstep.max())
    natommax = int(max(para['natomall']))
    dref = np.zeros((ntrain, natommax), dtype=float)
    dopt = np.zeros((ntrain, nstepmax, natommax), dtype=float)

    print('plot dipole values')
    for ifile in range(ntrain):
        nstep_ = int(nstep[ifile])
        nat = int(para['natomall'][ifile])
        dref[ifile, :nat] = np.fromfile(fpref, dtype=float, count=nat, sep=' ')
        dopt[ifile, :nstep_, :nat] = np.fromfile(
            fpopt, dtype=float, count=nat*nstep_, sep=' ').reshape(nstep_, nat)
    icount = 1
    for ifile in range(ntrain):
        nstep_ = int(nstep[ifile])
        count = np.linspace(icount, icount + nstep_ - 1, nstep_)
        for iat in range(int(para['natomall'][ifile])):
            plt.plot(count, abs(dref[ifile, iat] - dopt[ifile, :nstep_, iat]))
        icount += nstep_
    plt.xlabel('steps * molecules')
    plt.ylabel('polarizability absolute difference')
    plt.legend()
    plt.show()


def plot_pol_f(para):
    """Plot polarizability [ntrain, nsteps, natom] in .data."""
    if para['ref'] == 'aims':
        polref = '.data/polaims.dat'
    elif para['ref'] == 'dftbplus':
        polref = '.data/poldftbplus.dat'
    elif para['ref'] == 'dftb':
        polref = '.data/poldftb.dat'
    ntrain = para['nfile']
    fpopt, fpref = open('.data/polbp.dat', 'r'), open(polref, 'r')
    nstep = int(para['mlsteps'] / para['save_steps'])
    natommax = int(max(para['natomall']))
    dref = np.zeros((ntrain, natommax), dtype=float)
    dopt = np.zeros((ntrain, nstep, natommax), dtype=float)

    print('plot dipole values')
    for ifile in range(ntrain):
        nat = int(para['natomall'][ifile])
        dref[ifile, :nat] = np.fromfile(fpref, dtype=float, count=nat, sep=' ')
    for istep in range(nstep):
        for ifile in range(ntrain):
            dopt[ifile, istep, :nat] = np.fromfile(
                fpopt, dtype=float, count=nat, sep=' ')
    icount = 1
    for istep in range(nstep):
        count = np.linspace(icount, icount + ntrain - 1, ntrain)
        for iat in range(int(para['natomall'][ifile])):
            plt.plot(count, abs(dref[:, iat] - dopt[:, istep, iat]))
        icount += ntrain
    plt.xlabel('steps * molecules')
    plt.ylabel('polarizability absolute difference')
    plt.legend()
    plt.show()


def plot_energy(para):
    """Plot energy [ntrain, nsteps] in .data."""
    dire = para["dire_data"]
    if para['ref'] == 'aims':
        enerref = dire + '/energyaims.dat'
    elif para['ref'] == 'dftbplus' or para['ref'] == 'dftbase':
        enerref = dire + '/energydftbplus.dat'
    elif para['ref'] == 'dftb':
        enerref = dire + '/energydftb.dat'
    ntrain = para['ntrain']
    fpopt, fpref = open(dire + '/energybp.dat', 'r'), open(enerref, 'r')
    nstep = para['nsteps']
    nstepmax = int(nstep.max())
    dataref = np.zeros((ntrain), dtype=float)
    dataopt = np.zeros((ntrain, nstepmax), dtype=float)

    print('plot energy values')
    for ifile in range(0, ntrain):
        nstep_ = int(nstep[ifile])
        dataref[ifile] = np.fromfile(fpref, dtype=float, count=1, sep=' ')
        dataopt[ifile, :nstep_] = np.fromfile(
            fpopt, dtype=float, count=nstep_, sep=' ')
    icount = 1
    for ifile in range(ntrain):
        nstep_ = int(nstep[ifile])
        count = np.linspace(icount, icount + nstep_ - 1, nstep_)
        plt.plot(count, abs(dataref[ifile] - dataopt[ifile, :nstep_]))
        icount += nstep_
    plt.xlabel('steps * molecules')
    plt.ylabel('energy absolute difference')
    plt.legend()
    plt.show()
    print("dataref", dataref, "\n dataopt", dataopt)


def plot_energy_f(para):
    """Plot energy [ntrain, nsteps] in .data."""
    if para['ref'] == 'aims':
        enerref = '.data/energyaims.dat'
    elif para['ref'] == 'dftbplus':
        enerref = '.data/energydftbplus.dat'
    elif para['ref'] == 'dftb':
        enerref = '.data/energydftb.dat'
    ntrain = para['nfile']
    fpopt, fpref = open('.data/energybp.dat', 'r'), open(enerref, 'r')
    nstep = int(para['mlsteps'] / para['save_steps'])
    dataref = np.zeros((ntrain), dtype=float)
    dataopt = np.zeros((ntrain, nstep), dtype=float)

    print('plot energy values')
    for ifile in range(ntrain):
        dataref[ifile] = np.fromfile(fpref, dtype=float, count=1, sep=' ')
        dataopt[ifile, :] = np.fromfile(
            fpopt, dtype=float, count=nstep, sep=' ')
    icount = 1
    for istep in range(nstep):
        count = np.linspace(icount, icount + ntrain - 1, ntrain)
        plt.plot(count, abs(dataref[:] - dataopt[:, istep]))
        icount += ntrain
    plt.xlabel('steps * molecules')
    plt.ylabel('energy absolute difference')
    plt.legend()
    plt.show()


def plot_loss2(para, ml):
    fploss = open('.data/loss.dat', 'r')
    yp = np.fromfile(fploss, dtype=float, count=ml['mlSteps'], sep=' ')
    xp = np.linspace(1, len(yp), len(yp))
    plt.plot(xp, yp)
    plt.show()


def plot_loss(para, ml):
    """Plot energy [nfile, nsteps] in .data."""
    print('plot loss values')
    loss = np.fromfile('loss.dat', dtype=float, sep=' ')
    print('loss', loss, np.arange(1, len(loss) + 1))
    plt.plot(np.arange(1, len(loss) + 1), loss)
    plt.xlabel('steps * molecules')
    plt.ylabel('loss: abs(y_ref - y_pred) / natom')
    plt.legend()
    plt.show()


def plot_loss_f(para):
    """Plot energy [ntrain, nsteps] in .data."""
    ntrain = para['nfile']
    fploss = open('.data/lossbp.dat', 'r')
    nstep = int(para['mlsteps'] / para['save_steps'])
    dataloss = np.zeros((ntrain, nstep), dtype=float)

    print('plot loss values')
    for istep in range(nstep):
        dataloss[:, istep] = np.fromfile(
            fploss, dtype=float, count=ntrain, sep=' ')
    icount = 1
    for istep in range(nstep):
        ilabel = str(istep) + 'step'
        count = np.linspace(icount, icount + ntrain - 1, ntrain)
        plt.plot(count, dataloss[:, istep], label=ilabel)
    plt.xlabel('steps * molecules')
    plt.ylabel('loss: abs(y_ref - y_pred) / natom')
    plt.legend()
    plt.show()


def plot_dip_pred(para, dire, ref=None, dftbplus=None):
    """Plot dipole with various results.

    This is for test, will plot, DFTB+, DFT, DFTB-pred dipoles together.
    """
    ntrain = para['ntrain']
    npred = para['npred']
    if para['Lopt_step']:
        nsteps = para['nsteps']
    else:
        nsteps = int(para['mlsteps'] / para['save_steps'])

    dinit = np.zeros((ntrain, 3), dtype=float)
    dopt = np.zeros((ntrain, 3), dtype=float)
    diff_init = np.zeros((ntrain), dtype=float)
    diff_opt = np.zeros((ntrain), dtype=float)
    dref = np.zeros((npred, 3), dtype=float)
    dpred = np.zeros((npred, 3), dtype=float)
    ddftbplus = np.zeros((npred, 3), dtype=float)
    diff_pred = np.zeros((npred), dtype=float)
    diff_dftbplus = np.zeros((npred), dtype=float)

    print('plot dipole values')
    fppred = open(os.path.join(dire, 'dippred.dat'), 'r')
    fpinit = open(os.path.join(dire, 'dipbp.dat'), 'r')
    if ref == 'aims':
        fpref = open(os.path.join(dire, 'dipaims.dat'), 'r')
    if dftbplus is not None:
        fpdftbplus = open(os.path.join(dire, 'dipdftbplus.dat'), 'r')

    for ifile in range(ntrain):
        if para['Lopt_step']:
            nstep_ = int(nsteps[ifile])
        else:
            nstep_ = nsteps
        dinit_ = np.fromfile(fpinit, dtype=float, count=3*nstep_, sep=' ')
        dinit[ifile, :] = dinit_[:3]
        dopt[ifile, :] = dinit_[-3:]
    for ifile in range(npred):
        dpred[ifile, :] = np.fromfile(fppred, dtype=float, count=3, sep=' ')
        if ref == 'aims':
            dref[ifile, :] = np.fromfile(fpref, dtype=float, count=3, sep=' ')
        if dftbplus is not None:
            ddftbplus[ifile, :] = np.fromfile(
                    fpdftbplus, dtype=float, count=3, sep=' ')

    min_ = min(ntrain, npred)
    for ii in range(min_):
        diff_init[ii] = sum(abs(dref[ii, :] - dinit[ii, :]))
        diff_opt[ii] = sum(abs(dref[ii, :] - dopt[ii, :]))
        p1, = plt.plot(dref[ii, :], dinit[ii, :], 'xr')
        p2, = plt.plot(dref[ii, :], dopt[ii, :], 'vc')
    if ref == 'aims':
        for ii in range(npred):
            diff_pred[ii] = sum(abs(dref[ii, :] - dpred[ii, :]))
            p3, = plt.plot(dref[ii, :], dpred[ii, :], 'ob')
            diff_dftbplus[ii] = sum(abs(dref[ii, :] - ddftbplus[ii, :]))
            p4, = plt.plot(dref[ii, :], ddftbplus[ii, :], '*y')
        plt.legend([p1, p2, p3, p4],
                   ['dipole-init', 'dipole-opt',
                    'dipole-pred', 'dipole-DFTB+'])
        plt.xlabel('reference dipole')
        plt.ylabel('dipole with initial r vs. predict r')
        minref, maxref = np.min(dref), np.max(dref)
        refrange = np.linspace(minref, maxref)
        plt.plot(refrange, refrange, 'k')
        plt.show()
    else:
        for ii in range(min_):
            p1, = plt.plot(ddftbplus[ii], dinit[ii], 'xr')
            p2, = plt.plot(ddftbplus[ii], dpred[ii], 'ob')
            plt.legend([p1, p2], ['dipole-init', 'dipole-pred'])
        plt.xlabel('reference dipole')
        plt.ylabel('dipole with initial r vs. predict r')
        plt.show()

    p1 = plt.plot(np.arange(1, min_ + 1, 1), diff_init[: min_], marker='x',
                  color='r', label='difference-init')
    p2 = plt.plot(np.arange(1, min_ + 1, 1), diff_opt[: min_], marker='v',
                  color='c', label='difference-opt')
    p3 = plt.plot(np.arange(1, npred + 1, 1), diff_pred, marker='o',
                  color='b', label='difference-pred')
    if dftbplus is not None:
        p4 = plt.plot(np.arange(1, npred + 1, 1), diff_dftbplus, marker='*',
                      color='y', label='difference-dftbplus')
        pred_dftb = sum(diff_pred) / sum(diff_dftbplus)
        print('(prediction -reference) / (DFTB+ - reference):', pred_dftb)
    pred_init = sum(diff_pred) / (npred + 1) / (sum(diff_init) / (min_ + 1))
    print('(prediction -reference) / (initial - reference):', pred_init)
    pred_opt = sum(diff_pred) / (npred + 1) / (sum(diff_opt) / (min_ + 1))
    print('(prediction -reference) / (opt - reference):', pred_opt)
    plt.xlabel('molecule number')
    plt.ylabel('dipole difference between DFTB and aims')
    plt.legend()
    plt.show()
    para['dip_ratio_pred_dftb'] = pred_dftb


def plot_dip_pred_weight(para, dire, ref=None, dftbplus=None):
    """Plot dipole with various results.

    This is for test, will plot, DFTB+, DFT, DFTB-pred dipoles together.
    """
    ntrain = para['ntrain']
    npred = para['npred']

    dref = np.zeros((npred, 3), dtype=float)
    dpred = np.zeros((npred, 3), dtype=float)
    ddftbplus = np.zeros((npred, 3), dtype=float)
    diff_pred = np.zeros((npred), dtype=float)
    diff_dftbplus = np.zeros((npred), dtype=float)

    print('plot dipole values')
    fppred = open(os.path.join(dire, 'dippred.dat'), 'r')
    if ref == 'aims':
        fpref = open(os.path.join(dire, 'dipaims.dat'), 'r')
    if dftbplus is not None:
        fpdftbplus = open(os.path.join(dire, 'dipdftbplus.dat'), 'r')

    for ifile in range(npred):
        dpred[ifile, :] = np.fromfile(fppred, dtype=float, count=3, sep=' ')
        if ref == 'aims':
            dref[ifile, :] = np.fromfile(fpref, dtype=float, count=3, sep=' ')
        if dftbplus is not None:
            ddftbplus[ifile, :] = np.fromfile(
                    fpdftbplus, dtype=float, count=3, sep=' ')

    min_ = min(ntrain, npred)
    if ref == 'aims':
        for ii in range(npred):
            diff_pred[ii] = sum(abs(dref[ii, :] - dpred[ii, :]))
            p3, = plt.plot(dref[ii, :], dpred[ii, :], 'ob')
            diff_dftbplus[ii] = sum(abs(dref[ii, :] - ddftbplus[ii, :]))
            p4, = plt.plot(dref[ii, :], ddftbplus[ii, :], '*y')
        plt.legend([p3, p4], ['dipole-pred', 'dipole-DFTB+'])
        plt.xlabel('reference dipole')
        plt.ylabel('dipole with initial r vs. predict r')
        minref, maxref = np.min(dref), np.max(dref)
        refrange = np.linspace(minref, maxref)
        plt.plot(refrange, refrange, 'k')
        plt.show()
    else:
        for ii in range(min_):
            p2, = plt.plot(ddftbplus[ii], dpred[ii], 'ob')
            plt.legend([p2], ['dipole-pred'])
        plt.xlabel('reference dipole')
        plt.ylabel('dipole with initial r vs. predict r')
        plt.show()

    p3 = plt.plot(np.linspace(1, npred, npred), diff_pred, marker='o',
                  color='b', label='difference-pred')
    if dftbplus is not None:
        p4 = plt.plot(np.linspace(1, npred, npred), diff_dftbplus, marker='*',
                      color='y', label='difference-dftbplus')
        print('(prediction -reference) / (DFTB+ - reference):',
              sum(diff_pred) / sum(diff_dftbplus))
    plt.xlabel('molecule number')
    plt.ylabel('dipole difference between DFTB and aims')
    plt.legend()
    plt.show()


def plot_pol_pred(para, dire, ref=None, dftbplus=None):
    """Plot dipole with various results."""
    ntrain = para['ntrain']
    npred = para['npred']
    nmax = max(ntrain, npred)
    nmin = min(ntrain, npred)
    nsteps = para['nsteps']
    natommax = para['natommax']

    if ref == 'aims':
        fpref = open(os.path.join(dire, 'polaims.dat'), 'r')
    if dftbplus is not None:
        fpdftbplus = open(os.path.join(dire, 'poldftbplus.dat'), 'r')
    fppred = open(os.path.join(dire, 'polpred.dat'), 'r')
    fpinit = open(os.path.join(dire, 'polbp.dat'), 'r')
    dinit = np.zeros((ntrain, natommax), dtype=float)
    dopt = np.zeros((ntrain, natommax), dtype=float)
    diff_init = np.zeros((ntrain), dtype=float)
    diff_opt = np.zeros((ntrain), dtype=float)
    dref = np.zeros((nmax, natommax), dtype=float)
    ddftbplus = np.zeros((nmax, natommax), dtype=float)
    diff_dftbplus = np.zeros((nmax), dtype=float)
    dpred = np.zeros((npred, natommax), dtype=float)
    diff_pred = np.zeros((npred), dtype=float)

    print('plot polarizability values')
    for ifile in range(ntrain):
        iat = int(para['natomall'][ifile])
        nstep_ = int(nsteps[ifile])
        dinit_ = np.fromfile(fpinit, dtype=float, count=iat*nstep_, sep=' ')
        dinit[ifile, :iat] = dinit_[:iat]
        dopt[ifile, :iat] = dinit_[-iat:]

    for ifile in range(npred):
        if ref == 'aims':
            iat = int(para['natomall'][ifile])
            dref[ifile, :iat] = np.fromfile(
                fpref, dtype=float, count=iat, sep=' ')
        dpred[ifile, :iat] = np.fromfile(
            fppred, dtype=float, count=iat, sep=' ')
        ddftbplus[ifile, :iat] = np.fromfile(
                fpdftbplus, dtype=float, count=iat, sep=' ')

    for ii in range(nmin):
        iat = int(para['natomall'][ii])
        diff_init[ii] = sum(abs(dref[ii, :iat] - dinit[ii, :iat]))
        p1, = plt.plot(dref[ii, :iat], dinit[ii, :iat], 'xr')
        diff_opt[ii] = sum(abs(dref[ii, :iat] - dopt[ii, :iat]))
        p2, = plt.plot(dref[ii, :iat], dopt[ii, :iat], 'vc')
    if ref == 'aims':
        for ii in range(npred):
            iat = int(para['natomall'][ii])
            diff_pred[ii] = sum(abs(dref[ii, :iat] - dpred[ii, :iat]))
            p3, = plt.plot(dref[ii, :iat], dpred[ii, :iat], 'ob')
            diff_dftbplus[ii] = sum(abs(dref[ii, :iat] - ddftbplus[ii, :iat]))
            p4, = plt.plot(dref[ii, :iat], ddftbplus[ii, :iat], '*y')
        plt.legend([p1, p2, p3, p4],
                   ['polarizability-init', 'polarizability-opt',
                    'polarizability-pred', 'polarizability-DFTB+'])
        plt.xlabel('reference polarizability')
        plt.ylabel('polarizability with initial r vs. predict r')
        minref, maxref = np.min(dref), np.max(dref)
        refrange = np.linspace(minref, maxref)
        plt.plot(refrange, refrange, 'k')
        plt.show()
    else:
        for ii in range(nmin):
            plt.plot(ddftbplus[ii], ddftbplus[ii])
            p1, = plt.plot(ddftbplus[ii], dinit[ii], 'xr')
            p2, = plt.plot(ddftbplus[ii], dpred[ii], 'ob')
            plt.legend([p1, p2],
                       ['polarizability-init', 'polarizability-pred'])
        plt.xlabel('reference dipole')
        plt.ylabel('dipole with initial r vs. predict r')
        plt.show()

    p1 = plt.plot(np.arange(1, nmin + 1, 1), diff_init[: nmin], marker='x',
                  color='r', label='polarizability-init')
    p2 = plt.plot(np.arange(1, nmin + 1, 1), diff_opt[: nmin], marker='v',
                  color='c', label='polarizability-opt')
    p3 = plt.plot(np.arange(1, npred + 1, 1), diff_pred, marker='o',
                  color='b', label='polarizability-pred')
    if ref == 'aims':
        p4 = plt.plot(np.linspace(1, npred, npred), diff_dftbplus[:npred],
                      marker='*', color='y', label='polarizability-dftbplus')
        print('(prediction -reference) / (DFTB+ - reference):',
              sum(diff_pred) / sum(diff_dftbplus))
    print('(prediction -reference) / (opt - reference):',
          (sum(diff_pred) / npred) / (sum(diff_opt) / nmin))
    print('(prediction -reference) / (initial - reference):',
          (sum(diff_pred) / npred) / (sum(diff_init) / nmin))
    plt.xlabel('molecule number')
    plt.ylabel('polarizability difference between DFTB and aims')
    plt.legend()
    plt.show()


def plot_pol_pred_weight(para, dire, ref=None, dftbplus=None):
    """Plot dipole with various results."""
    ntrain = para['ntrain']
    npred = para['npred']
    nmax = max(ntrain, npred)
    nmin = min(ntrain, npred)
    natommax = int(max(para['natomall']))  # para['natommax']
    nsteps = int(para['mlsteps'] / para['save_steps'])

    if ref == 'aims':
        fpref = open(os.path.join(dire, 'polaims.dat'), 'r')
    if dftbplus is not None:
        fpdftbplus = open(os.path.join(dire, 'poldftbplus.dat'), 'r')
    fppred = open(os.path.join(dire, 'polpred.dat'), 'r')
    fpinit = open(os.path.join(dire, 'polbp.dat'), 'r')
    dref = np.zeros((npred, natommax), dtype=float)
    dpred = np.zeros((npred, natommax), dtype=float)
    ddftbplus = np.zeros((npred, natommax), dtype=float)
    diff_pred = np.zeros((npred), dtype=float)
    diff_dftbplus = np.zeros((npred), dtype=float)
    dinit = np.zeros((nsteps, nmin, natommax), dtype=float)
    diff_opt = np.zeros((nmin), dtype=float)
    print('plot polarizability values')
    for istep in range(nsteps):
        for ifile in range(nmin):
            iat = int(para['natomall'][ifile])
            dinit_ = np.fromfile(fpinit, dtype=float, count=iat, sep=' ')
            dinit[istep, ifile, :iat] = dinit_[:iat]
    for ifile in range(npred):
        iat = int(para['natomall'][ifile])
        if ref == 'aims':
            dref[ifile, :] = np.fromfile(
                fpref, dtype=float, count=natommax, sep=' ')
        dpred[ifile, :iat] = np.fromfile(
            fppred, dtype=float, count=iat, sep=' ')
        ddftbplus[ifile, :] = np.fromfile(
                fpdftbplus, dtype=float, count=natommax, sep=' ')

    if ref == 'aims':
        for ii in range(npred):
            iat = int(para['natomall'][ifile])
            if ii < nmin:
                p1, = plt.plot(dref[ii, :iat], dinit[0, ii, :iat], 'xr')
                p2, = plt.plot(dref[ii, :iat], dinit[-1, ii, :iat], 'vc')
                diff_opt[ii] = sum(abs(dinit[-1, ii, :iat] - dref[ii, :iat]))
            diff_pred[ii] = sum(abs(dref[ii, :iat] - dpred[ii, :iat]))
            p3, = plt.plot(dref[ii, :iat], dpred[ii, :iat], 'ob')
            diff_dftbplus[ii] = sum(abs(dref[ii, :iat] - ddftbplus[ii, :iat]))
            p4, = plt.plot(dref[ii, :iat], ddftbplus[ii, :iat], '*y')
        plt.legend([p1, p2, p3, p4], [
            'polarizability-init', 'polarizability-opt',
            'polarizability-pred', 'polarizability-DFTB+'])
        plt.xlabel('reference polarizability')
        plt.ylabel('polarizability with initial r vs. predict r')
        minref, maxref = np.min(dref), np.max(dref)
        refrange = np.linspace(minref, maxref)
        plt.plot(refrange, refrange, 'k')
        plt.show()
    else:
        for ii in range(npred):
            plt.plot(ddftbplus[ii], ddftbplus[ii])
            p2, = plt.plot(ddftbplus[ii], dpred[ii], 'ob')
            plt.legend([p2], ['polarizability-pred'])
        plt.xlabel('reference dipole')
        plt.ylabel('dipole with initial r vs. predict r')
        plt.show()

    p3 = plt.plot(np.linspace(1, npred, npred), diff_pred, marker='o',
                  color='b', label='polarizability-pred')
    if ref == 'aims':
        p3 = plt.plot(np.linspace(1, npred, npred), diff_opt, marker='v',
                      color='r', label='polarizability-opt')
        p4 = plt.plot(np.linspace(1, npred, npred), diff_dftbplus, marker='*',
                      color='y', label='polarizability-dftbplus')
        print('(prediction -reference) / (DFTB+ - reference):',
              sum(diff_pred) / sum(diff_dftbplus))
    plt.xlabel('molecule number')
    plt.ylabel('polarizability difference between DFTB and aims')
    plt.legend()
    plt.show()


def plot_compr(para):
    """Plot compression radius [ntrain, nstep, natom]."""
    dire = para["dire_data"]
    ntrain = para['ntrain']
    fpr = open(dire + '/comprbp.dat', 'r')
    nstep = para['nsteps']
    nstepmax = int(nstep.max())
    natommax = int(max(para['natomall']))
    datafpr = np.zeros((ntrain, nstepmax, natommax), dtype=float)

    print('plot compression R')
    for ifile in range(ntrain):
        natom = int(para['natomall'][ifile])
        nstep_ = int(nstep[ifile])
        datafpr[ifile, :nstep_, :natom] = \
            np.fromfile(fpr, dtype=float, count=natom*nstep_,
                        sep=' ').reshape(nstep_, natom)

    icount = 1
    for ifile in range(ntrain):
        nstep_ = int(nstep[ifile])
        count = np.linspace(icount, icount + nstep_ - 1, nstep_)
        natom = para['coorall'][ifile].shape[0]
        for iatom in range(natom):
            plt.plot(count, datafpr[ifile, :nstep_, iatom])
        icount += nstep_
    plt.ylabel('compression radius of each atom')
    plt.xlabel('steps * molecule')
    plt.show()


def plot_pdos(para):
    """Plot PDOS."""
    dire = para["dire_data"]
    ntrain = para['ntrain']
    fpr = open(dire + '/pdosref.dat', 'r')
    fpbp = open(dire + '/pdosbp.dat', 'r')

    nstep = para['nsteps']
    nstepmax = int(nstep.max())
    lenE1 = max(para['shape_pdos'][:, 0])
    lenE2 = max(para['shape_pdos'][:, 1])
    datafpr = np.zeros((ntrain, lenE1, lenE2), dtype=float)
    datafpbp = np.zeros((ntrain, nstepmax, lenE1, lenE2), dtype=float)

    print('plot PDOS')
    for ifile in range(ntrain):
        nstep_ = int(nstep[ifile])
        lenE1_, lenE2_ = para['shape_pdos'][ifile]
        datafpr[ifile, :, :] = \
            np.fromfile(fpr, dtype=float, count=lenE1_*lenE2_,
                        sep=' ').reshape(lenE1_, lenE2_)

        datafpbp[ifile, :nstep_, :, :] = \
            np.fromfile(fpbp, dtype=float, count=lenE1_*lenE2_*nstep_,
                        sep=' ').reshape(nstep_, lenE1_, lenE2_)

    # color = ['r', 'y', 'c', 'b']
    for ifile in range(ntrain):
        nstep_ = int(nstep[ifile])
        lenE1_, lenE2_ = para['shape_pdos'][ifile]

        for ilen in range(lenE1_):

            plt.plot(para['pdos_E'], datafpr[ifile, ilen, :], 'k')
            for istep in range(nstep_):
                plt.plot(para['pdos_E'], datafpbp[ifile, istep, ilen, :], '*r')

    plt.ylabel('projected density of states')
    plt.xlabel('E (eV)')
    plt.show()


def plot_compr_f(para):
    """Plot compression radius [ntrain, nstep, natom]."""
    ntrain = para['nfile']
    fpr = open('.data/comprbp.dat', 'r')
    nstep = int(para['mlsteps'] / para['save_steps'])
    natommax = int(max(para['natomall']))
    datafpr = np.zeros((ntrain, nstep, natommax), dtype=float)

    print('plot compression R')
    for istep in range(nstep):
        for ifile in range(ntrain):
            natom = int(para['natomall'][ifile])
            datafpr[ifile, istep, :natom] = \
                np.fromfile(fpr, dtype=float, count=natom, sep=' ')

    icount = 1
    for istep in range(nstep):
        for imol in range(ntrain):
            natom = para['coorall'][ifile].shape[0]
            xx = np.linspace(icount, icount, natom)
            plt.plot(xx, datafpr[imol, istep, :natom], 'x')
            icount += 1
    plt.ylabel('compression radius of each atom')
    plt.xlabel('steps * molecule')
    plt.show()


def plot_compr_(para, dire, ty):
    fpr = open(os.path.join(dire, 'compr.dat'), 'r')
    ntrain = int(para['n_dataset'][0])
    nsteps = int(para['mlsteps'] / para['save_steps'])
    max_natom = 10
    datafpr = np.zeros((ntrain, nsteps, max_natom))
    icount = 0
    natom = para['natom']
    print('plot compression R')
    for ifile in range(0, ntrain):
        for istep in range(0, nsteps):
            datafpr[ifile, istep, :natom] = np.fromfile(
                    fpr, dtype=float, count=natom, sep=' ')

    for ifile in range(0, ntrain):
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
    ntrain = int(para['n_dataset'][0])
    save_steps = para['save_steps']
    nsteps_ = int(nsteps / save_steps)
    fpcompr = open(os.path.join(dire, 'compr.dat'), 'r')
    fprad = open(os.path.join(dire, 'env_rad.dat'), 'r')
    fpang = open(os.path.join(dire, 'env_ang.dat'), 'r')
    print('plot compression radius values')
    compr = np.zeros((ntrain, 5))
    rad = np.zeros((ntrain, 5))
    ang = np.zeros((ntrain, 5))
    for ifile in range(0, ntrain):
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
    ntrain = int(para['n_dataset'][0])
    print('plot spline parameters')
    nsteps = int(para['mlsteps'] / para['save_steps'])
    icount = 0
    for ifile in range(ntrain):
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
    ntrain = int(para['n_dataset'][0])
    nsteps = int(para['mlsteps'] / para['save_steps'])
    fphamupdate = open('ham.dat', 'r')
    fphamref = open('hamref.dat', 'r')
    istep = 0
    for ifile in range(0, ntrain):
        hamref = np.fromfile(fphamref, dtype=float, count=36, sep=' ')
        for i in range(0, nsteps):
            hamupdate = np.fromfile(fphamupdate, dtype=float, count=36,
                                    sep=' ')
            plt.plot(istep, hamref[31], 'x', color='r')
            plt.plot(istep, hamupdate[31], 'o', color='b')
            istep += 1
    plt.show()

def plot_loss_():
    dire = '../data/results/200709weight_500mol_dip'
    fp = open(os.path.join(dire, 'lossbp.dat'), 'r')
    fpdata = np.fromfile(fp, dtype=float, count=500,  sep=' ')
    xx = np.linspace(1, 500, 500)
    fpdata_ = np.zeros(500)
    for i in range(500):
        fpdata_[i * 10: (i+1) * 10] = sum(fpdata[i * 10: (i+1) * 10]) / 10
    plt.xlabel('batch')
    plt.ylabel('loss during optimization')
    plt.plot(xx, fpdata, label='loss')
    plt.plot(xx, fpdata_, label='average every 10 points of loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    para = {}
    task = 'dftbml_bp'
    if task == 'dftbml_bp':
        para['dire_data'] = '../../dftbpy/data/results/test_CH4_dip_pol_para/dire11'
        initpara.init_dftb_ml(para)
        LoadData(para, int(para['n_dataset'][0]), int(para['n_test'][0]))
        ntrain = int(para['n_dataset'][0])
        npred = int(para['n_test'][0])
        if ntrain >= npred:
            para['ntrain'] = para['nhdf_max']
            para['npred'] = para['nhdf_min']
        else:
            para['npred'] = para['nhdf_max']
            para['ntrain'] = para['nhdf_min']
        para['ref'] = para['reference']
        # read_nstep(para)
        plot_homolumo_pred_weight(
                para, para['dire_data'], ref='aims', dftbplus='dftbplus')
        plot_dip_pred_weight(
                para, para['dire_data'], ref='aims', dftbplus='dftbplus')
        plot_pol_pred_weight(
                para, para['dire_data'], ref='aims', dftbplus='dftbplus')
