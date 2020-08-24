#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 10:59:55 2019
@author: guozheng
"""
import os
import numpy as np
import sys
from torch.autograd import Variable
import torch as t
import h5py
import write_output as write
import dftbtorch.dftb_torch as dftb_torch
import dftbtorch.parameters as parameters
import dftbtorch.slakot as slakot
import utils.plot as plot
import init_parameter as initpara
import ml.interface as interface
from ml.feature import ACSF as acsfml
from utils.load import LoadData
from utils.save import SaveData
from utils.ase import DFTB, Aims
import dftbtorch.parser as parser
# DireSK = '/home/gz_fan/Documents/ML/dftb/slko'
ATOMIND = {'H': 1, 'HH': 2, 'HC': 3, 'C': 4, 'CH': 5, 'CC': 6}
ATOMNUM = {'H': 1, 'C': 6, 'N': 7, 'O': 8}
HNUM = {'CC': 4, 'CH': 2, 'CO': 4, 'HC': 0, 'HH': 1, 'HO': 2, 'OC': 0,
        'OH': 0, 'OO': 4}
COMP_R = {'H': 3.0, 'C': 3.0}
VAL_ORB = {"H": 1, "C": 2, "N": 2, "O": 2, "Ti": 3}
AIMS_ENERGY = {"H": -0.45891649, "C": -37.77330663, "N": -54.46973501,
               "O": -75.03140052}
DFTB_ENERGY = {"H": -0.238600544, "C": -1.398493891, "N": -2.0621839400,
               "O": -3.0861916005}
HIRSH_VOL = {"H": 10.31539447, "C": 38.37861207}


def opt(para):
    """'DFTB-ML optimization."""
    # get the default para for dftb and ML, these para will maintain unchanged
    initpara.init_dftb_ml(para)

    # load dataset, here is hdf type
    LoadData(para, int(para['n_dataset'][0]))

    if para['dataType'] == 'ani':
        para['nfile'] = para['nhdf_max']
    elif para['dataType'] == 'json' or para['dataType'] == 'hdf':
        para['nfile'] = int(para['n_dataset'][0])
        para['ntrain'] = int(para['n_dataset'][0])

    # run reference calculations, either dft or dftb
    runml = RunML(para)
    para['ref'] = para['reference']
    runml.ref()

    # run dftb in ML process
    runml.mldftb(para)

    # plot data from ML
    if para['Lml_acsf']:
        plot.plot_ml_feature(para)
    else:
        plot.plot_ml_compr(para)


def testml(para):
    """'main function for testing DFTB-ML"""
    # get the default para for testing DFTB-ML
    initpara.init_dftb_ml(para)
    ml = interface.ML(para)
    runml = RunML(para)

    # load dataset, here is hdf type
    ntrain = int(para['n_dataset'][0])
    npred = int(para['n_test'][0])
    LoadData(para, int(para['n_dataset'][0]), int(para['n_test'][0]))
    if ntrain >= npred:
        para['ntrain'] = para['nhdf_max']
        para['npred'] = para['nhdf_min']
    else:
        para['npred'] = para['nhdf_max']
        para['ntrain'] = para['nhdf_min']

    dscribe = interface.Dscribe(para)
    if para['Lml_skf']:
        ml.dataprocess(para['dire_data'])
        ml.ml_compr()  # ML predict compression radius with optimized data
    elif para['Lml_acsf']:
        para['compr_pred'] = []
        dscribe.get_acsf_dim()  # get the dimension of features
        ml.get_test_para()  # get optimized weight parameter to predict compR
        for ibatch in range(para['npred']):
            para['coor'] = para['coorall'][ibatch]
            para['ibatch'] = ibatch
            ml.dataprocess_atom()
            para['compr_pred'].append(ml.ml_acsf())

    # if ntest > training dataset, recalculate reference
    if ntrain < npred:
        para['ref'] = para['reference']
        para['nfile'] = para['npred']
        runml.ref()
    # if refrence is not DFTB+, run and add DFTB+ results
    if para['reference'] != 'dftbplus':
        para['ref'] = 'dftbplus'
        para['nfile'] = para['npred']
        runml.ref()

    runml.test_pred_compr(para)

    if para['reference'] == 'aims' and para['Lml_skf']:
        plot.plot_dip_pred(
                para, para['dire_data'], ref='aims', dftbplus='dftbplus')
        plot.plot_pol_pred(
                para, para['dire_data'], ref='aims', dftbplus='dftbplus')
    elif para['reference'] == 'aims' and para['Lml_acsf']:
        plot.plot_homolumo_pred_weight(
                para, para['dire_data'], ref='aims', dftbplus='dftbplus')
        plot.plot_dip_pred_weight(
                para, para['dire_data'], ref='aims', dftbplus='dftbplus')
        plot.plot_pol_pred_weight(
                para, para['dire_data'], ref='aims', dftbplus='dftbplus')
    else:
        plot.plot_dip_pred(
                para, para['dire_data'], dftbplus='dftbplus')
        plot.plot_pol_pred(
                para, para['dire_data'], dftbplus='dftbplus')


class RunML:
    """Perform DFTB-ML optimization.

    loading reference data and dataset
    running calculations of reference method and DFTB method
    saving ml data
    Assumption: the atom specie in each dataset maintain unchanged, otherwise
    we have to check for each new moluecle, if there is new atom specie.
    """

    def __init__(self, para):
        """Initialize DFTB-ML optimization"""
        self.para = para
        self.save = SaveData(self.para)
        self.slako = slakot.SKinterp(self.para)
        self.genml = GenMLPara(self.para)
        self.runcal = RunCalc(self.para)

    def ref(self):
        """Run different reference calculations according to reference type"""
        os.system('rm .data/*.dat')
        # get the constant parameters for DFTB
        parameters.dftb_parameter(self.para)

        # run reference calculations (e.g., DFTB+ ...) before ML
        if self.para['run_reference']:
            self.nbatch = self.para['nfile']
            self.para['refhomo_lumo'] = t.zeros((self.nbatch, 2), dtype=t.float64)
            self.para['refenergy'] = t.zeros((self.nbatch), dtype=t.float64)
            self.para['refdipole'] = t.zeros((self.nbatch, 3), dtype=t.float64)
            self.para['specieall'] = []
            self.para['refeigval'] = []
            if self.para['ref'] == 'dftb':
                self.dftb_ref()
            elif self.para['ref'] == 'aims':
                self.aims_ref(self.para)
            elif self.para['ref'] == 'dftbplus':
                self.dftbplus_ref(self.para)

            # DFTB+ as reference
            elif self.para['ref'] == 'dftbase':
                DFTB(self.para, setenv=True).run_dftb(self.nbatch, para['coorall'])

            # FHI-aims as reference
            elif self.para['ref'] == 'aimsase':
                Aims(self.para).run_aims(self.nbatch, para['coorall'])

        # read reference properties from defined dataset
        elif not self.para['run_reference']:
            if self.para['dataType'] == 'hdf':
                self.get_hdf_data()

    def get_hdf_data(self):
        """Read data from hdf for reference or the following ML."""
        # join the path and hdf data
        hdffile = os.path.join(self.para['pythondata_dire'],
                               self.para['pythondata_file'])
        self.para['coorall'] = []
        self.para['refeigval'] = []
        self.para['refdipole'] = []
        self.para['refenergy'] = []

        # read global parameters
        with h5py.File(hdffile) as f:
            self.para['specie_all'] = f['globalgroup'].attrs['specie_all']
            molecule = [ikey.encode() for ikey in f.keys()]

            # get rid of b-prefix
            molecule2 = [istr.decode('utf-8') for istr in molecule]

            # delete group which is not related to atom species
            ind = molecule2.index('globalgroup')
            del molecule2[ind]

        self.nbatch = 0
        # get the coordinates
        if self.para['hdf_mixture']:
            for ibatch in range(self.para['nfile']):
                for igroup in molecule2:
                    with h5py.File(hdffile) as f:

                        # coordinates: not tensor now !!!
                        namecoor = str(ibatch) + 'coordinate'
                        self.para['coorall'].append(
                            t.from_numpy(f[igroup][namecoor].value))

                        # eigenvalue
                        nameeig = str(ibatch) + 'eigenvalue'
                        self.para['refeigval'].append(
                            t.from_numpy(f[igroup][nameeig].value))

                        # dipole
                        namedip = str(ibatch) + 'dipole'
                        self.para['refdipole'].append(
                            t.from_numpy(f[igroup][namedip].value))

                        # formation energy
                        nameEf = str(ibatch) + 'formationenergy'
                        self.para['refenergy'].append(f[igroup][nameEf].value)
                    self.nbatch += 1

    def dftb_ref(self):
        """Calculate reference (DFTB_torch)"""
        if para['task'] == 'optml':
            dire = '.data'
        elif para['task'] == 'test':
            dire = para['dire_data']
        os.system('rm ' + dire + '/*dftb.dat')

        for ibatch in range(self.nbatch):
            self.get_coor(ibatch)

            if self.para['Lml_skf']:
                # if read all .skf and build [N_ij, N_R1, N_R2, 20] matrix
                '''if self.para['atomspecie'] != self.para['atomspecie_old']:
                    self.genml.get_spllabel()
                else:
                    self.para['LreadSKFinterp'] = False'''
                dftb_torch.Initialization(self.para)
                self.slako.genskf_interp_ij()
                self.genml.genml_init_compr()
                self.para['LreadSKFinterp'] = False  # read SKF list only once

                self.para['compr_ml'] = self.para['compr_init'] - 1
                self.slako.genskf_interp_compr()
                self.runcal.idftb_torchspline()

            elif self.para['Lml_HS']:
                dftb_torch.Initialization(self.para)
                self.genml.get_specie_label()
                dftb_torch.Initialization(self.para)
                self.runcal.idftb_torchspline()
                if self.para['interptype'] == 'Polyspline':
                    self.save.save2D(self.para['splyall'],
                                     name='splref.dat', ty='w')
                    para['hs_all'] = self.para['splyall']
                elif self.para['interptype'] == 'Bspline':
                    self.save.save2D(self.para['cspline'],
                                     name='splref.dat', ty='w')
                self.save.save1D(self.para['hammat'].detach().numpy(),
                                 name='hamref.dat', ty='w')

            self.para['refhomo_lumo'][ibatch] = self.para['homo_lumo']
            self.para['refeigval'].append(self.para['eigenvalue'])
            self.para['refenergy'][ibatch] = self.para['energy']
            self.para['refdipole'][ibatch] = self.para['dipole'][:]
            self.save.save1D(self.para['eigenvalue'].detach().numpy(),
                             name='eigvaldftb.dat', dire=dire, ty='a')
            self.save.save1D(self.para['alpha_mbd'].detach().numpy(),
                             name='poldftb.dat', dire=dire, ty='a')
            self.save.save1D(self.para['natom'].numpy(),
                             name='natomdftb.dat', dire=dire, ty='a')
        self.save.save2D(self.para['refhomo_lumo'].detach().numpy(),
                         name='HLdftb.dat', dire=dire, ty='w')
        self.save.save2D(self.para['refdipole'].detach().numpy(),
                         name='dipdftb.dat', dire=dire, ty='w')
        self.save.save1D(self.para['refenergy'].detach().numpy(),
                         name='energydftb.dat', dire=dire, ty='w')

    def aims_ref(self, para):
        """Calculate reference (FHI-aims)"""
        self.pre_aims()

        for ibatch in range(self.nbatch):
            if 'eigval' in self.para['target']:
                print('Error: FHI-aims do not support eigenvalue optimization')
                sys.exit()
            if type(para['coorall'][ibatch]) is t.Tensor:
                para['coor'] = para['coorall'][ibatch]
            elif type(para['coorall'][ibatch]) is np.ndarray:
                para['coor'] = t.from_numpy(para['coorall'][ibatch])

            # check if atom specie is the same to the former
            self.runcal.aims(para, ibatch, self.dir_ref)

            # calculate formation energy
            energy = write.FHIaims(self.para).read_energy(
                    self.para, ibatch + 1, self.dir_ref)
            self.para['refenergy'][ibatch] = self.cal_for_energy(
                    energy[-1], para['coor'])
        self.save_aims()

    def pre_aims(self):
        """Pre-processing aims calculations."""
        self.direaims = os.getcwd() + '/ref/aims'
        if not os.path.exists(self.direaims):
            print('Warning: please make a folder "aims", prepare input files ',
                  'and a script which run calculations and extract results')
        self.dir_ref = os.getcwd() + '/ref/aims'
        os.system('rm ref/aims/*.dat')

    def save_aims(self):
        """save files for reference calculations"""
        homo_lumo = write.FHIaims(self.para).read_bandenergy(
                self.para, self.nbatch, self.dir_ref)
        dipole = write.FHIaims(self.para).read_dipole(
                self.para, self.nbatch, self.dir_ref, 'eang', 'eang')
        alpha_mbd = write.FHIaims(self.para).read_alpha(
                    self.para, self.nbatch, self.dir_ref)
        refvol = write.FHIaims(self.para).read_hirshfeld_vol(
                    self.para, self.nbatch, self.dir_ref)
        self.para['refhomo_lumo'] = homo_lumo
        self.para['refdipole'] = dipole
        self.para['refalpha_mbd'] = alpha_mbd
        self.para['refvol'] = refvol

        if para['task'] == 'opt':
            dire = '.data'
        elif para['task'] == 'test':
            dire = para['dire_data']
        os.system('rm ' + dire + '/*aims.dat')
        self.save.save2D(self.para['refhomo_lumo'].detach().numpy(),
                         name='HLaims.dat', dire=dire, ty='a')
        self.save.save2D(self.para['refdipole'].detach().numpy(),
                         name='dipaims.dat', dire=dire, ty='a')
        self.save.save2D(self.para['refalpha_mbd'].detach().numpy(),
                         name='polaims.dat', dire=dire, ty='a')
        self.save.save2D(self.para['refvol'].detach().numpy(),
                         name='volaims.dat', dire=dire, ty='a')
        self.save.save1D(self.para['refenergy'].detach().numpy(),
                         name='energyaims.dat', dire=dire, ty='a')
        self.save.save1D(np.asarray(self.para['natomall']),
                         name='natomaims.dat', dire=dire, ty='a')

    def cal_for_energy(self, energy, coor):
        """calculate formation energy for molecule"""
        natom = coor.shape[0]
        if self.para['ref'] == 'aims':
            for iat in range(0, natom):
                idx = int(coor[iat, 0])
                iname = list(ATOMNUM.keys())[list(ATOMNUM.values()).index(idx)]
                energy = energy - AIMS_ENERGY[iname]
        elif self.para['ref'] == 'dftb' or self.para['ref'] == 'dftbplus':
            for iat in range(0, natom):
                idx = int(coor[iat, 0])
                iname = list(ATOMNUM.keys())[list(ATOMNUM.values()).index(idx)]
                energy = energy - DFTB_ENERGY[iname]
        return energy

    def dftbplus_ref(self, para):
        """Calculate reference (DFTB+)"""
        self.pre_dftbplus()

        for ibatch in range(self.nbatch):
            # get the coor of ibatch, and start initialization
            para['ibatch'] = ibatch
            if type(para['coorall'][ibatch]) is t.Tensor:
                para['coor'] = para['coorall'][ibatch]
            elif type(para['coorall'][ibatch]) is np.ndarray:
                para['coor'] = t.from_numpy(para['coorall'][ibatch])

            # check if atom specie is the same to the former
            self.runcal.dftbplus(para, ibatch, self.dir_ref)

            # calculate formation energy
            energy = write.Dftbplus(self.para).read_energy(
                    self.para, ibatch + 1, self.dir_ref)
            self.para['refenergy'][ibatch] = self.cal_for_energy(
                    energy[-1], para['coor'])
        self.save_dftbplus()

    def pre_dftbplus(self):
        """pre-processing DFTB+ calculations"""
        self.diredftbplus = os.getcwd() + '/ref/dftbplus'
        if not os.path.exists(self.diredftbplus):
            print('Warning: please make a folder "dftbplus", prepare input ',
                  'files, a script to run calculations and data processing')
        self.dir_ref = os.getcwd() + '/ref/dftbplus'
        os.system('rm ref/dftbplus/*.dat')

    def save_dftbplus(self):
        """Save files for reference calculations"""
        dftb = write.Dftbplus(self.para)
        homo_lumo = dftb.read_bandenergy(
                self.para, self.para['nfile'], self.dir_ref)
        dipole = dftb.read_dipole(
                self.para, self.para['nfile'], self.dir_ref, 'debye', 'eang')
        alpha_mbd = write.Dftbplus(self.para).read_alpha(
                    self.para, self.para['nfile'], self.dir_ref)
        self.para['refhomo_lumo'] = homo_lumo
        self.para['refdipole'] = dipole
        self.para['refalpha_mbd'] = alpha_mbd

        if self.para['task'] == 'opt':
            dire = '.data'
        elif self.para['task'] == 'test':
            dire = self.para['dire_data']
        os.system('rm ' + dire + '/*dftbplus.dat')
        self.save.save2D(self.para['refhomo_lumo'].detach().numpy(),
                         name='HLdftbplus.dat', dire=dire, ty='a')
        self.save.save2D(self.para['refdipole'].detach().numpy(),
                         name='dipdftbplus.dat', dire=dire, ty='a')
        self.save.save2D(self.para['refalpha_mbd'].detach().numpy(),
                         name='poldftbplus.dat', dire=dire, ty='a')
        self.save.save1D(self.para['refenergy'].detach().numpy(),
                         name='energydftbplus.dat', dire=dire, ty='a')
        self.save.save1D(np.asarray(self.para['natomall']),
                         name='natomdftbplus.dat', dire=dire, ty='a')

    def get_coor(self, ibatch):
        """get the ith coor according to data type"""
        if type(self.para['coorall'][ibatch]) is t.Tensor:
            self.para['coor'] = self.para['coorall'][ibatch][:, :]
        elif type(self.para['coorall'][ibatch]) is np.ndarray:
            self.para['coor'] = \
                t.from_numpy(self.para['coorall'][ibatch][:, :])

    def cal_optfor_energy(self, energy, coor):
        natom = self.para['natom']
        for iat in range(0, natom):
            idx = int(coor[iat, 0])
            iname = list(ATOMNUM.keys())[list(ATOMNUM.values()).index(idx)]
            energy = energy - DFTB_ENERGY[iname]
        return energy

    def mldftb(self, para):
        """Run DFTB-ML with various targets, e.g:

            optimize compression radius
            optimize integrals...
        """
        os.system('rm .data/*bp.dat')

        if para['Lml_HS']:
            self.ml_interp_hs(para)
        elif para['Lml_skf']:
            self.ml_compr(para)
        elif para['Lml_compr']:
            self.ml_compr_interp(para)
        elif para['Lml_acsf']:
            self.ml_acsf(para)

    def ml_interp_hs(self, para):
        '''DFTB optimization for given dataset'''

        para['cal_ref'] = False
        # optimize selected para to get opt target
        if para['Lml_HS'] and para['interptype'] == 'Bspline':
            para['cspline'] = Variable(para['cspl_rand'], requires_grad=True)
            optimizer = t.optim.SGD([para['cspline']], lr=5e-7)
        elif para['Lml_HS'] and para['interptype'] == 'Polyspline':
            para['splyall_rand'] = Variable(para['splyall_rand'],
                                            requires_grad=True)
            optimizer = t.optim.SGD([para['splyall_rand']], lr=5e-7)
        save = SaveData(para)
        coorall = para['coorall']

        # calculate one by one to optimize para
        for ibatch in range(self.nbatch):
            eigref = para['refhomo_lumo'][ibatch]
            if 'dipole' in range(0, self.nbatch):
                dipref = t.zeros((3), dtype=t.float64)
                dipref[:] = para['refdipole'][ibatch][:]

            # for each molecule we will run mlsteps
            if type(coorall[ibatch]) is t.tensor:
                para['coor'] = coorall[ibatch]
            elif type(coorall[ibatch]) is np.array:
                para['coor'] = t.from_numpy(coorall[ibatch])
            for it in range(0, para['mlsteps']):

                # dftb calculations (choose scc or nonscc)
                self.runcal.idftb_torchspline()
                eigval = para['eigenvalue']

                # define loss function
                criterion = t.nn.L1Loss(reduction='sum')
                loss = criterion(eigval, eigref)

                # clear gradients and define back propagation
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

                if it % para['save_steps'] == 0:
                    print('ibatch: {} steps: {} loss:  {}'.format(
                              ibatch, it, loss.item()))
                    print('eigval: {}, eigref: {}'.format(eigval, eigref))
                    save.save1D(eigval.detach().numpy(), name='eigbp.dat',
                                ty='a')
                    save.save2D(para['splyall_rand'].detach().numpy(),
                                name='splbp.dat', ty='a')
                    save.save1D(para['hammat'].detach().numpy(),
                                name='hambp.dat', ty='a')

    def ml_compr(self, para):
        """DFTB optimization of compression radius for given dataset."""
        # calculate one by one to optimize para
        para['nsteps'] = t.zeros((self.nbatch), dtype=t.float64)
        para['shape_pdos'] = t.zeros((self.nbatch, 2), dtype=t.int32)

        for ibatch in range(self.nbatch):
            para['ibatch'] = ibatch
            self.get_coor(ibatch)
            nat = self.para['coor'].shape[0]
            dftb_torch.Initialization(self.para)
            # get natom * natom * [ncompr, ncompr, 20] for interpolation DFTB
            self.slako.genskf_interp_ij()

            # if use ML predict initial compression radius from optimized data
            if para['Lopt_ml_compr']:
                nbatchml_ = int(para['opt_ml_step'] * self.nbatch)
                if ibatch == 0:
                    if para['opt_ml_all']:  # use training compR from beginning
                        npred_ = min(ibatch + int(nbatchml_), self.nbatch)
                        self.para['ntrain'] = ibatch
                        self.para['npred'] = npred_
                        interface.ML(para)
                    else:
                        self.genml.genml_init_compr()
                elif ibatch % nbatchml_ == 0:
                    self.para['dire_data'] = '.data'
                    npred_ = min(ibatch + int(nbatchml_), self.nbatch)
                    self.para['ntrain'] = ibatch
                    self.para['npred'] = npred_
                    interface.ML(para)
                elif ibatch > nbatchml_:
                    para['compr_init'] = self.para['compr_pred'][ibatch]
            else:
                self.genml.genml_init_compr()
            self.para['LreadSKFinterp'] = False  # read SKF list only once

            # homo_lumo_ref = para['refhomo_lumo'][ibatch]
            # dipref = para['refdipole'][ibatch]
            if para['LMBD_DFTB']:
                pol_ref = para['refalpha_mbd'][ibatch][:nat]
                volref = para['refvol'][ibatch][:nat]
            if 'hstable' in para['target']:
                hatableref = para['refhammat'][ibatch]
            elif 'eigval' in para['target']:
                eigvalref = para['refeigval'][ibatch]
            elif 'qatomall' in para['target']:
                qatomall_ref = para['refqatom'][ibatch]
            elif 'energy' in para['target']:
                energy_ref = para['refenergy'][ibatch]

            if not para['Lml_compr_global']:
                para['compr_ml'] = \
                    para['compr_init'].detach().clone().requires_grad_(True)
                # optimizer = t.optim.SGD([para['compr_ml']], lr=para['lr'])
                optimizer = t.optim.Adam([para['compr_ml']], lr=para['lr'])
            elif para['Lml_compr_global']:
                optimizer = t.optim.Adam([para['compr_all']], lr=para['lr'])

            # for each molecule we will run mlsteps
            savestep_ = 0
            for it in range(para['mlsteps']):
                if para['Lml_compr_global']:
                    self.genml.genml_init_compr()

                # dftb calculations (choose scc or nonscc)
                self.slako.genskf_interp_compr()
                self.runcal.idftb_torchspline()
                self.para['energy'] = self.cal_optfor_energy(
                        self.para['energy'], self.para['coor'])

                if self.para['loss_function'] == 'MSELoss':
                    criterion = t.nn.MSELoss(reduction='sum')
                elif self.para['loss_function'] == 'L1Loss':
                    criterion = t.nn.L1Loss(reduction='sum')

                if len(para['target']) == 1:
                    homo_lumo = para['homo_lumo']
                    dipole = para['dipole']
                    gap = t.abs(homo_lumo[1] - homo_lumo[0])
                    # gapref = t.abs(homo_lumo_ref[1] - homo_lumo_ref[0])
                    eigval = para['eigenvalue']
                    qatomall = para['qatomall']
                    energy = para['energy']
                    if 'homo_lumo' in para['target']:
                        loss = criterion(homo_lumo, homo_lumo_ref)
                    elif 'gap' in para['target']:
                        loss = criterion(gap, gapref)
                    elif 'dipole' in para['target']:
                        loss = criterion(dipole, dipref)
                    elif 'hstable' in para['target']:
                        hstable = para['hammat']
                        loss = criterion(hstable, hatableref)
                    elif 'eigval' in para['target']:
                        print("eigval, eigvalref", eigval, eigvalref)
                        loss = criterion(eigval, eigvalref)
                    elif 'qatomall' in para['target']:
                        loss = criterion(qatomall, qatomall_ref)
                    elif 'energy' in para['target']:
                        loss = criterion(energy, energy_ref)
                    elif 'polarizability' in para['target']:
                        pol = para['alpha_mbd']
                        loss = criterion(pol, pol_ref)
                    elif 'cpa' in para['target']:
                        cpa = para['cpa']
                        vol_ratio_ref = self.get_hirsh_vol_ratio(volref)
                        loss = criterion(cpa, vol_ratio_ref)
                    elif 'pdos' in para['target']:
                        pdosref = para['pdosdftbplus'][ibatch]
                        loss = criterion(para['pdos'], pdosref)
                        para['shape_pdos'][ibatch][0] = para['pdos'].shape[0]
                        para['shape_pdos'][ibatch][1] = para['pdos'].shape[1]
                elif len(para['target']) == 2:
                    if 'dipole' and 'polarizability' in para['target']:
                        dipole = para['dipole']
                        pol = para['alpha_mbd']
                        loss = 2 * criterion(pol, pol_ref) + \
                            criterion(dipole, dipref)

                # clear gradients and define back propagation
                optimizer.zero_grad()
                # loss.requres_grad = True
                loss.backward(retain_graph=True)
                optimizer.step()

                # save and print information
                if (it + 1) % para['save_steps'] == 0 or it == 0:
                    savestep_ += 1
                    if self.para['loss_function'] == 'MSELoss':
                        loss_ = t.sqrt(loss.detach()) / nat
                    elif self.para['loss_function'] == 'L1Loss':
                        loss_ = loss.detach() / nat
                    para["nsteps"][ibatch] = savestep_
                    print('-' * 100)
                    print('ibatch: {} steps: {} target: {}'.format(
                              ibatch + 1, it + 1, para['target']))
                    print('average loss: {}'.format(loss_))
                    print('compr_ml.grad', para['compr_ml'].grad.detach())
                    print("para['compr_ml']", para['compr_ml'])
                    '''print('homo_lumo: {}, homo_lumo_ref: {}'.format(
                        homo_lumo, homo_lumo_ref))
                    self.save.save1D(para['homo_lumo'].detach().numpy(),
                                     name='HLbp.dat', dire='.data', ty='a')
                    if 'dipole' in para['target']:
                        print('dipole: {}, dipref: {}'.format(dipole, dipref))
                    if 'energy' in para['target']:
                        print('energy: {}, energyref: {}'.format(
                                energy, energy_ref))'''
                    if para['LMBD_DFTB']:
                        self.save.save1D(para['alpha_mbd'].detach().numpy(),
                                         name='polbp.dat', dire='.data', ty='a')
                        self.save.save1D(para['cpa'].detach().numpy(),
                                         name='cpa.dat', dire='.data', ty='a')
                    if para['Lpdos']:
                        self.save.save2D(para['pdos'].detach().numpy(),
                                         name='pdosbp.dat', dire='.data', ty='a')
                    self.save.save1D(para['dipole'].detach().numpy(),
                                     name='dipbp.dat', dire='.data', ty='a')
                    self.save.save1D(para['hammat'].detach().numpy(),
                                     name='hambp.dat', dire='.data', ty='a')
                    self.save.save1D(para['compr_ml'].detach().numpy(),
                                     name='comprbp.dat', dire='.data', ty='a')
                    self.save.save1D(para['qatomall'].detach().numpy(),
                                     name='qatombp.dat', dire='.data', ty='a')
                    self.save.save1D(para['energy'].detach().numpy(),
                                     name='energybp.dat', dire='.data', ty='a')
                    self.save.save1D(loss_,
                                     name='lossbp.dat', dire='.data', ty='a')
                    print('-' * 100)
                # convergence or break condition
                if para['Lopt_step'] and it > para['opt_step_min']:
                    if loss_ < para['opt_ml_tol']:
                        print('loss is < {}'.format(para['opt_ml_tol']))
                        break
                if t.lt(para['compr_ml'], para['compr_min']).all():
                    print("there is compression R smaller than {}".format(
                        para['compr_min']))
                    break
                if t.gt(para['compr_ml'], para['compr_max']).all():
                    print("there is compression larger than {}".format(
                        para['compr_max']))
                    break
        self.save.save1D(para['nsteps'].detach().numpy(),
                         name='nsave.dat', dire='.data', ty='a')

    def ml_acsf(self, para):
        """DFTB optimization of ACSF parameters radius for given dataset."""
        # calculate one by one to optimize para
        ml = interface.ML(para)
        acsfml(self.para)  # get acsf_dim
        acsf_dim = self.para['acsf_dim']
        # define linear ML weight parameters
        acsf_w = t.ones((acsf_dim), dtype=t.float64) / 5
        bias_ = t.Tensor([2])
        bias = bias_.cpu().requires_grad_(True)
        acsf_weight = acsf_w.cpu().requires_grad_(True)
        optimizer = t.optim.SGD([acsf_weight, bias], lr=para['lr'])

        for it in range(para['mlsteps']):
            for ibatch in range(self.nbatch):
                para['ibatch'] = ibatch
                self.get_coor(ibatch)
                nat = self.para['coor'].shape[0]
                dftb_torch.Initialization(self.para)
                # get natom * natom * [ncompr, ncompr, 20] by interpolation
                self.slako.genskf_interp_ij()
                self.para['LreadSKFinterp'] = False  # read SKF list only once

                # get compr by weight parameter and atomic structure parameter
                ml.dataprocess_atom()  # get atomic structure parameter
                para['compr_ml'] = (para['acsf_mlpara'] @ acsf_weight) + bias

                homo_lumo_ref = para['refhomo_lumo'][ibatch]
                dipref = para['refdipole'][ibatch]
                if para['LMBD_DFTB'] and para['reference'] == 'aims':
                    pol_ref = para['refalpha_mbd'][ibatch][:nat]
                    volref = para['refvol'][ibatch][:nat]
                if 'hstable' in para['target']:
                    hatableref = para['refhammat'][ibatch]
                elif 'eigval' in para['target']:
                    eigvalref = para['refeigval'][ibatch]
                elif 'qatomall' in para['target']:
                    qatomall_ref = para['refqatom'][ibatch]
                elif 'energy' in para['target']:
                    energy_ref = para['refenergy'][ibatch]

                # dftb calculations (choose scc or nonscc)
                self.slako.genskf_interp_compr()
                self.runcal.idftb_torchspline()
                self.para['energy'] = self.cal_optfor_energy(
                        self.para['energy'], self.para['coor'])

                if self.para['loss_function'] == 'MSELoss':
                    criterion = t.nn.MSELoss(reduction='sum')
                elif self.para['loss_function'] == 'L1Loss':
                    criterion = t.nn.L1Loss(reduction='sum')

                homo_lumo = para['homo_lumo']
                dipole = para['dipole']
                gap = t.abs(homo_lumo[1] - homo_lumo[0])
                gapref = t.abs(homo_lumo_ref[1] - homo_lumo_ref[0])
                eigval = para['eigenvalue']
                qatomall = para['qatomall']
                energy = para['energy']
                if len(para['target']) == 1:
                    if 'homo_lumo' in para['target']:
                        loss = criterion(homo_lumo, homo_lumo_ref)
                    elif 'gap' in para['target']:
                        loss = criterion(gap, gapref)
                    elif 'dipole' in para['target']:
                        loss = criterion(dipole, dipref)
                    elif 'hstable' in para['target']:
                        hstable = para['hammat']
                        loss = criterion(hstable, hatableref)
                    elif 'eigval' in para['target']:
                        loss = criterion(eigval, eigvalref)
                    elif 'qatomall' in para['target']:
                        loss = criterion(qatomall, qatomall_ref)
                    elif 'energy' in para['target']:
                        loss = criterion(energy, energy_ref)
                    elif 'polarizability' in para['target']:
                        pol = para['alpha_mbd']
                        loss = criterion(pol, pol_ref)
                    elif 'cpa' in para['target']:
                        cpa = para['cpa']
                        vol_ratio_ref = self.get_hirsh_vol_ratio(volref)
                        loss = criterion(cpa, vol_ratio_ref)
                elif len(para['target']) == 2:
                    pol_ratio = para['polarizability_loss_ratio']
                    dip_ratio = para['polarizability_loss_ratio']
                    if 'dipole' and 'polarizability' in para['target']:
                        dipole = para['dipole']
                        pol = para['alpha_mbd']
                        loss = pol_ratio * criterion(pol, pol_ref) + \
                            criterion(dipole, dipref) * dip_ratio

                # clear gradients and define back propagation
                optimizer.zero_grad()
                # loss.requres_grad = True
                loss.backward(retain_graph=True)
                optimizer.step()

                # save and print information
                if (it + 1) % para['save_steps'] == 0 or it == 0:
                    if self.para['loss_function'] == 'MSELoss':
                        loss_ = t.sqrt(loss.detach()) / nat
                    elif self.para['loss_function'] == 'L1Loss':
                        loss_ = loss.detach() / nat
                    print('-' * 100)
                    print('ibatch: {} steps: {} target: {}'.format(
                              ibatch + 1, it + 1, para['target']))
                    print('average loss: {}'.format(loss_))
                    print('acsf_weight', acsf_weight.detach().numpy())
                    print('acsf_bias', bias.detach().numpy())
                    print("para['compr_ml']", para['compr_ml'].detach().numpy())
                    print('homo_lumo: {}, homo_lumo_ref: {}'.format(
                        homo_lumo.detach().numpy(), homo_lumo_ref))
                    self.save.save1D(homo_lumo.detach().numpy(),
                                     name='HLbp.dat', dire='.data', ty='a')
                    print('dipole: {}, dipref: {}'.format(dipole, dipref))
                    if 'energy' in para['target']:
                        print('energy: {}, energyref: {}'.format(
                                energy, energy_ref))
                    self.save.save1D(para['dipole'].detach().numpy(),
                                     name='dipbp.dat', dire='.data', ty='a')
                    self.save.save1D(para['hammat'].detach().numpy(),
                                     name='hambp.dat', dire='.data', ty='a')
                    self.save.save1D(para['compr_ml'].detach().numpy()[:nat],
                                     name='comprbp.dat', dire='.data', ty='a')
                    self.save.save1D(para['qatomall'].detach().numpy(),
                                     name='qatombp.dat', dire='.data', ty='a')
                    self.save.save1D(para['energy'].detach().numpy(),
                                     name='energybp.dat', dire='.data', ty='a')
                    self.save.save1D(para['alpha_mbd'].detach().numpy(),
                                     name='polbp.dat', dire='.data', ty='a')
                    self.save.save1D(para['cpa'].detach().numpy(),
                                     name='cpabp.dat', dire='.data', ty='a')
                    self.save.save1D(loss_,
                                     name='lossbp.dat', dire='.data', ty='a')
                    self.save.save1D(acsf_weight.detach().numpy(),
                                     name='weight.dat', dire='.data', ty='a')
                    self.save.save1D(bias.detach().numpy(),
                                     name='bias.dat', dire='.data', ty='a')
                    print('-' * 100)
                # convergence or break condition
                if t.lt(para['compr_ml'], para['compr_min']).all():
                    print("there is compression R smaller than {}".format(
                        para['compr_min']))
                    break
                if t.gt(para['compr_ml'], para['compr_max']).all():
                    print("there is compression larger than {}".format(
                        para['compr_max']))
                    break

    def get_hirsh_vol_ratio(self, volume):
        """Get Hirshfeld volume ratio."""
        natom = self.para["natom"]
        coor = self.para["coor"]
        volumeratio = t.zeros((len(volume)), dtype=t.float64)
        for iat in range(natom):
            idx = int(coor[iat, 0])
            iname = list(ATOMNUM.keys())[list(ATOMNUM.values()).index(idx)]
            volumeratio[iat] = volume[iat] / HIRSH_VOL[iname]
        return volumeratio

    def test_pred_compr(self, para):
        '''DFTB optimization for given dataset'''
        dire = self.para['dire_data']
        os.system('rm ' + dire + '/*pred.dat')
        self.nbatch = self.para['npred']

        for ibatch in range(self.nbatch):
            para['ibatch'] = ibatch
            self.get_coor(ibatch)
            dftb_torch.Initialization(self.para)

            if self.para['atomspecie'] != self.para['atomspecie_old']:
                self.genml.get_spllabel()
            else:
                self.para['LreadSKFinterp'] = False
            dftb_torch.Initialization(self.para)
            self.slako.genskf_interp_ij()
            self.genml.genml_init_compr()

            para['compr_ml'] = self.para['compr_pred'][ibatch]

            # dftb calculations (choose scc or nonscc)
            # slako.genskf_interp_r(para)
            self.slako.genskf_interp_compr()
            self.runcal.idftb_torchspline()
            homo_lumo = para['homo_lumo']
            dipole = para['dipole']
            gap = t.abs(homo_lumo[1] - homo_lumo[0])
            self.save.save1D(
                    homo_lumo.numpy(), name='eigpred.dat', dire=dire, ty='a')
            self.save.save1D(
                    dipole.numpy(), name='dippred.dat', dire=dire, ty='a')
            self.save.save1D(para['qatomall'].numpy(),
                             name='qatompred.dat', dire=dire, ty='a')
            self.save.save1D(self.para['alpha_mbd'].numpy(),
                             name='polpred.dat', dire=dire, ty='a')
            self.save.save1D(homo_lumo.numpy(),
                             name='HLpred.dat', dire=dire, ty='a')
            self.save.save1D(gap.numpy(),
                             name='gappred.dat', dire=dire, ty='a')
            self.save.save1D(para['compr_ml'].numpy(),
                             name='comprpred.dat', dire=dire, ty='a')

    def ml_compr_interp(self, para):
        """Test the interpolation gradients."""

        # select the first molecule
        if type(para['coorall'][0]) is t.Tensor:
            para['coor'] = para['coorall'][0][:, :]
        elif type(para['coorall'][0]) is np.ndarray:
            para['coor'] = t.from_numpy(para['coorall'][0][:, :])
        dftb_torch.Initialization(para)
        # interpskf(para)
        self.genml.genml_init_compr()
        para['compr_ml'] = para['compr_init'] + 1
        self.slako.genskf_interp_ij(para)
        self.slako.genskf_interp_r(para)
        hs_ref = para['h_s_all']
        print('compr ref', para['compr_ml'])

        para['compr_ml'] = Variable(para['compr_init'], requires_grad=True)
        optimizer = t.optim.SGD([para['compr_ml']], lr=1e-1)

        # for each molecule we will run mlsteps
        for it in range(0, para['mlsteps']):
            self.slako.genskf_interp_r(para)

            # define loss function
            criterion = t.nn.MSELoss(reduction='sum')
            # criterion = t.nn.L1Loss()
            loss = criterion(para['h_s_all'], hs_ref)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            print('compr_ml.grad', para['compr_ml'].grad)
            optimizer.step()

            if it % para['save_steps'] == 0:
                print('-' * 50)
                print('steps: {} loss: {} target: {}'.format(
                      it, loss.item(), para['target']))
                print("para['compr_ml']", para['compr_ml'])
                self.save.save1D(para['compr_ml'].detach().numpy(),
                                 name='comprbp.dat', dire='.data', ty='a')


def readhs_ij_line(iat, jat, para):
    '''
    deal with the integrals each line in .skf
    '''
    nameij, nameji = iat + jat, jat + iat
    ncompr = int(t.sqrt(t.Tensor([para['nfile_rall' + nameij]])))
    lmax = max(VAL_ORB[iat], VAL_ORB[jat])
    lmin = min(VAL_ORB[iat], VAL_ORB[jat])
    if lmax == 2 and lmin == 1:
        if VAL_ORB[iat] == 2:
            for ii in range(0, ncompr):
                for jj in range(0, ncompr):
                    para['hs_all_rall' + nameij][ii, jj, :, 8] = \
                        -para['hs_all_rall' + nameji][jj, ii, :, 8]
                    para['hs_all_rall' + nameij][ii, jj, :, 18] = \
                        -para['hs_all_rall' + nameji][jj, ii, :, 18]
        elif VAL_ORB[jat] == 2:
            for ii in range(0, ncompr):
                for jj in range(0, ncompr):
                    para['hs_all_rall' + nameji][ii, jj, :, 8] = \
                        -para['hs_all_rall' + nameij][jj, ii, :, 8]
                    para['hs_all_rall' + nameji][ii, jj, :, 18] = \
                        -para['hs_all_rall' + nameij][jj, ii, :, 18]
    elif lmax == 2 and lmin == 2:
        pass


class GenMLPara:
    """Aims to get parameters for ML.

    genenvir: atomic environment parameters
    get_spllabel: get how many row lines of c parameters in bspline

    """

    def __init__(self, para):
        self.para = para

    def genmlpara0(self, para):
        pass

    def genml_init_compr(self):
        atomname = self.para['atomnameall']
        natom = self.para['natom']
        init_compr = t.zeros((natom), dtype=t.float64)
        icount = 0
        if self.para['Lml_compr_global']:
            for iatom in atomname:
                if iatom == 'H':
                    init_compr[icount] = self.para['compr_all'][0]
                elif iatom == 'C':
                    init_compr[icount] = self.para['compr_all'][5]
                icount += 1
            self.para['compr_ml'] = init_compr
        else:
            for iatom in atomname:
                init_compr[icount] = self.para[iatom + '_init_compr']
                icount += 1
            self.para['compr_init'] = init_compr

    def get_spllabel(self):
        """Check if atom specie is the same, if not, then update."""
        if self.para['atomspecie'] != self.para['atomspecie_old']:
            for iatomspecie in self.para['atomspecie']:
                if iatomspecie not in self.para['atomspecie_old']:
                    self.para['atomspecie_old'].append(iatomspecie)
            if self.para['Lml_HS']:
                h_spl_num = 0
                self.para['spl_label'] = []
                for iatom in self.para['atomspecie_old']:
                    for jatom in self.para['atomspecie_old']:
                        nameij = iatom + jatom
                        h_spl_num += HNUM[nameij]
                        if HNUM[nameij] > 0:
                            self.para['spl_label'].append(nameij)
                            self.para['spl_label'].append(h_spl_num)
                            self.para['spl_label'].append(HNUM[nameij])
                self.para['h_spl_num'] = h_spl_num

    def get_specie_label(self):
        """Check if atom specie is the same, if not, then update."""
        h_spl_num = 0
        self.para['spl_label'] = []
        for ispecie in self.para['atomspecie']:
            for jspecie in self.para['atomspecie']:
                nameij = ispecie + jspecie
                h_spl_num += HNUM[nameij]
                if HNUM[nameij] > 0:
                    self.para['spl_label'].append(nameij)
                    self.para['spl_label'].append(h_spl_num)
                    self.para['spl_label'].append(HNUM[nameij])
        print('initial H-table has {} rows'.format(h_spl_num))
        self.para['h_spl_num'] = h_spl_num


class RunCalc:
    """Run different DFT(B) calculations.

    with both ASE interface or code in write_output.py
    """

    def __init__(self, para):
        self.para = para

    def aims(self, para, ibatch, dire):
        """DFT means FHI-aims here."""
        coor = para['coor']
        self.para['natom'] = int(self.para['natomall'][ibatch])
        write.FHIaims(para).geo_nonpe_hdf(para, ibatch, coor[:, 1:])
        os.rename('geometry.in.{}'.format(ibatch), 'ref/aims/geometry.in')
        os.system('bash ' + dire + '/run.sh ' + dire + ' ' + str(ibatch) +
                  ' ' + str(self.para['natom']))

    def dftbplus(self, para, ibatch, dire):
        """Perform DFTB+ to calculate."""
        dftb = write.Dftbplus(para)
        coor = para['coor']
        self.para['natom'] = int(self.para['natomall'][ibatch])
        scc = para['scc']
        dftb.geo_nonpe(dire, coor)
        dftb.write_dftbin(self.para, dire, scc, coor)
        os.system('bash ' + dire + '/run.sh ' + dire + ' ' + str(ibatch))

    def dftbtorchrun(self, para, coor, DireSK):
        """Perform DFTB_python with reading SKF."""
        para['coor'] = t.from_numpy(coor)
        dipolemall = para['dipolemall']
        eigvalall = para['eigvalall']
        dipolem, eigval = dftb_torch.main(para)
        dipolemall.append(dipolem)
        eigvalall.append(eigval)
        para['dipolemall'] = dipolemall
        para['eigvalall'] = eigvalall
        return para

    def idftb_torchspline(self):
        """Perform DFTB_python with integrals."""
        # dftb_torch.Initialization(self.para).gen_sk_matrix(self.para)
        slakot.SKTran(self.para)
        dftb_torch.Rundftbpy(self.para)


if __name__ == "__main__":
    """Main function for optimizing DFTB parameters, testing DFTB."""
    t.autograd.set_detect_anomaly(True)
    t.set_printoptions(precision=15)
    para = {}
    parser.parser_cmd_args(para)
    if para['task'] == 'opt':
        opt(para)
    elif para['task'] == 'test':
        testml(para)
    elif para['task'] == 'envpara':
        initpara.init_dftb_ml(para)
        interface.get_env_para()
