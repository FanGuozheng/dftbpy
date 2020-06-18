#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 10:59:55 2019
@author: guozheng
"""
import os
import json
from collections import Counter
import numpy as np
import sys
from torch.autograd import Variable
import torch as t
import data.pyanitools as pya
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import write_output as write
import lattice_cell
import dftbtorch.dftb_torch as dftb_torch
import dftbtorch.slakot as slakot
import plot
from readt import ReadInt
import init_parameter as initpara
Directory = '/home/gz_fan/Documents/ML/dftb/ml'
DireSK = '/home/gz_fan/Documents/ML/dftb/slko'
ATOMIND = {'H': 1, 'HH': 2, 'HC': 3, 'C': 4, 'CH': 5, 'CC': 6}
ATOMNUM = {'H': 1, 'C': 6, 'N': 7, 'O': 8}
HNUM = {'CC': 4, 'CH': 2, 'CO': 4, 'HC': 0,  'HH': 1, 'HO': 2, 'OC': 0,
        'OH': 0, 'OO': 4}
COMP_R = {'H': 3.0, 'C': 3.0}
VAL_ORB = {"H": 1, "C": 2, "N": 2, "O": 2, "Ti": 3}
AIMS_ENERGY = {"H": -0.45891649, "C": -37.77330663, "N": -54.46973501,
               "O": -75.03140052}
DFTB_ENERGY = {"H": -0.238600544, "C": -1.398493891, "N": -2.0621839400,
               "O": -3.0861916005}
HIRSH_VOL = {"H": 10.31539447, "C": 38.37861207}


def optml(para):
    """'main function for DFTB-ML optimization"""
    # get the default para for dftb and ML, these para will maintain unchanged
    initpara.init_dftb_ml(para)

    # load dataset, here is hdf type
    LoadData(para)

    # run reference calculations, either dft or dftb
    runml = RunML(para)
    para['nfile'] = int(para['n_dataset'][0])
    runml.ref()

    # run dftb in ML process
    runml.mldftb(para)

    # plot data from ML
    plot.plot_main(para)


def testml(para):
    """'main function for testing DFTB-ML"""
    get_env_para(para)

    # get the default para for testing DFTB-ML
    initpara.init_dftb_ml(para)

    # load dataset, here is hdf type
    LoadData(para)

    # if refrence is not DFTB+, run and add DFTB+ results
    ML(para)

    runml = RunML(para)
    if int(para['n_dataset'][0]) < int(para['n_test'][0]):
        para['nfile'] = int(para['n_test'][0])
        runml.ref()

    if para['ref'] != 'dftbplus':
        para['ref'] = 'dftbplus'
        if int(para['n_dataset'][0]) == int(para['n_test'][0]):
            para['nfile'] = int(para['n_test'][0])
            runml.ref()
        elif int(para['n_dataset'][0]) < int(para['n_test'][0]):
            para['nfile'] = int(para['n_test'][0])
            runml.ref()

    runml.test_pred_compr(para)

    plot.plot_dip_pred(
            para, para['dire_data'], aims='aims', dftbplus='dftbplus')


def get_env_para(para):
    """get the environmental parameters"""
    dire_ = para['dire_data']
    os.system('rm ' + dire_ + '/env_rad.dat')
    os.system('rm ' + dire_ + '/env_ang.dat')

    genpara = GenMLPara(para)
    initpara.init_dftb_ml(para)
    load = LoadData(para)
    save = SaveData(para)

    load.loadhdfdata()
    if len(para['n_dataset']) == 1:
        nbatch = max(int(para['n_dataset'][0]), int(para['n_test'][0]))
    print('begin to calculate environmental parameters')
    symbols = para['symbols']
    for ibatch in range(0, nbatch):
        if type(para['coorall'][ibatch]) is np.array:
            coor = t.from_numpy(para['coorall'][ibatch])
        elif type(para['coorall'][ibatch]) is t.Tensor:
            coor = para['coorall'][ibatch]
        para['coor'] = coor[:]
        genpara.genenvir(nbatch, para, ibatch, coor, symbols)

        ReadInt(para).cal_coor()
        iang = para['ang_paraall'][ibatch]
        irad = para['rad_paraall'][ibatch]

        save.save1D(iang, name='env_ang.dat', dire=dire_, ty='a')
        save.save1D(irad, name='env_rad.dat', dire=dire_, ty='a')


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
        self.slako = slakot.SlaKo(self.para)
        self.genml = GenMLPara(self.para)
        self.runcal = RunCalc(self.para)

    def ref(self):
        """Run different reference calculations according to reference type"""

        os.system('rm .data/*.dat')

        self.nbatch = self.para['nfile']
        # self.para['nfile'] = self.nbatch
        self.para['refhomo_lumo'] = t.zeros((self.nbatch, 2), dtype=t.float64)
        self.para['refenergy'] = t.zeros((self.nbatch), dtype=t.float64)
        self.para['refdipole'] = t.zeros((self.nbatch, 3), dtype=t.float64)
        self.para['natomall'] = t.zeros((self.nbatch), dtype=t.float64)
        self.para['specieall'] = []
        self.para['refeigval'] = []

        if self.para['ref'] == 'dftb':
            self.dftb_ref()
        elif self.para['ref'] == 'aims':
            self.aims_ref(self.para)
        elif self.para['ref'] == 'dftbplus':
            self.dftbplus_ref(self.para)

    def dftb_ref(self):
        """Calculate reference (DFTB_torch)"""

        # self.para['cal_ref'] = True
        os.system('rm *.dat')

        for ibatch in range(0, self.para['nfile']):
            # get coor and related geometry information
            self.get_coor(ibatch)

            if self.para['Lml_skf']:
                # if read all .skf and build [N_ij, N_R1, N_R2, 20] matrix
                if self.para['atomspecie'] != self.para['atomspecie_old']:
                    self.genml.get_spllabel()
                else:
                    self.para['LreadSKFinterp'] = False
                dftb_torch.Initialization(self.para)
                self.get_compr_specie()

                self.para['compr_ml'] = self.para['compr_init'] - 1
                self.slako.genskf_interp_compr()
                self.runcal.idftb_torchspline()

            elif self.para['Lml_HS']:
                dftb_torch.Initialization(self.para)
                self.genml.get_specie_label()
                dftb_torch.Initialization(self.para).form_sk_spline()
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

            # read from .skf then use interp to generate H(S) mat for DFTB
            self.para['refhomo_lumo'][ibatch] = self.para['homo_lumo']
            self.para['refeigval'].append(self.para['eigenvalue'])
            self.para['refenergy'][ibatch] = self.para['energy']
            self.para['refdipole'][ibatch] = self.para['dipole'][:]
            self.para['natomall'][ibatch] = self.para['natom']
            self.save.save1D(self.para['eigenvalue'].detach().numpy(),
                             name='eigvalref.dat', dire='.data', ty='a')
        self.save.save1D(self.para['natomall'].detach().numpy(),
                         name='natom.dat', dire='.data', ty='a')
        self.save.save2D(self.para['refhomo_lumo'].detach().numpy(),
                         name='HLref.dat', dire='.data', ty='w')
        self.save.save2D(self.para['refdipole'].detach().numpy(),
                         name='dipref.dat', dire='.data', ty='w')
        self.save.save1D(self.para['refenergy'].detach().numpy(),
                         name='energyref.dat', dire='.data', ty='w')

    def aims_ref(self, para):
        """Calculate reference (FHI-aims)"""
        self.pre_aims()

        for ibatch in range(0, self.nbatch):
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
            self.para['natomall'][ibatch] = self.para['natom']
        self.save_aims()

    def pre_aims(self):
        """pre-processing aims calculations"""
        self.direaims = os.getcwd() + '/ref/aims'
        if not os.path.exists(self.direaims):
            print('Warning: please make a folder "aims", prepare input files ',
                  'and a script which run calculations and extract results')
        self.dir_ref = os.getcwd() + '/ref/aims'
        os.system('rm ref/aims/*.dat')

    def save_aims(self):
        """save files for reference calculations"""
        homo_lumo = write.FHIaims(self.para).read_bandenergy(
                self.para, self.para['nfile'], self.dir_ref)
        dipole = write.FHIaims(self.para).read_dipole(
                self.para, self.para['nfile'], self.dir_ref, 'eang', 'eang')
        alpha_mbd = write.FHIaims(self.para).read_alpha(
                    self.para, self.para['nfile'], self.dir_ref)
        refvol = write.FHIaims(self.para).read_hirshfeld_vol(
                    self.para, self.para['nfile'], self.dir_ref)
        '''energy = write.FHIaims(self.para).read_energy(
                 self.para, self.para['nfile'], self.dir_ref)'''
        self.para['refhomo_lumo'] = homo_lumo
        self.para['refdipole'] = dipole
        self.para['refalpha_mbd'] = alpha_mbd
        self.para['refvol'] = refvol

        if para['task'] == 'optml':
            dire = '.data'
        elif para['task'] == 'test':
            dire = para['dire_data']
        os.system('rm ' + dire + '/dipaims.dat')
        os.system('rm ' + dire + '/natom.dat')
        os.system('rm ' + dire + '/energyaims.dat')

        os.system('mv ' + os.path.join(self.direaims, 'bandenergy.dat') +
                  ' .data/HLaims.dat')
        os.system('mv ' + os.path.join(self.direaims, 'dip.dat') +
                  ' .data/dipaims.dat')
        os.system('mv ' + os.path.join(self.direaims, 'pol.dat') +
                  ' .data/polaims.dat')
        os.system('mv ' + os.path.join(self.direaims, 'vol.dat') +
                  ' .data/volaims.dat')
        self.save.save1D(self.para['refdipole'].detach().numpy(),
                         name='dipaims.dat', dire=dire, ty='a')
        self.save.save1D(self.para['refenergy'].detach().numpy(),
                         name='energyaims.dat', dire=dire, ty='a')
        self.save.save1D(self.para['natomall'].detach().numpy(),
                         name='natom.dat', dire=dire, ty='a')

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

        for ibatch in range(0, self.nbatch):
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
            self.para['natomall'][ibatch] = self.para['natom']

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
        # the run.sh will read debye dipolemoment from detailed.out
        dipole = dftb.read_dipole(
                self.para, self.para['nfile'], self.dir_ref, 'debye', 'eang')
        self.para['refhomo_lumo'] = homo_lumo
        self.para['refdipole'] = dipole

        if para['task'] == 'optml':
            dire = '.data'
        elif para['task'] == 'test':
            dire = para['dire_data']
        os.system('rm ' + dire + '/dipdftbplus.dat')
        os.system('rm ' + dire + '/natom.dat')
        os.system('rm ' + dire + '/energydftbplus.dat')

        os.system('cp ' + os.path.join(self.diredftbplus, 'bandenergy.dat ') +
                  dire + '/HLdftbplus.dat')
        '''os.system('cp ' + os.path.join(self.diredftbplus, 'dip.dat ') +
                  dire + '/dipdftbplus.dat')'''
        self.save.save1D(self.para['refdipole'].detach().numpy(),
                         name='dipdftbplus.dat', dire=dire, ty='a')
        self.save.save1D(self.para['refenergy'].detach().numpy(),
                         name='energydftbplus.dat', dire=dire, ty='a')
        self.save.save1D(self.para['natomall'].detach().numpy(),
                         name='natom.dat', dire=dire, ty='a')

    def dftb(self, para):
        '''this function is for dftb calculation'''
        if para['cal_ref']:
            para['cal_ref'] = False
            nbatch = para['nfile']
            coorall = para['coorall']
            save = SaveData(para)
            if para['Lml_HS'] and not para['Lml_skf']:
                save.save2D(para['splyall_rand'].detach().numpy(),
                            name='splref.dat', ty='a')
            for ibatch in range(0, nbatch):
                para['coor'] = t.from_numpy(coorall[ibatch])
                self.runcal.idftb_torchspline()
                eigval = para['homo_lumo']

                # save data
                save.save1D(eigval.detach().numpy(), name='eigref.dat', ty='a')
                save.save1D(para['hammat'].detach().numpy(),
                            name='hamref.dat', ty='a')
        elif not para['cal_ref']:
            nbatch = para['nfile']
            coorall = para['coorall']
            save = SaveData(para)
            if para['Lml_HS'] and not para['Lml_skf']:
                save.save2D(para['splyall_rand'].detach().numpy(),
                            name='splref.dat', ty='a')

            # calculate one by one to optimize para
            for ibatch in range(0, nbatch):
                para['coor'] = t.from_numpy(coorall[ibatch])
                self.runcal.idftb_torchspline()
                eigval = para['homo_lumo']

                # save data
                save.save1D(eigval.detach().numpy(), name='eigref.dat', ty='a')
                save.save1D(para['hammat'].detach().numpy(),
                            name='hamref.dat', ty='a')

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

    def get_compr_specie(self):
        '''
        1. read all .skf with various compR for all apecie
        2. get the itegrals matrix: natom * natom * [ncompr, ncompr, 20]
        3. get the initial compR
        '''
        self.slako.genskf_interp_ij()
        self.genml.genml_init_compr()

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
        for ibatch in range(0, self.nbatch):
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
        """DFTB optimization of compression radius for given dataset"""
        # calculate one by one to optimize para
        for ibatch in range(0, self.nbatch):
            para['ibatch'] = ibatch
            self.get_coor(ibatch)
            nat = self.para['coor'].shape[0]

            if self.para['atomspecie'] != self.para['atomspecie_old']:
                self.genml.get_spllabel()
            else:
                self.para['LreadSKFinterp'] = False
            dftb_torch.Initialization(self.para)
            self.get_compr_specie()

            # build the ref data
            if any(x in para['target'] for x in ['homo_lumo', 'gap']):
                homo_lumo_ref = t.zeros((2), dtype=t.float64)
                homo_lumo_ref[:] = para['refhomo_lumo'][ibatch]
            elif 'dipole' in para['target']:
                dipref = t.zeros((3), dtype=t.float64)
                dipref[:] = para['refdipole'][ibatch]
            elif 'hstable' in para['target']:
                hatableref = para['refhammat'][ibatch]
            elif 'eigval' in para['target']:
                eigvalref = para['refeigval'][ibatch]
            elif 'qatomall' in para['target']:
                qatomall_ref = para['refqatom'][ibatch]
            elif 'energy' in self.para['target']:
                energy_ref = self.para['refenergy'][ibatch]
            elif 'polarizability' in self.para['target']:
                pol_ref = self.para['refalpha_mbd'][ibatch][:nat]
            elif 'cpa' in self.para['target']:
                volref = self.para['refvol'][ibatch][:nat]

            if not para['Lml_compr_global']:
                para['compr_ml'] = \
                    para['compr_init'].detach().clone().requires_grad_(True)
                optimizer = t.optim.SGD([para['compr_ml']], lr=para['lr'])
            elif para['Lml_compr_global']:
                optimizer = t.optim.SGD([para['compr_all']], lr=para['lr'])

            # for each molecule we will run mlsteps
            for it in range(0, para['mlsteps']):
                if para['Lml_compr_global']:
                    self.genml.genml_init_compr()

                # dftb calculations (choose scc or nonscc)
                # slako.genskf_interp_r(para)
                self.slako.genskf_interp_compr()
                self.runcal.idftb_torchspline()
                self.para['energy'] = self.cal_optfor_energy(
                        self.para['energy'], self.para['coor'])

                # define loss function with different traget
                criterion = t.nn.L1Loss(reduction='sum')
                homo_lumo = para['homo_lumo']
                if 'homo_lumo' in para['target']:
                    loss = criterion(homo_lumo, homo_lumo_ref)
                if 'gap' in para['target']:
                    gap = t.abs(homo_lumo[1] - homo_lumo[0])
                    gapref = t.abs(homo_lumo_ref[1] - homo_lumo_ref[0])
                    loss = criterion(gap, gapref)
                elif 'dipole' in para['target']:
                    dipole = para['dipole']
                    loss = criterion(dipole, dipref)
                elif 'hstable' in para['target']:
                    hstable = para['hammat']
                    loss = criterion(hstable, hatableref)
                elif 'eigval' in para['target']:
                    eigval = para['eigenvalue']
                    loss = criterion(eigval, eigvalref)
                elif 'qatomall' in para['target']:
                    qatomall = para['qatomall']
                    loss = criterion(qatomall, qatomall_ref)
                elif 'energy' in para['target']:
                    energy = para['energy']
                    loss = criterion(energy, energy_ref)
                elif 'polarizability' in para['target']:
                    pol = para['alpha_ts']
                    loss = criterion(pol, pol_ref)
                elif 'cpa' in para['target']:
                    cpa = para['OnsitePopulation']
                    vol_ratio_ref = self.get_hirsh_vol_ratio(volref)
                    loss = criterion(cpa, vol_ratio_ref)

                # clear gradients and define back propagation
                optimizer.zero_grad()
                # loss.requres_grad = True
                loss.backward(retain_graph=True)
                print('compr_ml.grad', para['compr_ml'].grad)
                optimizer.step()

                # save and print information
                if it % para['save_steps'] == 0:
                    print('-' * 50)
                    print('ibatch: {} steps: {} loss: {} target: {}'.format(
                              ibatch, it, loss.item(), para['target']))
                    print("para['compr_ml']", para['compr_ml'])
                    if any(x in para['target'] for x in ['homo_lumo', 'gap']):
                        print('homo_lumo: {}, homo_lumo_ref: {}'.format(
                                homo_lumo, homo_lumo_ref))
                    self.save.save1D(homo_lumo.detach().numpy(),
                                     name='HLbp.dat', dire='.data', ty='a')
                    if 'dipole' in para['target']:
                        print('dipole: {}, dipref: {}'.format(dipole, dipref))
                    if 'energy' in para['target']:
                        print('energy: {}, energyref: {}'.format(
                                energy, energy_ref))
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
                    self.save.save1D(para['alpha_mbd'].detach().numpy(),
                                     name='pol.dat', dire='.data', ty='a')

    def get_hirsh_vol_ratio(self, volume):
        """Get Hirshfeld volume ratio."""
        natom = self.para["natom"]
        coor = self.para["coor"]
        for iat in range(natom):
            idx = int(coor[iat, 0])
            iname = list(ATOMNUM.keys())[list(ATOMNUM.values()).index(idx)]
            volume[iat] = volume[iat] / HIRSH_VOL[iname]
        return volume

    def test_pred_compr(self, para):
        '''DFTB optimization for given dataset'''
        self.nbatch = int(self.para['n_test'][0])
        dire = self.para['dire_data']
        os.system('rm ' + dire + '/eigpred.dat')
        os.system('rm ' + dire + '/dippred.dat')
        os.system('rm ' + dire + '/qatompred.dat')

        for ibatch in range(0, self.nbatch):
            para['ibatch'] = ibatch
            self.get_coor(ibatch)
            dftb_torch.Initialization(self.para)

            if self.para['atomspecie'] != self.para['atomspecie_old']:
                self.genml.get_spllabel()
            else:
                self.para['LreadSKFinterp'] = False
            dftb_torch.Initialization(self.para)
            self.get_compr_specie()

            para['compr_ml'] = self.para['compr_pred'][ibatch]

            # dftb calculations (choose scc or nonscc)
            # slako.genskf_interp_r(para)
            self.slako.genskf_interp_compr()
            self.runcal.idftb_torchspline()
            homo_lumo = para['homo_lumo']
            dipole = para['dipole']
            self.save.save1D(
                    homo_lumo.numpy(), name='eigpred.dat', dire=dire, ty='a')
            self.save.save1D(
                    dipole.numpy(), name='dippred.dat', dire=dire, ty='a')
            self.save.save1D(para['qatomall'].numpy(),
                             name='qatompred.dat', dire=dire, ty='a')

    def ml_compr_interp(self, para):
        '''test the interpolation gradients'''

        # select the first molecule
        if type(para['coorall'][0]) is t.Tensor:
            para['coor'] = para['coorall'][0][:, :]
        elif type(para['coorall'][0]) is np.ndarray:
            para['coor'] = t.from_numpy(para['coorall'][0][:, :])
        dftb_torch.Initialization(para)
        interpskf(para)
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

    def genenvir(self, nbatch, para, ifile, coor, symbols):
        rcut = para['rcut']
        r_s = para['r_s']
        eta = para['eta']
        tol = para['tol']
        zeta = para['zeta']
        lamda = para['lambda']
        rad_para = lattice_cell.rad_molecule2(coor, rcut, r_s, eta, tol,
                                              symbols)
        ang_para = lattice_cell.ang_molecule2(coor, rcut, r_s, eta, zeta,
                                              lamda, tol, symbols)
        para['ang_paraall'].append(ang_para)
        para['rad_paraall'].append(rad_para)
        '''for ispecie in speciedict:
            nbatch += speciedict[ispecie]
            for jspecie in speciedict:
                nbatch += speciedict[ispecie]*speciedict[jspecie]'''
        natom = len(coor)
        nbatch += natom * (natom+1)
        return para, nbatch

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


class LoadData:
    '''
    the input:
        dataType: the data type, hdf, json...
        hdf_num: how many dataset in one hdf file
    the output:
        coorall: all the coordination of molecule
        symbols: all the atoms in each molecule
        specie: the specie in molecule
        speciedict: Counter(symbols)
    '''

    def __init__(self, para):
        self.para = para
        if self.para['dataType'] == 'hdf':
            self.loadhdfdata()
        if self.para['dataType'] == 'json':
            self.load_json_data()

    def loadhdfdata(self):
        """"Load the data from hdf type input files."""
        ntype = self.para['hdf_num']
        hdf5filelist = self.para['hdffile']
        icount = 0
        self.para['coorall'] = []
        for hdf5file in hdf5filelist:
            adl = pya.anidataloader(hdf5file)
            for data in adl:
                icount += 1
                if icount == ntype:
                    for icoor in data['coordinates']:
                        row, col = np.shape(icoor)[0], np.shape(icoor)[1]
                        coor = t.zeros((row, col + 1), dtype=t.float64)
                        for iat in range(0, len(data['species'])):
                            coor[iat, 0] = ATOMNUM[data['species'][iat]]
                            coor[iat, 1:] = t.from_numpy(icoor[iat, :])
                        self.para['coorall'].append(coor)
                    symbols = data['species']
                    specie = set(symbols)
                    speciedict = Counter(symbols)
                    self.para['symbols'] = symbols
                    self.para['specie'] = specie
                    self.para['atomspecie'] = []
                    [self.para['atomspecie'].append(ispecie) for
                     ispecie in specie]
                    self.para['speciedict'] = speciedict

    def load_json_data(self):
        """"Load the data from json type input files."""
        dire = self.para['pythondata_dire']
        filename = self.para['pythondata_file']
        self.para['coorall'] = []

        with open(os.path.join(dire, filename), 'r') as fp:
            fpinput = json.load(fp)

            if 'symbols' in fpinput['general']:
                self.para['symbols'] = fpinput['general']['symbols'].split()
                self.para['speciedict'] = Counter(self.para['symbols'])

            specie = set(self.para['symbols'])
            self.para['specie'] = specie
            self.para['atomspecie'] = []
            [self.para['atomspecie'].append(ispe) for ispe in specie]

            for iname in fpinput['geometry']:
                icoor = fpinput['geometry'][iname]
                self.para['coorall'].append(t.from_numpy(np.asarray(icoor)))

    def loadrefdata(self, ref, dire, nfile):
        """"Load the data from DFT calculations."""
        if ref == 'aims':
            newdire = os.path.join(Directory, dire)
            if os.path.exists(os.path.join(newdire,
                                           'bandenergy.dat')):
                refenergy = Variable(t.zeros((nfile, 2), dtype=t.float64),
                                     requires_grad=False)
                fpenergy = open(os.path.join(newdire, 'bandenergy.dat'), 'r')
                for ifile in range(0, nfile):
                    energy = np.fromfile(fpenergy, dtype=float,
                                         count=3, sep=' ')
                    refenergy[ifile, :] = t.from_numpy(energy[1:])
        elif ref == 'dftbrand':
            newdire = os.path.join(Directory, dire)
            if os.path.exists(os.path.join(newdire, 'bandenergy.dat')):
                refenergy = Variable(t.zeros((nfile, 2), dtype=t.float64),
                                     requires_grad=False)
                fpenergy = open(os.path.join(newdire, 'bandenergy.dat'), 'r')
                for ifile in range(0, nfile):
                    energy = np.fromfile(fpenergy, dtype=float,
                                         count=3, sep=' ')
                    refenergy[ifile, :] = t.from_numpy(energy[1:])
        elif ref == 'VASP':
            pass
        return refenergy

    def loadenv(ref, DireSK, nfile, natom):
        """Load the data of atomic environment parameters."""
        if os.path.exists(os.path.join(DireSK, 'rad_para.dat')):
            rad = np.zeros((nfile, natom))
            fprad = open(os.path.join(DireSK, 'rad_para.dat'), 'r')
            for ifile in range(0, nfile):
                irad = np.fromfile(fprad, dtype=float, count=natom, sep=' ')
                rad[ifile, :] = irad[:]
        if os.path.exists(os.path.join(DireSK, 'ang_para.dat')):
            ang = np.zeros((nfile, natom))
            fpang = open(os.path.join(DireSK, 'ang_para.dat'), 'r')
            for ifile in range(0, nfile):
                iang = np.fromfile(fpang, dtype=float, count=natom, sep=' ')
                ang[ifile, :] = iang[:]
        return rad, ang


class RunCalc:
    """Run different DFT(B) calculations.

    with both ASE interface or code in write_output.py
    """

    def __init__(self, para):
        self.para = para

    def aims(self, para, ibatch, dire):
        """DFT means FHI-aims here."""
        coor = para['coor']
        self.para['natom'] = coor.shape[0]
        natom = np.shape(coor)[0]
        write.FHIaims(para).geo_nonpe_hdf(para, ibatch, coor[:, 1:])
        os.rename('geometry.in.{}'.format(ibatch), 'ref/aims/geometry.in')
        os.system('bash ' + dire + '/run.sh ' + dire + ' ' + str(ibatch) +
                  ' ' + str(natom))

    def dftbplus(self, para, ibatch, dire):
        """Perform DFTB+ to calculate."""
        dftb = write.Dftbplus(para)
        coor = para['coor']
        self.para['natom'] = coor.shape[0]
        specie = para['specie']
        scc = para['scc']

        dftb.geo_nonpe(dire, coor, specie)
        dftb.write_dftbin(dire, scc, coor, specie)
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


class SaveData:
    """Simple code dor saving data.

    data is numpy type matrix
    blank defines where we'll write blank line
    name is the name of the saved file
    save2D will save file line by line
    savetype: 'a': appendix; 'w': save as a new file (replace the old)
    """

    def __init__(self, para):
        self.para = para

    def save1D(self, data, name, blank='lower', dire=None, ty='w'):
        """Save 1D numpy array or tensor."""
        if len(data.shape) == 0:
            data = data.reshape(1)
        if dire is None:
            newdire = os.getcwd()
        else:
            newdire = dire
        with open(os.path.join(newdire, name), ty) as fopen:
            if blank == 'upper':
                fopen.write('\n')
            np.savetxt(fopen, data, newline=" ")
            fopen.write('\n')
            if blank == 'lower':
                fopen.write('\n')

    def save2D(self, data, name, blank='lower', dire=None, ty='w'):
        """Save 2D numpy array or tensor."""
        if dire is None:
            newdire = os.getcwd()
        else:
            newdire = dire
        with open(os.path.join(newdire, name), ty) as fopen:
            for idata in data:
                if blank == 'upper':
                    fopen.write('\n')
                np.savetxt(fopen, idata, newline=" ")
                fopen.write('\n')
                if blank == 'lower':
                    fopen.write('\n')

    def save_envir(para, Directory):
        """Save atomic environment data."""
        ang_paraall = para['ang_paraall']
        rad_paraall = para['rad_paraall']
        with open(os.path.join(Directory, 'rad_para.dat'), 'w') as fopen:
            np.savetxt(fopen, rad_paraall, fmt="%s", newline=' ')
            fopen.write('\n')
        with open(os.path.join(Directory, 'ang_para.dat'), 'w') as fopen:
            np.savetxt(fopen, ang_paraall, fmt="%s", newline=' ')
            fopen.write('\n')


class Read:
    """Simple reading code.

    """

    def __init__(self, para):
        self.para = para

    def read1d(self, dire, name, number, outtype='torch'):
        fp = open(os.path.join(dire, name), 'r')
        data = np.zeros(number)
        data[:] = np.fromfile(fp, dtype=int, count=number, sep=' ')
        if outtype == 'torch':
            return t.from_numpy(data)
        elif outtype == 'numpy':
            return data


class ML:
    """Machine learning with optimized data.

    process data
    perform ML prediction
    """

    def __init__(self, para):
        """Initialize ML process.

        nfile is the optimization dataset number
        ntest is the test dataset number
        """
        self.para = para
        self.read = Read(para)
        self.nfile = int(para['n_dataset'][0])
        self.ntest = int(para['n_test'][0])
        self.dataprocess(self.para['dire_data'])
        if self.para['testMLmodel'] == 'linear':
            self.linearmodel()

    def dataprocess(self, diredata):
        """Process the optimization dataset and data for the following ML.

        Returns:
            features of ML (X)
            traing target (Y, e.g, compression radius)

        """
        # dire = self.para['direfeature']
        nsteps_ = int(self.para['mlsteps'] / self.para['save_steps'])
        self.para['natomall'] = []
        [self.para['natomall'].append(len(coor))
         for coor in self.para['coorall']]
        # self.para['natomall'] = \
        #   self.read.read1d(diredata, 'natom.dat', self.nfile)

        if self.para['Lml_skf']:
            natom = int(max(self.para['natomall']))
            fpcompr = open(os.path.join(diredata, 'comprbp.dat'), 'r')
            compr = np.zeros((self.nfile, nsteps_, natom))

            for ifile in range(0, self.nfile):
                natom_ = int(self.para['natomall'][ifile])
                datafpcompr = np.fromfile(fpcompr, dtype=float,
                                          count=natom_*nsteps_, sep=' ')
                datafpcompr.shape = (nsteps_, natom_)
                compr[ifile, :, :natom_] = datafpcompr
            self.para['optRall'] = compr
        if self.para['feature'] == 'rad':
            fprad = open(os.path.join(diredata, 'env_rad.dat'), 'r')
            datafprad = np.fromfile(fprad, dtype=float,
                                    count=natom_*self.ntest,  sep=' ')
            datafprad.shape = (self.ntest, natom_)
            self.para['feature_data'] = datafprad

    def linearmodel(self):
        """Use the optimization dataset for training.

        Returns:
            linear ML method predicted DFTB parameters

        """
        reg = linear_model.LinearRegression()
        X = self.para['feature_data'][:self.nfile]
        X_pred = self.para['feature_data']
        y = self.para['optRall'][:, -1, :]
        X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.5)

        reg.fit(X_train, y_train)

        y_pred = reg.predict(X_pred)

        plt.scatter(X_train, X_train,  color='black')
        plt.plot(X_train, y_train, 'ob')
        plt.xlabel('feature of training dataset')
        plt.ylabel('traning compression radius')
        plt.show()

        plt.scatter(X_pred, y_pred,  color='black')
        plt.plot(X_pred, y_pred, 'ob')
        plt.xlabel('feature of prediction (tesing)')
        plt.ylabel('testing compression radius')
        plt.show()
        self.para['compr_pred'] = t.from_numpy(y_pred)

    def nnmodel(self):
        pass

    def svm(self):
        pass


if __name__ == "__main__":
    """main function for optimizing DFTB parameters, testing DFTB"""
    t.autograd.set_detect_anomaly(True)
    t.set_printoptions(precision=15)
    para = {}
    para['task'] = 'optml'

    if para['task'] == 'optml':
        optml(para)
    elif para['task'] == 'test':
        para['dire_data'] = '../data/200609compr_100mol_dip'
        testml(para)
    elif para['task'] == 'envpara':
        get_env_para()
