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
from torch.autograd import Variable
import torch as t
import pyanitools as pya
<<<<<<< HEAD
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
=======
>>>>>>> ad0a72eac1ab68c13c8d6d1ed31874c22bb490c3
from ase.calculators.dftb import Dftb
import write_output as write
import lattice_cell
import dftbtorch.dftb_torch as dftb_torch
import dftbtorch.slakot as slakot
from geninterpskf import SkInterpolator
from plot import plot_main
<<<<<<< HEAD
from readt import ReadInt
=======
>>>>>>> ad0a72eac1ab68c13c8d6d1ed31874c22bb490c3
Directory = '/home/gz_fan/Documents/ML/dftb/ml'
DireSK = '/home/gz_fan/Documents/ML/dftb/slko'
BOHR = 0.529177210903
ATOMIND = {'H': 1, 'HH': 2, 'HC': 3, 'C': 4, 'CH': 5, 'CC': 6}
<<<<<<< HEAD
ATOMNUM = {'H': 1, 'C': 6, 'O': 8}
HNUM = {'CC': 4, 'CH': 2, 'CO': 4, 'HC': 0,  'HH': 1, 'HO': 2, 'OC': 0,
        'OH': 0, 'OO': 4}
COMP_R = {'H': 3.0, 'C': 3.0}
VAL_ORB = {"H": 1, "C": 2, "N": 2, "O": 2, "Ti": 3}


def mainml(outpara):
    '''main function for DFTB-ML'''
=======
ATOMNUM = {'H': 1, 'C': 6}
HNUM = {'CC': 4, 'CH': 2, 'CO': 4, 'HC': 0,  'HH': 1, 'HO': 2, 'OC': 0,
        'OH': 0, 'OO': 4}
COMP_R = {'H': 3.0, 'C': 3.0}


def mainml(task):
    '''main function for DFTB-ML'''
    outpara = {}
>>>>>>> ad0a72eac1ab68c13c8d6d1ed31874c22bb490c3

    # get the default para for dftb and ML, these para will maintain unchanged
    getmlpara(outpara)

    # load dataset, here is hdf type
    LoadData(outpara)

    # run reference calculations, either dft or dftb
    runml = RunML(outpara)
    runml.ref()

    # run dftb in ML process
    runml.mldftb(outpara)

    # plot data from ML
    plot_main(outpara)


<<<<<<< HEAD
def testml(outpara):

    # get the default para for dftb and ML, these para will maintain unchanged
    getmlpara(outpara)

    # load dataset, here is hdf type
    LoadData(outpara)

    # run reference calculations, either dft or dftb
    ml, runml = ML(outpara), RunML(outpara)
    runml.test_compr(outpara)
    runml.test_pred_compr(outpara)
=======
def get_env_para():
    '''this function is to get the environmental parameters'''
    para = {}
    genpara = GenMLPara(para)
    load = LoadData(para)
    save = SaveData(para)

    getmlpara(para)
    load.loadhdfdata(para)

    os.system('rm para_ang.dat')
    nbatch = int(para['nfile_dataset'][0])
    print('begin to calculate environmental parameters')
    symbols = para['symbols']
    for ibatch in range(0, nbatch):
        coor = t.from_numpy(para['coorall'][ibatch])
        genpara.genenvir(nbatch, para, ibatch, coor, symbols)
        iang = para['ang_paraall'][ibatch]
        irad = para['rad_paraall'][ibatch]
        save.save1D(iang, name='env_ang.dat', ty='a')
        save.save1D(irad, name='env_rad.dat', ty='a')
>>>>>>> ad0a72eac1ab68c13c8d6d1ed31874c22bb490c3


def getmlpara(outpara):
    '''
    ref: the reference of ML
    datasettype: type of dataset
    hdf_num (int): for hdf data type, read which kind of molecule
<<<<<<< HEAD
    n_dataset (int): read how many molecules of each dataset
    optim_para: the optimized parameters
    '''

    # ------------------------- loading dataset -------------------------
    outpara['datasettype'] = 'json'  # hdf, json
    if outpara['datasettype'] == 'json':
        outpara['pythondata_dire'] = './data'
        outpara['pythondata_file'] = 'CH4_data2'
    elif outpara['datasettype'] == 'hdf':
        hdffilelist = []
        hdffilelist.append(
                '/home/gz_fan/Documents/ML/database/an1/ani_gdb_s01.h5')
        outpara['hdffile'] = hdffilelist
        outpara['hdf_num'] = 1
    outpara['n_dataset'] = ['1']
    outpara['optim_para'] = ['Hamiltonian']

    # ---------------------- environment parameters ----------------------
    outpara['rcut'] = 5
    outpara['r_s'] = 3
    outpara['eta'] = 0.1
=======
    nfile_dataset (int): read how many molecules of each dataset
    optim_para: the optimized parameters
    '''

    # ------------------------ loading hdf dataset ------------------------
    outpara['datasettype'] = 'json'
    if outpara['datasettype'] == 'json':
        outpara['nfile'] = 5
        outpara['pythondata_dire'] = './data'
        outpara['pythondata_file'] = 'CH4_data'
    hdffilelist = []
    hdffilelist.append('/home/gz_fan/Documents/ML/database/an1/ani_gdb_s01.h5')
    outpara['hdffile'] = hdffilelist
    outpara['hdf_num'] = 1
    outpara['nfile_dataset'] = ['1']
    outpara['optim_para'] = ['Hamiltonian']

    # ---------------------- environment parameters ----------------------
    outpara['rcut'] = 3
    outpara['r_s'] = 0.8
    outpara['eta'] = 1
>>>>>>> ad0a72eac1ab68c13c8d6d1ed31874c22bb490c3
    outpara['tol'] = 1E-4
    outpara['zeta'] = 1
    outpara['lambda'] = 1
    outpara['ang_paraall'] = []
    outpara['rad_paraall'] = []

    # ------------------------- Machine learning -------------------------
    # splinetype: Bspline, Polyspline
    # < zero_threshold: this value is treated as zero
    # rand_threshold: the coefficient para of added randn number
<<<<<<< HEAD
    outpara['ref'] = 'dftb'
    outpara['target'] = ['eigval']  # dipole, homo_lumo, gap, eigval, qatomall
    outpara['mlsteps'] = 5
    outpara['Lml'] = True
    outpara['Lml_skf'] = True  # if use interp to gen .skf with compress_r
    outpara['Lml_HS'] = False  # if use interp to gen HS mat (e.g Polyspline)
    outpara['Lml_compr'] = False  # test gradients of interp of SK table
    outpara['Lml_compr_global'] = False  # each spiece has the same compress_r
    outpara['save_steps'] = 1
    outpara['interptype'] = 'Polyspline'
    outpara['interpdist'] = 0.4
    outpara['interpcutoff'] = 10
    outpara['zero_threshold'] = 5E-3
    outpara['rand_threshold'] = 5E-2
    outpara['lr'] = 1e-1
    outpara['atomspecie_old'] = []
=======
    outpara['ref'] = 'aims'
    outpara['target'] = ['gap']  # dipole, homo_lumo, gap
    outpara['mlsteps'] = 100
    outpara['ml'] = True
    outpara['interpskf'] = True  # if use interp to gen .skf with compress_r
    outpara['opt_dataset_compr'] = False  # each spiece has the same compress_r
    outpara['interpHS'] = False  # if use interp to gen HS mat (e.g Polyspline)
    outpara['save_steps'] = 10
    outpara['interptype'] = 'Polyspline'
    outpara['interpdist'] = 0.4
    outpara['interpcutoff'] = 4
    outpara['zero_threshold'] = 5E-3
    outpara['rand_threshold'] = 5E-2
    outpara['atomname_set_old'] = []
>>>>>>> ad0a72eac1ab68c13c8d6d1ed31874c22bb490c3
    outpara['H_init_compr'] = 3.34
    outpara['C_init_compr'] = 3.34
    outpara['init_compr_all'] = t.Tensor([3.34, 3.34, 3.34, 3.34, 3.34, 3.34,
                                          3.34, 3.34, 3.34, 3.34, 3.34, 3.34])
<<<<<<< HEAD
    '''outpara['H_compr_grid'] = t.Tensor([2.00, 2.34, 2.77, 3.34, 4.07, 5.03,
                                       6.28, 7.90, 10.00])
    outpara['C_compr_grid'] = t.Tensor([2.00, 2.34, 2.77, 3.34, 4.07, 5.03,
                                       6.28, 7.90, 10.00])'''
    outpara['H_compr_grid'] = t.Tensor([2.00, 2.50, 3.00, 3.50, 4.00, 4.50,
                                        5.00, 5.50, 6.00])
    outpara['C_compr_grid'] = t.Tensor([2.00, 2.50, 3.00, 3.50, 4.00, 4.50,
                                        5.00, 5.50, 6.00])
    outpara['testMLmodel'] = 'linear'
    outpara['feature'] = 'rad'
    outpara['direfeature'] = '.'

    # ----------------------------- DFTB -----------------------------
    outpara['LReadInput'] = False
    outpara['convergenceType'], outpara['energy_tol'] = 'energy', 1e-6
    outpara['scf'] = True
    outpara['scc'] = 'nonscc'
    outpara['task'] = 'ground'
    outpara['HSsym'] = 'symall_chol'  # symall, symhalf. important!!!!!!
=======
    outpara['H_compr_grid'] = t.Tensor([2.00, 2.34, 2.77, 3.34, 4.07, 5.03,
                                       6.28, 7.90, 10.00])
    outpara['C_compr_grid'] = t.Tensor([2.00, 2.34, 2.77, 3.34, 4.07, 5.03,
                                       6.28, 7.90, 10.00])

    # ----------------------------- DFTB -----------------------------
    outpara['readInput'] = False
    outpara['scf'] = True
    outpara['scc'] = True
    outpara['task'] = 'ground'
>>>>>>> ad0a72eac1ab68c13c8d6d1ed31874c22bb490c3
    outpara['ninterp'] = 8
    outpara['grid0'] = 0.4
    outpara['dist_tailskf'] = 1.0
    outpara['mixMethod'] = 'anderson'
    outpara['mixFactor'] = 0.2
    outpara['tElec'] = 0
    outpara['maxIter'] = 60
<<<<<<< HEAD
    outpara['Lperiodic'] = False
    outpara['Ldipole'] = True
    outpara['Lrepulsive'] = True
=======
    outpara['periodic'] = False
    outpara['ldipole'] = True
>>>>>>> ad0a72eac1ab68c13c8d6d1ed31874c22bb490c3
    outpara['coorType'] = 'C'
    path = os.getcwd()
    outpara['filename'] = 'dftb_in'
    outpara['direInput'] = os.path.join(path, 'dftbtorch')
    outpara['direSK'] = os.path.join(path, 'dftbtorch/slko')
<<<<<<< HEAD
    outpara['dire_interpSK'] = os.path.join(path, 'slko/sk_den3')
=======
    outpara['dire_interpSK'] = os.path.join(path, 'slko/sk_den')
>>>>>>> ad0a72eac1ab68c13c8d6d1ed31874c22bb490c3

    # ---------------------- plotting and others ----------------------
    outpara['plot_ham'] = True
    outpara['hamold'] = 0
    return outpara


class RunML:
    '''
    This is class for ML process (loading Ref data and dataset, running
    calculations of ref method and dftb method, saving ml data).
    len(outpara['nfile']) is to get optimize how many dataset.
    Assumption: the atom specie in each dataset maintain unchanged, otherwise
    we have to check for each new moluecle, if there is new atom specie.
    '''
    def __init__(self, para):
        self.para = para
<<<<<<< HEAD
        self.n_dataset = self.para['n_dataset']
        self.save = SaveData(self.para)
        self.slako = slakot.SlaKo(self.para)
        self.genml = GenMLPara(self.para)
        self.runcal = RunCalc(self.para)
=======
        # self.initialization()

    def initialization(self):
        '''
        this function aims to initialize ML parameters: read and construct
        all data dor the following dftb calculations'''
        slako = slakot.SlaKo(self.para)
        genml = GenMLPara(self.para)

        if self.para['datasettype'] == 'hdf':
            self.para['nspecie'] = len(self.para['specie'])
        if self.para['interpHS']:
            genml.get_specie_label()
            slako.read_skdata(self.para)
            slako.get_sk_spldata(self.para)
>>>>>>> ad0a72eac1ab68c13c8d6d1ed31874c22bb490c3

    def ref(self):
        '''
        according to reference type, this function will run different
<<<<<<< HEAD
        reference calculations
        '''

        # if run one dataset or multi dataset
        self.nbatch = int(self.para['n_dataset'][0])
        self.para['nfile'] = self.nbatch

        if any(x in self.para['target'] for x in ['homo_lumo', 'gap']):
            self.para['refhomo_lumo'] = t.zeros(self.nbatch, 2)
        elif 'dipole' in self.para['target']:
            self.para['refdipole'] = t.zeros(self.nbatch, 3)

        if self.para['ref'] == 'dftb':
            self.dftb_ref()
=======
        reference calculations'''

        # if run one dataset or multi dataset
        if len(self.para['nfile_dataset']) == 1 and self.para['datasettype'] == 'hdf':
            self.nbatch = int(self.para['nfile_dataset'][0])
            self.para['nfile'] = self.nbatch
        elif self.para['datasettype'] == 'json':
            self.nbatch = self.para['nfile']
        if 'homo_lumo' or 'gap' in self.para['target']:
            self.para['refhomo_lumo'] = t.zeros(self.nbatch, 2)
        if 'dipole' in self.para['target']:
            self.para['refdipole'] = t.zeros(self.nbatch, 3)

        if self.para['ref'] == 'dftb':
            self.dftb_ref(self.para)
>>>>>>> ad0a72eac1ab68c13c8d6d1ed31874c22bb490c3
        elif self.para['ref'] == 'aims':
            self.aims_ref(self.para)
        elif self.para['ref'] == 'dftbplus':
            self.dftbplus_ref(self.para)

<<<<<<< HEAD
    def dftb_ref(self):
        '''
        calculate reference for ML according to opt target
        '''
        self.para['cal_ref'] = True

        for ibatch in range(0, self.para['nfile']):
            # get coor and related geometry information
            self.get_coor(ibatch)
            dftb_torch.Initialization(self.para)

            if self.para['Lml_skf']:
                # if read all .skf and build [N_ij, N_R1, N_R2, 20] matrix
                if self.para['atomspecie'] != self.para['atomspecie_old']:
                    self.get_compr_specie()

                self.genml.get_spllabel()
                self.para['compr_ml'] = self.para['compr_init']

                if 'hstable' in self.para['target']:
                    self.para['compr_ml'] = self.para['compr_init'] - 1
                    self.slako.genskf_interp_compr()
                    self.runcal.idftb_torchspline()
                elif 'homo_lumo' in self.para['target']:
                    self.para['compr_ml'] = self.para['compr_init'] - 1
                    self.slako.genskf_interp_compr()
                    self.runcal.idftb_torchspline()
                elif 'eigval' in self.para['target']:
                    self.para['compr_ml'] = self.para['compr_init'] - 1
                    self.slako.genskf_interp_compr()
                    self.runcal.idftb_torchspline()
                elif 'qatomall' in self.para['target']:
                    self.para['compr_ml'] = self.para['compr_init'] - 1
                    self.slako.genskf_interp_compr()
                    self.runcal.idftb_torchspline()
                elif 'dipole' in self.para['target']:
                    self.para['compr_ml'] = self.para['compr_init'] - 1
                    self.slako.genskf_interp_compr()
                    self.runcal.idftb_torchspline()

            # read from .skf then use interp to generate H(S) mat for DFTB
            if any(x in self.para['target'] for x in ['homo_lumo', 'gap']):
                self.para['refhomo_lumo'][ibatch, :] = self.para['homo_lumo']
                self.save.save1D(self.para['homo_lumo'].detach().numpy(),
                                 name='eigref.dat', ty='w')
            elif 'dipole' in self.para['target']:
                self.para['refdipole'][ibatch] = self.para['dipole'][:]
                self.save.save1D(self.para['dipole'][:].detach().numpy(),
                                 name='dipref.dat', ty='w')

            if self.para['Lml_HS']:
                if self.para['interptype'] == 'Polyspline':
                    self.save.save2D(self.para['splyall'],
                                     name='splref.dat', ty='w')
                elif self.para['interptype'] == 'Bspline':
                    self.save.save2D(self.para['cspline'],
                                     name='splref.dat', ty='w')
                self.save.save1D(self.para['hammat'].detach().numpy(),
                                 name='hamref.dat', ty='w')
=======
    def dftb_ref(self, para):
        '''calculate ref (reference) for ML (machine learning)'''

        save = SaveData(para)
        slako = slakot.SlaKo(para)
        genml = GenMLPara(para)
        runcal = RunCalc(para)

        # run reference (dft/dftb) calculations according to opt target
        para['cal_ref'] = True

        for ibatch in range(0, self.nbatch):
            # get the coor of ibatch, and start initialization
            para['ibatch'] = ibatch
            para['coor'] = t.from_numpy(para['coorall'][ibatch])
            dftb_torch.Initialization(para)

            # check if atom specie is the same to the former
            if para['atomname_set'] != para['atomname_set_old']:

                # use interpolation to generate .skf for further interp and ML
                if para['interpskf']:
                    genml.get_spllabel()
                    genml.genml_compr()
                    interpskf(para)
                    para['compr_ml'] = para['compr_init']
                    slako.genskf_interp_ij(para)
                    slako.genskf_interp_r(para)
                    runcal.idftb_torchspline(para)

                # read from .skf then use interp to generate H(S) mat for DFTB
                else:
                    genml.get_spllabel()
                    slako.read_skdata(para)
                    slako.get_sk_spldata(para)
                    runcal.idftb_torchspline(para)

                # save data
                if 'homo_lumo' or 'gap' in para['target']:
                    print(para['homo_lumo'])
                    para['refhomo_lumo'][ibatch, :] = para['homo_lumo']
                    save.save1D(para['homo_lumo'].detach().numpy(),
                                name='eigref.dat', ty='w')
                if 'dipole' in para['target']:
                    para['refdipole'][ibatch] = para['dipole'][:]
                    save.save1D(para['dipole'][:].detach().numpy(),
                                name='dipref.dat', ty='w')

                if para['interptype'] == 'Polyspline' and not \
                                         para['interpskf']:
                    save.save2D(para['splyall'], name='splref.dat', ty='w')
                elif para['interptype'] == 'Bspline' and not para['interpskf']:
                    save.save2D(para['cspline'], name='splref.dat', ty='w')
                save.save1D(para['hammat'].detach().numpy(),
                            name='hamref.dat', ty='w')

            # atom speice is the same sa the former atom
            else:

                # use init compr to gen .skf for further interp and ML
                if para['interpskf']:
                    genml.genml_compr()
                    interpskf(para)
                    para['compr_ml'] = para['compr_init']
                    slako.genskf_interp_ij(para)
                    slako.genskf_interp_r(para)
                    runcal.idftb_torchspline(para)

                # read from .skf then use interp to generate H(S) mat for DFTB
                else:
                    runcal.idftb_torchspline(para)

                if 'homo_lumo' or 'gap' in para['target']:
                    para['refhomo_lumo'][ibatch, :] = para['homo_lumo']
                    save.save1D(para['homo_lumo'].detach().numpy(),
                                name='eigref.dat', ty='a')
                if 'dipole' in para['target']:
                    para['refdipole'][ibatch] = para['dipole'][:]
                    save.save1D(para['dipole'][:].detach().numpy(),
                                name='dipref.dat', ty='a')
                save.save1D(para['hammat'].detach().numpy(),
                            name='hamref.dat', ty='a')
>>>>>>> ad0a72eac1ab68c13c8d6d1ed31874c22bb490c3

    def aims_ref(self, para):
        '''FHI-aims as reference, run FHI-aims and data processing'''
        runcal = RunCalc(para)

        self.pre_aims()

        for ibatch in range(0, self.nbatch):
            # get the coor of ibatch, and start initialization
            para['ibatch'] = ibatch
            if type(para['coorall'][ibatch]) is t.Tensor:
                para['coor'] = para['coorall'][ibatch]
            elif type(para['coorall'][ibatch]) is np.ndarray:
                para['coor'] = t.from_numpy(para['coorall'][ibatch])

            # check if atom specie is the same to the former
            runcal.aims(para, ibatch, self.dir_ref)

        self.save_aims()

    def pre_aims(self):
        '''pre-processing aims calculations'''
        dire = os.getcwd() + '/aims'
        if not os.path.exists(dire):
            print('Warning: please make a folder "aims", prepare input files ',
                  'and a script which run calculations and extract results')
        self.dir_ref = os.getcwd() + '/aims'
        os.system('rm ./aims/*.dat')

    def save_aims(self):
        '''save files for reference calculations'''
        save = SaveData(self.para)
<<<<<<< HEAD
        # if any(x in self.para['target'] for x in ['homo_lumo', 'gap']):
        homo_lumo = write.FHIaims(self.para).read_bandenergy(
                self.para, self.para['nfile'], self.dir_ref)
        self.para['refhomo_lumo'] = t.from_numpy(homo_lumo)
        save.save1D(homo_lumo, name='eigref.dat', ty='w')
=======
        if 'homo_lumo' or 'gap' in self.para['target']:
            homo_lumo = write.FHIaims(self.para).read_bandenergy(
                    self.para, self.para['nfile'], self.dir_ref)
            self.para['refhomo_lumo'][:, :] = t.from_numpy(homo_lumo)
            save.save1D(homo_lumo, name='eigref.dat', ty='w')
>>>>>>> ad0a72eac1ab68c13c8d6d1ed31874c22bb490c3
        if 'dipole' in self.para['target']:
            dipole = write.FHIaims(self.para).read_dipole(
                    self.para, self.para['nfile'], self.dir_ref)
            self.para['refdipole'][:, :] = t.from_numpy(dipole)
            save.save1D(dipole, name='dipref.dat', ty='w')

    def dftbplus_ref(self, para):
        '''DFTB+ as reference, run DFTB+ and data processing'''
        runcal = RunCalc(para)

        self.pre_dftbplus()

        for ibatch in range(0, self.nbatch):
            # get the coor of ibatch, and start initialization
            para['ibatch'] = ibatch
            para['coor'] = t.from_numpy(para['coorall'][ibatch])

            # check if atom specie is the same to the former
            runcal.dftbplus(para, ibatch, self.dir_ref)

        self.save_dftbplus()

    def pre_dftbplus(self):
        '''pre-processing dftb+ calculations'''
        dire = os.getcwd() + '/dftbplus'
        if not os.path.exists(dire):
            print('Warning: please make a folder "dftbplus", prepare input ',
                  'files, a script to run calculations and data processing')
        self.dir_ref = os.getcwd() + '/dftbplus'
        os.system('rm ./dftbplus/*.dat')

    def save_dftbplus(self):
        '''save files for reference calculations'''
        save = SaveData(self.para)
<<<<<<< HEAD
        # if any(x in self.para['target'] for x in ['homo_lumo', 'gap']):
        homo_lumo = write.Dftbplus(self.para).read_bandenergy(
                self.para, self.para['nfile'], self.dir_ref)
        self.para['refhomo_lumo'] = t.from_numpy(homo_lumo)
        save.save1D(homo_lumo, name='eigref.dat', ty='w')
=======
        if 'homo_lumo' or 'gap' in self.para['target']:
            homo_lumo = write.Dftbplus(self.para).read_bandenergy(
                    self.para, self.para['nfile'], self.dir_ref)
            self.para['refhomo_lumo'][:, :] = t.from_numpy(homo_lumo)
            save.save1D(homo_lumo, name='eigref.dat', ty='w')
>>>>>>> ad0a72eac1ab68c13c8d6d1ed31874c22bb490c3
        if 'dipole' in self.para['target']:
            dipole = write.Dftbplus(self.para).read_dipole(
                    self.para, self.para['nfile'], self.dir_ref)
            self.para['refdipole'][:, :] = t.from_numpy(dipole)
            save.save1D(dipole, name='dipref.dat', ty='w')

    def dftb(self, para):
        '''this function is for dftb calculation'''
        if para['cal_ref']:
            para['cal_ref'] = False
            nbatch = para['nfile']
            coorall = para['coorall']
            save = SaveData(para)
<<<<<<< HEAD
            if para['Lml_HS'] and not para['Lml_skf']:
=======
            if para['interpHS'] and not para['interpskf']:
>>>>>>> ad0a72eac1ab68c13c8d6d1ed31874c22bb490c3
                save.save2D(para['splyall_rand'].detach().numpy(),
                            name='splref.dat', ty='a')
            for ibatch in range(0, nbatch):
                para['coor'] = t.from_numpy(coorall[ibatch])
                RunCalc(para).idftb_torchspline(para)
                eigval = para['homo_lumo']

                # save data
                save.save1D(eigval.detach().numpy(), name='eigref.dat',
                            ty='a')
                save.save1D(para['hammat'].detach().numpy(),
                            name='hamref.dat', ty='a')
        elif not para['cal_ref']:
            nbatch = para['nfile']
            coorall = para['coorall']
            save = SaveData(para)
<<<<<<< HEAD
            if para['Lml_HS'] and not para['Lml_skf']:
=======
            if para['interpHS'] and not para['interpskf']:
>>>>>>> ad0a72eac1ab68c13c8d6d1ed31874c22bb490c3
                save.save2D(para['splyall_rand'].detach().numpy(),
                            name='splref.dat', ty='a')

            # calculate one by one to optimize para
            for ibatch in range(0, nbatch):
                para['coor'] = t.from_numpy(coorall[ibatch])
                RunCalc(para).idftb_torchspline(para)
                eigval = para['homo_lumo']

                # save data
                save.save1D(eigval.detach().numpy(), name='eigref.dat',
                            ty='a')
                save.save1D(para['hammat'].detach().numpy(),
                            name='hamref.dat', ty='a')

<<<<<<< HEAD
    def get_coor(self, ibatch):
        '''
        get the ith coor according to data type
        '''
        if type(self.para['coorall'][ibatch]) is t.Tensor:
            self.para['coor'] = self.para['coorall'][ibatch][:, :]
        elif type(self.para['coorall'][ibatch]) is np.ndarray:
            self.para['coor'] = \
                t.from_numpy(self.para['coorall'][ibatch][:, :])

    def get_compr_specie(self):
        '''
        1. read all .skf with various compR for all apecie
        2. get the itegrals matrix: natom * natom * [ncompr, ncompr, 20]
        3. get the initial compR
        '''
        interpskf(self.para)
        self.slako.genskf_interp_ij()
        self.genml.genml_init_compr()

    def mldftb(self, para):
        '''this function will run ML of DFTB with various targets'''
        if len(self.n_dataset) == 1 and self.para['datasettype'] == 'hdf':
            self.nbatch = int(self.para['n_dataset'][0])
            self.para['nfile'] = self.nbatch
        elif self.para['datasettype'] == 'json':
            self.nbatch = self.para['nfile']
        elif self.para['datasettype'] == 'input_data':
            self.nbatch = self.para['nfile']

        if para['Lml_HS']:
            self.ml_interp_hs(para)
        elif para['Lml_skf']:
            self.ml_compr(para)
        elif para['Lml_compr']:
            self.ml_compr_interp(para)
=======
    def mldftb(self, para):
        '''this function will run ML of DFTB with various targets'''
        if para['interpHS']:
            self.ml_interp_hs(para)
        elif para['interpskf']:
            self.ml_compr(para)
>>>>>>> ad0a72eac1ab68c13c8d6d1ed31874c22bb490c3

    def ml_interp_hs(self, para):
        '''DFTB optimization for given dataset'''

        para['cal_ref'] = False
        # optimize selected para to get opt target
<<<<<<< HEAD
        if para['Lml_HS'] and para['interptype'] == 'Bspline':
            para['cspline'] = Variable(para['cspl_rand'], requires_grad=True)
            optimizer = t.optim.SGD([para['cspline']], lr=5e-7)
        elif para['Lml_HS'] and para['interptype'] == 'Polyspline':
=======
        if para['interpHS'] and para['interptype'] == 'Bspline':
            para['cspline'] = Variable(para['cspl_rand'], requires_grad=True)
            optimizer = t.optim.SGD([para['cspline']], lr=5e-7)
        elif para['interpHS'] and para['interptype'] == 'Polyspline':
>>>>>>> ad0a72eac1ab68c13c8d6d1ed31874c22bb490c3
            para['splyall_rand'] = Variable(para['splyall_rand'],
                                            requires_grad=True)
            optimizer = t.optim.SGD([para['splyall_rand']], lr=5e-7)

        # rm the old file
        os.system('rm eigbp.dat ham.dat spl.dat compr.dat')
        save = SaveData(para)
        nbatch = para['nfile']
        coorall = para['coorall']
        print('nbatch', nbatch, para['refhomo_lumo'][0], coorall[0])

        # calculate one by one to optimize para
        for ibatch in range(0, nbatch):
<<<<<<< HEAD
            if any(x in para['target'] for x in ['homo_lumo', 'gap']):
=======
            if 'homo_lumo' or 'gap' in para['target']:
>>>>>>> ad0a72eac1ab68c13c8d6d1ed31874c22bb490c3
                eigref = t.zeros(2)
                eigref[:] = para['refhomo_lumo'][ibatch]
            if 'dipole' in range(0, nbatch):
                dipref = t.zeros(3)
                dipref[:] = para['refdipole'][ibatch][:]

            # for each molecule we will run mlsteps
            para['coor'] = t.from_numpy(coorall[ibatch])
            for it in range(0, para['mlsteps']):

                # dftb calculations (choose scc or nonscc)
                RunCalc(para).idftb_torchspline(para)
<<<<<<< HEAD
                if any(x in para['target'] for x in ['homo_lumo', 'gap']):
=======
                if 'homo_lumo' or 'gap' in para['target']:
>>>>>>> ad0a72eac1ab68c13c8d6d1ed31874c22bb490c3
                    eigval = para['homo_lumo']
                elif 'dipole' in para['target']:
                    dipole = para['dipole'][:]

                # define loss function
<<<<<<< HEAD
                criterion = t.nn.L1Loss(reduction='sum')
=======
                criterion = t.nn.MSELoss(reduction='sum')
>>>>>>> ad0a72eac1ab68c13c8d6d1ed31874c22bb490c3
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
                                name='spl.dat', ty='a')
                    save.save1D(para['hammat'].detach().numpy(),
                                name='ham.dat', ty='a')

    def ml_compr(self, para):
        '''DFTB optimization for given dataset'''

        # rm the old file
        os.system('rm eigbp.dat ham.dat spl.dat compr.dat qatom.dat')
        os.system('rm dipref.dat, dipbp.dat')
<<<<<<< HEAD

        # calculate one by one to optimize para
        for ibatch in range(0, self.nbatch):
            para['ibatch'] = ibatch
            self.get_coor(ibatch)
            dftb_torch.Initialization(self.para)

            if self.para['atomspecie'] != self.para['atomspecie_old']:
                self.get_compr_specie()

            # build the ref data
            if any(x in para['target'] for x in ['homo_lumo', 'gap']):
                homo_lumo_ref = t.zeros(2)
                homo_lumo_ref[:] = para['refhomo_lumo'][ibatch]
            elif 'dipole' in para['target']:
                dipref = t.zeros(3)
                dipref[:] = para['refdipole'][ibatch]
            elif 'hstable' in para['target']:
                homo_lumo_ref = para['homo_lumo']
                hatableref = para['hammat']
            elif 'eigval' in para['target']:
                eigvalref = para['eigenvalue']
            elif 'qatomall' in para['target']:
                qatomall_ref = para['qatomall']
                print('compr0_ref:', para['compr_ml'])

            if not para['Lml_compr_global']:
                '''para['compr_ml'] = Variable(
                        para['compr_init'], requires_grad=True)'''
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
=======
        save = SaveData(para)
        runcal = RunCalc(para)
        genml = GenMLPara(para)
        slako = slakot.SlaKo(para)

        # calculate one by one to optimize para
        for ibatch in range(0, self.nbatch):
            if not para['opt_dataset_compr']:
                para['ibatch'] = ibatch
                if para['coorall'][ibatch] is t.Tensor:
                    para['coor'] = para['coorall'][ibatch]
                elif para['coorall'][ibatch] is np.ndarray:
                    para['coor'] = t.from_numpy(para['coorall'][ibatch])
                dftb_torch.Initialization(para)
                interpskf(para)
                genml.genml_compr()
                para['compr_ml'] = Variable(para['compr_init'],
                                            requires_grad=True)
                optimizer = t.optim.SGD([para['compr_ml']], lr=1e-4)
            elif para['opt_dataset_compr']:
                para['ibatch'] = ibatch
                if para['coorall'][ibatch] is t.Tensor:
                    para['coor'] = para['coorall'][ibatch]
                elif para['coorall'][ibatch] is np.ndarray:
                    para['coor'] = t.from_numpy(para['coorall'][ibatch])
                dftb_torch.Initialization(para)
                interpskf(para)
                optimizer = t.optim.SGD([para['compr_all']], lr=5e-4)

            # from whole SK data interpolate SK at certain distance
            slako.genskf_interp_ij(para)

            if 'homo_lumo' or 'gap' in para['target']:
                eigref = t.zeros(2)
                eigref[:] = para['refhomo_lumo'][ibatch]
            if 'dipole' in para['target']:
                dipref = t.zeros(3)
                dipref[:] = para['refdipole'][ibatch]

            # for each molecule we will run mlsteps
            for it in range(0, para['mlsteps']):
                if para['opt_dataset_compr']:
                    genml.genml_compr()

                # dftb calculations (choose scc or nonscc)
                slako.genskf_interp_r(para)
                runcal.idftb_torchspline(para)

                # define loss function
                criterion = t.nn.MSELoss(reduction='sum')
                if 'homo_lumo' or 'gap' in para['target']:
                    eigval = para['homo_lumo']
                    if 'homo_lumo' in para['target']:
                        loss = criterion(eigval, eigref)
                    if 'gap' in para['target']:
                        gap = t.abs(eigval[1] - eigval[0])
                        gapref = t.abs(eigref[1] - eigref[0])
                        loss = criterion(gap, gapref)
                if 'dipole' in para['target']:
                    dipole = para['dipole']
                    print('dipole', dipole, dipref)
                    loss = criterion(dipole, dipref)
                print("para['target']", para['target'])
                print("para['compr_ml']", para['compr_ml'])
>>>>>>> ad0a72eac1ab68c13c8d6d1ed31874c22bb490c3

                # clear gradients and define back propagation
                optimizer.zero_grad()
                # loss.requres_grad = True
                loss.backward(retain_graph=True)
<<<<<<< HEAD
                print('compr_ml.grad', para['compr_ml'].grad)
=======
>>>>>>> ad0a72eac1ab68c13c8d6d1ed31874c22bb490c3
                optimizer.step()

                # save and print information
                if it % para['save_steps'] == 0:
<<<<<<< HEAD
                    print('-' * 50)
                    print('ibatch: {} steps: {} loss: {} target: {}'.format(
                              ibatch, it, loss.item(), para['target']))
                    print('speciedict', para['speciedict'])
                    print("para['compr_ml']", para['compr_ml'])
                    if any(x in para['target'] for x in ['homo_lumo', 'gap']):
                        print('homo_lumo: {}, homo_lumo_ref: {}'.format(
                                homo_lumo, homo_lumo_ref))
                    self.save.save1D(homo_lumo.detach().numpy(),
                                     name='eigbp.dat', ty='a')
                    if 'dipole' in para['target']:
                        print('dipole: {}, dipref: {}'.format(dipole, dipref))
                        self.save.save1D(dipole.detach().numpy(),
                                         name='dipbp.dat', ty='a')
                    self.save.save1D(para['hammat'].detach().numpy(),
                                     name='ham.dat', ty='a')
                    self.save.save1D(para['compr_ml'].detach().numpy(),
                                     name='compr.dat', ty='a')
                    self.save.save1D(para['qatomall'].detach().numpy(),
                                     name='qatom.dat', ty='a')

    def test_compr(self, para):
        '''DFTB optimization for given dataset'''

        # rm the old file
        os.system('rm qatom.dat dipbp.dat')
        self.nbatch = int(self.para['n_dataset'][0])

        # calculate one by one to optimize para
        for ibatch in range(0, self.nbatch):
            para['ibatch'] = ibatch
            self.get_coor(ibatch)
            dftb_torch.Initialization(self.para)

            if self.para['atomspecie'] != self.para['atomspecie_old']:
                self.get_compr_specie()

            para['compr_ml'] = para['compr_init']

            # dftb calculations (choose scc or nonscc)
            # slako.genskf_interp_r(para)
            self.slako.genskf_interp_compr()
            self.runcal.idftb_torchspline()
            homo_lumo = para['homo_lumo']
            if 'dipole' in para['target']:
                dipole = para['dipole']
            self.save.save1D(homo_lumo.numpy(), name='eigbp.dat', ty='a')
            if 'dipole' in para['target']:
                self.save.save1D(dipole.numpy(), name='dipbp.dat', ty='a')
            self.save.save1D(para['qatomall'].numpy(), name='qatom.dat', ty='a')

    def test_pred_compr(self, para):
        '''DFTB optimization for given dataset'''
        os.system('rm qatompred.dat dippred.dat eigpred.dat')
        # rm the old file
        self.nbatch = int(self.para['n_dataset'][0])

        # calculate one by one to optimize para
        for ibatch in range(0, self.nbatch):
            para['ibatch'] = ibatch
            self.get_coor(ibatch)
            dftb_torch.Initialization(self.para)

            if self.para['atomspecie'] != self.para['atomspecie_old']:
                self.get_compr_specie()

            para['compr_ml'] = self.para['compr_pred'][ibatch]

            # dftb calculations (choose scc or nonscc)
            # slako.genskf_interp_r(para)
            self.slako.genskf_interp_compr()
            self.runcal.idftb_torchspline()
            homo_lumo = para['homo_lumo']
            if 'dipole' in para['target']:
                dipole = para['dipole']
            self.save.save1D(homo_lumo.numpy(), name='eigpred.dat', ty='a')
            if 'dipole' in para['target']:
                self.save.save1D(dipole.numpy(),
                                 name='dippred.dat', ty='a')
            self.save.save1D(para['qatomall'].numpy(),
                             name='qatompred.dat', ty='a')

    def ml_compr_interp(self, para):
        '''test the interpolation gradients'''

        # rm the old file
        os.system('rm interp.dat compr.dat qatom.dat')

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
                                 name='compr.dat', ty='a')


def interpskf(para):
    '''
    read .skf data from skgen with various compR
    '''
    for namei in para['atomspecie']:
        for namej in para['atomspecie']:
            SkInterpolator(para, gridmesh=0.2).readskffile(
                    namei, namej, para['dire_interpSK'])


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
=======
                    print('ibatch: {} steps: {} loss:  {}'.format(
                              ibatch, it, loss.item()))
                    print("para['compr_ml']", para['compr_ml'])
                    if 'homo_lumo' or 'gap' in para['target']:
                        print('eigval: {}, eigref: {}'.format(eigval, eigref))
                        save.save1D(eigval.detach().numpy(), name='eigbp.dat',
                                    ty='a')
                    if 'dipole' in para['target']:
                        print('dipole: {}, dipref: {}'.format(dipole, dipref))
                        save.save1D(dipole.detach().numpy(), name='dipbp.dat',
                                    ty='a')
                    save.save1D(para['hammat'].detach().numpy(),
                                name='ham.dat', ty='a')
                    save.save1D(para['compr_ml'].detach().numpy(),
                                name='compr.dat', ty='a')
                    if para['scc']:
                        for iq in range(0, len(para['qatomall'])):
                            save.save1D(para['qatomall'][iq].detach().numpy(),
                                        name='qatom.dat', ty='a')


def interpskf(para):
    '''read .skf data from skgen'''
    dire = para['dire_interpSK']
    for namei in para['atomname_set']:
        for namej in para['atomname_set']:
            skinter = SkInterpolator(para, grid0=0.4, gridmesh=0.2)
            skinter.readskffile(namei, namej, dire)

    # Example2: generate whole skf file
    # we will generate 4 skf files, therefore num will be 0, 1, 2 and 3
    '''for num in range(0, 4):
        skffile = skinter.readskffile(num, nameall[num], dire)
        hs_skf, ninterpline = skinter.getallgenintegral(num, skffile, ri, rj,
                                                        x0, y0)
        hs_skf = skinter.polytozero(hs_skf, ninterpline)
        # polytozero function adds 5 lines to make the tail more smooth
        skinter.saveskffile(num, nameall[num], skffile, hs_skf, ninterpline+5)
        num += 1'''
>>>>>>> ad0a72eac1ab68c13c8d6d1ed31874c22bb490c3


class GenMLPara:
    '''
    this class aims to form parameters for ML
    genenvir: atomic environment parameters
    get_spllabel: get how many row lines of c parameters in bspline
    '''
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

<<<<<<< HEAD
    def genml_init_compr(self):
=======
    def genml_compr(self):
>>>>>>> ad0a72eac1ab68c13c8d6d1ed31874c22bb490c3
        atomname = self.para['atomnameall']
        natom = self.para['natom']
        init_compr = t.empty(natom)
        icount = 0
<<<<<<< HEAD
        if self.para['Lml_compr_global']:
=======
        if self.para['opt_dataset_compr']:
>>>>>>> ad0a72eac1ab68c13c8d6d1ed31874c22bb490c3
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
        return self.para

    def get_spllabel(self):
        '''
        check if atom specie is the same, if not, then update
        '''
<<<<<<< HEAD
        if self.para['atomspecie'] != self.para['atomspecie_old']:
            for iatomspecie in self.para['atomspecie']:
                if iatomspecie not in self.para['atomspecie_old']:
                    self.para['atomspecie_old'].append(iatomspecie)
            h_spl_num = 0
            self.para['spl_label'] = []
            for iatom in self.para['atomspecie_old']:
                for jatom in self.para['atomspecie_old']:
=======
        if self.para['atomname_set'] != self.para['atomname_set_old']:
            for iatomspecie in self.para['atomname_set']:
                if iatomspecie not in self.para['atomname_set_old']:
                    self.para['atomname_set_old'].append(iatomspecie)
            h_spl_num = 0
            self.para['spl_label'] = []
            for iatom in self.para['atomname_set_old']:
                for jatom in self.para['atomname_set_old']:
>>>>>>> ad0a72eac1ab68c13c8d6d1ed31874c22bb490c3
                    nameij = iatom + jatom
                    h_spl_num += HNUM[nameij]
                    if HNUM[nameij] > 0:
                        self.para['spl_label'].append(nameij)
                        self.para['spl_label'].append(h_spl_num)
                        self.para['spl_label'].append(HNUM[nameij])
            print('initial H-table has {} rows'.format(h_spl_num))
            self.para['h_spl_num'] = h_spl_num
        return self.para

    def get_specie_label(self):
        '''
        check if atom specie is the same, if not, then update
        '''
        h_spl_num = 0
        self.para['spl_label'] = []
<<<<<<< HEAD
        for ispecie in self.para['atomspecie']:
            for jspecie in self.para['atomspecie']:
=======
        for ispecie in self.para['atomname_set']:
            for jspecie in self.para['atomname_set']:
>>>>>>> ad0a72eac1ab68c13c8d6d1ed31874c22bb490c3
                nameij = ispecie + jspecie
                h_spl_num += HNUM[nameij]
                if HNUM[nameij] > 0:
                    self.para['spl_label'].append(nameij)
                    self.para['spl_label'].append(h_spl_num)
                    self.para['spl_label'].append(HNUM[nameij])
        print('initial H-table has {} rows'.format(h_spl_num))
        self.para['h_spl_num'] = h_spl_num
        return self.para


class LoadData:
    '''
<<<<<<< HEAD
    the input:
        datasettype: the data type, hdf, json...
        hdf_num: how many dataset in one hdf file
    the output:
        coorall: all the coordination of molecule
        symbols: all the atoms in each molecule
        specie: the specie in molecule
        speciedict: Counter(symbols)
=======
    In hdf data type, each ntype represents one type of molecule, such as for
        ntype = 1, the molecules are all CH4
>>>>>>> ad0a72eac1ab68c13c8d6d1ed31874c22bb490c3
    '''
    def __init__(self, outpara):
        self.outpara = outpara
        if self.outpara['datasettype'] == 'hdf':
            self.loadhdfdata()
        if self.outpara['datasettype'] == 'json':
            self.load_json_data()

    def loadhdfdata(self):
        ntype = self.outpara['hdf_num']
        hdf5filelist = self.outpara['hdffile']
        icount = 0
<<<<<<< HEAD
        self.outpara['coorall'] = []
=======
>>>>>>> ad0a72eac1ab68c13c8d6d1ed31874c22bb490c3
        for hdf5file in hdf5filelist:
            adl = pya.anidataloader(hdf5file)
            for data in adl:
                icount += 1
                if icount == ntype:
<<<<<<< HEAD
                    for icoor in data['coordinates']:
                        row, col = np.shape(icoor)[0], np.shape(icoor)[1]
                        coor = t.zeros(row, col + 1)
                        for iat in range(0, len(data['species'])):
                            coor[iat, 0] = ATOMNUM[data['species'][iat]]
                            coor[iat, 1:] = t.from_numpy(icoor[iat, :])
                        self.outpara['coorall'].append(coor)
                    symbols = data['species']
                    specie = set(symbols)
                    speciedict = Counter(symbols)
                    self.outpara['symbols'] = symbols
                    self.outpara['specie'] = specie
                    self.outpara['atomspecie'] = []
                    [self.outpara['atomspecie'].append(ispecie) for
=======
                    coorall = data['coordinates']
                    symbols = data['species']
                    specie = set(symbols)
                    speciedict = Counter(symbols)
                    self.outpara['coorall'] = coorall
                    self.outpara['symbols'] = symbols
                    self.outpara['specie'] = specie
                    self.outpara['atomname_set'] = []
                    [self.outpara['atomname_set'].append(ispecie) for
>>>>>>> ad0a72eac1ab68c13c8d6d1ed31874c22bb490c3
                     ispecie in specie]
                    self.outpara['speciedict'] = speciedict

    def load_json_data(self):
        dire = self.outpara['pythondata_dire']
        filename = self.outpara['pythondata_file']
        self.outpara['coorall'] = []

        with open(os.path.join(dire, filename), 'r') as fp:
            fpinput = json.load(fp)

            if 'symbols' in fpinput['general']:
                self.outpara['symbols'] = fpinput['general']['symbols'].split()
                self.outpara['speciedict'] = Counter(self.outpara['symbols'])

            specie = set(self.outpara['symbols'])
            self.outpara['specie'] = specie
<<<<<<< HEAD
            self.outpara['atomspecie'] = []
            [self.outpara['atomspecie'].append(ispe) for ispe in specie]
=======
            self.outpara['atomname_set'] = []
            [self.outpara['atomname_set'].append(ispe) for ispe in specie]
>>>>>>> ad0a72eac1ab68c13c8d6d1ed31874c22bb490c3

            for iname in fpinput['geometry']:
                icoor = fpinput['geometry'][iname]
                self.outpara['coorall'].append(t.from_numpy(np.asarray(icoor)))

    def loadrefdata(self, ref, dire, nfile):
<<<<<<< HEAD
        '''
        load the data from DFT calculations
        '''
=======
>>>>>>> ad0a72eac1ab68c13c8d6d1ed31874c22bb490c3
        if ref == 'aims':
            newdire = os.path.join(Directory, dire)
            if os.path.exists(os.path.join(newdire,
                                           'bandenergy.dat')):
                refenergy = Variable(t.empty(nfile, 2),
                                     requires_grad=False)
                fpenergy = open(os.path.join(newdire, 'bandenergy.dat'), 'r')
                for ifile in range(0, nfile):
                    energy = np.fromfile(fpenergy, dtype=float,
                                         count=3, sep=' ')
                    refenergy[ifile, :] = t.from_numpy(energy[1:])
        elif ref == 'dftbrand':
            newdire = os.path.join(Directory, dire)
            if os.path.exists(os.path.join(newdire, 'bandenergy.dat')):
                refenergy = Variable(t.empty(nfile, 2),
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
<<<<<<< HEAD
        '''
        load the data of atomic environment parameters
        '''
=======
>>>>>>> ad0a72eac1ab68c13c8d6d1ed31874c22bb490c3
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
    '''
    this class aims to run different DFT(B) calculations, to write the input,
    we will use ASE interface
    '''
    def __init__(self, outpara):
        self.outpara = outpara

    def aims(self, para, ibatch, dire):
        '''here dft means FHI-aims'''
        coor = para['coor']
        natom = np.shape(coor)[0]
<<<<<<< HEAD
        write.FHIaims(para).geo_nonpe_hdf(para, ibatch, coor[:, 1:])
=======
        write.FHIaims(para).geo_nonpe_hdf(para, ibatch, coor)
>>>>>>> ad0a72eac1ab68c13c8d6d1ed31874c22bb490c3
        os.rename('geometry.in.{}'.format(ibatch), 'aims/geometry.in')
        os.system('bash '+dire+'/run.sh '+dire+' '+str(ibatch)+' '+str(natom))

    def dftbplus(self, para, ibatch, dire):
        '''use dftb+ to calculate'''
        coor = para['coor']
        specie, speciedict = para['specie'], para['speciedict']
<<<<<<< HEAD
        write.Dftbplus(para).geo_nonpe2(ibatch, coor[:, 1:], specie, speciedict)
=======
        write.Dftbplus(para).geo_nonpe2(ibatch, coor, specie, speciedict)
>>>>>>> ad0a72eac1ab68c13c8d6d1ed31874c22bb490c3
        os.rename('geo.gen.{}'.format(ibatch), 'dftbplus/geo.gen')
        print('run dftbplus')
        os.system('bash '+dire+'/run.sh '+dire+' '+str(ibatch))

    def dftbtorchrun(self, outpara, coor, DireSK):
        '''
        use dftb_python and read SK from whole .skf file, coor as input and
        do not have to read coor from geo.gen or other input files
        '''
        outpara['coor'] = t.from_numpy(coor)
        dipolemall = outpara['dipolemall']
        eigvalall = outpara['eigvalall']
        dipolem, eigval = dftb_torch.main(outpara)
        dipolemall.append(dipolem)
        eigvalall.append(eigval)
        outpara['dipolemall'] = dipolemall
        outpara['eigvalall'] = eigvalall
        return outpara

<<<<<<< HEAD
    def idftb_torchspline(self):
=======
    def idftb_torchspline(self, outpara):
>>>>>>> ad0a72eac1ab68c13c8d6d1ed31874c22bb490c3
        '''
        use dftb_python and read SK from whole .skf file, coor as input and
        do not have to read coor from geo.gen or other input files
        '''
<<<<<<< HEAD
        dftb_torch.Initialization(self.outpara).gen_sk_matrix(self.outpara)
        dftb_torch.Rundftbpy(self.outpara)
=======
        dftb_torch.Initialization(outpara).gen_sk_matrix(outpara)
        dftb_torch.Rundftbpy(outpara)
>>>>>>> ad0a72eac1ab68c13c8d6d1ed31874c22bb490c3


class SaveData:
    '''
    data is numpy type matrix
    blank defines where we'll write blank line
    name is the name of the saved file
    save2D will save file line by line
    savetype: 'a': appendix; 'w': save as a new file (replace the old)
    '''
    def __init__(self, outpara):
        self.outpara = outpara

    def save1D(self, data, name, blank='lower', dire=None, ty='w'):
<<<<<<< HEAD
        '''
        save 1D numpy array or tensor
        '''
=======
>>>>>>> ad0a72eac1ab68c13c8d6d1ed31874c22bb490c3
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

    def save2D(self, data, name, blank='lower', dire='.', ty='w'):
<<<<<<< HEAD
        '''
        save 2D numpy array or tensor
        '''
=======
>>>>>>> ad0a72eac1ab68c13c8d6d1ed31874c22bb490c3
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

    def save_envir(outpara, Directory):
<<<<<<< HEAD
        '''
        save atomic environment parameters
        '''
=======
>>>>>>> ad0a72eac1ab68c13c8d6d1ed31874c22bb490c3
        ang_paraall = outpara['ang_paraall']
        rad_paraall = outpara['rad_paraall']
        with open(os.path.join(Directory, 'rad_para.dat'), 'w') as fopen:
            np.savetxt(fopen, rad_paraall, fmt="%s", newline=' ')
            fopen.write('\n')
        with open(os.path.join(Directory, 'ang_para.dat'), 'w') as fopen:
            np.savetxt(fopen, ang_paraall, fmt="%s", newline=' ')
            fopen.write('\n')


<<<<<<< HEAD
class ML:

    def __init__(self, para):
        self.para = para
        self.nfile = int(para['n_dataset'][0])
        self.dataprocess()
        if self.para['testMLmodel'] == 'linear':
            self.linearmodel()

    def dataprocess(self):
        dire = self.para['direfeature']
        nsteps_ = int(self.para['mlsteps'] / self.para['save_steps'])
        natom_ = 5
        if self.para['Lml_skf']:
            fpcompr = open(os.path.join(dire, 'compr.dat'), 'r')
            compr = np.zeros((self.nfile, nsteps_, 5))
            for ifile in range(0, self.nfile):
                datafpcompr = np.fromfile(fpcompr, dtype=float,
                                          count=natom_*nsteps_, sep=' ')
                datafpcompr.shape = (nsteps_, natom_)
                compr[ifile, :, :] = datafpcompr
            self.para['compr_opt_data'] = compr
        if self.para['feature'] == 'rad':
            fprad = open(os.path.join(dire, 'env_rad.dat'), 'r')
            datafprad = np.fromfile(fprad, dtype=float,
                                    count=natom_*self.nfile,  sep=' ')
            datafprad.shape = (self.nfile, natom_)
            self.para['feature_data'] = datafprad

    def linearmodel(self):
        reg = linear_model.LinearRegression()
        X = self.para['feature_data']
        y = self.para['compr_opt_data'][:, -1, :]
        X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.5)
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X)
        plt.scatter(X_test[:, 0], y_test[:, 0],  color='black')
        plt.plot(X[:, 0], y_pred[:, 0], 'ob')
        plt.xlabel('feature of Carbon')
        plt.ylabel('compression radius of Carbon')
        plt.show()
        plt.scatter(X_test[:, 1:], y_test[:, 1:],  color='black')
        plt.plot(X[:, 1:], y_pred[:, 1:], 'ob')
        plt.xlabel('feature of Hydrogen')
        plt.ylabel('compression radius of Hydrogen')
        plt.show()
        self.para['compr_pred'] = t.from_numpy(y_pred)

    def nnmodel(self):
        pass

    def svm(self):
        pass


=======
>>>>>>> ad0a72eac1ab68c13c8d6d1ed31874c22bb490c3
class Net(t.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Net, self).__init__()
        self.linear1 = t.nn.Linear(D_in, H)
        self.linear2 = t.nn.Linear(H, D_out)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


<<<<<<< HEAD
def get_env_para():
    '''this function is to get the environmental parameters'''
    para = {}
    genpara = GenMLPara(para)
    getmlpara(para)
    load = LoadData(para)
    save = SaveData(para)

    load.loadhdfdata()

    os.system('rm env_rad.dat env_ang.dat')
    nbatch = int(para['n_dataset'][0])
    print('begin to calculate environmental parameters')
    symbols = para['symbols']
    for ibatch in range(0, nbatch):
        if type(para['coorall'][ibatch]) is np.array:
            coor = t.from_numpy(para['coorall'][ibatch])
        elif type(para['coorall'][ibatch]) is t.Tensor:
            coor = para['coorall'][ibatch]
        para['coor'] = coor[:]
        genpara.genenvir(nbatch, para, ibatch, coor, symbols)
        # ReadInt(para).get_coor()
        ReadInt(para).cal_coor()
        iang = para['ang_paraall'][ibatch]
        irad = para['rad_paraall'][ibatch]
        # irad = para['distance'][0]
        # irad = irad / (max(irad) + min(irad[1:]))
        save.save1D(iang, name='env_ang.dat', ty='a')
        save.save1D(irad, name='env_rad.dat', ty='a')


if __name__ == '__main__':
    t.autograd.set_detect_anomaly(True)
    t.set_printoptions(precision=15)
    para = {}
    para['task'] = 'dftbml'
    if para['task'] == 'dftbml':
        mainml(para)
    elif para['task'] == 'test':
        testml(para)
    elif para['task'] == 'envpara':
        get_env_para()
=======
if __name__ == '__main__':
    Task = 'dftbml'
    mainml(Task)
    # get_env_para()
>>>>>>> ad0a72eac1ab68c13c8d6d1ed31874c22bb490c3
