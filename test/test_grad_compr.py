#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 10:59:55 2019
@author: guozheng
"""
import os
import numpy as np
from torch.autograd import Variable
import torch as t
import h5py
import time
import write_output as write
import dftbtorch.dftb_torch as dftb_torch
import dftbtorch.slakot as slakot
import utils.plot as plot
import dftbtorch.init_parameter as initpara
import ml.interface as interface
from ml.feature import ACSF as acsfml
from ml.padding import pad1d
from IO.load import LoadData
from IO.save import SaveData
from utils.aset import DFTB, Aims
import dftbtorch.parser as parser
import dftbmalt.utils.maths as maths
from ml.padding import pad1d, pad2d
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
HIRSH_VOL = {"H": 10.31539447, "C": 38.37861207, "N": 29.90025370,
             "O": 23.60491416}


def opt(para):
    """'DFTB-ML optimization."""
    # get the default para for dftb and ML, these para will maintain unchanged
    time_begin = time.time()

    # get the initial parameters
    para['Lbatch'] = True
    para = initpara.dftb_parameter(para)
    skf = initpara.skf_parameter()
    para, ml, dataset = initpara.init_ml(para)
    dataset = initpara.init_dataset(dataset)

    # load dataset, here is hdf type
    ml['ref'] = ml['reference']
    if not ml['ref'] == 'hdf':

        if dataset['dataType'] == 'ani':
            LoadData(para, dataset['n_dataset'][0]).load_ani()
            LoadData(para, dataset['n_dataset'][0]).get_specie_all()

        # dataset is qm7
        elif dataset['dataType'] == 'qm7':
            LoadData(para, dataset['n_dataset'][0]).loadqm7()

        # dataset is json
        elif dataset['dataType'] == 'json':
            LoadData(para, dataset['n_dataset'][0]).load_json_data()
            LoadData(para, dataset['n_dataset'][0]).get_specie_all()

    # run reference calculations, either dft or dftb
    runml = RunML(para, ml, dataset, skf)

    runml.ref()

    # run dftb in ML process
    runml.mldftb()
    time_end = time.time()
    print("total training time:", time_end - time_begin)

    # plot data from ML
    if ml['mlType'] == 'ACSF':
        plot.plot_ml_feature(para)
    else:
        plot.plot_loss2(para, ml)
        # plot.plot_ml_compr(para, ml)


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
            para['coor'] = para['coordinate'][ibatch]
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
    saving ml data (numpy.save type or hdf type)
    Assumption: the atom specie in each dataset maintain unchanged, otherwise
    we have to check for each new moluecle, if there is new atom specie.
    """

    def __init__(self, para, ml, dataset, skf):
        """Initialize DFTB-ML optimization."""
        self.para = para
        self.ml = ml
        self.dataset = dataset
        self.skf = skf

        # some outside class or functions used during DFTB-ML
        self.save = SaveData(self.para)
        self.slako = slakot.SKinterp(self.para, self.dataset, self.skf, self.ml)
        self.genml = GenMLPara(self.para, self.ml)
        self.runcal = RunCalc(self.para, self.dataset, self.skf, self.ml)

    def ref(self):
        """Run different reference calculations according to reference type.

        Args:
            run_reference (L): if run calculations or read from defined file
            nfile (Int): number of total molecules
        """
        # check data and path, rm *.dat, get path for saving data
        self.dire_res = check_data(self.para, rmdata=True)

        # build reference data
        self.build_ref_data()

        # run reference calculations (e.g., DFTB+ ...) before ML
        if self.ml['run_reference']:

            # run DFTB python code as reference
            if self.ml['ref'] == 'dftb':
                self.dftb_ref()

            # run DFTB+ as reference
            elif self.ml['ref'] == 'dftbplus':
                self.dftbplus_ref(self.para)

            # run DFTB+ with ASE interface as reference
            elif self.ml['ref'] == 'dftbase':
                DFTB(self.para, setenv=True).run_dftb(
                    self.nbatch, para['coordinate'])

            # FHI-aims as reference
            elif self.ml['ref'] == 'aims':
                self.aims_ref(self.para)

            # FHI-aims as reference
            elif self.ml['ref'] == 'aimsase':
                Aims(self.para).run_aims(self.nbatch, para['coordinate'])

        # read reference properties from defined dataset
        elif not self.ml['run_reference']:

            # read reference from hdf
            if self.ml['ref'] == 'hdf':
                self.get_hdf_data()

    def build_ref_data(self):
        """Build reference data."""
        # self.nbatch = self.para['nfile']
        self.dataset['refhomo_lumo'] = []
        self.dataset['refenergy'] = []
        self.dataset['refdipole'] = []
        self.dataset['specieall'] = []
        self.dataset['refeigval'] = []

    def get_hdf_data(self):
        """Read data from hdf for reference or the following ML."""
        # join the path and hdf data
        hdffile = os.path.join(self.dataset['path_dataset'],
                               self.dataset['name_dataset'])
        self.dataset['coordinateAll'] = []
        self.dataset['natomall'] = []
        self.dataset['atomnameall'] = []
        # element number of each atom
        self.dataset['atomNumber'] = []
        self.dataset['refcharge'] = []
        self.dataset['refFormEnergy'] = []
        self.dataset['refTotEenergy'] = []
        self.dataset['refhirshfeldvolume'] = []
        self.dataset['refalpha_mbd'] = []
        # the number of atom specie in each system
        self.dataset['numberatom'] = []

        # read global parameters
        with h5py.File(hdffile, 'r') as f:

            # get all the molecule species
            self.skf['specie_all'] = f['globalgroup'].attrs['specie_all']

            # get the molecule species
            molecule = [ikey.encode() for ikey in f.keys()]

            # get rid of b-prefix
            molecule2 = [istr.decode('utf-8') for istr in molecule]

            # delete group which is not related to atom species
            ind = molecule2.index('globalgroup')
            del molecule2[ind]

        # read and mix different molecules
        self.nbatch = 0
        if self.dataset['LdatasetMixture']:

            # loop for molecules in each molecule specie
            for ibatch in range(int(self.dataset['n_dataset'][0])):

                # each molecule specie
                for igroup in molecule2:

                    # add atom name and atom number
                    self.dataset['atomnameall'].append([ii for ii in igroup])
                    self.dataset['natomall'].append(len(igroup))

                    with h5py.File(hdffile, 'r') as f:

                        # coordinates: not tensor now !!!
                        namecoor = str(ibatch) + 'coordinate'
                        self.dataset['coordinateAll'].append(
                            t.from_numpy(f[igroup][namecoor][()]))
                        self.dataset['atomNumber'].append(
                            f[igroup].attrs['atomNumber'])
                        self.dataset['numberatom'].append(t.tensor(
                            [np.count_nonzero(f[igroup].attrs['atomNumber'] == i)
                             for i in [1, 6, 7, 8]], dtype=t.float64))

                        # HOMO LUMO
                        namehl = str(ibatch) + 'humolumo'
                        self.dataset['refhomo_lumo'].append(
                            t.from_numpy(f[igroup][namehl][()]))

                        # charge
                        namecharge = str(ibatch) + 'charge'
                        self.dataset['refcharge'].append(
                            t.from_numpy(f[igroup][namecharge][()]))

                        # dipole
                        namedip = str(ibatch) + 'dipole'
                        self.dataset['refdipole'].append(
                            t.from_numpy(f[igroup][namedip][()]))

                        # formation energy
                        nameEf = str(ibatch) + 'formationenergy'
                        self.dataset['refFormEnergy'].append(f[igroup][nameEf][()])

                        # total energy
                        nameEf = str(ibatch) + 'totalenergy'
                        self.dataset['refTotEenergy'].append(f[igroup][nameEf][()])

                        # get refhirshfeld volume ratios
                        namehv = str(ibatch) + 'hirshfeldvolume'
                        volume = f[igroup][namehv][()]
                        volume = self.get_hirsh_vol_ratio(volume, self.nbatch)
                        self.dataset['refhirshfeldvolume'].append(volume)

                        # polarizability
                        namepol = str(ibatch) + 'alpha_mbd'
                        self.dataset['refalpha_mbd'].append(
                            f[igroup][namepol][()])

                    # total molecule number
                    self.nbatch += 1
                    self.save_ref_idata('hdf', ibatch,
                                        LWHL=self.para['LHL'],
                                        LWeigenval=self.para['Leigval'],
                                        LWenergy=self.para['Lenergy'],
                                        LWdipole=self.para['Ldipole'],
                                        LWpol=self.para['LMBD_DFTB'])
            self.para['nfile'] = self.nbatch
            self.dataset['refFormEnergy'] = t.tensor(self.dataset['refFormEnergy'])

        # read single molecule specie
        elif not self.para['hdf_mixture']:
            # loop for molecules in each molecule specie
            for ibatch in range(int(self.para['n_dataset'][0])):

                # each molecule specie
                iatomspecie = int(para['hdf_num'][0])
                igroup = molecule2[iatomspecie]
                self.para['nfile'] = int(self.para['n_dataset'][0])

                # add atom name
                self.dataset['atomnameall'].append([ii for ii in igroup])
                self.dataset['natomall'].append(len(igroup))

                # read the molecule specie data
                with h5py.File(hdffile) as f:

                    # coordinates: not tensor now !!!
                    namecoor = str(ibatch) + 'coordinate'
                    self.dataset['coordinateAll'].append(
                        t.from_numpy(f[igroup][namecoor][()]))

                    # eigenvalue
                    nameeig = str(ibatch) + 'eigenvalue'
                    self.dataset['refeigval'].append(
                        t.from_numpy(f[igroup][nameeig][()]))

                    # charge
                    namecharge = str(ibatch) + 'charge'
                    self.dataset['refeigval'].append(
                        t.from_numpy(f[igroup][namecharge][()]))

                    # dipole
                    namedip = str(ibatch) + 'dipole'
                    self.dataset['refdipole'].append(
                        t.from_numpy(f[igroup][namedip][()]))

                    # formation energy
                    nameEf = str(ibatch) + 'formationenergy'
                    self.dataset['refenergy'].append(f[igroup][nameEf][()])

                    # total molecule number
                    self.nbatch += 1
                    self.save_ref_idata('hdf', ibatch,
                                        LWHL=self.para['LHL'],
                                        LWeigenval=self.para['Leigval'],
                                        LWenergy=self.para['Lenergy'],
                                        LWdipole=self.para['Ldipole'],
                                        LWpol=self.para['LMBD_DFTB'])

    def dftb_ref(self):
        """Calculate reference with DFTB(torch)"""
        for ibatch in range(self.nbatch):
            # get the tensor type coordinates
            self.get_coor(ibatch)

            # interpolation with compression radius to generate skf data
            if self.para['Lml_skf']:

                # initialize DFTB calculation data
                dftb_torch.Initialization(self.para)

                # get inital compression radius
                self.slako.genskf_interp_dist()

                # interpolate along compression radius (2) dimension
                self.genml.genml_init_compr()
                self.para['LreadSKFinterp'] = False  # read SKF list only once

                # minus compression redius to test the following gradient in ML
                self.para['compr_ml'] = self.para['compr_init'] - 1

                # interpolate along distance dimension
                self.slako.genskf_interp_compr()

                # run DFTB calculations
                self.runcal.idftb_torchspline()

            elif self.para['Lml_HS']:
                dftb_torch.Initialization(self.para)
                self.runcal.idftb_torchspline()

            self.save_ref_idata(ref='dftb', LWHL=self.para['LHL'],
                                LWeigenval=self.para['Leigval'],
                                LWenergy=self.para['Lenergy'],
                                LWdipole=self.para['Ldipole'],
                                LWpol=self.para['LMBD_DFTB'])

    def dftbplus_ref(self, para):
        """Calculate reference (DFTB+)"""
        # get the binary aims
        dftb = write.Dftbplus(self.para)
        bdftb = os.path.join(self.para['dftb_ase_path'], self.para['dftb_bin'])

        # copy executable dftb+ as ./dftbplus/dftb+
        self.dir_ref = os.getcwd() + '/dftbplus'
        os.system('cp ' + bdftb + ' ./dftbplus/dftb+')

        # check binary FHI-aims
        if os.path.isfile(bdftb) is False:
            raise FileNotFoundError("Could not find binary, executable DFTB+")

        for ibatch in range(self.nbatch):
            # get and check the nth coordinates
            self.get_coor(ibatch)

            # check if atom specie is the same to the former
            self.runcal.dftbplus(para, ibatch, self.dir_ref)

        # calculate formation energy
        self.para['totalenergy'] = write.Dftbplus(self.para).read_energy(
            self.para, self.nbatch, self.dir_ref)
        self.para['refenergy'] = self.cal_for_energy(
            self.para['totalenergy'], para['coor'])
        self.para['homo_lumo'] = dftb.read_bandenergy(
            self.para, self.para['nfile'], self.dir_ref)
        self.para['refdipole'] = dftb.read_dipole(
            self.para, self.para['nfile'], self.dir_ref, 'debye', 'eang')
        self.para['alpha_mbd'] = dftb.read_alpha(
            self.para, self.para['nfile'], self.dir_ref)

        # save results for each single molecule
        self.save_ref_data(ref='dftbplus', LWHL=self.para['LHL'],
                           LWeigenval=self.para['Leigval'],
                           LWenergy=self.para['Lenergy'],
                           LWdipole=self.para['Ldipole'],
                           LWpol=self.para['LMBD_DFTB'])

    def aims_ref(self, para):
        """Calculate reference (FHI-aims)"""
        # get the binary aims
        baims = os.path.join(self.para['aims_ase_path'], self.para['aims_bin'])

        # check binary FHI-aims
        if os.path.isfile(baims) is False:
            raise FileNotFoundError("Could not find binary FHI-aims")

        if os.path.isdir('aims'):

            # if exist aims folder, remove files
            os.system('rm ./aims/*.dat')

        elif os.path.isdir('aims'):
            # if exist aims folder
            os.system('mkdir aims')

        # copy executable aims as ./aims/aims
        self.dir_ref = os.getcwd() + '/aims'
        os.system('cp ' + baims + ' ./aims/aim')

        for ibatch in range(self.nbatch):
            # get the nth coordinates
            self.get_coor(ibatch)

            # check if atom specie is the same to the former
            self.runcal.aims(para, ibatch, self.dir_ref)

        # read results, including energy, dipole ...
        self.para['totalenergy'] = write.FHIaims(self.para).read_energy(
            self.para, self.nbatch, self.dir_ref)
        if self.para['mlenergy'] == 'formationenergy':
            self.para['refenergy'] = self.cal_for_energy(
                self.para['totalenergy'], para['coor'])
        self.para['refdipole'] = write.FHIaims(self.para).read_dipole(
            self.para, self.nbatch, self.dir_ref, 'eang', 'eang')
        self.para['homo_lumo'] = write.FHIaims(self.para).read_bandenergy(
            self.para, self.nbatch, self.dir_ref)
        self.para['alpha_mbd'] = write.FHIaims(self.para).read_alpha(
            self.para, self.nbatch, self.dir_ref)
        self.para['refvol'] = write.FHIaims(self.para).read_hirshfeld_vol(
            self.para, self.nbatch, self.dir_ref)

        # save results
        self.save_ref_data(ref='aims', LWHL=self.para['LHL'],
                           LWeigenval=self.para['Leigval'],
                           LWenergy=self.para['Lenergy'],
                           LWdipole=self.para['Ldipole'],
                           LWpol=self.para['LMBD_DFTB'])

    def cal_for_energy(self, energy, coor):
        """calculate formation energy for molecule"""
        if self.para['ref'] == 'aims':
            for ibatch in range(self.nbatch):
                natom = len(self.para['coordinate'][ibatch])
                for iat in range(natom):
                    idx = int(self.para['coordinate'][ibatch][iat, 0])
                    iname = list(ATOMNUM.keys())[list(
                        ATOMNUM.values()).index(idx)]
                    energy[ibatch] -= AIMS_ENERGY[iname]
        elif self.para['ref'] == 'dftb' or self.para['ref'] == 'dftbplus':
            for ibatch in range(self.nbatch):
                natom = len(self.para['coordinate'][ibatch])
                for iat in range(0, natom):
                    idx = int(coor[iat, 0])
                    iname = list(ATOMNUM.keys())[list(ATOMNUM.values()).index(idx)]
                    energy = energy - DFTB_ENERGY[iname]
        return energy

    def get_coor(self, ibatch=None):
        """get the ith coor according to data type"""
        # for batch system
        if ibatch is None:
            if type(self.dataset['coordinateAll']) is t.Tensor:
                coordinate = self.dataset['coordinateAll']
            elif type(self.dataset['coordinateAll']) is np.ndarray:
                coordinate = t.from_numpy(self.dataset['coordinateAll'])
            elif type(self.dataset['coordinateAll']) is list:
                coordinate = self.dataset['coordinateAll']
            self.dataset['coordinate'] = pad2d(coordinate)
        # for single system
        else:
            if type(self.dataset['coordinateAll'][ibatch]) is t.Tensor:
                self.dataset['coordinate'] = self.dataset['coordinateAll'][ibatch][:, :]
            elif type(self.dataset['coordinateAll'][ibatch]) is np.ndarray:
                self.dataset['coordinate'] = \
                    t.from_numpy(self.dataset['coordinateAll'][ibatch][:, :])

    def cal_optfor_energy(self, energy, ibatch):
        natom = self.dataset['natomall'][ibatch]
        for iat in range(natom):
            idx = int(self.dataset['atomNumber'][ibatch][iat])
            iname = list(ATOMNUM.keys())[list(ATOMNUM.values()).index(idx)]
            energy = energy - DFTB_ENERGY[iname]
        return energy

    def cal_offset_energy(self, energy, refenergy):
        A = pad2d(self.dataset['numberatom'])
        B = t.tensor(refenergy) - pad1d(energy)
        offset, _ = t.lstsq(B, A)
        return A @ offset

    def save_ref_idata(self, ref, ibatch, LWHL=False, LWeigenval=False,
                       LWenergy=False, LWdipole=False, LWpol=False):
        """Save data for single molecule calculation result."""
        if LWHL:
            self.save.save1D(self.dataset['refhomo_lumo'][ibatch].numpy(),
                             name='HL'+ref+'.dat', dire=self.dire_res, ty='a')
        if LWeigenval:
            self.para['refeigval'] = self.para['eigenvalue']
            self.save.save1D(self.para['eigenvalue'].detach().numpy(),
                             name='eigval'+ref+'.dat',
                             dire=self.dire_res, ty='a')
        if LWenergy:
            self.save.save1D(self.dataset['refTotEenergy'][ibatch],
                             name='totalenergy'+ref+'.dat',
                             dire=self.dire_res, ty='a')
        if LWdipole:
            self.save.save1D(self.dataset['refdipole'][ibatch].detach().numpy(),
                             name='dip'+ref+'.dat', dire=self.dire_res, ty='a')
        if LWpol:
            self.save.save1D(self.dataset['refalpha_mbd'][ibatch],
                             name='pol'+ref+'.dat', dire=self.dire_res, ty='a')

    def save_ref_data(self, ref, LWHL=False, LWeigenval=False,
                      LWenergy=False, LWdipole=False, LWpol=False):
        """Save data for all molecule calculation results."""
        if LWHL:
            self.para['refhomo_lumo'] = self.para['homo_lumo']
            self.save.save2D(self.para['homo_lumo'].detach().numpy(),
                             name='HL'+ref+'.dat', dire=self.dire_res, ty='a')
        if LWeigenval:
            self.para['refeigval'] = self.para['eigenvalue']
            self.save.save2D(self.para['eigenvalue'].detach().numpy(),
                             name='eigval'+ref+'.dat',
                             dire=self.dire_res, ty='a')
        if LWenergy:
            self.save.save1D(self.para['refenergy'],
                             name='refenergy'+ref+'.dat',
                             dire=self.dire_res, ty='a')
            self.save.save1D(self.para['totalenergy'],
                             name='totalenergy'+ref+'.dat',
                             dire=self.dire_res, ty='a')
        if LWdipole:
            self.save.save2D(self.para['refdipole'].detach().numpy(),
                             name='dip'+ref+'.dat', dire=self.dire_res, ty='a')
        if LWpol:
            self.save.save2D(self.para['alpha_mbd'].detach().numpy(),
                             name='pol'+ref+'.dat', dire=self.dire_res, ty='a')

    def mldftb(self):
        """Run DFTB-ML with various targets, e.g:

            optimize compression radius
            optimize integrals...
        """
        if self.ml['mlType'] == 'integral':
            self.ml_integral()
        elif self.ml['mlType'] == 'compressionRadius' and self.para['Lbatch']:
            self.ml_compr_batch()
        elif self.ml['mlType'] == 'compressionRadius' and not self.para['Lbatch']:
            self.ml_compr()
        elif self.ml['mlType'] == 'integral':
            self.ml_compr_interp()
        elif self.ml['mlType'] == 'ACSF':
            self.ml_acsf(para)

    def ml_integral(self):
        '''DFTB optimization for given dataset'''
        # get the ith coordinates
        self.get_coor()

        # initialize DFTB calculations with datasetmetry and input parameters
        # read skf according to global atom species
        dftb_torch.Initialization(self.para, self.dataset, self.skf)

        # get natom * natom * [ncompr, ncompr, 20] for interpolation DFTB
        self.skf['hs_compr_all_'] = []
        self.para['compr_init_'] = []

        # get spline integral
        ml_variable = self.slako.skf_integral_spline_parameter()  # 0.2 s, too long!!!

        # get loss function type
        if self.ml['loss_function'] == 'MSELoss':
            self.criterion = t.nn.MSELoss(reduction='sum')
        elif self.ml['loss_function'] == 'L1Loss':
            self.criterion = t.nn.L1Loss(reduction='sum')

        # get optimizer
        if self.ml['optimizer'] == 'SCG':
            optimizer = t.optim.SGD(ml_variable, lr=self.ml['lr'])
        elif self.ml['optimizer'] == 'Adam':
            optimizer = t.optim.Adam(ml_variable, lr=self.ml['lr'])

        # calculate one by one to optimize para
        for istep in range(self.ml['mlsteps']):
            loss = 0.
            for ibatch in range(self.nbatch):
                print("step:", istep + 1, "ibatch:", ibatch + 1)

                # do not perform batch calculation
                self.para['Lbatch'] = False

                # get integral at certain distance, read raw integral from binary hdf
                # SK transformations
                slakot.SKTran(self.para, self.dataset, self.skf, self.ml, ibatch)

                # run each DFTB calculation separatedly
                dftb_torch.Rundftbpy(self.para, self.dataset, self.skf, ibatch)

                # define loss function
                # get loss function
                if 'dipole' in self.ml['target']:
                    loss += self.criterion(self.para['dipole'].squeeze(), self.dataset['refdipole'][ibatch])

                # clear gradients and define back propagation
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
            self.save.save1D(np.array([loss]), name='loss.dat', dire='.data', ty='a')

    def ml_compr_batch(self):
        """DFTB optimization of compression radius for given dataset."""
        # clear some documents
        os.system('rm .data/loss.dat .data/compr.dat')

        # calculate one by one to optimize para
        self.para['nsteps'] = t.zeros((self.nbatch), dtype=t.float64)
        self.para['shape_pdos'] = t.zeros((self.nbatch, 2), dtype=t.int32)

        # get the ith coordinates
        self.get_coor()

        # initialize DFTB calculations with datasetmetry and input parameters
        # read skf according to global atom species
        dftb_torch.Initialization(self.para, self.dataset, self.skf)
        maxorb = max(self.dataset['norbital'])

        # get natom * natom * [ncompr, ncompr, 20] for interpolation DFTB
        self.skf['hs_compr_all_'] = []
        self.para['compr_init_'] = []

        # get integral and compression radius
        for ibatch in range(self.nbatch):
            if self.ml['ref'] == 'hdf':
                self.dataset['natom'] = self.dataset['natomall']
                natom = self.dataset['natom'][ibatch]
                atomname = self.dataset['atomnameall'][ibatch]

                # get integral at certain distance, read raw integral from binary hdf
                self.slako.genskf_interp_dist_hdf(ibatch, natom)
                self.skf['hs_compr_all_'].append(self.skf['hs_compr_all'])

                # get initial compression radius
                self.genml.genml_init_compr(ibatch, atomname)
                self.para['compr_init_'].append(self.para['compr_init'])

        self.para['compr_ml'] = \
            pad1d(self.para['compr_init_']).clone().requires_grad_(True)

        # get optimizer
        if self.ml['optimizer'] == 'SCG':
            optimizer = t.optim.SGD([self.para['compr_ml']], lr=self.ml['lr'])
        elif self.ml['optimizer'] == 'Adam':
            optimizer = t.optim.Adam([self.para['compr_ml']], lr=self.ml['lr'])

        for istep in range(self.ml['mlsteps']):
            ham = t.zeros((self.nbatch, maxorb, maxorb), dtype=t.float64)
            over = t.zeros((self.nbatch, maxorb, maxorb), dtype=t.float64)

            # dftb_torch.Initialization(self.para, Lreadskf=False)
            for ibatch in range(self.nbatch):
                self.skf['hs_compr_all'] = self.skf['hs_compr_all_'][ibatch]
                if self.ml['ref'] == 'hdf':
                    self.slako.genskf_interp_compr(ibatch)

                # SK transformations
                slakot.SKTran(self.para, self.dataset, self.skf, self.ml, ibatch)
                iorb = self.dataset['norbital'][ibatch]
                ham[ibatch, :iorb, :iorb] = self.skf['hammat']
                over[ibatch, :iorb, :iorb] = self.skf['overmat']
            self.skf['hammat_'] = ham
            self.skf['overmat_'] = over
            dftb_torch.Rundftbpy(self.para, self.dataset, self.skf, self.nbatch)

            # dftb formation energy calculations
            self.para['formation_energy'] = self.cal_optfor_energy(
                self.para['electronic_energy'], ibatch)

            # get loss function type
            if self.ml['loss_function'] == 'MSELoss':
                self.criterion = t.nn.MSELoss(reduction='sum')
            elif self.ml['loss_function'] == 'L1Loss':
                self.criterion = t.nn.L1Loss(reduction='sum')

            # get loss function
            if 'dipole' in self.ml['target']:
                loss = self.criterion(self.para['dipole'], pad1d(self.dataset['refdipole']))
            print("istep:", istep, '\n loss', loss)
            print(self.para['dipole'] - pad1d(self.dataset['refdipole']))

            # save data
            self.save.save1D(np.array([loss]), name='loss.dat', dire='.data', ty='a')

            # clear gradients and define back propagation
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
        print(self.para['compr_ml'].grad)

    def ml_compr_batch2(self):
        """DFTB optimization of compression radius for given dataset."""
        # clear some documents
        os.system('rm .data/loss.dat .data/compr.dat')

        # calculate one by one to optimize para
        self.para['nsteps'] = t.zeros((self.nbatch), dtype=t.float64)
        self.para['shape_pdos'] = t.zeros((self.nbatch, 2), dtype=t.int32)

        # get the ith coordinates
        self.get_coor()

        # initialize DFTB calculations with datasetmetry and input parameters
        # read skf according to global atom species
        dftb_torch.Initialization(self.para, self.dataset, self.skf)

        # get natom * natom * [ncompr, ncompr, 20] for interpolation DFTB
        self.skf['hs_compr_all_'] = []
        self.para['compr_init_'] = []

        # get integral and compression radius
        for ibatch in range(self.nbatch):
            if self.ml['ref'] == 'hdf':
                natom = self.dataset['natomall'][ibatch]
                atomname = self.dataset['atomnameall'][ibatch]

                # get integral at certain distance, read raw integral from binary hdf
                self.slako.genskf_interp_dist_hdf(ibatch, natom)
                self.skf['hs_compr_all_'].append(self.skf['hs_compr_all'])

                # get initial compression radius
                self.genml.genml_init_compr(ibatch, atomname)
                self.para['compr_init_'].append(self.para['compr_init'])

        self.para['compr_ml'] = \
            Variable(pad1d(self.para['compr_init_']), requires_grad=True)
        # pad1d(self.para['compr_init_']).requires_grad_(True)

        # get optimizer
        if self.ml['optimizer'] == 'SCG':
            optimizer = t.optim.SGD([self.para['compr_ml']], lr=self.ml['lr'])
        elif self.ml['optimizer'] == 'Adam':
            optimizer = t.optim.Adam([self.para['compr_ml']], lr=self.ml['lr'])

        # get loss function type
        if self.ml['loss_function'] == 'MSELoss':
            self.criterion = t.nn.MSELoss(reduction='sum')
        elif self.ml['loss_function'] == 'L1Loss':
            self.criterion = t.nn.L1Loss(reduction='sum')

        # do not perform batch calculation
        self.para['Lbatch'] = False
        for istep in range(self.ml['mlsteps']):

            # set compr_ml every several steps, to avoid memory problem
            # self.para['compr_ml'] = self.para['compr_ml'].clone().requires_grad_(True)
            loss = 0
            if 'offsetenergy' in self.ml['target']:
                initenergy = []

            # dftb_torch.Initialization(self.para, Lreadskf=False)
            for ibatch in range(self.nbatch):
                self.skf['hs_compr_all'] = self.skf['hs_compr_all_'][ibatch]
                if self.ml['ref'] == 'hdf':
                    self.slako.genskf_interp_compr(ibatch)

                # SK transformations
                slakot.SKTran(self.para, self.dataset, self.skf, self.ml, ibatch)

                # run each DFTB calculation separatedly
                dftb_torch.Rundftbpy(self.para, self.dataset, self.skf, ibatch)

                # get loss function
                if 'dipole' in self.ml['target']:
                    loss += self.criterion(self.para['dipole'].squeeze(), self.dataset['refdipole'][ibatch])
                elif 'charge' in self.ml['target']:
                    loss += self.criterion(self.para['charge'].squeeze(), self.dataset['refcharge'][ibatch])
                elif 'homo_lumo' in self.ml['target']:
                    loss += self.criterion(self.para['homo_lumo'].squeeze(), self.dataset['refhomo_lumo'][ibatch])
                elif 'formationenergy' in self.ml['target']:
                    self.para['formation_energy'] = self.cal_optfor_energy(
                        self.para['electronic_energy'], ibatch)
                    loss += self.criterion(self.para['formation_energy'], self.dataset['refFormEnergy'][ibatch])
                elif 'offsetenergy' in self.ml['target']:
                    initenergy.append(self.para['electronic_energy'])
                elif 'cpa' in self.ml['target']:
                    loss += self.criterion(self.para['homo_lumo'].squeeze(), self.dataset['refhirshfeldvolume'][ibatch])

                print("*" * 50, "\n istep:", istep + 1, "\n ibatch", ibatch + 1)
                print("loss:", loss, self.para['compr_ml'].grad)

            # get loss function
            if 'offsetenergy' in self.ml['target']:
                offset = self.cal_offset_energy(initenergy, self.dataset['refFormEnergy'])
                self.para['offsetEnergy'] = pad1d(initenergy) + offset
                print(self.para['offsetEnergy'], self.dataset['refFormEnergy'])
                loss = self.criterion(self.para['offsetEnergy'], self.dataset['refFormEnergy'])

            # save data
            self.save.save1D(np.array([loss]), name='loss.dat', dire='.data', ty='a')
            self.save.save2D(self.para['compr_ml'].detach().numpy(),
                             name='compr.dat', dire='.data', ty='a')

            # clear gradients and define back propagation
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

    def ml_compr(self):
        """DFTB optimization of compression radius for given dataset."""
        # calculate one by one to optimize para
        self.ml['nsteps'] = t.zeros((self.nbatch), dtype=t.float64)
        self.para['shape_pdos'] = t.zeros((self.nbatch, 2), dtype=t.int32)

        for ibatch in range(self.nbatch):

            # if batch calculation
            assert not self.para['Lbatch']

            # temporal ibatch, if calculate system by system, the 1st
            # dimension of DFTB data will be 1 and then ibatch_ will be 0
            ibatch_ = 0

            # get the ith coordinates
            self.get_coor(ibatch)

            # number of atoms in molecule
            nat = self.dataset['coordinate'].shape[0]

            # initialize DFTB calculations with geometry and input parameters
            # read skf according to global atom species
            dftb_torch.Initialization(self.para, self.dataset, self.skf)
            atomname = self.dataset['atomnameall'][ibatch]

            # only read skf data once when calculate the first molecule
            self.para['LreadSKFinterp'] = False

            # get natom * natom * [ncompr, ncompr, 20] for interpolation DFTB
            if self.ml['ref'] == 'hdf':
                self.slako.genskf_interp_dist_hdf(ibatch_, nat)
            else:
                self.slako.genskf_interp_dist(ibatch)

            # update (ML predict) compression radius during ML process
            # if self.ml['Lopt_ml_compr']:
            #    self.update_compr_para()
            # do not update compression radius parameters
            # else:
            self.genml.genml_init_compr(ibatch, atomname)

            # get the reference data
            self.get_iref(ibatch, nat)

            self.para['compr_ml'] = \
                self.para['compr_init'].clone().requires_grad_(True)

            # get optimizer
            if self.ml['optimizer'] == 'SCG':
                optimizer = t.optim.SGD([self.para['compr_ml']], lr=self.ml['lr'])
            elif self.ml['optimizer'] == 'Adam':
                optimizer = t.optim.Adam([self.para['compr_ml']], lr=self.ml['lr'])

            # for each molecule we will run mlsteps
            savestep_ = 0
            for it in range(self.ml['mlsteps']):
                # 2D interpolation with compression radius of atom pairs
                if self.ml['ref'] == 'hdf':
                    self.slako.genskf_interp_compr(ibatch_)
                else:
                    self.slako.genskf_interp_compr(ibatch)

                # dftb calculations
                self.runcal.idftb_torchspline(ibatch)
                self.para['formation_energy'] = self.cal_optfor_energy(
                        self.para['electronic_energy'], ibatch_)

                # get loss function type
                if self.ml['loss_function'] == 'MSELoss':
                    self.criterion = t.nn.MSELoss(reduction='sum')
                elif self.ml['loss_function'] == 'L1Loss':
                    self.criterion = t.nn.L1Loss(reduction='sum')

                # get loss function
                loss = self.get_loss(ibatch_)

                # clear gradients and define back propagation
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

                # save and print information
                if (it + 1) % self.ml['save_steps'] == 0 or it == 0:
                    savestep_ += 1
                    if self.ml['loss_function'] == 'MSELoss':
                        self.loss_ = t.sqrt(loss.detach()) / nat
                    elif self.ml['loss_function'] == 'L1Loss':
                        self.loss_ = loss.detach() / nat
                    self.ml["nsteps"][ibatch_] = savestep_
                    print('-' * 100)
                    print('ibatch: {} steps: {} target: {}'.format(
                              ibatch + 1, it + 1, self.ml['target']))
                    print('average loss: {}'.format(self.loss_))
                    print('compr_ml.grad', para['compr_ml'].grad.detach())
                    print("para['compr_ml']", para['compr_ml'])
                    print('-' * 100)
                    self.save_idftbml()

                # convergence or break condition
                if self.ml['Lopt_step'] and it > self.ml['opt_step_min']:
                    if self.loss_ < self.ml['opt_ml_tol']:
                        print('loss is < {}'.format(self.ml['opt_ml_tol']))
                        break
                if t.lt(self.para['compr_ml'], self.ml['compr_min']).all():
                    print("there is compression R smaller than {}".format(
                        para['compr_min']))
                    break
                if t.gt(self.para['compr_ml'], self.ml['compr_max']).all():
                    print("there is compression larger than {}".format(
                        self.ml['compr_max']))
                    break
        self.save.save1D(self.ml['nsteps'].detach().numpy(),
                         name='nsave.dat', dire='.data', ty='a')

    def update_compr_para(self, ibatch):
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

    def get_iref(self, ibatch, nat):
        """Get reference data for each single molecule."""
        # self.homo_lumo_ref = self.para['refhomo_lumo'][ibatch]
        self.dipref = self.dataset['refdipole'][ibatch]
        if para['LMBD_DFTB']:
            self.pol_ref = self.para['refalpha_mbd'][ibatch][:nat]
            self.volref = self.para['refvol'][ibatch][:nat]
        if 'hstable' in self.ml['target']:
            self.hatableref = self.para['refhammat'][ibatch]
        if 'eigval' in self.ml['target']:
            self.eigvalref = self.para['refeigval'][ibatch]
        if 'qatomall' in self.ml['target']:
            self.qatomall_ref = para['refqatom'][ibatch]
        if self.para['Lenergy']:
            self.energy_ref = self.para['refenergy'][ibatch]

    def get_loss(self, ibatch):
        if len(self.ml['target']) == 1:
            homo_lumo = para['homo_lumo']
            dipole = para['dipole']
            gap = t.abs(homo_lumo[ibatch][1] - homo_lumo[ibatch][0])
            eigval = para['eigenvalue']
            qatomall = para['charge']
            energy = para['energy']
            if 'homo_lumo' in self.ml['target']:
                loss = self.criterion(self.homo_lumo, self.homo_lumo_ref)
            elif 'gap' in self.ml['target']:
                loss = self.criterion(gap, self.gapref)
            elif 'dipole' in self.ml['target']:
                loss = self.criterion(dipole, self.dipref)
            elif 'hstable' in self.ml['target']:
                hstable = self.skf['hammat']
                loss = self.criterion(hstable, self.hatableref)
            elif 'eigval' in self.ml['target']:
                loss = self.criterion(eigval, self.eigvalref)
            elif 'qatomall' in self.ml['target']:
                loss = self.criterion(qatomall, self.qatomall_ref)
            elif 'energy' in self.ml['target']:
                loss = self.criterion(energy, self.energy_ref)
            elif 'polarizability' in self.ml['target']:
                pol = self.para['alpha_mbd']
                loss = self.criterion(pol, self.pol_ref)
            elif 'cpa' in self.ml['target']:
                cpa = self.para['cpa']
                vol_ratio_ref = self.get_hirsh_vol_ratio(self.volref)
                loss = self.criterion(cpa, vol_ratio_ref)
            elif 'pdos' in self.para['target']:
                pdosref = self.para['pdosdftbplus'][ibatch]
                loss = maths.hellinger(para['pdos'], pdosref).sum()
                # loss = self.criterion(para['pdos'], pdosref)
                self.para['shape_pdos'][ibatch][0] = self.para['pdos'].shape[0]
                self.para['shape_pdos'][ibatch][1] = self.para['pdos'].shape[1]
            elif len(self.ml['target']) == 2:
                if 'dipole' and 'polarizability' in self.ml['target']:
                    dipole = self.para['dipole']
                    pol = self.para['alpha_mbd']
                    loss = 2 * self.criterion(pol, self.pol_ref) + \
                        self.criterion(dipole, self.dipref)
            return loss

    def save_idftbml(self):
        #self.save.save1D(para['homo_lumo'].detach().numpy(),
        #                 name='HLbp.dat', dire='.data', ty='a')
        if self.para['Ldipole']:
            print('dipole: {}, dipref: {}'.format(
                self.para['dipole'], self.dipref))
            self.save.save1D(self.para['dipole'].detach().numpy(),
                             name='dipbp.dat', dire='.data', ty='a')
        if self.para['Lenergy']:
            print('energy: {}, energyref: {}'.format(
                self.para['formation_energy'], self.energy_ref))
            self.save.save1D(self.para['formation_energy'].detach().numpy().squeeze(),
                             name='form_energybp.dat', dire='.data', ty='a')
            if self.para['Lrepulsive']:
                self.save.save1D(self.para['rep_energy'].detach().numpy(),
                                 name='repenergybp.dat', dire='.data', ty='a')
        if self.para['LMBD_DFTB']:
            self.save.save1D(para['alpha_mbd'].detach().numpy(),
                             name='polbp.dat', dire='.data', ty='a')
            self.save.save1D(self.para['cpa'].detach().numpy(),
                             name='cpa.dat', dire='.data', ty='a')
        if self.para['Lpdos']:
            self.save.save2D(self.para['pdos'].detach().numpy(),
                             name='pdosbp.dat', dire='.data', ty='a')
            self.save.save1D(self.para['hammat'].detach().numpy(),
                             name='hambp.dat', dire='.data', ty='a')
            self.save.save1D(self.para['compr_ml'].detach().numpy(),
                             name='comprbp.dat', dire='.data', ty='a')
            self.save.save1D(self.para['charge'].detach().numpy(),
                             name='qatombp.dat', dire='.data', ty='a')
        self.save.save1D(self.loss_, name='lossbp.dat', dire='.data', ty='a')

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
                self.slako.genskf_interp_dist()
                self.para['LreadSKFinterp'] = False  # read SKF list only once

                # get compr by weight parameter and atomic structure parameter
                ml.dataprocess_atom()  # get atomic structure parameter
                para['compr_ml'] = (para['acsf_mlpara'] @ acsf_weight) + bias

                # dftb calculations (choose scc or nonscc)
                if para['Lml_compr_global']:
                    self.genml.genml_init_compr()

                # 2D interpolation with compression radius of atom pairs
                if self.para['dataType'] == 'hdf':
                    self.slako.genskf_interp_compr_vec()
                else:
                    self.slako.genskf_interp_compr()

                self.runcal.idftb_torchspline()
                self.para['formation_energy'] = self.cal_optfor_energy(
                        self.para['electronic_energy'], self.para['coor'])

                # get optimizer
                if self.para['optimizer'] == 'SCG':
                    optimizer = t.optim.SGD([para['compr_ml']], lr=para['lr'])
                elif self.para['optimizer'] == 'Adam':
                    optimizer = t.optim.Adam([para['compr_ml']], lr=para['lr'])

                # for each molecule we will run mlsteps
                savestep_ = 0
                # get loss function type
                if self.para['loss_function'] == 'MSELoss':
                    self.criterion = t.nn.MSELoss(reduction='sum')
                elif self.para['loss_function'] == 'L1Loss':
                    self.criterion = t.nn.L1Loss(reduction='sum')

                # get loss function
                loss = self.get_loss(ibatch)

                # clear gradients and define back propagation
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

                # save and print information
                if (it + 1) % para['save_steps'] == 0 or it == 0:
                    savestep_ += 1
                    if self.para['loss_function'] == 'MSELoss':
                        self.loss_ = t.sqrt(loss.detach()) / nat
                    elif self.para['loss_function'] == 'L1Loss':
                        self.loss_ = loss.detach() / nat
                    self.para["nsteps"][ibatch] = savestep_
                    print('-' * 100)
                    print('ibatch: {} steps: {} target: {}'.format(
                              ibatch + 1, it + 1, para['target']))
                    print('average loss: {}'.format(self.loss_))
                    print('compr_ml.grad', para['compr_ml'].grad.detach())
                    print("para['compr_ml']", para['compr_ml'])
                    # print('homo_lumo: {}, homo_lumo_ref: {}'.format(
                    #      self.para['homo_lumo'], self.homo_lumo_ref))
                    print('-' * 100)
                    self.save_idftbml()
                    self.save.save1D(bias.detach().numpy(),
                                     name='bias.dat', dire='.data', ty='a')
                    self.save.save1D(acsf_weight.detach().numpy(),
                                     name='weight.dat', dire='.data', ty='a')

                # convergence or break condition
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

    def get_hirsh_vol_ratio(self, volume, ibatch=0):
        """Get Hirshfeld volume ratio."""
        natom = self.dataset["natomall"][ibatch]
        for iat in range(natom):
            idx = int(self.dataset['atomNumber'][ibatch][iat])
            iname = list(ATOMNUM.keys())[list(ATOMNUM.values()).index(idx)]
            volume[iat] = volume[iat] / HIRSH_VOL[iname]
        return volume

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
            self.slako.genskf_interp_dist()
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
            self.save.save1D(para['charge'].numpy(),
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
        if type(para['coordinate'][0]) is t.Tensor:
            para['coor'] = para['coordinate'][0][:, :]
        elif type(para['coordinate'][0]) is np.ndarray:
            para['coor'] = t.from_numpy(para['coordinate'][0][:, :])
        dftb_torch.Initialization(para)
        # interpskf(para)
        self.genml.genml_init_compr()
        para['compr_ml'] = para['compr_init'] + 1
        self.slako.genskf_interp_dist(para)
        self.slako.genskf_interp_r(para)
        hs_ref = para['h_s_all']

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

    def __init__(self, para, ml):
        self.para = para
        self.ml = ml

    def genmlpara0(self, para):
        pass

    def genml_init_compr(self, ibatch, atomname):
        """Get initial compression radius for each atom in system."""
        self.para['compr_init'] = t.tensor(
            [self.ml[ia + '_init_compr'] for ia in atomname], dtype=t.float64)


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


class RunCalc:
    """Run different DFT(B) calculations.

    with both ASE interface or code in write_output.py
    """

    def __init__(self, para, dataset, skf, ml):
        self.para = para
        self.skf = skf
        self.dataset = dataset
        self.ml = ml

    def aims(self, para, ibatch, dire):
        """DFT means FHI-aims here."""
        coor = para['coor']
        self.para['natom'] = int(self.para['natomall'][ibatch])
        write.FHIaims(para).geo_nonpe_hdf(para, ibatch, coor[:, 1:])
        os.rename('geometry.in.{}'.format(ibatch), 'aims/geometry.in')
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
        dipolem, eigval = dftb_torch.dftb(para)
        dipolemall.append(dipolem)
        eigvalall.append(eigval)
        para['dipolemall'] = dipolemall
        para['eigvalall'] = eigvalall
        return para

    def idftb_torchspline(self, ibatch=None):
        """Perform DFTB_python with integrals."""
        # dftb_torch.Initialization(self.para).gen_sk_matrix(self.para)
        slakot.SKTran(self.para, self.dataset, self.skf, self.ml, ibatch)
        dftb_torch.Rundftbpy(self.para, self.dataset, self.skf, ibatch)


def check_data(para, rmdata=False):
    """Check and build folder/data before run or read reference part."""
    if 'dire_result' in para.keys():
        dire_res = para['dire_result']

        # rm .dat file
        if rmdata:
            os.system('rm ' + dire_res + '/*.dat')

    # build new directory for saving results
    else:
        # do not have .data folder
        if os.path.isdir('.data'):

            # rm all the .dat files
            if rmdata:
                os.system('rm .data/*.dat')

        # .data folder exist
        else:
            os.system('mkdir .data')

        # new path for saving result
        dire_res = os.path.join(os.getcwd(), '.data')

    return dire_res


if __name__ == "__main__":
    """Main function for optimizing DFTB parameters, testing DFTB."""
    # automatically test the gradient problem
    t.autograd.set_detect_anomaly(True)

    # set the print precision
    t.set_printoptions(precision=15)

    # set the data type precision
    t.set_default_dtype(d=t.float64)
    para = {}

    # interface to shell terminal and get the default task
    parser.parser_cmd_args(para)
    if para['task'] == 'opt':
        opt(para)
    elif para['task'] == 'test':
        testml(para)
    elif para['task'] == 'envpara':
        initpara.init_dftb_ml(para)
        interface.get_env_para()
    time_end = time.time()
