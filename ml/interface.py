"""Interface to some popular ML framework."""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import torch as t
from ase import Atoms
from sklearn import linear_model, svm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from dscribe.descriptors import ACSF
from dscribe.descriptors import CoulombMatrix
from IO.readt import ReadIn
from ml.feature import ACSF as acsfml
from IO.load import LoadData
from IO.save import SaveData
ATOMNUM = {'H': 1, 'C': 6, 'N': 7, 'O': 8}


class ML:
    """Machine learning with optimized data.

    process data.
    perform ML prediction.
    """

    def __init__(self, para):
        """Initialize ML process.

        nfile is the optimization dataset number
        ntest is the test dataset number
        """
        self.para = para
        self.read = Read(para)
        self.dscribe = Dscribe(self.para)
        # self.dataprocess(self.para['dire_data'])  # generate ML X, Y data
        # self.ml_compr()

    def dataprocess(self, diredata):
        """Process the optimization dataset and data for the following ML.

        Returns:
            features of ML (X)
            traing target (Y, e.g, compression radius)

        """
        ntrain = self.para['ntrain']  # training number
        ntest = self.para['npred']  # number for prediction
        if self.para['Lopt_step']:
            fp = open(os.path.join(self.para['dire_data'], 'nsave.dat'), 'r')
            self.para['nsteps'] = np.fromfile(fp, dtype=float, count=-1,
                                              sep=' ')
            nsteps_ = self.para['nsteps']
        else:
            nsteps_ = int(self.para['mlsteps'] / self.para['save_steps'])

        if self.para['Lml_skf']:
            self.para['optRall'] = self.read.read3d_rand(
                diredata, 'comprbp.dat', self.para['natomall'], nsteps_)

        if self.para['featureType'] == 'rad':
            get_env_para(self.para)
            self.para['feature_data'] = self.para['x_rad'][:ntrain]
            self.para['feature_test'] = self.para['x_rad'][:ntest]
            self.para['feature_target'] = self.para['optRall'][:, -1, :]
        elif self.para['featureType'] == 'cm':
            self.dscribe.pro_()
            self.get_target_to1d(self.para['natomall'],
                                 self.para['optRall'], ntrain)
        elif self.para['featureType'] == 'acsf':
            self.dscribe.pro_()
            self.get_target_to1d(self.para['natomall'],
                                 self.para['optRall'], ntrain)

    def dataprocess_atom(self):
        """ML process for compression radius."""
        if self.para['featureType'] == 'acsf':
            self.para['acsf_mlpara'] = self.dscribe.pro_molecule()

    def ml_compr(self):
        """ML process for compression radius."""
        if self.para['testMLmodel'] == 'linear':
            self.linearmodel()
        elif self.para['testMLmodel'] == 'schnet':
            self.schnet()
        elif self.para['testMLmodel'] == 'svm':
            self.svm_model()

    def ml_acsf(self):
        """Generate opreeR with optimized ML parameters and fingerprint."""
        weight = self.para['test_weight']
        bias = self.para['test_bias']
        return self.para['acsf_mlpara'] @ weight + bias

    def get_test_para(self):
        """Read optimized parameters(wieght, bias...) with given ratio."""
        dire = self.para['dire_data']
        ratio = self.para['opt_para_test']
        fpw = open(os.path.join(dire, 'weight.dat'))
        fpb = open(os.path.join(dire, 'bias.dat'))
        dim = self.para['acsf_dim']
        bias = t.from_numpy(np.fromfile(fpb, dtype=float, count=-1, sep=' '))
        nbatch_step = bias.shape[0]
        weight = t.from_numpy(np.fromfile(
            fpw, dtype=float, count=-1, sep=' ').reshape(nbatch_step, dim))
        num = int(ratio * (nbatch_step - 1))
        self.para['test_weight'] = weight[num]
        self.para['test_bias'] = bias[num]

    def linearmodel(self):
        """Use the optimization dataset for training.

        Returns:
            linear ML method predicted DFTB parameters
        shape[0] of feature_data is defined by the optimized compression R
        shape[0] of feature_test is the defined by para['n_test']

        """
        reg = linear_model.LinearRegression()
        X = self.para['feature_data']
        X_pred = self.para['feature_test']
        y = self.para['feature_target']
        X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.5)

        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_pred)
        if self.para['featureType'] == 'rad':
            self.para['compr_pred'] = t.from_numpy(y_pred)
        elif self.para['featureType'] == 'acsf':
            self.para['compr_pred'] = \
                self.get_target_to2d(self.para['natomall'], y_pred)
        elif self.para['featureType'] == 'cm':
            self.para['compr_pred'] = \
                self.get_target_to2d(self.para['natomall'], y_pred)

        if self.para['Lplot_feature']:
            plt.scatter(X_train, X_train,  color='black')
            plt.plot(X_train, y_train, 'ob')
            plt.xlabel('feature of training dataset')
            plt.ylabel('traning compression radius')
            plt.show()
            # plt.scatter(X_pred, y_pred,  color='black')
            if self.para['featureType'] == 'rad':
                plt.plot(X_pred, y_pred, 'ob')
            elif self.para['featureType'] == 'acsf':
                plt.plot(X_pred[:, 0], y_pred, 'ob')
            elif self.para['featureType'] == 'cm':
                plt.plot(X_pred[:, 0], y_pred, 'ob')
            plt.xlabel('feature of prediction (tesing)')
            plt.ylabel('testing compression radius')
            plt.show()

    def svm_model(self):
        """ML process with support vector machine method."""
        reg = svm.SVR()
        X = self.para['feature_data']
        X_pred = self.para['feature_test']
        y = self.para['feature_target']
        X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.5)

        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_pred)
        if self.para['featureType'] == 'rad':
            self.para['compr_pred'] = t.from_numpy(y_pred)
        elif self.para['featureType'] == 'acsf':
            self.para['compr_pred'] = \
                self.get_target_to2d(self.para['natomall'], y_pred)
        elif self.para['featureType'] == 'cm':
            self.para['compr_pred'] = \
                self.get_target_to2d(self.para['natomall'], y_pred)

        if self.para['Lplot_feature']:
            plt.scatter(X_train, X_train,  color='black')
            plt.plot(X_train, y_train, 'ob')
            plt.xlabel('feature of training dataset')
            plt.ylabel('traning compression radius')
            plt.show()
            # plt.scatter(X_pred, y_pred,  color='black')
            if self.para['featureType'] == 'rad':
                plt.plot(X_pred, y_pred, 'ob')
            elif self.para['featureType'] == 'acsf':
                plt.plot(X_pred[:, 0], y_pred, 'ob')
            elif self.para['featureType'] == 'cm':
                plt.plot(X_pred[:, 0], y_pred, 'ob')
            plt.xlabel('feature of prediction (tesing)')
            plt.ylabel('testing compression radius')
            plt.show()

    def schnet(self):
        """ML process with schnetpack."""
        pass

    def get_target_to1d(self, nall, comprall, ntrain):
        """Transfer target fro [ntrain, natom] to [ntrain * natom] type.

        ntrain should be equal to the feature size, and smaller than the
        optimized dataset size.
        """
        nmax = self.para['natommax']
        nsteps = self.para["nsteps"]
        # nmax_ = comprlast.shape[1]
        # assert nmax == nmax_
        feature_target = t.zeros((ntrain * nmax), dtype=t.float64)
        for ibatch in range(ntrain):
            comprlast = comprall[ibatch, int(nsteps[ibatch]) - 1, :]
            nat = int(nall[ibatch])
            feature_target[ibatch * nmax: ibatch * nmax + nat] = \
                comprlast[:nat]
        self.para['feature_target'] = feature_target

    def get_target_to2d(self, nall, comprall):
        """Transfer target from [nbatch, natom] to [nbatch * natom] type."""
        nmax = self.para['natommax']
        nbatch = int(comprall.shape[0] / nmax)
        if type(comprall) is np.ndarray:
            comprall = t.from_numpy(comprall)
        feature_target = t.zeros((nbatch, nmax), dtype=t.float64)
        for ibatch in range(nbatch):
            nat = int(nall[ibatch])
            feature_target[ibatch, :nat] = \
                comprall[ibatch * nmax: ibatch * nmax + nat]
        return feature_target


class Dscribe:
    """Interface to Dscribe.

    Returns:
        features for machine learning

    """

    def __init__(self, para):
        """Initialize the parameters."""
        self.para = para

    def pro_(self):
        """Process data for Dscribe."""
        nbatch = self.para['npred']
        ndataset = self.para['ntrain']
        nfile = max(nbatch, ndataset)
        nmax = int(max(self.para['natomall']))

        if self.para['featureType'] == 'cm':  # flatten=True for all!!!!!
            features = t.zeros((nfile * nmax, nmax), dtype=t.float64)
        elif self.para['featureType'] == 'acsf':
            self.get_acsf_dim()
            col = self.para['acsf_dim']  # get ACSF feature dimension
            features = t.zeros((nfile * nmax, col), dtype=t.float64)
        for ibatch in range(nfile):
            if type(self.para['coorall'][ibatch]) is np.array:
                self.para['coor'] = t.from_numpy(self.para['coorall'][ibatch])
            elif type(self.para['coorall'][ibatch]) is t.Tensor:
                self.para['coor'] = self.para['coorall'][ibatch]
            nat_ = int(self.para['natomall'][ibatch])

            if self.para['featureType'] == 'cm':
                features[ibatch * nmax: ibatch * nmax + nat_, :nat_] = \
                    self.coulomb(n_atoms_max_=nmax)[:nat_, :nat_]
            elif self.para['featureType'] == 'acsf':
                features[ibatch * nmax: ibatch * nmax + nat_, :] = self.acsf()
        self.para['natommax'] = nmax
        self.para['feature_test'] = features[:nbatch * nmax, :]
        self.para['feature_data'] = features[:ndataset * nmax, :]

    def pro_molecule(self):
        """Get atomic environment parameter only for single molecule."""
        nmax = int(max(self.para['natomall']))
        if self.para['featureType'] == 'cm':  # flatten=True for all!!!!!
            features = t.zeros((nmax, nmax), dtype=t.float64)
        elif self.para['featureType'] == 'acsf':
            col = self.para['acsf_dim']
            features = t.zeros((nmax, col), dtype=t.float64)
        if type(self.para['coor']) is np.array:
            self.para['coor'] = t.from_numpy(self.para['coor'])
        elif type(self.para['coor']) is t.Tensor:
            pass
        nat_ = self.para['coor'].shape[0]

        if self.para['featureType'] == 'cm':
            features[:nat_, :nat_] = \
                self.coulomb(n_atoms_max_=nmax)[:nat_, :nat_]
        elif self.para['featureType'] == 'acsf':
            features[:nat_, :] = self.acsf()
        return features

    def get_acsf_dim(self):
        """Get the dimension (column) of ACSF method."""
        nspecie = len(self.para['specie_all'])
        col = 0
        if nspecie == 1:
            n_types, n_type_pairs = 1, 1
        elif nspecie == 2:
            n_types, n_type_pairs = 2, 3
        elif nspecie == 3:
            n_types, n_type_pairs = 3, 6
        elif nspecie == 4:
            n_types, n_type_pairs = 4, 10
        elif nspecie == 5:
            n_types, n_type_pairs = 5, 15
        col += n_types  # G0
        if self.para['Lacsf_g2']:
            col += len(self.para['acsf_g2']) * n_types  # G2
        if self.para['Lacsf_g4']:
            col += (len(self.para['acsf_g4'])) * n_type_pairs  # G4
        self.para['acsf_dim'] = col

    def coulomb(self, rcut=6.0, nmax=8, lmax=6, n_atoms_max_=6):
        """Coulomb method for atomic environment.

        Phys. Rev. Lett., 108:058301, Jan 2012.
        """
        cm = CoulombMatrix(n_atoms_max=n_atoms_max_)
        coor = self.para['coor']
        atomspecie = []
        for iat in range(coor.shape[0]):
            idx = int(coor[iat, 0])
            atomspecie.append(
                list(ATOMNUM.keys())[list(ATOMNUM.values()).index(idx)])
        atom = Atoms(atomspecie, positions=coor[:, 1:])
        cm_test = cm.create(atom)
        return t.from_numpy(cm_test)

    def sine(self):
        pass

    def ewald(self):
        pass

    def acsf(self):
        """Atom-centered Symmetry Functions method for atomic environment.

        J. chem. phys., 134.7 (2011): 074106.
        You should define all the atom species to fix the feature dimension!
        """
        coor = self.para['coor']
        rcut_ = self.para['acsf_rcut']
        specie_global = self.para['specie_all']
        if self.para['Lacsf_g2']:
            g2_params_ = self.para['acsf_g2']
        else:
            g2_params_ = None
        if self.para['Lacsf_g4']:
            g4_params_ = self.para['acsf_g4']
        else:
            g4_params_ = None
        acsf = ACSF(species=specie_global,
                    rcut=rcut_,
                    g2_params=g2_params_,
                    g4_params=g4_params_,
                    )

        atomspecie = []
        for iat in range(coor.shape[0]):
            idx = int(coor[iat, 0])
            atomspecie.append(
                list(ATOMNUM.keys())[list(ATOMNUM.values()).index(idx)])
        test_module = Atoms(atomspecie, positions=coor[:, 1:])
        acsf_test = acsf.create(test_module)
        return t.from_numpy(acsf_test)

    def soap(self):
        pass

    def manybody(self):
        pass

    def localmanybody(self):
        pass

    def kernels(self):
        pass


class Schnetpack:
    """Interface to Schnetpack for NN."""

    def __init__():
        pass


class Read:
    """Simple reading code.

    """

    def __init__(self, para):
        """Initialize data."""
        self.para = para

    def read1d(self, dire, name, number, outtype='torch'):
        """Read one dimentional data."""
        fp = open(os.path.join(dire, name), 'r')
        data = np.zeros((number), dtype=float)
        data[:] = np.fromfile(fp, dtype=int, count=number, sep=' ')
        if outtype == 'torch':
            return t.from_numpy(data)
        elif outtype == 'numpy':
            return data

    def read2d(self, dire, name, num1, num2, outtype='torch'):
        """Read two dimentional data."""
        data = np.zeros((num1, num2), dtype=float)
        fp = open(os.path.join(dire, name), 'r')
        for inum1 in range(0, num1):
            idata_ = np.fromfile(fp, dtype=float, count=num2, sep=' ')
            data[inum1, :] = idata_
        if outtype == 'torch':
            return t.from_numpy(data)
        elif outtype == 'numpy':
            return data

    def read2d_rand(self, dire, name, nall, num1, outtype='torch'):
        """Read two dimentional data, which may not be all filled."""
        nmax = int((nall).max())
        data = np.zeros((num1, nmax), dtype=float)
        fp = open(os.path.join(dire, name), 'r')
        for inum1 in range(num1):
            num_ = int(nall[inum1])
            idata_ = np.fromfile(fp, dtype=float, count=num_, sep=' ')
            data[inum1, :num_] = idata_
        if outtype == 'torch':
            return t.from_numpy(data)
        elif outtype == 'numpy':
            return data

    def read3d_rand(self, dire, name, nall, ntemp, outtype='torch'):
        """Read three dimentional data, which the out may not be all filled.

        Args:
            nall: the atom number of all batch
            ntemp: usually the saved steps for one batch, this can be fixed or
            not depends on Lopt_step parameter

        """
        nmax = int(max(nall))
        ntrain = self.para['ntrain']
        if self.para['Lopt_step']:
            nstepmax = int(max(ntemp))
            data = np.zeros((ntrain, nstepmax, nmax), dtype=float)
        else:
            ntemp_ = ntemp
            data = np.zeros((ntrain, ntemp, nmax), dtype=float)

        # read data
        fp = open(os.path.join(dire, name), 'r')
        for inum1 in range(ntrain):
            num_ = int(nall[inum1])
            ntemp_ = int(ntemp[inum1])
            idata_ = np.fromfile(fp, dtype=float, count=num_*ntemp_, sep=' ')
            idata_.shape = (ntemp_, num_)
            data[inum1, :ntemp_, :num_] = idata_

        if outtype == 'torch':
            return t.from_numpy(data)
        elif outtype == 'numpy':
            return data


def get_env_para(para):
    """Get the environmental parameters."""
    dire_ = para['dire_data']
    natomall = para['natomall']
    nmax = int(natomall.max())
    os.system('rm ' + dire_ + '/env_rad.dat')
    os.system('rm ' + dire_ + '/env_ang.dat')
    load = LoadData(para)
    save = SaveData(para)
    read = ReadInt(para)

    if len(para['n_dataset']) == 1:
        nbatch = max(para['ntrain'], para['npred'])
    rad = t.zeros((nbatch, nmax), dtype=t.float64)
    ang = t.zeros((nbatch, nmax), dtype=t.float64)

    load.loadhdfdata()
    print('begin to calculate environmental parameters')
    symbols = para['symbols']
    for ibatch in range(nbatch):
        if type(para['coorall'][ibatch]) is np.array:
            coor = t.from_numpy(para['coorall'][ibatch])
        elif type(para['coorall'][ibatch]) is t.Tensor:
            coor = para['coorall'][ibatch]
        nat_ = int(natomall[ibatch])
        para['coor'] = coor[:]
        genenvir(nbatch, para, ibatch, coor, symbols)
        read.cal_coor()
        ang[ibatch, :nat_] = t.from_numpy(para['ang_paraall'][ibatch])
        rad[ibatch, :nat_] = t.from_numpy(para['rad_paraall'][ibatch])

        save.save1D(ang[ibatch, :nat_], name='env_ang.dat', dire=dire_, ty='a')
        save.save1D(rad[ibatch, :nat_], name='env_rad.dat', dire=dire_, ty='a')
    para['x_rad'] = rad
    para['x_ang'] = ang

def genenvir(nbatch, para, ifile, coor, symbols):
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
    natom = len(coor)
    nbatch += natom * (natom+1)
    return para, nbatch
