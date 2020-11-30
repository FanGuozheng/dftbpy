"""Interface to some popular ML framework."""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import torch as t
from sklearn import linear_model, svm
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from ml.feature import Dscribe
from IO.save import Save1D, Save2D
ATOMNUM = {'H': 1, 'C': 6, 'N': 7, 'O': 8}


class MLPara:
    """Machine learning with optimized data.

    process data.
    perform ML prediction.
    """

    def __init__(self, para, dataset, ml):
        """Initialize ML process.

        nfile is the optimization dataset number
        ntest is the test dataset number
        """
        self.para = para
        self.dataset = dataset
        self.ml = ml
        self.read = Read(para)
        # get fearure data according to geometry
        self.dscribe = Dscribe(self.para, self.dataset, self.ml)
        self.dataprocess()
        if self.para['task'] == 'testCompressionR':
            self.ml_compr()

    def dataprocess(self):
        """Process the optimization dataset and data for the following ML.

        Returns:
            features of ML (X)
            traing target (Y, e.g, compression radius)

        """
        ntrain = self.dataset['nbatch']  # training number
        ntest = self.dataset['ntest']  # number for prediction
        self.dscribe.pro_(ntrain, ntest)
        self.para['feature_target'] = t.flatten(self.ml['optCompressionR'])

    def dataprocess_atom(self):
        """ML process for compression radius."""
        if self.para['featureType'] == 'acsf':
            self.para['acsf_mlpara'] = self.dscribe.pro_molecule()

    def ml_compr(self):
        """ML process for compression radius."""
        if self.ml['MLmodel'] == 'linear':
            self.linearmodel()
        elif self.ml['MLmodel'] == 'svm':
            self.svm_model()
        elif self.ml['MLmodel'] == 'nn':
            self.nn_model()

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
        bias = t.from_numpy(np.fromfile(fpb, count=-1, sep=' '))
        nbatch_step = bias.shape[0]
        weight = t.from_numpy(np.fromfile(
            fpw, count=-1, sep=' ').reshape(nbatch_step, dim))
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
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
        reg.fit(X, y)
        y_pred = reg.predict(X_pred)
        self.para['compr_pred'] = t.clamp(self.get_target_to2d(y_pred),
                                          self.ml['compressionRMin'],
                                          self.ml['compressionRMax'])

    def svm_model(self):
        """ML process with support vector machine method."""
        reg = svm.SVR()
        X = self.para['feature_data']
        X_pred = self.para['feature_test']
        y = self.para['feature_target']
        # X_train, X_test, y_train, y_test = train_test_split(
        #         X, y, test_size=0.5)
        reg.fit(X, y)
        y_pred = reg.predict(X_pred)
        self.para['compr_pred'] = t.clamp(self.get_target_to2d(y_pred),
                                          self.ml['compressionRMin'],
                                          self.ml['compressionRMax'])
    def nn_model(self):
        """ML process with support vector machine method."""
        clf = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
        X = self.para['feature_data']
        X_pred = self.para['feature_test']
        y = self.para['feature_target']
        clf.fit(X.numpy(), y.numpy())
        y_pred = t.from_numpy(clf.predict(X_pred.numpy()))
        self.para['compr_pred'] = t.clamp(self.get_target_to2d(y_pred),
                                          self.ml['compressionRMin'],
                                          self.ml['compressionRMax'])

    def schnet(self):
        """ML process with schnetpack."""
        pass

    def get_target_to2d(self, compression_r):
        """Transfer target from [nbatch, natom] to [nbatch * natom] type."""
        nmax = self.para['natommax']
        nbatch = int(compression_r.shape[0] / nmax)
        if type(compression_r) is np.ndarray:
            compression_r = t.from_numpy(compression_r)

        return compression_r.resize(nbatch, nmax)


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
        data = np.zeros(number)
        data[:] = np.fromfile(fp, dtype=int, count=number, sep=' ')
        if outtype == 'torch':
            return t.from_numpy(data)
        elif outtype == 'numpy':
            return data

    def read2d(self, dire, name, num1, num2, outtype='torch'):
        """Read two dimentional data."""
        data = np.zeros(num1, num2)
        fp = open(os.path.join(dire, name), 'r')
        for inum1 in range(0, num1):
            idata_ = np.fromfile(fp, count=num2, sep=' ')
            data[inum1, :] = idata_
        if outtype == 'torch':
            return t.from_numpy(data)
        elif outtype == 'numpy':
            return data

    def read2d_rand(self, dire, name, nall, num1, outtype='torch'):
        """Read two dimentional data, which may not be all filled."""
        nmax = int((nall).max())
        data = np.zeros(num1, nmax)
        fp = open(os.path.join(dire, name), 'r')
        for inum1 in range(num1):
            num_ = int(nall[inum1])
            idata_ = np.fromfile(fp, count=num_, sep=' ')
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
            data = np.zeros(ntrain, nstepmax, nmax)
        else:
            ntemp_ = ntemp
            data = np.zeros(ntrain, ntemp, nmax)

        # read data
        fp = open(os.path.join(dire, name), 'r')
        for inum1 in range(ntrain):
            num_ = int(nall[inum1])
            ntemp_ = int(ntemp[inum1])
            idata_ = np.fromfile(fp, count=num_*ntemp_, sep=' ')
            idata_.shape = (ntemp_, num_)
            data[inum1, :ntemp_, :num_] = idata_

        if outtype == 'torch':
            return t.from_numpy(data)
        elif outtype == 'numpy':
            return data

