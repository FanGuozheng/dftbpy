"""Load data."""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import os
import scipy
import scipy.io
import numpy as np
import torch as t
from torch.autograd import Variable
from collections import Counter
import utils.pyanitools as pya
ATOMNUM = {'H': 1, 'C': 6, 'N': 7, 'O': 8}


class LoadData:
    """Load data.

    Args:
        dataType: the data type, hdf, json...
        hdf_num: how many dataset in one hdf file
    Returns:
        coorall: all the coordination of molecule
        symbols: all the atoms in each molecule
        specie: the specie in molecule
        speciedict: Counter(symbols)

    """

    def __init__(self, para):
        """Initialize parameters."""
        self.para = para
        if self.para['dataType'] == 'ani':
            self.load_ani()
        elif self.para['dataType'] == 'qm7':
            self.loadqm7()
        elif self.para['dataType'] == 'json':
            self.load_json_data()

    def load_ani(self):
        """Load the data from hdf type input files."""
        ntype = self.para['hdf_num']
        hdf5filelist = self.para['hdffile']
        nfile = int(self.para['n_dataset'][0])
        ntest = int(self.para['n_test'][0])
        assert ntest >= nfile
        ntrain = 0
        npred = 0
        icount = 0
        self.para['coorall'] = []
        self.para['natomall'] = []
        self.para['symbols'] = []
        self.para['specie'] = []
        self.para['specie_global'] = []
        self.para['speciedict'] = []
        for hdf5file in hdf5filelist:
            adl = pya.anidataloader(hdf5file)
            if ntype == 'all':
                for data in adl:
                    for ifile in range(nfile):
                        # for icoor in data['coordinates']:
                        ntrain += 1
                        npred += 1
                        icoor = data['coordinates'][ifile]
                        row, col = np.shape(icoor)[0], np.shape(icoor)[1]
                        coor = t.zeros((row, col + 1), dtype=t.float64)
                        for iat in range(len(data['species'])):
                            coor[iat, 0] = ATOMNUM[data['species'][iat]]
                            coor[iat, 1:] = t.from_numpy(icoor[iat, :])
                            ispe = data['species'][iat]
                            if ispe not in self.para['specie_global']:
                                self.para['specie_global'].append(ispe)
                        self.para['natomall'].append(coor.shape[0])
                        self.para['coorall'].append(coor)
                        self.para['symbols'].append(data['species'])
                        self.para['specie'].append(set(data['species']))
                        speciedict = Counter(data['species'])
                        self.para['speciedict'].append(speciedict)
                for data in adl:
                    for ifile in range(nfile, ntest):
                        npred += 1
                        icoor = data['coordinates'][ifile]
                        row, col = np.shape(icoor)[0], np.shape(icoor)[1]
                        coor = t.zeros((row, col + 1), dtype=t.float64)
                        for iat in range(len(data['species'])):
                            coor[iat, 0] = ATOMNUM[data['species'][iat]]
                            coor[iat, 1:] = t.from_numpy(icoor[iat, :])
                            ispe = data['species'][iat]
                            if ispe not in self.para['specie_global']:
                                self.para['specie_global'].append(ispe)
                        self.para['natomall'].append(coor.shape[0])
                        self.para['coorall'].append(coor)
                        self.para['symbols'].append(data['species'])
                        self.para['specie'].append(set(data['species']))
                        speciedict = Counter(data['species'])
                        self.para['speciedict'].append(speciedict)
            else:
                ntype = int(ntype[0])
                for data in adl:
                    icount += 1
                    if icount == ntype:
                        # for icoor in data['coordinates']:
                        for ifile in range(nfile):
                            ntrain += 1
                            icoor = data['coordinates'][ifile]
                            row, col = np.shape(icoor)[0], np.shape(icoor)[1]
                            coor = t.zeros((row, col + 1), dtype=t.float64)

                            for iat in range(len(data['species'])):
                                coor[iat, 0] = ATOMNUM[data['species'][iat]]
                                coor[iat, 1:] = t.from_numpy(icoor[iat, :])
                                ispe = data['species'][iat]
                                if ispe not in self.para['specie_global']:
                                    self.para['specie_global'].append(ispe)

                            self.para['natomall'].append(coor.shape[0])
                            self.para['coorall'].append(coor)
                            self.para['symbols'].append(data['species'])
                            self.para['specie'].append(set(data['species']))
                            speciedict = Counter(data['species'])
                            self.para['speciedict'].append(speciedict)
        self.para['ntrain'] = ntrain
        self.para['npred'] = npred

    def load_ani_old(self):
        """Load the data from hdf type input files."""
        ntype = self.para['hdf_num']
        hdf5filelist = self.para['hdffile']
        icount = 0
        self.para['coorall'] = []
        self.para['natomall'] = []
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
                        self.para['natomall'].append(coor.shape[0])
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
        """Load the data from json type input files."""
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

    def loadrefdata(self, ref, Directory, dire, nfile):
        """Load the data from DFT calculations."""
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

    def loadqm7(self):
        """Load QM7 type data."""
        dataset = scipy.io.loadmat(self.para['qm7_data'])
        n_dataset_ = int(self.para['n_dataset'][0])
        coor_ = dataset['R']
        qatom_ = dataset['Z']
        train_specie = self.para['train_specie']  # not training all species
        n_dataset = 0
        self.para['coorall'] = []
        self.para['natomall'] = []
        self.para['specie'] = []
        self.para['symbols'] = []
        self.para['specie_global'] = []
        self.para['speciedict'] = []
        for idata in range(n_dataset_):
            icoor = coor_[idata]
            natom_ = 0
            train_ = 'yes'
            symbols_ = []
            for iat in qatom_[idata]:
                if iat > 0.0:
                    natom_ += 1
                    idx = int(iat)
                    ispe = \
                        list(ATOMNUM.keys())[list(ATOMNUM.values()).index(idx)]
                    symbols_.append(ispe)
                    if iat in train_specie:
                        if ispe not in self.para['specie_global']:
                            self.para['specie_global'].append(ispe)
                    else:
                        train_ = 'no'
            if train_ == 'yes':
                n_dataset += 1
                row, col = natom_, 4
                coor = t.zeros((row, col), dtype=t.float64)
                coor[:, 0] = t.from_numpy(qatom_[idata][:natom_])
                coor[:, 1:] = t.from_numpy(icoor[:natom_, :])
                self.para['natomall'].append(coor.shape[0])
                self.para['coorall'].append(coor)
                self.para['symbols'].append(symbols_)
                self.para['specie'].append(set(symbols_))
                self.para['speciedict'].append(Counter(symbols_))
        self.para['n_dataset'][0] = str(n_dataset)
