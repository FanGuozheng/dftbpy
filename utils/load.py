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

    def __init__(self, para, itrain, itest=0):
        """Initialize parameters."""
        self.para = para
        self.itrain = itrain
        self.itest = itest
        if self.para['dataType'] == 'ani':
            self.load_ani()
            self.get_specie_all()
        elif self.para['dataType'] == 'qm7':
            self.loadqm7()
        elif self.para['dataType'] == 'json':
            self.load_json_data()
            self.get_specie_all()

    def load_ani(self):
        """Load the data from hdf type input files."""
        hdf5filelist = self.para['hdffile']
        maxnfile = max(self.itrain, self.itest)
        minnfile = min(self.itrain, self.itest)
        self.para['coorall'] = []
        self.para['natomall'] = []
        self.para['symbols'] = []
        self.para['specie'] = []
        self.para['specie_global'] = []
        self.para['speciedict'] = []
        coorall_ = []
        specie_hdf = []
        ihdf = 0
        nmolecule = 0  # number of molecule type
        nmin_ = 0
        nmax_ = 0
        assert len(hdf5filelist) == len(self.para['hdf_num'])
        for hdf5file in hdf5filelist:
            ntype = self.para['hdf_num'][ihdf]
            ihdf += 1
            iadl = 0
            adl = pya.anidataloader(hdf5file)
            for data in adl:
                iadl += 1
                if 'all' in ntype or str(iadl) in ntype:
                    coorall_.append(data['coordinates'][:maxnfile])
                    specie_hdf.append(data['species'])
                    nmolecule += 1
        if self.para['hdf_mixture']:  # if mix different type of molecule
            for ifile in range(minnfile):
                for imolecule in range(nmolecule):
                    icoor = coorall_[imolecule][ifile]
                    ispecie = specie_hdf[imolecule]
                    row, col = np.shape(icoor)[0], np.shape(icoor)[1]
                    coor = t.zeros((row, col + 1), dtype=t.float64)
                    coor[:, 1:] = t.from_numpy(icoor[:, :])
                    # check global atom specie
                    for iat in range(len(ispecie)):
                        coor[iat, 0] = ATOMNUM[ispecie[iat]]
                        ispe = ispecie[iat]
                        if ispe not in self.para['specie_global']:
                            self.para['specie_global'].append(ispe)
                    self.para['natomall'].append(coor.shape[0])
                    self.para['coorall'].append(coor)
                    self.para['symbols'].append(ispecie)
                    self.para['specie'].append(list(dict.fromkeys(ispecie)))
                    self.para['speciedict'].append(Counter(ispecie))
                    nmin_ += 1
                    nmax_ += 1
            for ifile in range(minnfile, maxnfile):
                for imolecule in range(nmolecule):
                    icoor = coorall_[imolecule][ifile]
                    ispecie = specie_hdf[imolecule]
                    row, col = np.shape(icoor)[0], np.shape(icoor)[1]
                    coor = t.zeros((row, col + 1), dtype=t.float64)
                    coor[:, 1:] = t.from_numpy(icoor[:, :])
                    for iat in range(len(ispecie)):
                        coor[iat, 0] = ATOMNUM[ispecie[iat]]
                        ispe = ispecie[iat]
                        if ispe not in self.para['specie_global']:
                            self.para['specie_global'].append(ispe)
                    self.para['natomall'].append(coor.shape[0])
                    self.para['coorall'].append(coor)
                    self.para['symbols'].append(ispecie)
                    self.para['specie'].append(list(dict.fromkeys(ispecie)))
                    self.para['speciedict'].append(Counter(ispecie))
                    nmax_ += 1
        else:
            for imolecule in range(nmolecule):
                for ifile in range(minnfile):
                    icoor = coorall_[imolecule][ifile]
                    ispecie = specie_hdf[imolecule]
                    row, col = np.shape(icoor)[0], np.shape(icoor)[1]
                    coor = t.zeros((row, col + 1), dtype=t.float64)
                    coor[:, 1:] = t.from_numpy(icoor[:, :])
                    '''for iat in range(len(ispecie)):
                        coor[iat, 0] = ATOMNUM[ispecie[iat]]
                        ispe = ispecie[iat]
                        if ispe not in self.para['specie_global']:
                            self.para['specie_global'].append(ispe)'''
                    self.para['natomall'].append(coor.shape[0])
                    self.para['coorall'].append(coor)
                    self.para['symbols'].append(ispecie)
                    self.para['specie'].append(list(dict.fromkeys(ispecie)))
                    self.para['speciedict'].append(Counter(ispecie))
                    nmin_ += 1
                    nmax_ += 1
                for ifile in range(minnfile, maxnfile):
                    icoor = coorall_[imolecule][ifile]
                    ispecie = specie_hdf[imolecule]
                    row, col = np.shape(icoor)[0], np.shape(icoor)[1]
                    coor = t.zeros((row, col + 1), dtype=t.float64)
                    coor[:, 1:] = t.from_numpy(icoor[:, :])
                    '''for iat in range(len(ispecie)):
                        coor[iat, 0] = ATOMNUM[ispecie[iat]]
                        ispe = ispecie[iat]
                        if ispe not in self.para['specie_global']:
                            self.para['specie_global'].append(ispe)'''
                    self.para['natomall'].append(coor.shape[0])
                    self.para['coorall'].append(coor)
                    self.para['symbols'].append(ispecie)
                    self.para['specie'].append(list(dict.fromkeys(ispecie)))
                    self.para['speciedict'].append(Counter(ispecie))
                    nmax_ += 1
        self.para['nhdf_max'] = nmax_
        self.para['nhdf_min'] = nmin_

        # return training dataset number
        if self.para['task'] == 'opt':
            self.para['ntrain'] = nmax_


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
        self.para['natomall'] = []
        self.para['specie'] = []
        self.para['speciedict'] = []

        with open(os.path.join(dire, filename), 'r') as fp:
            fpinput = json.load(fp)

            if 'symbols' in fpinput['general']:
                self.para['symbols'] = fpinput['general']['symbols'].split()
                # self.para['speciedict'] = Counter(self.para['symbols'])

            specie = list(set(self.para['symbols']))
            self.para['specie'] = specie
            self.para['atomspecie'] = []
            [self.para['atomspecie'].append(ispe) for ispe in specie]
            self.para['specie'].append(self.para['atomspecie'])
            self.para['speciedict'].append(Counter(self.para['atomspecie']))

            for iname in fpinput['geometry']:
                icoor = fpinput['geometry'][iname]
                self.para['coorall'].append(t.from_numpy(np.asarray(icoor)))
                self.para['natomall'].append(len(icoor))
            self.para['ntrain'] = int(self.para['n_dataset'][0])

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
        n_dataset_ = self.para['n_dataset'][0]
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
                self.para['specie'].append(list(set(symbols_)))
                self.para['speciedict'].append(Counter(symbols_))
        self.para['n_dataset'][0] = str(n_dataset)

    def get_specie_all(self):
        """Get all the atom species in dataset before running Dscribe."""
        atomspecieall = []
        for coor in self.para['coorall']:
            for iat in range(coor.shape[0]):
                idx = int(coor[iat, 0])
                ispe = list(ATOMNUM.keys())[list(ATOMNUM.values()).index(idx)]
                if ispe not in atomspecieall:
                    atomspecieall.append(ispe)
        self.para['specie_all'] = atomspecieall
