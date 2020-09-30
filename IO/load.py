"""Load data."""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import os
import scipy
import scipy.io
import numpy as np
import torch as t
import h5py
from torch.autograd import Variable
from collections import Counter
import IO.pyanitools as pya
from utils.aset import DFTB, RunASEAims
ATOMNUM = {'H': 1, 'C': 6, 'N': 7, 'O': 8}


class LoadData:
    """Load data.

    Args:
        dataType: the data type, hdf, json...
        hdf_num: how many dataset in one hdf file

    Returns:
        coorall: all the coordination of molecule
        symbols: all the atoms in each molecule
        specie: the specie in each molecule
        specie_global: all the species in dataset
        speciedict: Counter(symbols)
    """

    def __init__(self, para, geo, itrain):
        """Initialize parameters."""
        self.para = para
        self.geo = geo

        # load training data: interger number or string 'all'
        self.itrain = itrain

    def load_ani(self):
        """Load the data from hdf type input files."""
        # define the output
        self.geo['coorall'] = []
        self.geo['natomall'] = []
        self.geo['symbols'] = []
        self.geo['specie'] = []
        self.geo['specie_global'] = []
        self.geo['speciedict'] = []

        # temporal coordinates for all
        coorall_ = []

        # temporal molecule species for all
        specie_hdf = []

        # temporal number of molecules in all molecule species
        nmolecule = []

        # number which control reading ani_gdb_s01~08.h5
        ihdf = 0

        # number of molecule type
        nmoleculespecie = 0

        # hdf data: ani_gdb_s01.h5 ~ ani_gdb_s08.h5
        for hdf5file in self.para['hdffile']:
            ntype = self.para['hdf_num'][ihdf]

            # update ihdf, define iadl
            ihdf += 1
            iadl = 0

            # load each ani_gdb_s0*.h5 data in datalist
            adl = pya.anidataloader(hdf5file)

            # loop for each molecule specie
            for data in adl:
                # iadl represent the nth molecule specie
                iadl += 1

                # read molecule specie according to input parameters
                if 'all' in ntype or str(iadl) in ntype:

                    # get the number of molecule in each molecule specie
                    if self.itrain == 'all':
                        imol = len(data['coordinates'])
                    else:
                        imol = int(self.itrain)

                    nmolecule.append(imol)
                    coorall_.append(data['coordinates'][:imol])
                    specie_hdf.append(data['species'])
                    nmoleculespecie += 1

        # get the min number of molecules in all molecule specie
        minnmolecule = min(nmolecule)

        # mix different type of molecule specie
        if self.para['hdf_mixture']:
            for ifile in range(minnmolecule):

                # all molecule specie
                for imolecule in range(nmoleculespecie):
                    icoor = coorall_[imolecule][ifile]
                    ispecie = specie_hdf[imolecule]
                    row, col = np.shape(icoor)[0], np.shape(icoor)[1]
                    coor = t.zeros((row, col + 1), dtype=t.float64)
                    coor[:, 1:] = t.from_numpy(icoor[:, :])

                    # get the global atom species
                    for iat in range(len(ispecie)):
                        coor[iat, 0] = ATOMNUM[ispecie[iat]]
                        ispe = ispecie[iat]
                        if ispe not in self.para['specie_global']:
                            self.para['specie_global'].append(ispe)
                    self.geo['natomall'].append(coor.shape[0])
                    self.geo['coorall'].append(coor)
                    self.geo['symbols'].append(ispecie)
                    self.geo['specie'].append(list(dict.fromkeys(ispecie)))
                    self.geo['speciedict'].append(Counter(ispecie))

        # do not mix molecule species
        else:
            for imolecule in range(nmoleculespecie):

                # get the global atom species
                specie0 = specie_hdf[imolecule]

                for iat in range(len(specie0)):
                    if specie0[iat] not in self.para['specie_global']:
                        self.para['specie_global'].append(ispe)

                for ifile in range(minnmolecule):
                    icoor = coorall_[imolecule][ifile]
                    ispecie = specie_hdf[imolecule]
                    row, col = np.shape(icoor)[0], np.shape(icoor)[1]
                    coor = t.zeros((row, col + 1), dtype=t.float64)
                    coor[:, 1:] = t.from_numpy(icoor[:, :])
                    self.geo['natomall'].append(coor.shape[0])
                    self.geo['coorall'].append(coor)
                    self.geo['symbols'].append(ispecie)
                    self.geo['specie'].append(list(dict.fromkeys(ispecie)))
                    self.geo['speciedict'].append(Counter(ispecie))

        # return training dataset number
        self.para['nfile'] = len(self.geo['coorall'])

    def load_data_hdf(self, path=None, filename=None):
        """Load data from original dataset, ani ... and write as hdf."""
        if path is not None and filename is not None:
            path_file = os.path.join(path, filename)
            os.system('rm ' + path_file)

        # default path is '.', default name is 'testfile.hdf5'
        else:
            path_file = 'testfile.hdf5'
            os.system('rm ' + path_file)

        # the global atom species in dataset (training data)
        self.geo['specie_all'] = []

        # number of molecules in dataset
        for hdf5file in self.para['hdffile']:
            adl = pya.anidataloader(hdf5file)
            for data in adl:

                # this loop will write information of the same molecule with
                # different geometries
                atom_num = []

                # get atom number of each atom
                [atom_num.append(ATOMNUM[spe]) for spe in data['species']]

                # number of molecule in current molecule species
                imol = len(data['coordinates'])

                # the code will write molecule from range(0, end)
                nend = min(self.itrain, imol)

                # number of atom in molecule
                natom = len(data['coordinates'][0])

                # write all coordinates to list "corrall" and first column is
                # atom number
                self.geo['coorall'] = []
                self.geo['coorall'].extend(
                    [np.insert(coor, 0, atom_num, axis=1)
                     for coor in np.asarray(
                             data['coordinates'][:nend], dtype=float)])

                # write the current atom species in molecule to list "symbols"
                # use metadata instead
                ispecie = ''.join(data['species'])

                # write global atom species
                for ispe in data['species']:
                    if ispe not in self.geo['specie_all']:
                        self.geo['specie_all'].append(ispe)

                # write metadata
                with h5py.File(path_file, 'a') as self.f:
                    self.g = self.f.create_group(ispecie)
                    self.g.attrs['specie'] = ispecie
                    self.g.attrs['natom'] = natom

                    # run dftb with ase interface
                    if self.para['reference'] == 'dftbase':
                        DFTB(self.para, setenv=True).run_dftb(
                            nend, self.geo['coorall'],
                            hdf=self.f, group=self.g)

                    # run FHI-aims with ase interface
                    elif self.para['reference'] == 'aimsase':
                        RunASEAims(self.para, setenv=True).run_aims(
                            nend, self.geo['coorall'],
                            hdf=self.f, group=self.g)

    def load_json_data(self):
        """Load the data from json type input files."""
        dire = self.para['pythondata_dire']
        filename = self.para['pythondata_file']
        self.geo['coorall'] = []
        self.geo['natomall'] = []
        self.geo['specie'] = []
        self.geo['speciedict'] = []

        with open(os.path.join(dire, filename), 'r') as fp:
            fpinput = json.load(fp)

            if 'symbols' in fpinput['general']:
                self.para['symbols'] = fpinput['general']['symbols'].split()
                # self.para['speciedict'] = Counter(self.para['symbols'])

            specie = list(set(self.geo['symbols']))
            self.geo['specie'] = specie
            self.geo['atomspecie'] = []
            [self.geo['atomspecie'].append(ispe) for ispe in specie]
            self.geo['specie'].append(self.geo['atomspecie'])
            self.geo['speciedict'].append(Counter(self.geo['atomspecie']))

            for iname in fpinput['geometry']:
                icoor = fpinput['geometry'][iname]
                self.geo['coorall'].append(t.from_numpy(np.asarray(icoor)))
                self.geo['natomall'].append(len(icoor))
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
