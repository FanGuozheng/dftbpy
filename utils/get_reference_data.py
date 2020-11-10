"""Build hdf type data for machine learning reference.

The data will include:
    geometry information,
    physical properties

define para['dataType'] as original dataset, e.g 'ani'

"""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import h5py
import dftbtorch.initparams as initpara
import IO.pyanitools as pya
from utils.aset import DFTB, AseAims
from IO.dataloader import LoadData
import dftbtorch.parameters as constpara
ATOMNUM = {'H': 1, 'C': 6, 'N': 7, 'O': 8}


class RefDFTB:
    """Physical properties are from DFTB-ASE calculations.

    Note:
    -----
    set para['reference'] = 'dftbase'
    """

    def __init__(self, para):
        """Load original dataset and then write into hdf type."""
        # define ml dictionary
        self.para = para

        # load constant parameters
        constpara.constant_parameter(self.para)
        self.ml = {}
        self.dataset = {}
        self.ml['reference'] = 'dftbase'
        self.dataset['dataType'] = 'ani'
        self.dataset['LdatasetMixture'] = False

        # get parameters for generating reference data
        self.para = initpara.dftb_parameter(self.para)
        self.skf = initpara.skf_parameter()
        self.dataset = initpara.init_dataset(self.dataset)
        self.para, self.dataset, self.skf, self.ml = \
            initpara.init_ml(self.para, self.dataset, self.skf, self.ml)

        # load hdf data
        LoadData(self.para, self.dataset, self.ml).load_data_hdf()

    def load_data_hdf(self):
        """Load data from original dataset, ani ..."""
        os.system('rm testfile.hdf5')

        # the global atom species in dataset (training data)
        self.para['specie_all'] = []

        # number of molecules in dataset
        nmol = 0
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

                # number of atom in molecule
                natom = len(data['coordinates'][0])

                # write all coordinates to list "corrall" and first column is
                # atom number
                self.para['coorall'] = []
                self.para['coorall'].extend(
                    [np.insert(coor, 0, atom_num, axis=1)
                     for coor in np.asarray(data['coordinates'], dtype=float)])

                # write the current atom species in molecule to list "symbols"
                # use metadata instead
                # self.para['symbols'].extend(
                #     data['species'] for ii in range(imol))
                ispecie = ''.join(data['species'])

                # write global atom species
                for ispe in data['species']:
                    if ispe not in self.para['specie_all']:
                        self.para['specie_all'].append(ispe)

                # write number of atoms
                # self.para['natomall'].extend(natom for ii in range(imol))

                # write metadata
                with h5py.File('testfile.hdf5', 'a') as self.f:
                    self.g = self.f.create_group(ispecie)
                    self.g.attrs['specie'] = ispecie
                    self.g.attrs['natom'] = natom

                    # run reference claculations
                    try:
                        if self.para['reference'] == 'dftbase':
                            self.run_dftbplus(imol, self.para['coorall'])
                    except ValueError:
                        print("Please set reference == dftbase!")

                nmol += imol

        # save rest to global attrs
        with h5py.File('testfile.hdf5', 'a') as f:
            g = f.create_group('globalgroup')
            g.attrs['specie_all'] = self.para['specie_all']
            g.attrs['totalmolecule'] = nmol

    def run_dftbplus(self, end, coorall):
        """Run DFTB-ase and save as hdf data."""
        # DFTB+ as reference
        DFTB(self.para, setenv=True).run_dftb(end, coorall, hdf=self.f,
                                              group=self.g)

    def run_dftbase():
        pass

    def get_property():
        pass

    def save_data():
        pass


class RefAims:
    """Run FHI-aims calculations and save results as reference data for ML.

    Note:
    -----
    set para['reference'] = 'aimsase'
    """

    def __init__(self, para):
        """Load geometry data, run FHI-aims and analyze the results."""
        # load constant parameters
        constpara.constant_parameter()

        # define ml dictionary
        self.para = para
        self.ml = {}
        self.dataset = {}
        self.ml['reference'] = 'aimsase'

        # runani: read dataset and run FHI-aims calculations
        # writeinput: read dataset and write FHI-aims input without calculation
        self.dataset['datasetType'] = 'runani'

        # read and run different molecule species dataset size
        self.dataset['sizeDataset'] = ['200']
        self.dataset['LdatasetMixture'] = False
        self.dataset['dataset'] = ['../data/dataset/an1/ani_gdb_s02.h5']

        # get parameters for generating reference data
        self.para = initpara.dftb_parameter(self.para)
        self.skf = initpara.skf_parameter(self.para)
        self.dataset = initpara.init_dataset(self.dataset)
        self.para, self.dataset, self.skf, self.ml = \
            initpara.init_ml(self.para, self.dataset, self.skf, self.ml)

        # LoadData will load ani dataset (call function load_ani)
        # then run load_data_hdf
        LoadData(self.para, self.dataset, self.ml)

    def load_data_hdf(self):
        """Load original data for FHI-aims calculations."""
        pass

    def run_aims(self):
        """Run FHI-aims calculations."""
        pass

if __name__ == '__main__':
    para = {}
    para['task'] = 'get_aims_hdf'
    if para['task'] == 'get_dftb_hdf':
        RefDFTB(para)
    elif para['task'] == 'get_aims_hdf':
        RefAims(para)
