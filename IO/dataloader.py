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
import IO.write_output as write
from utils.aset import DFTB, AseAims
ATOMNUM = {'H': 1, 'C': 6, 'N': 7, 'O': 8}
HIRSH_VOL = {"H": 10.31539447, "C": 38.37861207, "N": 29.90025370,
             "O": 23.60491416}


class LoadData:
    """Load dataset.

    Args:
        dataType: the data type, hdf, json...
        hdf_num: how many dataset in one hdf file

    Returns:
        positions: all the coordination of molecule
        symbols: all the atoms in each molecule
        specie: the specie in each molecule
        specie_global: all the species in dataset
        speciedict: Counter(symbols)
    """

    def __init__(self, para=None, dataset=None, ml=None, train_sample=None):
        """Initialize parameters."""
        self.dataset = [dataset, {}][dataset is None]
        self.para = [para, {}][para is None]
        self.ml = [ml, {}][ml is None]

        if self.dataset['datasetType'] == 'ani':
            self.load_ani()
        # load dataset and directly run DFTB+ or aims calculations
        elif self.dataset['datasetType'] in ('runani', 'writeinput'):
            self.load_run_ani()
        elif self.dataset['datasetType'] == 'qm7':
            self.loadqm7()

    def load_ani(self):
        """Load the data from hdf type input files."""
        # define the output
        self.dataset['positions'] = []
        self.dataset['natomAll'] = []
        self.dataset['symbols'] = []
        self.dataset['specie'] = []
        self.dataset['specieGlobal'] = []
        self.dataset['speciedict'] = []
        self.dataset['numbers'] = []

        # temporal coordinates for all
        _coorall = []

        # temporal molecule species for all
        _specie = []

        # temporal number of molecules in all molecule species
        nmolecule = []

        # hdf data: ani_gdb_s01.h5 ~ ani_gdb_s08.h5
        for itype, hdf5file in enumerate(self.dataset['Dataset']):
            # sizeDataset is input dataset size parameters
            nn = self.dataset['sizeDataset'][itype]

            # load each ani_gdb_s0*.h5 data in datalist
            adl = pya.anidataloader(hdf5file)

            # loop for each molecule specie
            # such as for ani_gdb_s01.h5, there are 3 species: CH4, NH3, H2O
            for iadl, data in enumerate(adl):

                # get each molecule specie size
                _isize = len(data['coordinates'])
                imol = _isize if 'all' in nn else int(nn)

                # global species
                for ispe in data['species']:
                    if ispe not in self.dataset['specieGlobal']:
                        self.dataset['specieGlobal'].append(ispe)

                # check if imol (size of this molecule specie) out of range
                imol = _isize if imol > _isize else imol

                # size of each molecule specie
                nmolecule.append(imol)

                # selected coordinates of each molecule specie
                _coorall.append(data['coordinates'][:imol])

                # add atom species in each molecule specie
                _specie.append(data['species'])

        # get the min number of molecules in all molecule specie
        # this makes sure each molecule specie size is the same
        minsize = min(nmolecule)

        # mix different molecule species
        if self.dataset['LdatasetMixture']:
            for ifile in range(minsize):
                # loop over each molecule specie
                for ispe in range(len(_specie)):
                    natom = len(_coorall[ispe][ifile])
                    self.dataset['positions'].append(
                        t.from_numpy(_coorall[ispe][ifile]))

                    # number of atom in each molecule
                    self.dataset['natomAll'].append(natom)

                    # get symbols of each atom
                    self.dataset['symbols'].append(_specie[ispe])

                    # get set of symbols according to order of appearance
                    self.dataset['specie'].append(
                        list(dict.fromkeys(_specie[ispe])))
                    self.dataset['speciedict'].append(Counter(_specie[ispe]))

                    # atom number of each atom in molecule
                    self.dataset['numbers'].append(
                        [ATOMNUM[ispe] for ispe in _specie[ispe]])

        # do not mix molecule species
        else:
            for ispe, isize in enumerate(nmolecule):
                # get the first length in molecule specie and '* size'
                self.dataset['natomAll'].extend([len(_coorall[ispe][0])] * isize)
                natom = len(_coorall[ispe][0])

                # get symbols of each atom
                self.dataset['symbols'].extend([_specie[ispe]] * isize)

                # add coordinates
                self.dataset['positions'].append(
                    [t.from_numpy(icoor) for icoor in _coorall[ispe][:isize]])

                # get set of symbols according to order of appearance
                self.dataset['specie'].append(
                    [list(dict.fromkeys(_isp)) for _isp in [_specie[ispe]] * isize])
                self.dataset['speciedict'].append(
                    [Counter(_isp) for _isp in [_specie[ispe]] * isize])

                # atom number of each atom in molecule
                self.dataset['numbers'].extend(
                    [ATOMNUM[ispe] for ispe in _specie[ispe] * isize])

        # return training dataset number
        self.dataset['nfile'] = len(self.dataset['natomAll'])

    def load_run_ani(self, path=None, filename=None):
        """Load data from original dataset, ani ... and write as hdf."""
        if path is not None and filename is not None:
            self.path_file = os.path.join(path, filename)
            os.system('rm ' + self.path_file)

        # default path is '.', default name is 'testfile.hdf5'
        else:
            self.path_file = 'testfile.hdf5'
            os.system('rm ' + self.path_file)

        # the global atom species in dataset (training data)
        self.dataset['specieGlobal'] = []

        # if only define the first specie in dataset, it will qutomatically
        # extend to all spcies by using the defined size
        if len(self.dataset['sizeDataset']) == 1:
            extend_dataset_seize = True

        # number of molecules in dataset
        adl = pya.anidataloader(self.dataset['dataset'][0])
        for idata, data in enumerate(adl):
            # extend dataset
            if extend_dataset_seize:
                self.dataset['sizeDataset'].append(self.dataset['sizeDataset'][0])

            # this loop will write information of the same molecule with
            # different datasetmetries
            self.dataset['numbers'] = []
            self.dataset['symbols'] = []

            # get atom number of each atom
            [self.dataset['numbers'].append(ATOMNUM[spe]) for spe in data['species']]

            # number of molecule in current molecule species
            imol = len(data['coordinates'])

            # the code will write molecule from range(0, end)
            _isize = self.dataset['sizeDataset'][idata]
            _isize = imol if _isize == 'all' else int(_isize)
            self.nend = min(_isize, imol)

            # number of atom in molecule
            self.natom = len(data['coordinates'][0])

            # write all coordinates to list "corrall" and first column is
            # atom number
            self.dataset['positions'] = []
            self.dataset['positions'].extend(
                [coor for coor in np.asarray(data['coordinates'][:self.nend], dtype=float)])

            # write the current atom species in molecule to list "symbols"
            # use metadata instead
            self.ispecie = ''.join(data['species'])
            self.dataset['symbols'].extend([data['species']] * self.nend)

            # write global atom species
            for ispe in data['species']:
                if ispe not in self.dataset['specieGlobal']:
                    self.dataset['specieGlobal'].append(ispe)

            # write aims input (do not save) and run FHI-aims calculation
            if self.dataset['datasetType'] == 'runani':
                self.write_hdf()
            # write aims input (and save)
            elif self.dataset['datasetType'] == 'writeinput':
                self.write_aims_input()

        if self.dataset['datasetType'] == 'runani':
            self.write_hdf_global()

    def write_hdf(self):
        """Write metadata."""
        with h5py.File(self.path_file, 'a') as self.f:
            self.g = self.f.create_group(self.ispecie)
            self.g.attrs['specie'] = self.ispecie
            self.g.attrs['natom'] = self.natom
            self.g.attrs['atomNumber'] = self.dataset['numbers']

            # run dftb with ase interface
            if self.ml['reference'] == 'dftbase':
                DFTB(self.para, self.dataset, self.ml, setenv=True).run_dftb(
                    self.nend, self.dataset['positions'],
                    hdf=self.f, group=self.g)

            # run FHI-aims with ase interface
            elif self.ml['reference'] == 'aimsase':
                AseAims(self.para, self.dataset, self.ml,
                        setenv=True).run_aims(self.nend, hdf=self.f, group=self.g)

    def write_aims_input(self):
        write.FHIaims(self.dataset).geo_nonpe_hdf_batch(self.nend)

    def write_hdf_global(self):
        # save rest to global attrs
        with h5py.File(self.path_file, 'a') as f:
            g = f.create_group('globalgroup')
            g.attrs['specieGlobal'] = self.dataset['specieGlobal']
            # g.attrs['totalmolecule'] = nmoldftbase

    def load_json_data(self):
        """Load the data from json type input files."""
        dire = self.para['pythondata_dire']
        filename = self.para['pythondata_file']
        self.dataset['positions'] = []
        self.dataset['natomall'] = []
        self.dataset['specie'] = []
        self.dataset['speciedict'] = []

        with open(os.path.join(dire, filename), 'r') as fp:
            fpinput = json.load(fp)

            if 'symbols' in fpinput['general']:
                self.para['symbols'] = fpinput['general']['symbols'].split()
                # self.para['speciedict'] = Counter(self.para['symbols'])

            specie = list(set(self.dataset['symbols']))
            self.dataset['specie'] = specie
            self.dataset['atomspecie'] = []
            [self.dataset['atomspecie'].append(ispe) for ispe in specie]
            self.dataset['specie'].append(self.dataset['atomspecie'])
            self.dataset['speciedict'].append(Counter(self.dataset['atomspecie']))

            for iname in fpinput['geometry']:
                icoor = fpinput['geometry'][iname]
                self.dataset['positions'].append(t.from_numpy(np.asarray(icoor)))
                self.dataset['natomall'].append(len(icoor))
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
        self.para['positionsAll'] = []
        self.para['natomAll'] = []
        self.para['specie'] = []
        self.para['symbols'] = []
        self.para['specieGlobal'] = []
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
                        if ispe not in self.para['specieGlobal']:
                            self.para['specie_global'].append(ispe)
                    else:
                        train_ = 'no'
            if train_ == 'yes':
                n_dataset += 1
                row, col = natom_, 4
                coor = t.zeros((row, col), dtype=t.float64)
                coor[:, 0] = t.from_numpy(qatom_[idata][:natom_])
                coor[:, 1:] = t.from_numpy(icoor[:natom_, :])
                self.para['natomAll'].append(coor.shape[0])
                self.para['positions'].append(coor)
                self.para['symbols'].append(symbols_)
                self.para['specie'].append(list(set(symbols_)))
                self.para['speciedict'].append(Counter(symbols_))
        self.para['n_dataset'][0] = str(n_dataset)

    def get_specie_all(self):
        """Get all the atom species in dataset before running Dscribe."""
        atomspecieall = []
        for coor in self.para['positions']:
            for iat in range(coor.shape[0]):
                idx = int(coor[iat, 0])
                ispe = list(ATOMNUM.keys())[list(ATOMNUM.values()).index(idx)]
                if ispe not in atomspecieall:
                    atomspecieall.append(ispe)
        self.para['specieGlobal'] = atomspecieall


class LoadReferenceData:
    """Load reference data from hdf5 type."""

    def __init__(self, para, dataset, skf, ml):
        self.para = para
        self.dataset = dataset
        self.skf = skf
        self.ml = ml
        if self.ml['reference'] == 'hdf':
            self.get_hdf_data()

    def get_hdf_data(self):
        """Read data from hdf for reference or the following ML."""
        # join the path and hdf data
        hdffile = self.para['referenceDataset']
        if not os.path.isfile(hdffile):
            raise FileExistsError('reference dataset do not exist')
        self.dataset['positions'] = []
        self.dataset['natomAll'] = []
        self.dataset['symbols'] = []
        self.dataset['refHomoLumo'] = []
        self.dataset['numbers'] = []
        self.dataset['refCharge'] = []
        self.dataset['refFormEnergy'] = []
        self.dataset['refTotEenergy'] = []
        self.dataset['refHirshfeldVolume'] = []
        self.dataset['refMBDAlpha'] = []
        self.dataset['refDipole'] = []
        self.dataset['numberatom'] = []
        if type(self.dataset['sizeDataset']) is list:
            # all molecule size is same for ML
            size_dataset = int(self.dataset['sizeDataset'][0])
        else:
            size_dataset = self.dataset['sizeDataset']

        # read global parameters
        with h5py.File(hdffile, 'r') as f:

            # get all the molecule species
            self.dataset['specieGlobal'] = f['globalgroup'].attrs['specie_all']

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
            for ibatch in range(size_dataset):

                # each molecule specie
                for igroup in molecule2:

                    # add atom name and atom number
                    self.dataset['symbols'].append([ii for ii in igroup])
                    self.dataset['natomAll'].append(len(igroup))

                    with h5py.File(hdffile, 'r') as f:

                        # coordinates: not tensor now !!!
                        namecoor = str(ibatch) + 'coordinate'
                        self.dataset['positions'].append(
                            t.from_numpy(f[igroup][namecoor][()]))
                        self.dataset['numbers'].append(
                            list(f[igroup].attrs['atomNumber']))

                        self.dataset['numberatom'].append(t.tensor(
                            [np.count_nonzero(f[igroup].attrs['atomNumber'] == i)
                             for i in [1, 6, 7, 8]], dtype=t.float64))

                        # HOMO LUMO
                        namehl = str(ibatch) + 'humolumo'
                        self.dataset['refHomoLumo'].append(
                            t.from_numpy(f[igroup][namehl][()]))

                        # charge
                        namecharge = str(ibatch) + 'charge'
                        self.dataset['refCharge'].append(
                            t.from_numpy(f[igroup][namecharge][()]))

                        # dipole
                        namedip = str(ibatch) + 'dipole'
                        self.dataset['refDipole'].append(
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
                        self.dataset['refHirshfeldVolume'].append(volume)

                        # polarizability
                        namepol = str(ibatch) + 'alpha_mbd'
                        self.dataset['refMBDAlpha'].append(
                            f[igroup][namepol][()])

                    # total molecule number
                    self.nbatch += 1
                    '''self.save_ref_idata('hdf', ibatch,
                                        LWHL=self.para['LHomoLumo'],
                                        LWeigenval=self.para['Leigval'],
                                        LWenergy=self.para['Lenergy'],
                                        LWdipole=self.para['Ldipole'],
                                        LWpol=self.para['LMBD_DFTB'])'''
            self.dataset['nfile'] = self.nbatch
            self.dataset['refFormEnergy'] = t.tensor(self.dataset['refFormEnergy'])

        # read single molecule specie
        elif not self.para['hdf_mixture']:
            # loop for molecules in each molecule specie
            for ibatch in range(size_dataset):

                # each molecule specie
                iatomspecie = int(self.para['hdf_num'][0])
                igroup = molecule2[iatomspecie]

                # add atom name
                self.dataset['symbols'].append([ii for ii in igroup])
                self.dataset['natomAll'].append(len(igroup))

                # read the molecule specie data
                with h5py.File(hdffile) as f:

                    # coordinates: not tensor now !!!
                    namecoor = str(ibatch) + 'coordinate'
                    self.dataset['positions'].append(
                        t.from_numpy(f[igroup][namecoor][()]))

                    # eigenvalue
                    nameeig = str(ibatch) + 'eigenvalue'
                    self.dataset['refEigval'].append(
                        t.from_numpy(f[igroup][nameeig][()]))

                    # charge
                    namecharge = str(ibatch) + 'charge'
                    self.dataset['refEigval'].append(
                        t.from_numpy(f[igroup][namecharge][()]))

                    # dipole
                    namedip = str(ibatch) + 'dipole'
                    self.dataset['refDipole'].append(
                        t.from_numpy(f[igroup][namedip][()]))

                    # formation energy
                    nameEf = str(ibatch) + 'formationenergy'
                    self.dataset['refEnergy'].append(f[igroup][nameEf][()])

                    # total molecule number
                    self.nbatch += 1
                    self.save_ref_idata('hdf', ibatch,
                                        LWHL=self.para['LHomoLumo'],
                                        LWeigenval=self.para['Leigval'],
                                        LWenergy=self.para['Lenergy'],
                                        LWdipole=self.para['Ldipole'],
                                        LWpol=self.para['LMBD_DFTB'])
            self.dataset['nfile'] = self.nbatch
            self.dataset['refFormEnergy'] = t.tensor(self.dataset['refFormEnergy'])

    def get_hirsh_vol_ratio(self, volume, ibatch=0):
        """Get Hirshfeld volume ratio."""
        natom = self.dataset["natomAll"][ibatch]
        for iat in range(natom):
            idx = int(self.dataset['numbers'][ibatch][iat])
            iname = list(ATOMNUM.keys())[list(ATOMNUM.values()).index(idx)]
            volume[iat] = volume[iat] / HIRSH_VOL[iname]
        return volume


class Split:
    """Split tensor according to chunks of split_sizes.

    Parameters
    ----------
    tensor : `torch.Tensor`
        Tensor to be split
    split_sizes : `list` [`int`], `torch.tensor` [`int`]
        Size of the chunks
    dim : `int`
        Dimension along which to split tensor

    Returns
    -------
    chunked : `tuple` [`torch.tensor`]
        List of tensors viewing the original ``tensor`` as a
        series of ``split_sizes`` sized chunks.

    Raises
    ------
    KeyError
        If number of elements requested via ``split_sizes`` exceeds hte
        the number of elements present in ``tensor``.
    """
    def __init__(tensor, split_sizes, dim=0):
        if dim < 0:  # Shift dim to be compatible with torch.narrow
            dim += tensor.dim()

        # Ensure the tensor is large enough to satisfy the chunk declaration.
        if tensor.size(dim) != split_sizes.sum():
            raise KeyError(
                'Sum of split sizes fails to match tensor length along specified dim')

        # Identify the slice positions
        splits = t.cumsum(t.Tensor([0, *split_sizes]), dim=0)[:-1]

        # Return the sliced tensor. use torch.narrow to avoid data duplication
        return tuple(tensor.narrow(int(dim), int(start), int(length))
                     for start, length in zip(splits, split_sizes))
