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
HIRSH_VOL = [10.31539447, 0., 0., 0., 0., 38.37861207, 29.90025370, 23.60491416]


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
        elif self.dataset['datasetType'] in ('runaims', 'rundftbplus', 'writeinput'):
            self.load_run_ani()

        # load QM7 dataset
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
        # default path is '.', default name is 'testfile.hdf5'
        else:
            self.path_file = 'testfile.hdf5'
        os.system('rm ' + self.path_file)

        # the global atom species in dataset (training data)
        self.specieGlobal, self.groupName = [], []

        # number of molecules in dataset
        adl = pya.anidataloader(self.dataset['dataset'])
        for idata, data in enumerate(adl):
            # get atom number of each atom
            self._number = [ATOMNUM[spe] for spe in data['species']]

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
            self.positions = [coor for coor in np.asarray(
                data['coordinates'][:self.nend])]

            # write the current atom species in molecule to list "symbols"
            # use metadata instead
            self.ispecie = ''.join(data['species'])
            self.groupName.append(self.ispecie)
            self.dataset['symbols'] = [data['species']] * self.nend

            # write global atom species
            for ispe in data['species']:
                if ispe not in self.specieGlobal:
                    self.specieGlobal.append(ispe)

            # write aims input (do not save) and run FHI-aims calculation
            if self.dataset['datasetType'] in ('runaims', 'rundftbplus'):
                self.write_hdf_group()

            # write aims input (and save)
            elif self.dataset['datasetType'] == 'writeinput':
                self.write_aims_input()

        self.NmoleculeSpecie = idata + 1
        if self.dataset['datasetType'] in ('runaims', 'rundftbplus'):
            self.write_hdf_global()

    def write_hdf_group(self):
        """Write general metadata for each molecule specie.

        Returns:
            specie: joint atom species of molecule (for CH4, CHHHH)
            numberAtom: number of atom in molecule
            atomNumber: each atom number in molecule
        """
        with h5py.File(self.path_file, 'a') as self.f:
            print('self.ispecie', self.ispecie)
            if self.ispecie not in self.f.keys():
                self.g = self.f.create_group(self.ispecie)
                self.g.attrs['specie'] = self.ispecie
                self.g.attrs['numberAtom'] = self.natom
                self.g.attrs['atomNumber'] = self._number
                self.g.attrs['numberMolecule'] = self.nend

                # run dftb with ase interface
                if self.ml['reference'] == 'dftbase':
                    DFTB(self.ml['dftbplus'], self.para['directorySK']).run_dftb(
                        self.nend, self.positions, self.g, self.dataset['symbols'])

                # run FHI-aims with ase interface
                elif self.ml['reference'] == 'aimsase':
                    AseAims(self.ml['aims'], self.ml['aimsSpecie']).run_aims(
                        self.nend, self.positions, self.g, self.dataset['symbols'])

    def write_aims_input(self):
        write.FHIaims(self.positions, self.dataset['symbols'])

    def write_hdf_global(self):
        """Save rest to global attrs."""
        with h5py.File(self.path_file, 'a') as f:
            g = f.create_group('globalGroup')
            g.attrs['specieGlobal'] = self.specieGlobal
            g.attrs['NmoleculeSpecie'] = self.NmoleculeSpecie
            g.attrs['groupName'] = self.groupName

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
            if os.path.exists(os.path.join(newdire, 'bandenergy.dat')):
                refenergy = t.zeros(nfile, 2)
                fpenergy = open(os.path.join(newdire, 'bandenergy.dat'), 'r')
                for ifile in range(0, nfile):
                    energy = np.fromfile(fpenergy, count=3, sep=' ')
                    refenergy[ifile, :] = t.from_numpy(energy[1:])
        elif ref == 'dftbrand':
            newdire = os.path.join(Directory, dire)
            if os.path.exists(os.path.join(newdire, 'bandenergy.dat')):
                refenergy = t.zeros(nfile, 2)
                fpenergy = open(os.path.join(newdire, 'bandenergy.dat'), 'r')
                for ifile in range(0, nfile):
                    energy = np.fromfile(fpenergy, count=3, sep=' ')
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
                irad = np.fromfile(fprad, count=natom, sep=' ')
                rad[ifile, :] = irad[:]
        if os.path.exists(os.path.join(DireSK, 'ang_para.dat')):
            ang = np.zeros((nfile, natom))
            fpang = open(os.path.join(DireSK, 'ang_para.dat'), 'r')
            for ifile in range(0, nfile):
                iang = np.fromfile(fpang, count=natom, sep=' ')
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
                coor = t.zeros(row, col)
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
        self.device = self.para['device']
        self.dtype = self.para['precision']
        if self.ml['reference'] == 'hdf':
            self.get_hdf_data()

    def get_hdf_data(self):
        """Read data from hdf for reference or the following ML."""
        # join the path and hdf data
        hdffile = self.ml['referenceDataset']
        if not os.path.isfile(hdffile):
            raise FileExistsError('dataset {} do not exist'.format(hdffile))
        self.dataset['positions'] = []
        self.dataset['natomAll'] = []
        self.dataset['symbols'] = []
        self.dataset['refHOMOLUMO'] = []
        self.dataset['numbers'] = []
        self.dataset['refCharge'] = []
        self.dataset['refFormEnergy'] = []
        self.dataset['refTotEenergy'] = []
        self.dataset['refHirshfeldVolume'] = []
        self.dataset['refMBDAlpha'] = []
        self.dataset['refDipole'] = []
        self.dataset['numberatom'] = []

        # read global parameters
        with h5py.File(hdffile, 'r') as f:

            # get all the molecule species
            self.dataset['specieGlobal'] = f['globalGroup'].attrs['specieGlobal']
            self.dataset['NmoleculeSpecie'] = f['globalGroup'].attrs['NmoleculeSpecie']

            # get the molecule species
            molecule = [ikey.encode() for ikey in f.keys()]

            # get rid of b-prefix
            molecule2 = [istr.decode('utf-8') for istr in molecule]

            # delete group which is not related to atom species
            ind = molecule2.index('globalGroup')
            del molecule2[ind]

        if type(self.dataset['sizeDataset']) is list:
            # all molecule size is same for ML
            if self.para['task'] == 'testCompressionR':
                self.dataset['ntest'] = sum(self.dataset['sizeTest'])
                self.dataset['nbatch'] = sum(self.dataset['sizeDataset'])
                size_dataset = [max(ii, jj) for ii, jj in zip(
                    self.dataset['sizeTest'], self.dataset['sizeDataset'])]
                self.dataset['nfile'] = self.dataset['ntest']
            else:
                size_dataset = self.dataset['sizeDataset']
                self.dataset['nfile'] = sum(self.dataset['sizeDataset'])

        # read and mix different molecules
        self.nbatch = 0
        if self.dataset['LdatasetMixture']:
            # loop for molecules in each molecule specie
            for ibatch in range(max(size_dataset)):
                # each molecule specie
                for igroup in molecule2:
                    with h5py.File(hdffile, 'r') as f:
                        # add atom name and atom number
                        self.dataset['symbols'].append([ii for ii in igroup])
                        self.dataset['natomAll'].append(len(igroup))

                        # coordinates: not tensor now !!!
                        namecoor = str(ibatch) + 'positions'
                        self.dataset['positions'].append(
                            t.from_numpy(f[igroup][namecoor][()]).type(self.dtype))
                        self.dataset['numbers'].append(
                            list(f[igroup].attrs['atomNumber']))

                        self.dataset['numberatom'].append(t.tensor(
                            [np.count_nonzero(f[igroup].attrs['atomNumber'] == i)
                             for i in [1, 6, 7, 8]]))

                        # HOMO LUMO
                        namehl = str(ibatch) + 'HOMOLUMO'
                        self.dataset['refHOMOLUMO'].append(
                            t.from_numpy(f[igroup][namehl][()]).type(self.dtype))

                        # charge
                        namecharge = str(ibatch) + 'charge'
                        self.dataset['refCharge'].append(
                            t.from_numpy(f[igroup][namecharge][()]).type(self.dtype))

                        # dipole
                        namedip = str(ibatch) + 'dipole'
                        self.dataset['refDipole'].append(
                            t.from_numpy(f[igroup][namedip][()]).type(self.dtype))

                        # formation energy
                        nameEf = str(ibatch) + 'formationEnergy'
                        self.dataset['refFormEnergy'].append(f[igroup][nameEf][()])

                        # total energy
                        nameEf = str(ibatch) + 'totalEnergy'
                        self.dataset['refTotEenergy'].append(f[igroup][nameEf][()])

                        # get refhirshfeld volume ratios, optional
                        namehv = str(ibatch) + 'hirshfeldVolume'
                        if namehv in f[igroup].keys():
                            volume = f[igroup][namehv][()]
                            volume = self.get_hirsh_vol_ratio(volume, self.nbatch)
                            self.dataset['refHirshfeldVolume'].append(volume)

                        # polarizability
                        namepol = str(ibatch) + 'alpha_mbd'
                        if namepol in f[igroup].keys():
                            self.dataset['refMBDAlpha'].append(
                                f[igroup][namepol][()])

                    # total molecule number
                    self.nbatch += 1
            self.dataset['nfile'] = self.nbatch
            self.dataset['refFormEnergy'] = t.tensor(self.dataset['refFormEnergy'])

        # read single molecule specie
        elif not self.dataset['LdatasetMixture']:
            # loop for molecules in each molecule specie
            for ibatch in size_dataset:

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
                        t.from_numpy(f[igroup][namecoor][()]).type(self.dtype))

                    # eigenvalue
                    nameeig = str(ibatch) + 'eigenvalue'
                    self.dataset['refEigval'].append(
                        t.from_numpy(f[igroup][nameeig][()])).type(self.dtype)

                    # charge
                    namecharge = str(ibatch) + 'charge'
                    self.dataset['refEigval'].append(
                        t.from_numpy(f[igroup][namecharge][()]).type(self.dtype))

                    # dipole
                    namedip = str(ibatch) + 'dipole'
                    self.dataset['refDipole'].append(
                        t.from_numpy(f[igroup][namedip][()]).type(self.dtype))

                    # formation energy
                    nameEf = str(ibatch) + 'formationenergy'
                    self.dataset['refEnergy'].append(f[igroup][nameEf][()])

                    # total molecule number
                    self.nbatch += 1
            self.dataset['nfile'] = self.nbatch
            self.dataset['refFormEnergy'] = t.tensor(self.dataset['refFormEnergy'])

    def get_hirsh_vol_ratio(self, volume, ibatch=0):
        """Get Hirshfeld volume ratio."""
        idx = self.dataset['numbers'][ibatch]
        volume = t.from_numpy(volume).type(self.dtype) if type(volume) is np.ndarray else volume
        return volume / t.tensor([HIRSH_VOL[num - 1] for num in idx])


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
