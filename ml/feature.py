"""Transfer atomic structures into ML fingerprints."""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch as t
from ase import Atoms
from dscribe.descriptors import CoulombMatrix, ACSF
ATOMNUM = {'H': 1, 'C': 6, 'N': 7, 'O': 8}


class Dscribe:
    """Interface to Dscribe.

    Returns:
        features for machine learning

    """

    def __init__(self, para, dataset, ml):
        """Initialize the parameters."""
        self.para = para
        self.dataset = dataset
        self.ml = ml

    def pro_(self, ndataset, ntest):
        """Process data for Dscribe."""
        nfile = max(ntest, ndataset)
        nmax = int(max(self.dataset['natomAll']))
        print("ntest, ndataset", ntest, ndataset)
        if self.ml['featureType'] == 'cm':  # flatten=True for all!!!!!
            features = t.zeros(nfile * nmax, nmax)
        elif self.ml['featureType'] == 'acsf':
            self.get_acsf_dim()
            col = self.para['acsf_dim']  # get ACSF feature dimension
            features = t.zeros(nfile * nmax, col)

        for ibatch in range(nfile):
            if type(self.dataset['positions'][ibatch]) is np.array:
                self.para['coor'] = t.from_numpy(self.dataset['positions'][ibatch])
            elif type(self.dataset['positions'][ibatch]) is t.Tensor:
                self.para['coor'] = self.dataset['positions'][ibatch]
            nat_ = int(self.dataset['natomAll'][ibatch])

            if self.ml['featureType'] == 'cm':
                features[ibatch * nmax: ibatch * nmax + nat_, :nat_] = \
                    self.coulomb(n_atoms_max_=nmax)[:nat_, :nat_]
            elif self.ml['featureType'] == 'acsf':
                features[ibatch * nmax: ibatch * nmax + nat_, :] = self.acsf(
                    self.dataset['symbols'][ibatch])
        self.para['natommax'] = nmax
        self.para['feature_test'] = features[:ntest * nmax, :]
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
        nspecie = len(self.dataset['specieGlobal'])
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
        if self.ml['LacsfG2']:
            col += len(self.ml['acsfG2']) * n_types  # G2
        if self.ml['LacsfG4']:
            col += (len(self.ml['acsfG4'])) * n_type_pairs  # G4
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

    def acsf(self, atomspecie):
        """Atom-centered Symmetry Functions method for atomic environment.

        J. chem. phys., 134.7 (2011): 074106.
        You should define all the atom species to fix the feature dimension!
        """
        coor = self.para['coor']
        rcut_ = self.ml['acsfRcut']
        specie_global = self.dataset['specieGlobal']
        if self.ml['LacsfG2']:
            g2_params_ = self.ml['acsfG2']
        else:
            g2_params_ = None
        if self.ml['LacsfG4']:
            g4_params_ = self.ml['acsfG4']
        else:
            g4_params_ = None
        acsf = ACSF(species=specie_global,
                    rcut=rcut_,
                    g2_params=g2_params_,
                    g4_params=g4_params_,
                    )

        test_module = Atoms(atomspecie, positions=coor)
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


class ASCF_:

    def cutoff(self):
        coor = self.dataset['positions']
        row, col = coor.shape
        dist = self.dataset['distance']
        for iatom in range(row):
            for jatom in range(row):
                fc = 0.5 * (np.cos(np.pi * dist[iatom, jatom] / self.rcut) + 1)
                jdx = coor[jatom, 0]
                jname = list(ATOMNUM.keys())[list(ATOMNUM.values()).index(jdx)]
                jcut = self.specie_all.index(jname)
                self.para['acsf_mlpara'][iatom, jcut] = fc

    def rad(self):
        eta = self.para['acsf_g2'][0][0]
        r_s = self.para['acsf_g2'][0][1]
        coor = self.para['coor']
        row, col = coor.shape
        dist = self.para['distance']
        for iatom in range(row):
            for jatom in range(row):
                if dist[iatom, jatom] > self.rcut:
                    pass
                else:
                    fc = 0.5 * (np.cos(np.pi * dist[iatom, jatom] / self.rcut) + 1)
                    jdx = coor[jatom, 0]
                    jname = list(ATOMNUM.keys())[list(ATOMNUM.values()).index(jdx)]
                    jcut = self.specie_all.index(jname)
                    self.para['acsf_mlpara'][iatom, jcut + self.nspecie] = \
                        np.exp(-eta * (dist[iatom, jatom] - r_s) ** 2) * fc
