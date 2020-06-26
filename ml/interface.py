"""Interface to some popular ML framework."""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch as t
from ase import Atoms
from dscribe.descriptors import ACSF
from dscribe.descriptors import CoulombMatrix
ATOMNUM = {'H': 1, 'C': 6, 'N': 7, 'O': 8}


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
        nbatch = int(self.para['n_test'][0])
        ndataset = int(self.para['n_dataset'][0])
        nfile = max(nbatch, ndataset)
        nmax = int(self.para['natomall'].max())

        if self.para['featureType'] == 'cm':
            # This requires flatten=True !!!
            features = t.zeros((nfile * nmax, nmax), dtype=t.float64)
        elif self.para['featureType'] == 'acsf':
            features = t.zeros((nfile * nmax, 2), dtype=t.float64)

        for ibatch in range(nfile):
            if type(self.para['coorall'][ibatch]) is np.array:
                coor = t.from_numpy(self.para['coorall'][ibatch])
            elif type(self.para['coorall'][ibatch]) is t.Tensor:
                coor = self.para['coorall'][ibatch]

            nat_ = int(self.para['natomall'][ibatch])
            self.para['coor'] = coor[:]

            if self.para['featureType'] == 'cm':
                features[ibatch * nmax: ibatch * nmax + nat_, :nat_] = \
                    self.coulomb(n_atoms_max_=nmax)[:nat_, :nat_]
            elif self.para['featureType'] == 'acsf':
                features[ibatch * nmax: ibatch * nmax + nat_, :] = self.acsf()
        self.para['natommax'] = nmax
        self.para['feature_test'] = features[:nbatch * nmax, :]
        self.para['feature_data'] = features[:ndataset * nmax, :]

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
        """
        coor = self.para['coor']
        atomspecie = []
        for iat in range(coor.shape[0]):
            idx = int(coor[iat, 0])
            atomspecie.append(
                list(ATOMNUM.keys())[list(ATOMNUM.values()).index(idx)])
        test_module = Atoms(atomspecie, positions=coor[:, 1:])
        acsf = ACSF(species=atomspecie, rcut=6.0,
                    # g2_params=[[1, 1]],
                    # g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]],
                    )
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
