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
        nmax = int(self.para['natomall'].max())

        if self.para['featureType'] == 'CoulombMatrix':
            # This requires flatten=True !!!
            features = t.zeros((nbatch, nmax * nmax), dtype=t.float64)
        elif self.para['featureType'] == 'ACSF':
            features = t.zeros((nbatch, nmax), dtype=t.float64)

        for ibatch in range(nbatch):
            if type(self.para['coorall'][ibatch]) is np.array:
                coor = t.from_numpy(self.para['coorall'][ibatch])
            elif type(self.para['coorall'][ibatch]) is t.Tensor:
                coor = self.para['coorall'][ibatch]
            nat_ = int(self.para['natomall'][ibatch])
            self.para['coor'] = coor[:]
            if self.para['featureType'] == 'CoulombMatrix':
                features[ibatch, :nat_ * nat_] = \
                    self.coulomb()[0, :nat_ * nat_]
            elif self.para['featureType'] == 'ACSF':
                print(self.coulomb().shape)
                features[ibatch, :nat_ ] = self.coulomb()[:nat_]
        return features

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
        return t.from_numpy(cm.create(atom))

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
        acsf = ACSF(atomspecie, positions=coor[:, 1:])
        return t.from_numpy(acsf.create(acsf))

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
