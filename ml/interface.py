#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
interface.py contains some interface to some popular machine learning
framework, such as schnet, dscribe
"""
from dscribe.descriptors import CoulombMatrix


class Dscribe:

    def __init__(self, para):
        self.para = para

    def get_features():
        pass

    def coulomb(self, rcut=None, nmax=None, lmax=None):
        coor = self.para['coor']
        atom_number = coor[:, 0]
        cm = CoulombMatrix(n_atoms_max=6,)
        coulomb_matrices = cm.create(samples)

    def sine(self):
        pass

    def ewald(self):
        pass

    def acsf(self):
        pass

    def soap(self):
        pass

    def manybody(self):
        pass

    def localmanybody(self):
        pass

    def kernels(self):
        pass


class Schnetpack:

    def __init__():
        pass
