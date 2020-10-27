"""Transfer atomic structures into ML fingerprints."""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch as t
ATOMNUM = {'H': 1, 'C': 6, 'N': 7, 'O': 8}


class ACSF:
    """ACSF atomic environment."""

    def __init__(self, para, rcut=6.0):
        self.para = para
        self.rcut = rcut
        self.specie_all = self.para['specie_all']
        self.nspecie = len(self.para['specie_all'])
        self.get_acsf_dim()

    def get_acsf(self):
        self.cutoff()
        if self.para['Lacsf_g2']:
            self.rad()

    def get_acsf_dim(self):
        col = 0
        if self.nspecie == 1:
            n_types, n_type_pairs = 1, 1
        elif self.nspecie == 2:
            n_types, n_type_pairs = 2, 3
        elif self.nspecie == 3:
            n_types, n_type_pairs = 3, 6
        elif self.nspecie == 4:
            n_types, n_type_pairs = 4, 10
        elif self.nspecie == 5:
            n_types, n_type_pairs = 5, 15
        if self.para['Lacsf_g2']:
            col += (1 + len(self.para['acsf_g2'])) * n_types
        if self.para['Lacsf_g4']:
            col += (len(self.para['acsf_g4'])) * n_type_pairs
        self.para['acsf_dim'] = col
        nat = self.para['coor'].shape[0]
        self.para['acsf_mlpara'] = t.zeros((nat, col), dtype=t.float64)


    def cutoff(self):
        coor = self.para['coor']
        row, col = coor.shape
        dist = self.para['distance']
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
