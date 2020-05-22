#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch as t


class Periodic:

    def __init__(self, para):
        self.para = para

    def get_neighbour(self, cutoff):
        '''
        Input:
            distance between all atoms
        '''
        atomspecie = self.para['atomspecie']
        atomnameall = self.para['atomnameall']
        natom = self.para['natom']
        self.para['neighbour'] = t.zeros(natom)
        for iat in range(0, natom):
            icount = 0
            for jat in range(0, natom):
                if iat < jat:
                    nameij = atomnameall[iat] + atomnameall[jat]
                    if cutoff == 'repulsive':
                        cutoff_ = self.para['cutoff_rep' + nameij]
                    if self.para['distance'][iat, jat] < cutoff_:
                        icount += 1
            self.para['neighbour'][iat] = icount
