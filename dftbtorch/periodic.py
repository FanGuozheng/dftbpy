#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch as t


class Periodic:

    def __init__(self, para):
        self.para = para

    def get_neighbour(self, cutoff):
        """
        Get the number of neighbouring atoms.

        Parameters:
            distance between all atoms
        """
        natom = self.para['natom']
        self.para['neighbour'] = t.zeros(natom)
        for iat in range(natom):
            icount = 0
            for jat in range(natom):
                if iat < jat:

                    # the cutoff will be the same for different atom pairs
                    if cutoff == 'repulsive' and not self.para['cutoff_atom_resolve']:
                        cutoff_ = self.para['cutoff_rep']
                        if self.para['distance'][iat, jat] < cutoff_:
                            icount += 1
            self.para['neighbour'][iat] = icount
