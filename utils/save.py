"""Saved data."""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np


class SaveData:
    """Simple code dor saving data.

    data is numpy type matrix
    blank defines where we'll write blank line
    name is the name of the saved file
    save2D will save file line by line
    savetype: 'a': appendix; 'w': save as a new file (replace the old)

    """

    def __init__(self, para):
        """Initialize data."""
        self.para = para

    def save1D(self, data, name, blank='lower', dire=None, ty='w'):
        """Save 0D/1D numpy array or tensor."""
        if len(data.shape) == 0:
            data = data.reshape(1)
        if dire is None:
            newdire = os.getcwd()
        else:
            newdire = dire
        with open(os.path.join(newdire, name), ty) as fopen:
            if blank == 'upper':
                fopen.write('\n')
            np.savetxt(fopen, data, newline=" ")
            fopen.write('\n')
            if blank == 'lower':
                fopen.write('\n')

    def save2D(self, data, name, blank='lower', dire=None, ty='w'):
        """Save 2D numpy array or tensor."""
        if dire is None:
            newdire = os.getcwd()
        else:
            newdire = dire
        with open(os.path.join(newdire, name), ty) as fopen:
            for idata in data:
                if blank == 'upper':
                    fopen.write('\n')
                np.savetxt(fopen, idata, newline=" ")
                fopen.write('\n')
                if blank == 'lower':
                    fopen.write('\n')

    def save_envir(para, Directory):
        """Save atomic environment data."""
        ang_paraall = para['ang_paraall']
        rad_paraall = para['rad_paraall']
        with open(os.path.join(Directory, 'rad_para.dat'), 'w') as fopen:
            np.savetxt(fopen, rad_paraall, fmt="%s", newline=' ')
            fopen.write('\n')
        with open(os.path.join(Directory, 'ang_para.dat'), 'w') as fopen:
            np.savetxt(fopen, ang_paraall, fmt="%s", newline=' ')
            fopen.write('\n')
