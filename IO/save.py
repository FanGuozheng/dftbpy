"""Saved data."""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np


class Save1D(object):
    """Simple code dor saving data.

    data is numpy type matrix
    blank defines where we'll write blank line
    name is the name of the saved file
    save2D will save file line by line
    savetype: 'a': appendix; 'w': save as a new file (replace the old)

    """

    def __init__(self, data, name, blank='lower', dire=None, ty='w'):
        """Save 0D/1D numpy array or tensor."""
        if len(data.shape) == 0:
            data = data.reshape(1)
        if dire is None:
            newdire = os.getcwd()
        else:
            newdire = dire
        with open(os.path.join(newdire, name), ty) as fopen:
            np.savetxt(fopen, data, newline=" ")
            fopen.write('\n')

class Save2D(object):

    def __init__(self, data, name, blank='lower', dire=None, ty='w'):
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

