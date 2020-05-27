#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch as t
import numpy as np
import sys
import os
import re
import data.pyanitools as pya
sys.path.append(os.path.join('../'))
from dftbtorch.geninterpskf import SkInterpolator
ATOMNUM = {'H': 1, 'C': 6, 'N': 7, 'O': 8}


def loadhdfdata(para):
    ntype = para['hdf_num']
    hdf5filelist = para['hdffile']
    para['coorall'] = []
    nlist = 0
    for hdf5file in hdf5filelist:
        icount = 0
        nlist += 1
        adl = pya.anidataloader(hdf5file)
        print('nlist:', nlist, hdf5file)
        for data in adl:
            icount += 1
            print(icount)
            if icount == ntype:
                for icoor in data['coordinates']:
                    row, col = np.shape(icoor)[0], np.shape(icoor)[1]
                    coor = t.zeros(row, col + 1)
                    for iat in range(0, len(data['species'])):
                        coor[iat, 0] = ATOMNUM[data['species'][iat]]
                        coor[iat, 1:] = t.from_numpy(icoor[iat, :])
                    para['coorall'].append(coor)
                symbols = data['species']
                specie = set(symbols)
                # speciedict = Counter(symbols)
                para['symbols'] = symbols
                para['specie'] = specie
                para['atomspecie'] = []
                [para['atomspecie'].append(ispecie) for ispecie in specie]
                # para['speciedict'] = speciedict
                print(icount, len(para['coorall']), data['species'])
                # print(para['coorall'])


def interpskf(para):
    '''
    read .skf data from skgen with various compR
    '''
    print('** read skf file with all compR **')
    for namei in para['atomspecie']:
        for namej in para['atomspecie']:
            SkInterpolator(para, gridmesh=0.2).readskffile(
                    namei, namej, para['dire_interpSK'])


if __name__ == '__main__':
    para = {}
    # 1: CH4, 2: NH3, 3: H2O, 4: C2H6 5: CNH5, 6: N2 7: NH3O 8: NHO 9: COH4
    # 10: C2H4, 11: CH2O, 12: O2 13: H2O2, 14: C2H2, 15: N2H4, 16: N2H2
    # 17: C3H8, 18: C2NH7, 19: CNOH3, 20: CO2H2, 21: CN2H4, 22: CNOH3, 23: C3H4
    # 24: C3H6, 25: C2NH5, 26: C2OH4 (OSerror), 27: NO2H (OSerror), 28: O3H2 ()
    # 29: C2OH6 (), 30: C2NH7 (), 31: C2OH6 (), 32: C3H6 (), 33: C2OH4 ()
    # 34: CO2 (), 35
    para['hdf_num'] = 3
    hdffilelist = []
    filelist = os.listdir('data/an1')
    filelist.sort()
    [hdffilelist.append(os.path.join('data/an1', f))
        for f in filelist if re.match(r'ani+.*\.h5', f)]
    para['hdffile'] = hdffilelist
    loadhdfdata(para)
