#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import os
import subprocess

default_r = {"H": 3, "C": 3.5, "N": 2.2, "O": 2.3, "S": 3.8}

class dftbplus:
    def __init__(self):
        pass

    def geo_nonpe(self, file, coor, specie, speciedict, symbols):
        row, col = np.shape(coor)
        with open('geo.gen.{}'.format(file), 'w') as f:
            f.write(str(row)+" "+"C")
            f.write('\n')
            for iname in range(0, len(specie)):
                f.write(symbols[iname]+' ')
            f.write('\n')
            ispecie = 0
            iatom = 0
            for atom in specie:
                ispecie += 1
                print(specie, speciedict)
                for natom in range(0, speciedict[atom]):
                    iatom += 1
                    f.write(str(iatom)+" "+str(ispecie)+" ")
                    np.savetxt(f, coor[iatom-1], fmt="%s", newline=" ")
                    f.write('\n')


    def geo_nonpe_ml(self, file, coor, specie, speciedict, symbols):
        row, col = np.shape(coor)
        compressr0 = np.zeros(row)
        print(specie, speciedict)
        with open('geo.gen.{}'.format(file), 'w') as f:
            f.write(str(row)+" "+"C")
            f.write('\n')
            for iname in specie:
                atom_i = 0
                for iatom in range(0, speciedict[iname]):
                    atom_i += 1
                    f.write(iname+str(atom_i)+' ')
            f.write('\n')
            iatom = 0
            for atom in specie:
                for natom in range(0, speciedict[atom]):
                    compressr0[iatom] = default_r[atom]
                    iatom += 1
                    f.write(str(iatom)+' '+str(iatom)+' ')
                    np.savetxt(f, coor[iatom-1], fmt="%s", newline=" ")
                    f.write('\n')
        return compressr0

    def geo_nonpe2(self, file, coor, specie, speciedict):
        """this code is used to ml process, each atom will be added a number
        so that every atom will be different"""
        row, col = np.shape(coor)
        with open('geo.gen.{}'.format(file), 'w') as f:
            f.write(str(row)+" "+"C")
            f.write('\n')
            for iname in specie:
                f.write(iname+' ')
            f.write('\n')
            ispecie = 0
            iatom = 0
            for atom in specie:
                ispecie += 1
                for natom in range(0, speciedict[atom]):
                    iatom += 1
                    f.write(str(iatom)+" "+str(ispecie)+" ")
                    np.savetxt(f, coor[iatom-1], fmt="%s", newline=" ")
                    f.write('\n')

    def geo_pe():
        pass

    def dftbin_nonpe_ml(self, direct, coor, specie, speciedict):
        row, col = np.shape(coor)
        cmd = 'cp dftb_in.hsd dftb_in.temp'
        results = subprocess.run(cmd, shell=True, universal_newlines=True, check=True)
        cmd = "sed -i '15, 16d' dftb_in.temp"
        results = subprocess.run(cmd, shell=True, universal_newlines=True, check=True)
        print('This is dftb_in')
        for iname in specie:
            atom_i = 0
            for iatom in range(0, speciedict[iname]):
                atom_i += 1
                if iname == 'H':
                    cmd = "sed -i '15i\H"+str(iatom+1)+'='+'"s"'+"' dftb_in.temp"
                    results = subprocess.run(cmd, shell=True, universal_newlines=True, check=True)
                if iname == 'C':
                    cmd = "sed -i '15i\C"+str(iatom+1)+'='+'"p"'+"' dftb_in.temp"
                    results = subprocess.run(cmd, shell=True, universal_newlines=True, check=True)
        cmd = 'mv dftb_in.temp '+direct+'/dftb_in.hsd'
        results = subprocess.run(cmd, shell=True, universal_newlines=True, check=True)

class FHIaims:
    def __init__(self):
        pass

    def geo_nonpe(self, file, coor, specie, speciedict, symbols):
        row, col = np.shape(coor)
        with open('geometry.in.{}'.format(file), 'w') as f:
            ispecie = 0
            iatom = 0
            for atom in specie:
                ispecie += 1
                for natom in range(0, speciedict[atom]):
                    iatom += 1
                    f.write('atom ')
                    np.savetxt(f, coor[iatom-1], fmt='%s', newline=' ')
                    f.write(symbols[iatom-1])
                    f.write('\n')


    def geo_pe():
        pass


class NWchem:
    pass
