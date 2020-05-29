#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import os
import torch as t
import subprocess

default_r = {"H": 3, "C": 3.5, "N": 2.2, "O": 2.3, "S": 3.8}


class Dftbplus:
    def __init__(self, para):
        self.para = para

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

    def read_bandenergy(self, para, nfile, dire,
                        inunit='hartree', outunit='hartree'):
        '''read file bandenergy.dat, which is HOMO and LUMO data'''
        fp = open(os.path.join(dire, 'bandenergy.dat'))
        bandenergy = np.zeros((nfile, 2))
        for ifile in range(0, nfile):
            ibandenergy = np.fromfile(fp, dtype=float, count=2, sep=' ')
            if inunit == outunit:
                bandenergy[ifile, :] = ibandenergy[:]
        return bandenergy

    def read_dipole(self, para, nfile, dire, unit='eang', outunit='debye'):
        '''read file dip.dat, which is dipole data'''
        fp = open(os.path.join(dire, 'dip.dat'))
        dipole = np.zeros((nfile, 3))
        for ifile in range(0, nfile):
            idipole = np.fromfile(fp, dtype=float, count=3, sep=' ')
            if unit == 'eang' and outunit == 'debye':
                dipole[ifile, :] = idipole[:] / 0.2081943
            elif unit == outunit:
                dipole[ifile, :] = idipole[:]
            elif unit == 'debye' and outunit == 'eang':
                dipole[ifile, :] = idipole[:] * 0.2081943
        return dipole

    def read_energy(self, para, nfile, dire,
                    inunit='hartree', outunit='hartree'):
        '''read file dip.dat, which is dipole data'''
        fp = open(os.path.join(dire, 'energy.dat'))
        energy = np.zeros(nfile)
        for ifile in range(0, nfile):
            iener = np.fromfile(fp, dtype=float, count=1, sep=' ')
            if inunit == outunit:
                energy[ifile] = iener
        return energy

    def read_hstable(self, para, nfile, dire):
        '''read file bandenergy.dat, which is HOMO and LUMO data'''
        fp = open(os.path.join(dire, 'hstable_ref'))
        hstable_ref = np.zeros(36)
        for ifile in range(0, nfile):
            ibandenergy = np.fromfile(fp, dtype=float, count=36, sep=' ')
            hstable_ref[:] = ibandenergy[:]
        return t.from_numpy(hstable_ref)


class FHIaims:

    def __init__(self, para):
        self.para = para

    def geo_nonpe_hdf(self, para, ibatch, coor):
        '''input is from hdf data, output is FHI-aims input: geo.in'''
        specie, speciedict, symbols = para['specie'], para['speciedict'], \
            para['symbols']
        with open('geometry.in.{}'.format(ibatch), 'w') as fp:
            ispecie = 0
            iatom = 0
            for atom in specie:
                ispecie += 1
                for natom in range(0, speciedict[atom]):
                    iatom += 1
                    fp.write('atom ')
                    np.savetxt(fp, coor[iatom - 1], fmt='%s', newline=' ')
                    fp.write(symbols[iatom - 1])
                    fp.write('\n')

    def geo_pe():
        pass

    def read_bandenergy(self, para, nfile, dire,
                        inunit='hartree', outunit='hartree'):
        '''read file bandenergy.dat, which is HOMO and LUMO data'''
        fp = open(os.path.join(dire, 'bandenergy.dat'))
        bandenergy = np.zeros((nfile, 2))
        for ifile in range(0, nfile):
            ibandenergy = np.fromfile(fp, dtype=float, count=2, sep=' ')
            if inunit == outunit:
                bandenergy[ifile, :] = ibandenergy[:]
        return bandenergy

    def read_dipole(self, para, nfile, dire, unit='eang', outunit='debye'):
        '''read file dip.dat, which is dipole data'''
        fp = open(os.path.join(dire, 'dip.dat'))
        dipole = np.zeros((nfile, 3))
        for ifile in range(0, nfile):
            idipole = np.fromfile(fp, dtype=float, count=3, sep=' ')
            if unit == 'eang' and outunit == 'debye':
                dipole[ifile, :] = idipole[:] / 0.2081943
            elif unit == outunit:
                dipole[ifile, :] = idipole[:]
            elif unit == 'debye' and outunit == 'eang':
                dipole[ifile, :] = idipole[:] * 0.2081943
        return dipole

    def read_energy(self, para, nfile, dire,
                    inunit='hartree', outunit='hartree'):
        '''read file dip.dat, which is dipole data'''
        fp = open(os.path.join(dire, 'energy.dat'))
        energy = np.zeros(nfile)
        for ifile in range(0, nfile):
            iener = np.fromfile(fp, dtype=float, count=1, sep=' ')
            if inunit == outunit:
                energy[ifile] = iener[:]
        return energy

    def read_qatom(self, para, nfile, dire, inunit='e', outunit='e'):
        '''read file dip.dat, which is dipole data'''
        fp = open(os.path.join(dire, 'qatomref.dat'))
        nmaxatom = 10
        qatom = np.zeros((nfile, nmaxatom))
        for ifile in range(0, nfile):
            natom = para['coorall'][ifile].shape[0]
            iqatom = np.fromfile(fp, dtype=float, count=natom, sep=' ')
            if inunit == outunit:
                qatom[ifile, :natom] = iqatom[:]
        return qatom


class NWchem:
    pass
