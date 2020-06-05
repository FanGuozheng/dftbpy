#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import torch as t
default_r = {"H": 3, "C": 3.5, "N": 2.2, "O": 2.3, "S": 3.8}
ATOMNUM = {"H": 1, "C": 6, "N": 7, "O": 8}


class Dftbplus:
    '''
    interface to DFTB+
    '''

    def __init__(self, para):
        '''
        read output from DFTB+, write DFTB+ input files
        '''
        self.para = para

    def geo_nonpe_ml(self, file, coor, specie, speciedict, symbols):
        '''write geo.gen in a dataset for non-periodic condition'''
        row, col = np.shape(coor)
        compressr0 = np.zeros(row)
        with open('geo.gen.{}'.format(file), 'w') as fp:
            fp.write(str(row)+" "+"C")
            fp.write('\n')
            for iname in specie:
                atom_i = 0
                for iatom in range(0, speciedict[iname]):
                    atom_i += 1
                    fp.write(iname+str(atom_i)+' ')
            fp.write('\n')
            iatom = 0
            for atom in specie:
                for natom in range(0, speciedict[atom]):
                    compressr0[iatom] = default_r[atom]
                    iatom += 1
                    fp.write(str(iatom)+' '+str(iatom)+' ')
                    np.savetxt(fp, coor[iatom-1], fmt="%s", newline=" ")
                    fp.write('\n')
        return compressr0

    def geo_nonpe(self, dire, coor, specie):
        '''
        this code is used to ml process, each atom will be added a number
        so that every atom will be different'''
        row, col = np.shape(coor)
        with open(os.path.join(dire, 'geo.gen'), 'w') as fp:

            # lst line, number of atoms
            fp.write(str(row) + " " + "C")
            fp.write('\n')

            # 2nd line: atom specie
            for ispe in specie:
                fp.write(ispe + ' ')
            fp.write('\n')

            # coordination
            iatom = 0
            nspe = 0
            for ispe in specie:
                nspe += 1
                for jatom in range(row):
                    if ATOMNUM[ispe] == coor[jatom, 0]:
                        iatom += 1
                        fp.write(str(iatom) + " " + str(nspe) + " ")
                        np.savetxt(fp, coor[jatom, 1:], fmt="%s", newline=" ")
                        fp.write('\n')

    def geo_pe():
        '''write geo.gen in a dataset for periodic condition'''
        pass

    def write_dftbin(self, dire, scc, coor, specie):
        '''write dftb_in.hsd'''
        with open(os.path.join(dire, 'dftb_in.hsd'), 'w') as fp:
            fp.write('Geometry = GenFormat { \n')
            fp.write('  <<< "geo.gen" \n } \n')
            fp.write('Driver {} \n')
            fp.write('Hamiltonian = DFTB { \n')
            if scc in ['scc', 'mbdscc']:
                fp.write('Scc = Yes \n')
                fp.write('SccTolerance = 1e-8 \n MaxSccIterations = 100 \n')
                fp.write('Mixer = Broyden { \n MixingParameter = 0.2 } \n')
            elif scc == 'nonscc':
                fp.write('Scc = No \n')
            fp.write('MaxAngularMomentum { \n')
            for ispe in specie:
                if ispe == 'H':
                    fp.write('H="s" \n')
                elif ispe == 'C':
                    fp.write('C="p" \n')
                elif ispe == 'N':
                    fp.write('N="p" \n')
                elif ispe == 'O':
                    fp.write('O="p" \n')
            fp.write('} \n Charge = 0.0 \n SpinPolarisation {} \n')
            fp.write('Eigensolver = DivideAndConquer {} \n')
            fp.write('Filling = Fermi { \n')
            fp.write('Temperature [Kelvin] = 0.0 \n } \n')
            fp.write('SlaterKosterFiles = Type2FileNames { \n')
            fp.write('Separator = "-" \n Suffix = ".skf" \n } \n')
            if scc == 'mbdscc':
                fp.write('Dispersion = Mbd { \n')
                fp.write('ReferenceSet = "ts" \n')
                fp.write('NOmegaGrid = 15 \n')
                fp.write('Beta = 1.05 \n KGrid = 3 3 3 \n')
                fp.write('VacuumAxis = Yes Yes Yes \n } \n')
            fp.write('} \n')
            fp.write('Analysis { \n CalculateForces = Yes \n } \n')
            fp.write('Options { \n WriteAutotestTag = Yes \n } \n')
            fp.write('ParserOptions { \n ParserVersion = 5 \n } \n')
            fp.write('Parallel { \n UseOmpThreads = Yes \n } \n')

        '''row, col = np.shape(coor)
        os.sys('cp dftb_in.hsd dftb_in.temp')
        os.sys("sed -i '15, 16d' dftb_in.temp")
        for ispe in specie:
            atom_i = 0
            if iname == 'H':
                os.sys("sed -i '15i\H" + str(iatom + 1) + '=' + '"s"' +
                       "' dftb_in.temp")
            if iname == 'C':
                os.sys("sed -i '15i\C" + str(iatom + 1) + '=' +
                       '"p"'+"' dftb_in.temp")
        os.sys('mv dftb_in.temp '+direct+'/dftb_in.hsd')'''

    def read_bandenergy(self, para, nfile, dire, inunit='H', outunit='H'):
        '''
        read file bandenergy.dat, which is HOMO and LUMO data
        H: Hartree
        '''
        fp = open(os.path.join(dire, 'bandenergy.dat'))
        bandenergy = np.zeros((nfile, 2), dtype=float)
        for ifile in range(0, nfile):
            ibandenergy = np.fromfile(fp, dtype=float, count=2, sep=' ')
            if inunit == outunit:
                bandenergy[ifile, :] = ibandenergy[:]
        return t.from_numpy(bandenergy)

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
        return t.from_numpy(dipole)

    def read_energy(self, para, nfile, dire, inunit='H', outunit='H'):
        '''read file dip.dat, which is dipole data'''
        fp = open(os.path.join(dire, 'energy.dat'))
        energy = np.zeros((nfile), dtype=float)

        for ifile in range(nfile):
            iener = np.fromfile(fp, dtype=float, count=1, sep=' ')
            if inunit == outunit:
                energy[ifile] = iener
        return t.from_numpy(energy)

    def read_hstable(self, para, nfile, dire):
        '''read file bandenergy.dat, which is HOMO and LUMO data'''
        fp = open(os.path.join(dire, 'hstable_ref'))
        hstable_ref = np.zeros((36), dtype=float)

        for ifile in range(0, nfile):
            ibandenergy = np.fromfile(fp, dtype=float, count=36, sep=' ')
            hstable_ref[:] = ibandenergy[:]
        return t.from_numpy(hstable_ref)


class FHIaims:
    '''interface to FHI-aims'''

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

    def read_bandenergy(self, para, nfile, dire, inunit='H', outunit='H'):
        '''read file bandenergy.dat, which is HOMO and LUMO data'''
        fp = open(os.path.join(dire, 'bandenergy.dat'))
        bandenergy = np.zeros((nfile, 2))
        for ifile in range(0, nfile):
            ibandenergy = np.fromfile(fp, dtype=float, count=2, sep=' ')
            if inunit == outunit:
                bandenergy[ifile, :] = ibandenergy[:]
        return t.from_numpy(bandenergy)

    def read_dipole(self, para, nfile, dire, unit='eang', outunit='debye'):
        '''read file dip.dat, which is dipole data'''
        fp = open(os.path.join(dire, 'dip.dat'))
        dipole = np.zeros((nfile, 3), dtype=float)

        for ifile in range(nfile):
            idipole = np.fromfile(fp, dtype=float, count=3, sep=' ')
            if unit == 'eang' and outunit == 'debye':
                dipole[ifile, :] = idipole[:] / 0.2081943
            elif unit == outunit:
                dipole[ifile, :] = idipole[:]
            elif unit == 'debye' and outunit == 'eang':
                dipole[ifile, :] = idipole[:] * 0.2081943
        return t.from_numpy(dipole)

    def read_energy(self, para, nfile, dire, inunit='H', outunit='H'):
        '''read file dip.dat, which is dipole data'''
        fp = open(os.path.join(dire, 'energy.dat'))
        energy = np.zeros((nfile), dtype=float)

        for ifile in range(0, nfile):
            iener = np.fromfile(fp, dtype=float, count=1, sep=' ')
            if inunit == outunit:
                energy[ifile] = iener[:]
        return t.from_numpy(energy)

    def read_qatom(self, para, nfile, dire, inunit='e', outunit='e'):
        '''read file dip.dat, which is dipole data'''
        nmaxatom = 10
        qatom = np.zeros((nfile, nmaxatom), dtype=float)

        fp = open(os.path.join(dire, 'qatomref.dat'))
        for ifile in range(0, nfile):
            natom = para['coorall'][ifile].shape[0]
            iqatom = np.fromfile(fp, dtype=float, count=natom, sep=' ')
            if inunit == outunit:
                qatom[ifile, :natom] = iqatom[:]
        return t.from_numpy(qatom)


class NWchem:
    pass
