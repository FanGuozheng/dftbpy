#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is python code for DFTB method
This part is for reading all kinds of input data
"""

import json
import os
import numpy as np
import torch as t
err = 1E-4
BOHR = 0.529177210903
ATOMNAME = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg",
            "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr",
            "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br",
            "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd",
            "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La",
            "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er",
            "Tm", "Yb", "Lu", "Hf", "Ta", "W ", "Re", "Os", "Ir", "Pt", "Au",
            "Hg", "Tl", "Pb", "Bi", "Po", "At"]
VAL_ORB = {"H": 1, "C": 2, "N": 2, "O": 2, "Ti": 3}


class ReadInt:
    """This class will read from .hsd input file, and return these file for
    further calculations"""

    def __init__(self, para):
        self.para = para
        self.get_constant()

    def get_constant(self):
        self.para['boltzmann_constant_H'] = 3.166811429e-6  # Hartree / K
        self.para['t_zero_max'] = 5.0

    def get_task(self, para):
        """this def will read the general information from .json file"""
        filename = para['filename']
        direct = para['direInput']
        with open(os.path.join(direct, filename), 'r') as fp:
            fpinput = json.load(fp)

            # parameter: scf
            if 'scf' in fpinput['general']:
                scf = fpinput['general']['scf']
                if scf in ('True', 'T', 'true'):
                    para['scf'] = True
                elif scf in ('False', 'F', 'false'):
                    para['scf'] = False
                else:
                    raise ImportError('Error: scf not defined correctly')
            else:
                para['scf'] = True

            # parameter: task
            if 'task' in fpinput['general']:
                para['task'] = fpinput['general']['task']
            else:
                para['task'] = 'ground'

            # parameter: scc
            if 'scc' in fpinput['general']:
                scc = fpinput['general']['scc']
<<<<<<< HEAD
                if scc in ('scc', 'nonscc', 'xlbomd'):
                    para['scc'] = scc
=======
                if scc in ('True', 'T', 'true'):
                    para['scc'] = True
                elif scc in ('False', 'F', 'false'):
                    para['scc'] = False
>>>>>>> ad0a72eac1ab68c13c8d6d1ed31874c22bb490c3
                else:
                    raise ImportError('Error: scc not defined correctly')
            else:
                para['scc'] = False

<<<<<<< HEAD
            if 'Lml' in fpinput['general']:
                Lml = fpinput['general']['Lml']
                if Lml in ('T', 'True', 'true'):
                    para['Lml'] = True
                elif Lml in ('F', 'False', 'false'):
                    para['Lml'] = False
                else:
                    raise ImportError('Error: scc not defined correctly')
            else:
                para['Lml'] = False
=======
            if 'ml' in fpinput['general']:
                scc = fpinput['general']['ml']
                if scc in ('True', 'T', 'true'):
                    para['ml'] = True
                elif scc in ('False', 'F', 'false'):
                    para['ml'] = False
                else:
                    raise ImportError('Error: scc not defined correctly')
            else:
                para['ml'] = False
>>>>>>> ad0a72eac1ab68c13c8d6d1ed31874c22bb490c3

            # parameter: mixFactor
            if 'mixFactor' in fpinput['general']:
                para['mixFactor'] = fpinput['general']['mixFactor']
            else:
                para['mixFactor'] = 0.2

            # parameter: tElec
            if 'tElec' in fpinput['general']:
                para['tElec'] = fpinput['general']['tElec']
            else:
                para['tElec'] = 0

            # parameter: mixing parameters
            if 'mixMethod' in fpinput['general']:
                para['mixMethod'] = fpinput['general']['mixMethod']
            else:
                para['mixMethod'] = 'anderson'
            if 'maxIter' in fpinput['general']:
                para['maxIter'] = fpinput['general']['maxIter']
            else:
                para['maxIter'] = 60

<<<<<<< HEAD
            # convergence
            if 'convergenceType' in fpinput['general']:
                para['convergenceType'] = fpinput['general']['convergenceType']
            else:
                para['convergenceType'] = 'charge'
            if 'energy_tol' in fpinput['general']:
                para['energy_tol'] = fpinput['general']['energy_tol']
            else:
                para['energy_tol'] = 1e-6

=======
>>>>>>> ad0a72eac1ab68c13c8d6d1ed31874c22bb490c3
            # --------------------------skf----------------------------
            # ninterp: the number of points for interp when read .skf
            if 'ninterp' in fpinput['skf']:
                para['ninterp'] = fpinput['skf']['ninterp']
            else:
                para['ninterp'] = 8

            if 'grid0' in fpinput['skf']:
                para['grid0'] = fpinput['skf']['grid0']
            else:
                para['grid0'] = 0.4

            if 'dist_tailskf' in fpinput['skf']:
                para['dist_tailskf'] = fpinput['skf']['dist_tailskf']
            else:
                para['dist_tailskf'] = 1.0

            # --------------------------analysis----------------------------
            if 'dipole' in fpinput['analysis']:
                dipole = fpinput['analysis']['dipole']
                if dipole in ('True', 'T', 'true'):
                    para['dipole'] = True
                elif dipole in ('False', 'F', 'false'):
                    para['dipole'] = False
                else:
                    ImportError('Error: dipole not defined correctly')
            else:
                para['dipole'] = False

            # --------------------------geometry----------------------------
            # parameter: periodic
            if 'periodic' in fpinput['general']:
                period = fpinput['general']['periodic']
                if period in ('True', 'T', 'true'):
                    para['periodic'] = True
                elif period in ('False', 'F', 'false'):
                    para['periodic'] = False
                else:
                    para['periodic'] = False
                    print('Warning: periodic not defined correctly')
            else:
                para['periodic'] = False

            if 'type' in fpinput['geometry']:
                para['coorType'] = fpinput['geometry']['periodic']
            else:
                para['coorType'] = 'C'

            # interphs: use interpolation to generate SK instead of read .skf
<<<<<<< HEAD
            if 'Lml_HS' in fpinput['general']:
                scc = fpinput['general']['Lml_HS']
                if scc in ('True', 'T', 'true'):
                    para['Lml_HS'] = True
                elif scc in ('False', 'F', 'false'):
                    para['Lml_HS'] = False
                else:
                    raise ImportError('Error: Lml_HS not defined correctly')
            else:
                para['Lml_HS'] = False
        return para

    def get_coor(self):
        '''
        this function will (read) / process coor info
        '''
        filename = self.para['filename']
        direct = self.para['direInput']
        with open(os.path.join(direct, filename), 'r') as f:
            fpinput = json.load(f)
            try:
                coor = fpinput['geometry']['coor']
            except IOError:
                print('please define the coordination')
        self.para['coor'] = t.from_numpy(np.asarray(coor))

    def cal_coor(self):
        '''
        generate vector, ditance according to input geometry
        Input:
            coor: [natom, 4], the first column is atom number
        Output:
            natomtype: the type of atom, the 1st is 0, the 2nd different is 1
            atomind: how many orbitals of each atom in DFTB calculations
        '''
        coor = self.para['coor']
        natom = np.shape(coor)[0]
        distance = t.zeros(natom, natom)
        dnorm, dvec = t.zeros(natom, natom, 3), t.zeros(natom, natom, 3)
        atomnamelist, atom_lmax, atomind = [], [], []
        self.para['natomtype'] = []

        atomind.append(0)
        [atomnamelist.append(ATOMNAME[int(num) - 1]) for num in coor[:, 0]]

        for iat in range(0, natom):
            atom_lmax.append(VAL_ORB[ATOMNAME[int(coor[iat, 0] - 1)]])
            atomind.append(int(atomind[iat] + atom_lmax[iat]**2))
            for jat in range(0, natom):
                [xx, yy, zz] = (coor[jat, 1:] - coor[iat, 1:]) / BOHR
                dd = np.sqrt(xx * xx + yy * yy + zz * zz)
                distance[iat, jat] = dd
                if dd > err:
                    dnorm[iat, jat, :] = t.Tensor([xx,  yy, zz]) / dd
                    dvec[iat, jat, :] = t.Tensor([xx, yy, zz])

        dictat = dict(zip(dict(enumerate(set(atomnamelist))).values(),
                          dict(enumerate(set(atomnamelist))).keys()))
        [self.para['natomtype'].append(dictat[ati]) for ati in atomnamelist]
        self.para['atomind2'] = int(atomind[natom] * (atomind[natom] + 1) / 2)
        self.para['atomspecie'] = list(set(atomnamelist))
        self.para['distance'], self.para['dnorm'] = distance, dnorm
        self.para['dvec'], self.para['natom'] = dvec, natom
        self.para['lmaxall'], self.para['atomind'] = atom_lmax, atomind
        self.para['atomnameall'] = atomnamelist

        self.cal_neighbour()

    def cal_neighbour(self):
        natom = self.para['natom']
        # cutoff =
        self.para['Nneighbour'] = t.zeros(natom)


class ReadSKt:
    '''
    this class is to read .skf file
    Input:
        namei, namej: name of atom i, j
        direSK: directory of .skf file
        '''
    def __init__(self, para):
        '''
        read .skf file and save in dict para
        '''
        self.para = para
        # self.read_sk()
        # self.get_cutoff()

    def read_skf_specie(self):
        '''
        read .skf according to atom species and its combinations
        Input:
            atom specie
            directory of .skf and .skf files with format atom-atom.skf
        Output (suffix with names of between atoms):
            grid distance
            number of total grids
            onsite, SPE and occupations
            mass, c, d parameters
            intergrals
        '''
        atomspecie, diresk = self.para['atomspecie'], self.para['direSK']
        nspecie = len(atomspecie)

        for iat in range(0, nspecie):
            for jat in range(0, nspecie):

                nameij = atomspecie[iat] + atomspecie[jat]
                skname = atomspecie[iat] + '-' + atomspecie[jat] + '.skf'
                fp = open(os.path.join(diresk, skname), "r")
                words = fp.readline().split()
                self.para['grid_dist' + nameij] = float(words[0])
                self.para['ngridpoint' + nameij] = int(words[1])
                nitem = int(words[1]) * 20

                if atomspecie[iat] == atomspecie[jat]:  # Homo-nuclear
                    fp_line = [float(ii) for ii in fp.readline().split()]
                    fp_line_ = t.from_numpy(np.asarray(fp_line))
                    self.para['onsite' + nameij] = fp_line_[0:3]
                    self.para['spe' + nameij] = fp_line_[3]
                    self.para['uhubb' + nameij] = fp_line_[4:7]
                    self.para['occ_skf' + nameij] = fp_line_[7:10]
                    data = np.fromfile(fp, dtype=float, count=20, sep=' ')
                    self.para['mass_cd' + nameij] = t.from_numpy(data)
                    hs_all = np.fromfile(fp, dtype=float, count=nitem, sep=' ')
                    hs_all.shape = (int(words[1]), 20)
                    self.para['hs_all' + nameij] = hs_all

                    spline = fp.readline().split()
                    if 'Spline' in spline:
                        nInt_cutoff = fp.readline().split()
                        nint_ = int(nInt_cutoff[0])
                        self.para['nint_rep' + nameij] = nint_
                        self.para['cutoff_rep' + nameij] = float(nInt_cutoff[1])
                        a123 = fp.readline().split()
                        self.para['a1_rep' + nameij] = float(a123[0])
                        self.para['a2_rep' + nameij] = float(a123[1])
                        self.para['a3_rep' + nameij] = float(a123[2])
                        datarep = np.fromfile(fp, dtype=float,
                                              count=(nint_-1)*6, sep=' ')
                        datarep.shape = (nint_ - 1, 6)
                        self.para['rep' + nameij] = t.from_numpy(datarep)
                        datarepend = np.fromfile(fp, dtype=float,
                                                 count=8, sep=' ')
                        self.para['repend' + nameij] = t.from_numpy(datarepend)
                else:  # Hetero-nuclear
                    data = np.fromfile(fp, dtype=float, count=20, sep=' ')
                    self.para['mass_cd' + nameij] = t.from_numpy(data)
                    hs_all = np.fromfile(fp, dtype=float, count=nitem, sep=' ')
                    hs_all.shape = (int(words[1]), 20)
                    self.para['hs_all' + nameij] = hs_all
                    # self.para['skf_rest' + nameij] = fp.read()

                    spline = fp.readline().split()
                    if 'Spline' in spline:
                        nInt_cutoff = fp.readline().split()
                        nint_ = int(nInt_cutoff[0])
                        self.para['nint_rep' + nameij] = nint_
                        self.para['cutoff_rep' + nameij] = float(nInt_cutoff[1])
                        a123 = fp.readline().split()
                        self.para['a1_rep' + nameij] = float(a123[0])
                        self.para['a2_rep' + nameij] = float(a123[1])
                        self.para['a3_rep' + nameij] = float(a123[2])
                        datarep = np.fromfile(fp, dtype=float,
                                              count=(nint_-1)*6, sep=' ')
                        datarep.shape = (nint_ - 1, 6)
                        self.para['rep' + nameij] = t.from_numpy(datarep)
                        datarepend = np.fromfile(fp, dtype=float,
                                                 count=8, sep=' ')
                        self.para['repend' + nameij] = t.from_numpy(datarepend)

    def read_skf_specie_old(self):
        atomspecie, diresk = self.para['atomspecie'], self.para['direSK']
        nspecie = len(atomspecie)

        for iat in range(0, nspecie):
            for jat in range(0, iat + 1):
                nameij = atomspecie[iat] + atomspecie[jat]
                if atomspecie[iat] == atomspecie[jat]:   # Homo-nuclear
                    skname = atomspecie[iat] + '-' + atomspecie[jat] + '.skf'
                    fp = open(os.path.join(diresk, skname))
                    skdata = []
                    try:
                        for line in fp:
                            each_line = line.strip().split()
                            skdata.append(each_line)
                    except IOError:
                        print('failed to open Slater-Koster file')
                    self.para['grid_dist' + nameij] = float(skdata[0][0])
                    self.para['ngridpoint' + nameij] = int(skdata[0][1])
                    line1_temp, hs_all = [], []
                    self.para['mass_cd' + nameij] = []
                    [line1_temp.append(float(ix)) for ix in skdata[1]]
                    self.para['onsite' + nameij] = line1_temp[0:3]
                    self.para['spe' + nameij] = line1_temp[3]
                    self.para['uhubb' + nameij] = line1_temp[4:7]
                    self.para['occ_skf' + nameij] = line1_temp[7:10]
                    for imass_cd in skdata[2]:
                        self.para['mass_cd' + nameij].append(float(imass_cd))
                    for iline in range(0, int(skdata[0][1])):
                        hs_all.append([float(ij) for ij in skdata[iline + 3]])
                    self.para['hs_all' + nameij] = hs_all

                else:  # Hetero-nuclear
                    nameji = atomspecie[jat] + atomspecie[iat]
                    self.para['mass_cd' + nameij] = []
                    self.para['mass_cd' + nameji] = []
                    sknameij = atomspecie[iat] + '-' + atomspecie[jat] + '.skf'
                    sknameji = atomspecie[jat] + '-' + atomspecie[iat] + '.skf'
                    fp_ij = open(os.path.join(diresk, sknameij))
                    fp_ji = open(os.path.join(diresk, sknameji))
                    skdataij, skdataji, hs_ij, hs_ji = [], [], [], []
                    try:
                        for line in fp_ij:
                            each_line = line.strip().split()
                            skdataij.append(each_line)
                        for line in fp_ji:
                            each_line = line.strip().split()
                            skdataji.append(each_line)
                    except IOError:
                        print('failed to open Slater-Koster file')
                    self.para['grid_dist' + nameij] = float(skdataij[0][0])
                    self.para['ngridpoint' + nameij] = int(skdataij[0][1])
                    self.para['grid_dist' + nameji] = float(skdataji[0][0])
                    self.para['ngridpoint' + nameji] = int(skdataji[0][1])
                    for imass_cdij in skdataij[1]:
                        self.para['mass_cd' + nameij].append(float(imass_cdij))
                    for imass_cdji in skdataji[1]:
                        self.para['mass_cd' + nameji].append(float(imass_cdji))
                    assert int(skdataij[0][1]) == int(skdataji[0][1])
                    for iline in range(0, int(skdataij[0][1])):
                        self.readhs_ij_line(
                            iline, skdataij, skdataji, atomspecie[iat],
                            atomspecie[jat], hs_ij, hs_ji)
                    self.para['hs_all' + nameij] = hs_ij
                    self.para['hs_all' + nameji] = hs_ji

    def readhs_ij_line(self, iline, skdataij, skdataji, iat, jat, hsij, hsji):
        '''
        deal with the integrals each line in .skf
        '''
        lmax = max(VAL_ORB[iat], VAL_ORB[jat])
        lmin = min(VAL_ORB[iat], VAL_ORB[jat])
        if lmax == 1:
            [hsij.append(float(ij)) for ij in skdataij[iline + 2]]
            [hsji.append(float(ji)) for ji in skdataji[iline + 2]]
        elif lmax == 2 and lmin == 1:
            if VAL_ORB[iat] == 2:
                hsji.append([float(ji) for ji in skdataji[iline + 2]])
                hsij.append(hsji[-1])
                hsij[-1][8], hsij[-1][18] = -hsij[-1][8], -hsij[-1][18]
            elif VAL_ORB[jat] == 2:
                hsij.append([float(ij) for ij in skdataij[iline + 2]])
                hsji.append(hsij[-1])
                hsji[-1][8], hsji[-1][18] = -hsji[-1][8], -hsji[-1][18]
        elif lmax == 2 and lmax == 1:
            pass

    def read_sk(self, namei, namej):
        '''read homo- type .skf file'''
        print(namei, namej)
        nameij = namei + namej
        skfname = namei + '-' + namej + '.skf'
=======
            if 'interphs' in fpinput['general']:
                scc = fpinput['general']['interphs']
                if scc in ('True', 'T', 'true'):
                    para['interphs'] = True
                elif scc in ('False', 'F', 'false'):
                    para['interphs'] = False
                else:
                    raise ImportError('Error: interphs not defined correctly')
            else:
                para['interphs'] = False
        return para

    def get_coor(self, para):
        '''this function will (read) / process coor info'''
        if para['readInput']:
            filename = para['filename']
            direct = para['direInput']
            with open(os.path.join(direct, filename), 'r') as f:
                fpinput = json.load(f)
                try:
                    para['coor'] = fpinput['geometry']['coor']
                except IOError:
                    print('please define the coordination')
            coor = np.asarray(para['coor'])
            natom = np.shape(coor)[0]
            coor = t.from_numpy(coor)
        else:
            coor0 = para['coor']
            natom = np.shape(coor0)[0]
            coor = t.zeros((natom, 4))
            coor[:, 1:] = coor0[:, :]
            icount = 0
            for iname in para['symbols']:
                coor[icount, 0] = ATOMNAME.index(iname) + 1
                icount += 1
        distance = t.zeros((natom, natom))
        dnorm = t.zeros((natom, natom, 3))
        dvec = t.zeros((natom, natom, 3))
        atomnamelist = []
        atom_lmax = []
        atomind = []
        atomind.append(0)
        for i in range(0, natom):
            atomunm = int(coor[i, 0] - 1)
            atomnamelist.append(ATOMNAME[atomunm])
            atom_lmax.append(VAL_ORB[ATOMNAME[atomunm]])
            atomind.append(int(atomind[i] + atom_lmax[i]**2))
            for j in range(0, i):
                xx = (coor[j, 1] - coor[i, 1]) / BOHR
                yy = (coor[j, 2] - coor[i, 2]) / BOHR
                zz = (coor[j, 3] - coor[i, 3]) / BOHR
                dd = np.sqrt(xx * xx + yy * yy + zz * zz)
                distance[i, j] = dd
                if dd < err:
                    pass
                else:
                    dnorm[i, j, 0], dnorm[i, j, 1], dnorm[i, j, 2] = xx / dd, \
                        yy / dd, zz / dd
                    dvec[i, j, 0], dvec[i, j, 1], dvec[i, j, 2] = xx, yy, zz
        para['natomtype'] = []
        dictatom = dict(zip(dict(enumerate(set(atomnamelist))).values(),
                            dict(enumerate(set(atomnamelist))).keys()))
        for atomi in atomnamelist:
            para['natomtype'].append(dictatom[atomi])
        para['atomind2'] = int(atomind[natom] * (atomind[natom] + 1) / 2)
        para['atomname_set'] = list(set(atomnamelist))
        para['distance_norm'] = dnorm
        para['distance'] = distance
        para['dvec'] = dvec
        para['natom'] = natom
        para['lmaxall'] = atom_lmax
        para['atomnameall'] = atomnamelist
        para['atomind'] = atomind
        return para


class ReadSKt:
    '''this class is to read .skf file'''
    def __init__(self, para, namei, namej):
        self.para = para
        self.namei = namei
        self.namej = namej
        self.read_sk(namei, namej)
        self.get_cutoff(namei, namej)

    def read_sk(self, namei, namej):
        '''read homo- type .skf file'''
        nameij = namei + namej
        skfname = namei+'-'+namej+'.skf'
>>>>>>> ad0a72eac1ab68c13c8d6d1ed31874c22bb490c3
        direc = self.para['direSK']
        fp_sk = open(os.path.join(direc, skfname))
        allskfdata = []
        try:
            for line in fp_sk:
                each_line = line.strip().split()
                allskfdata.append(each_line)
        except IOError:
            print('open Slater-Koster file ERROR')
        grid_dist = float(allskfdata[0][0])
        ngridpoint = int(allskfdata[0][1])
        mass_cd = []
<<<<<<< HEAD
        hs_all = []
=======
        h_s_all = []
>>>>>>> ad0a72eac1ab68c13c8d6d1ed31874c22bb490c3
        if namei == namej:
            line1_temp = []
            [line1_temp.append(float(ix)) for ix in allskfdata[1]]
            self.para['onsite' + nameij] = line1_temp[0:3]
            self.para['spe' + nameij] = line1_temp[3]
            self.para['uhubb' + nameij] = line1_temp[4:7]
            self.para['occ_skf' + nameij] = line1_temp[7:10]
            for imass_cd in allskfdata[2]:
                mass_cd.append(float(imass_cd))
            for iline in range(0, ngridpoint):
<<<<<<< HEAD
                hs_all.append([float(ii) for ii in allskfdata[iline + 3]])
        else:
            for imass_cd in allskfdata[1]:
                mass_cd.append(float(imass_cd))
            for iline in range(0, int(ngridpoint)):
                hs_all.append([float(ii) for ii in allskfdata[iline + 2]])
        self.para['grid_dist' + nameij] = grid_dist
        self.para['ngridpoint' + nameij] = ngridpoint
        self.para['mass_cd' + nameij] = mass_cd
        self.para['hs_all' + nameij] = hs_all

    def read_sk_(self):
        '''read homo- type .skf file'''
        grid_dist = float(self.para['grid_dist' + self.nameij][0])
        ngridpoint = int(self.para['grid_dist' + self.nameij][1])
        mass_cd = []
        if self.namei == self.namej:
            espd_uspd = []
            for ispe in self.para['espd_uspd' + self.nameij]:
                espd_uspd.append(float(ispe))
            self.para['espd_uspd' + self.nameij] = espd_uspd
            for imass_cd in self.para['mass_cd' + self.nameij]:
                mass_cd.append(float(imass_cd))
        self.para['grid_dist' + self.nameij] = grid_dist
        self.para['ngridpoint' + self.nameij] = ngridpoint
        self.para['mass_cd' + self.nameij] = mass_cd

    def get_cutoff(self, namei, namej):
        '''get the cutoff of atomi-atomj in .skf file'''
        nameij = namei + namej
        grid = self.para['grid_dist' + nameij]
        ngridpoint = self.para['ngridpoint' + nameij]
        disttailsk = self.para['dist_tailskf']
        cutoff = grid * ngridpoint + disttailsk
        lensk = grid * ngridpoint
        self.para['cutoffsk' + nameij] = cutoff
        self.para['lensk' + nameij] = lensk

    def get_cutoff_all(self):
        '''get the cutoff of atomi-atomj in .skf file'''
        atomspecie = self.para['atomspecie']
        for iat in range(0, len(atomspecie)):
            for jat in range(0, len(atomspecie)):
                nameij = atomspecie[iat] + atomspecie[jat]
                grid = self.para['grid_dist' + nameij]
                ngridpoint = self.para['ngridpoint' + nameij]
                disttailsk = self.para['dist_tailskf']
                cutoff = grid * ngridpoint + disttailsk
                lensk = grid * ngridpoint
                self.para['cutoffsk' + nameij] = cutoff
                self.para['lensk' + nameij] = lensk
=======
                h_s_all.append([float(ii) for ii in allskfdata[iline + 3]])
        else:
            for imass_cd in allskfdata[1]:
                mass_cd.append(float(imass_cd))
        for iline in range(0, int(ngridpoint)):
            h_s_all.append([float(ii) for ii in allskfdata[iline + 2]])
        self.para['grid_dist' + nameij] = grid_dist
        self.para['ngridpoint' + nameij] = ngridpoint
        self.para['mass_cd' + nameij] = mass_cd
        self.para['h_s_all' + nameij] = h_s_all
        return self.para

    def read_sk_(self, namei, namej):
        '''read homo- type .skf file'''
        grid_dist = float(self.para['grid_dist'+namei+namej][0])
        ngridpoint = int(self.para['grid_dist'+namei+namej][1])
        mass_cd = []
        if namei == namej:
            espd_uspd = []
            for ispe in self.para['espd_uspd'+namei+namej]:
                espd_uspd.append(float(ispe))
            self.para['espd_uspd'+namei+namej] = espd_uspd
            for imass_cd in self.para['mass_cd'+namei+namej]:
                mass_cd.append(float(imass_cd))
        self.para['grid_dist'+namei+namej] = grid_dist
        self.para['ngridpoint'+namei+namej] = ngridpoint
        self.para['mass_cd'+namei+namej] = mass_cd
        return self.para

    def get_cutoff(self, namei, namej):
        '''get the cutoff of atomi-atomj in .skf file'''
        grid = self.para['grid_dist'+namei+namej]
        ngridpoint = self.para['ngridpoint'+namei+namej]
        disttailsk = self.para['dist_tailskf']
        cutoff = grid * ngridpoint + disttailsk
        lensk = grid * ngridpoint
        self.para['cutoffsk'+namei+namej] = cutoff
        self.para['lensk'+namei+namej] = lensk
        return self.para
>>>>>>> ad0a72eac1ab68c13c8d6d1ed31874c22bb490c3
