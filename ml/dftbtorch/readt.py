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
                if scc in ('True', 'T', 'true'):
                    para['scc'] = True
                elif scc in ('False', 'F', 'false'):
                    para['scc'] = False
                else:
                    raise ImportError('Error: scc not defined correctly')
            else:
                para['scc'] = False

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
        h_s_all = []
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
