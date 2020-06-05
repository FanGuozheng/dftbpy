#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is python code for DFTB method
This part is for reading all kinds of input data
"""
import numpy as np
import json
import os
tol4 = 1E-4
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


class ReadInput:
    """
    This class will read from .hsd input file, and return these file for
    further calculations
    """

    def __init__(self, para):
        """
        according to different Args, call different functions
        """
        self.para = para
        self.get_task()
        self.cal_coor()

    def get_task(self):
        """this def will read the general information from .json file"""
        filename = self.para['filename']
        direct = self.para['dire']

        with open(os.path.join(direct, filename), 'r') as f:
            datatask = json.load(f)

            # general
            try:
                self.para['ty'] = datatask['general']['ty']
            except IOError:
                print('please define ty: true or false')

            try:
                self.para['scf'] = datatask['general']['scf']
            except IOError:
                print('please define scf: true or false')

            try:
                self.para['task'] = datatask['general']['task']
            except IOError:
                print('please define task')

            if 'mixMethod' in datatask['general']:
                self.para['mixMethod'] = datatask['general']['mixMethod']
            else:
                self.para['mixMethod'] = 'simple'

            if 'mixFactor' in datatask['general']:
                self.para['mixFactor'] = datatask['general']['mixFactor']
            else:
                self.para['mixFactor'] = 0.2

            if 'tElec' in datatask['general']:
                self.para['tElec'] = datatask['general']['tElec']
            else:
                self.para['tElec'] = 0

            if 'maxIter' in datatask['general']:
                self.para['maxIter'] = datatask['general']['maxIter']
            else:
                self.para['maxIter'] = 60

            if 'convergenceType' in datatask['general']:
                self.para['convergenceType'] = \
                    datatask['general']['convergenceType']
            else:
                self.para['convergenceType'] = 'energy'

            if 'periodic' in datatask['general']:
                period = datatask['general']['periodic']
                if period == 'True' or period == 'T' or period == 'true':
                    self.para['periodic'] = True
                elif period == 'False' or period == 'F' or period == 'false':
                    self.para['periodic'] = False
                else:
                    self.para['periodic'] = False
                    print('Warning: periodic not defined correctly')
            else:
                self.para['periodic'] = False

            if 'scc' in datatask['general']:
                scc = datatask['general']['scc']
                if scc == 'True' or scc == 'T' or scc == 'true':
                    self.para['scc'] = True
                elif scc == 'False' or scc == 'F' or scc == 'false':
                    self.para['scc'] = False
                else:
                    self.para['scc'] = False
                    print('Warning: scc not defined correctly')
            else:
                self.para['scc'] = False

            # analysis
            if 'dipole' in datatask['analysis']:
                dipole = datatask['analysis']['dipole']
                if dipole == 'True' or dipole == 'T' or dipole == 'true':
                    self.para['dipole'] = True
                elif dipole == 'False' or dipole == 'F' or dipole == 'false':
                    self.para['dipole'] = False
                else:
                    print('Warning: dipole parameter format')
            else:
                self.para['dipole'] = False

            # geometry
            if self.para['periodic'] is True:
                try:
                    self.para['ksample'] = datatask['geometry']['ksample']
                except IOError:
                    print('please define K-mesh points')
                try:
                    self.para['unit'] = datatask['geometry']['unit']
                except ImportError:
                    print('please define unit')

            if 'type' in datatask['geometry']:
                self.para['coorType'] = datatask['geometry']['periodic']
            else:
                self.para['coorType'] = 'C'

            if self.para['ty'] in ['dftb']:
                try:
                    self.para['coor'] = np.asarray(
                            datatask['geometry']['coor'])
                except IOError:
                    print('please define coor: true or false')

    def read_cal_coor(self):
        '''
        reading and processing geometry data
        '''
        filename = self.para['filename']
        direct = self.para['dire']
        with open(os.path.join(direct, filename), 'r') as f:
            datatask = json.load(f)
            try:
                self.para['coor'] = datatask['geometry']['coor']
            except IOError:
                print('please define the coordination')
        coor = np.asarray(self.para['coor'])
        natom = np.shape(coor)[0]
        distance = np.zeros((natom, natom), dtype=float)
        distance_norm = np.zeros((natom, natom, 3), dtype=float)
        distance_vec = np.zeros((natom, natom, 3), dtype=float)
        atomnamelist = []
        atom_lmax = []
        atomind = np.zeros((natom + 1), dtype=int)
        for iat in range(natom):
            atomunm = int(coor[iat, 0] - 1)
            atomnamelist.append(ATOMNAME[atomunm])
            atom_lmax.append(VAL_ORB[ATOMNAME[atomunm]])
            atomind[iat + 1] = atomind[iat] + atom_lmax[iat] ** 2
            for jat in range(iat):
                [xx, yy, zz] = (coor[jat, 1:] - coor[iat, 1:]) / BOHR
                dd = np.sqrt(xx ** 2 + yy ** 2 + zz ** 2)
                distance[iat, jat] = dd
                if dd < tol4:
                    pass
                else:
                    distance_norm[iat, jat, :] = [xx, yy, zz] / dd
                    distance_vec[iat, jat, :] = [xx, yy, zz]
        natomtype = []
        dictatom = dict(zip(dict(enumerate(set(atomnamelist))).values(),
                            dict(enumerate(set(atomnamelist))).keys()))
        for atomi in atomnamelist:
            natomtype.append(dictatom[atomi])
        self.para['distance_norm'] = distance_norm
        self.para['distance'] = distance
        self.para['distance_vec'] = distance_vec
        self.para['natom'] = natom
        self.para['lmaxall'] = atom_lmax
        self.para['atomnameall'] = atomnamelist
        self.para['atomind'] = atomind
        self.para['natomtype'] = natomtype

    def cal_coor(self):
        '''
        with raw geometry data and processing
        '''
        coor0 = self.para['coor']
        natom = np.shape(coor0)[0]
        coor = np.zeros((natom, 4), dtype=float)
        distance = np.zeros((natom, natom), dtype=float)
        distance_norm = np.zeros((natom, natom, 3), dtype=float)
        distance_vec = np.zeros((natom, natom, 3), dtype=float)
        atomnamelist = []
        atom_lmax = []
        atomind = np.zeros((natom + 1), dtype=int)

        coor[:, :] = coor0[:, :]
        for iat in range(0, natom):
            atomunm = int(coor[iat, 0] - 1)
            atomnamelist.append(ATOMNAME[atomunm])
            atom_lmax.append(VAL_ORB[ATOMNAME[atomunm]])
            atomind[iat + 1] = atomind[iat] + atom_lmax[iat] ** 2
            for jat in range(iat):
                xx = (coor[jat, 1] - coor[iat, 1])/BOHR
                yy = (coor[jat, 2] - coor[iat, 2])/BOHR
                zz = (coor[jat, 3] - coor[iat, 3])/BOHR
                dd = np.sqrt(xx*xx + yy*yy + zz*zz)
                distance[iat, jat] = dd
                if dd < tol4:
                    pass
                else:
                    distance_norm[iat, jat, :] = [xx, yy, zz] / dd
                    distance_vec[iat, jat, :] = [xx, yy, zz]
        natomtype = []
        dictatom = dict(zip(dict(enumerate(set(atomnamelist))).values(),
                            dict(enumerate(set(atomnamelist))).keys()))
        for atomi in atomnamelist:
            natomtype.append(dictatom[atomi])
        self.para['coor'] = coor
        self.para['distance_norm'] = distance_norm
        self.para['distance'] = distance
        self.para['distance_vec'] = distance_vec
        self.para['natom'] = natom
        self.para['lmaxall'] = atom_lmax
        self.para['atomnameall'] = atomnamelist
        self.para['atomind'] = atomind
        self.para['natomtype'] = natomtype


class ReadSK:

    def __init__(self, para, namei, namej):
        self.para = para
        self.namei = namei
        self.namej = namej
        self.para['grid0'] = 0.4
        self.para['disttailsk'] = 1.0
        self.para['ninterp'] = 8
        if self.para['ty'] in ['dftb', 'dftbpy']:
            self.ReadSK0(namei, namej)
        elif self.para['ty'] in ['ml']:
            self.ReadSK5(namei, namej)
        self.getCutoff(namei, namej)

    def ReadSK0(self, namei, namej):
        '''read homo- type .skf file'''
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
            Espd_Uspd = []
            for ispe in allskfdata[1]:
                Espd_Uspd.append(float(ispe))
            self.para['Espd_Uspd'+namei+namej] = Espd_Uspd
            for imass_cd in allskfdata[2]:
                mass_cd.append(float(imass_cd))
            for iline in range(0, int(ngridpoint)):
                h_s_all.append([float(ii) for ii in allskfdata[iline+3]])
        else:
            for imass_cd in allskfdata[1]:
                mass_cd.append(float(imass_cd))
        for iline in range(0, int(ngridpoint)):
            h_s_all.append([float(ii) for ii in allskfdata[iline+2]])
        self.para['grid_dist' + namei + namej] = grid_dist
        self.para['ngridpoint' + namei + namej] = ngridpoint
        self.para['mass_cd' + namei + namej] = mass_cd
        self.para['h_s_all' + namei + namej] = h_s_all

    def ReadSK5(self, namei, namej):
        '''read homo- type .skf file'''
        grid_dist = float(self.para['grid_dist' + namei + namej][0])
        ngridpoint = int(self.para['grid_dist' + namei + namej][1])
        mass_cd = []
        if namei == namej:
            Espd_Uspd = []
            for ispe in self.para['Espd_Uspd' + namei + namej]:
                Espd_Uspd.append(float(ispe))
            self.para['Espd_Uspd' + namei + namej] = Espd_Uspd
            for imass_cd in self.para['mass_cd'+namei+namej]:
                mass_cd.append(float(imass_cd))
        self.para['grid_dist' + namei + namej] = grid_dist
        self.para['ngridpoint' + namei + namej] = ngridpoint
        self.para['mass_cd' + namei + namej] = mass_cd
        self.para['h_s_all'] = self.para['h_s_all']

    def getCutoff(self, namei, namej):
        grid = self.para['grid_dist' + namei + namej]
        ngridpoint = self.para['ngridpoint' + namei + namej]
        disttailsk = self.para['disttailsk']
        cutoff = grid * ngridpoint + disttailsk
        lensk = grid * ngridpoint
        self.para['cutoffsk' + namei + namej] = cutoff
        self.para['lensk' + namei + namej] = lensk
