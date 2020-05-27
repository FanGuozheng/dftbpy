#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is python code for DFTB method
This part is for reading all kinds of input data
"""
import numpy as np
import json
import os
import torch
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


class ReadInput(object):
    """This class will read from .hsd input file, and return these file for
    further calculations"""

    def __init__(self, generalpara):
        self.generalpara = generalpara
        self.get_task(self.generalpara)
        if self.generalpara['ty'] == 5:
            self.get_coor5(self.generalpara)
        elif self.generalpara['ty'] == 1:
            self.get_coor5(self.generalpara)
        elif self.generalpara['ty'] == 0:
            self.get_coor(self.generalpara)

    def get_task(self, generalpara):
        """this def will read the general information from .json file"""
        filename = generalpara['filename']
        direct = generalpara['dire']
        with open(os.path.join(direct, filename), 'r') as f:
            datatask = json.load(f)
            # --------------------------general----------------------------- #
            try:
                generalpara['scf'] = datatask['general']['scf']
            except IOError:
                print('please define scf: true or false')
            try:
                generalpara['task'] = datatask['general']['task']
            except IOError:
                print('please define task')
            if 'mixFactor' in datatask['general']:
                generalpara['mixFactor'] = datatask['general']['mixFactor']
            else:
                generalpara['mixFactor'] = 0.2
            if 'tElec' in datatask['general']:
                generalpara['tElec'] = datatask['general']['tElec']
            else:
                generalpara['tElec'] = 0
            if 'maxIter' in datatask['general']:
                generalpara['maxIter'] = datatask['general']['maxIter']
            else:
                generalpara['maxIter'] = 60
            if 'periodic' in datatask['general']:
                period = datatask['general']['periodic']
                if period == 'True' or period == 'T' or period == 'true':
                    generalpara['periodic'] = True
                elif period == 'False' or period == 'F' or period == 'false':
                    generalpara['periodic'] = False
                else:
                    generalpara['periodic'] = False
                    print('Warning: periodic not defined correctly')
            else:
                generalpara['periodic'] = False
            if 'scc' in datatask['general']:
                scc = datatask['general']['scc']
                if scc == 'True' or scc == 'T' or scc == 'true':
                    generalpara['scc'] = True
                elif scc == 'False' or scc == 'F' or scc == 'false':
                    generalpara['scc'] = False
                else:
                    generalpara['scc'] = False
                    print('Warning: scc not defined correctly')
            else:
                generalpara['scc'] = False
            # --------------------------analysis---------------------------- #
            if 'dipole' in datatask['analysis']:
                dipole = datatask['analysis']['dipole']
                if dipole == 'True' or dipole == 'T' or dipole == 'true':
                    generalpara['dipole'] = True
                elif dipole == 'False' or dipole == 'F' or dipole == 'false':
                    generalpara['dipole'] = False
                else:
                    print('Warning: dipole parameter format')
            else:
                generalpara['dipole'] = False
            # --------------------------geometry---------------------------- #
            if generalpara['periodic'] is True:
                try:
                    generalpara['ksample'] = datatask['geometry']['ksample']
                except IOError:
                    print('please define K-mesh points')
                try:
                    generalpara['unit'] = datatask['geometry']['unit']
                except ImportError:
                    print('please define unit')
            if 'type' in datatask['geometry']:
                generalpara['coorType'] = datatask['geometry']['periodic']
            else:
                generalpara['coorType'] = 'C'
        return generalpara

    def get_coor(self, generalpara):
        filename = generalpara['filename']
        direct = generalpara['dire']
        with open(os.path.join(direct, filename), 'r') as f:
            datatask = json.load(f)
            try:
                generalpara['coor'] = datatask['geometry']['coor']
            except IOError:
                print('please define the coordination')
        coor = np.asarray(generalpara['coor'])
        natom = np.shape(coor)[0]
        coor = torch.from_numpy(coor)
        distance = torch.zeros((natom, natom))
        distance_norm = torch.zeros((natom, natom, 3))
        distance_vec = torch.zeros((natom, natom, 3))
        atomnamelist = []
        atom_lmax = []
        atomind = np.zeros(natom+1)
        for i in range(0, natom):
            atomunm = int(coor[i, 0]-1)
            atomnamelist.append(ATOMNAME[atomunm])
            atom_lmax.append(VAL_ORB[ATOMNAME[atomunm]])
            atomind[i+1] = atomind[i] + atom_lmax[i]**2
            for j in range(0, i):
                xx = (coor[j, 1] - coor[i, 1])/BOHR
                yy = (coor[j, 2] - coor[i, 2])/BOHR
                zz = (coor[j, 3] - coor[i, 3])/BOHR
                dd = np.sqrt(xx*xx + yy*yy + zz*zz)
                distance[i, j] = dd
                if dd < tol4:
                    pass
                else:
                    distance_norm[i, j, 0] = xx/dd
                    distance_norm[i, j, 1] = yy/dd
                    distance_norm[i, j, 2] = zz/dd
                    distance_vec[i, j, 0] = xx
                    distance_vec[i, j, 1] = yy
                    distance_vec[i, j, 2] = zz
        natomtype = []
        dictatom = dict(zip(dict(enumerate(set(atomnamelist))).values(),
                            dict(enumerate(set(atomnamelist))).keys()))
        for atomi in atomnamelist:
            natomtype.append(dictatom[atomi])
        generalpara['distance_norm'] = distance_norm
        generalpara['distance'] = distance
        generalpara['distance_vec'] = distance_vec
        generalpara['natom'] = natom
        generalpara['lmaxall'] = atom_lmax
        generalpara['atomnameall'] = atomnamelist
        generalpara['atomind'] = atomind
        generalpara['natomtype'] = natomtype
        return generalpara

    def get_coor5(self, generalpara):
        coor0 = generalpara['coor']
        natom = np.shape(coor0)[0]
        coor = np.zeros((natom, 4))
        coor[:, 1:] = coor0[:, :]
        icount = 0
        for iname in generalpara['symbols']:
            coor[icount, 0] = ATOMNAME.index(iname)+1
            icount += 1
        distance = np.zeros((natom, natom))
        distance_norm = np.zeros((natom, natom, 3))
        distance_vec = np.zeros((natom, natom, 3))
        atomnamelist = []
        atom_lmax = []
        atomind = np.zeros(natom+1)
        for i in range(0, natom):
            atomunm = int(coor[i, 0]-1)
            atomnamelist.append(ATOMNAME[atomunm])
            atom_lmax.append(VAL_ORB[ATOMNAME[atomunm]])
            atomind[i+1] = atomind[i] + atom_lmax[i]**2
            for j in range(0, i):
                xx = (coor[j, 1] - coor[i, 1])/BOHR
                yy = (coor[j, 2] - coor[i, 2])/BOHR
                zz = (coor[j, 3] - coor[i, 3])/BOHR
                dd = np.sqrt(xx*xx + yy*yy + zz*zz)
                distance[i, j] = dd
                if dd < tol4:
                    pass
                else:
                    distance_norm[i, j, 0] = xx/dd
                    distance_norm[i, j, 1] = yy/dd
                    distance_norm[i, j, 2] = zz/dd
                    distance_vec[i, j, 0] = xx
                    distance_vec[i, j, 1] = yy
                    distance_vec[i, j, 2] = zz
        natomtype = []
        dictatom = dict(zip(dict(enumerate(set(atomnamelist))).values(),
                            dict(enumerate(set(atomnamelist))).keys()))
        for atomi in atomnamelist:
            natomtype.append(dictatom[atomi])
        generalpara['coor'] = coor
        generalpara['distance_norm'] = distance_norm
        generalpara['distance'] = distance
        generalpara['distance_vec'] = distance_vec
        generalpara['natom'] = natom
        generalpara['lmaxall'] = atom_lmax
        generalpara['atomnameall'] = atomnamelist
        generalpara['atomind'] = atomind
        generalpara['natomtype'] = natomtype
        return generalpara


class ReadSK(object):
    def __init__(self, generalpara, outpara, namei, namej):
        self.generalpara = generalpara
        self.namei = namei
        self.namej = namej
        self.generalpara['grid0'] = 0.4
        self.generalpara['disttailsk'] = 1.0
        self.generalpara['ninterp'] = 8
        if generalpara['ty'] == 0:
            self.ReadSK0(namei, namej)
        elif generalpara['ty'] == 1:
            self.ReadSK0(namei, namej)
        elif generalpara['ty'] == 5:
            self.ReadSK5(outpara, namei, namej)
        self.getCutoff(generalpara, namei, namej)

    def ReadSK0(self, namei, namej):
        '''read homo- type .skf file'''
        skfname = namei+'-'+namej+'.skf'
        direc = self.generalpara['direSK']
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
            self.generalpara['Espd_Uspd'+namei+namej] = Espd_Uspd
            for imass_cd in allskfdata[2]:
                mass_cd.append(float(imass_cd))
            for iline in range(0, int(ngridpoint)):
                h_s_all.append([float(ii) for ii in allskfdata[iline+3]])
        else:
            for imass_cd in allskfdata[1]:
                mass_cd.append(float(imass_cd))
        for iline in range(0, int(ngridpoint)):
            h_s_all.append([float(ii) for ii in allskfdata[iline+2]])
        self.generalpara['grid_dist'+namei+namej] = grid_dist
        self.generalpara['ngridpoint'+namei+namej] = ngridpoint
        self.generalpara['mass_cd'+namei+namej] = mass_cd
        self.generalpara['h_s_all'+namei+namej] = h_s_all
        return self.generalpara

    def ReadSK5(self, outpara, namei, namej):
        '''read homo- type .skf file'''
        grid_dist = float(outpara['grid_dist'+namei+namej][0])
        ngridpoint = int(outpara['grid_dist'+namei+namej][1])
        mass_cd = []
        if namei == namej:
            Espd_Uspd = []
            for ispe in outpara['Espd_Uspd'+namei+namej]:
                Espd_Uspd.append(float(ispe))
            self.generalpara['Espd_Uspd'+namei+namej] = Espd_Uspd
            for imass_cd in outpara['mass_cd'+namei+namej]:
                mass_cd.append(float(imass_cd))
        self.generalpara['grid_dist'+namei+namej] = grid_dist
        self.generalpara['ngridpoint'+namei+namej] = ngridpoint
        self.generalpara['mass_cd'+namei+namej] = mass_cd
        self.generalpara['h_s_all'] = outpara['h_s_all']
        return self.generalpara

    def getCutoff(self, generalpara, namei, namej):
        grid = self.generalpara['grid_dist'+namei+namej]
        ngridpoint = self.generalpara['ngridpoint'+namei+namej]
        disttailsk = self.generalpara['disttailsk']
        cutoff = grid*ngridpoint+disttailsk
        lensk = grid*ngridpoint
        self.generalpara['cutoffsk'+namei+namej] = cutoff
        self.generalpara['lensk'+namei+namej] = lensk
        return self.generalpara
