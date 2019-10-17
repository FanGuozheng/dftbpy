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
                    print('Warning: periodic parameter format')
            else:
                generalpara['periodic'] = False
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
                xx = coor[j, 1] - coor[i, 1]
                yy = coor[j, 2] - coor[i, 2]
                zz = coor[j, 3] - coor[i, 3]
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
                xx = coor[j, 1] - coor[i, 1]
                yy = coor[j, 2] - coor[i, 2]
                zz = coor[j, 3] - coor[i, 3]
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
        if generalpara['ty'] == 0:
            self.ReadSK0(namei, namej)
        elif generalpara['ty'] == 1:
            self.ReadSK0(namei, namej)
        elif generalpara['ty'] == 5:
            self.ReadSK5(outpara, namei, namej)

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

    def ReadSK2(self, namei, namej):
        '''read hetero- type .skf file'''
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
        for ispe in allskfdata[1]:
            mass_cd.append(float(ispe))
        for iline in range(0, int(ngridpoint)):
            h_s_all.append([float(ii) for ii in allskfdata[iline+2]])
        self.generalpara['grid_dist'+namei+namej] = grid_dist
        self.generalpara['ngridpoint'+namei+namej] = ngridpoint
        self.generalpara['mass_cd'+namei+namej] = mass_cd
        self.generalpara['h_s_all'+namei+namej] = h_s_all
        return self.generalpara


"""def get_task(self, generalpara):
        read the main task from input file, the default of scf is
        scf=.T., the default of task is task='ground',
        scf = 'T'
        datalist = self.read_in_file(GEN_PARA)
        # print(data)
        if 'scf=.true.' in datalist[0] or 'scf=.T.' in datalist[0]:
            scf = 'T'
        else:
            scf = 'F'
        task = 'ground'
        if "task='steady'" in datalist[0]:
            task = 'ground'
        elif "task='lead'" in datalist[0]:
            task = 'steady'
        else:
            task = 'lead'
        # print("the SCF is", self.scf,"\n")
        generalpara["scf"] = scf
        generalpara["task"] = task

    def get_para(self, generalpara, GEN_PARA):
        # read the general parameters needed for dftb calculations, the
        # default value of mixfactor (mixf) is 0.2, maxinter (maxi) ia 100
        mixf = 0.2
        maxi = 100
        telect = 0.0
        periodic = 'T'
        ksampling = np.empty(3)
        datalist = self.read_in_file(GEN_PARA)
        generalpara = self.get_line_coordinate(generalpara, GEN_PARA)
        if 'mixFactor=0.2' in datalist[0]:
            pass
            # print(self.datalist[0])
        str0 = ''.join(datalist[0])
        strmixf = ''.join(re.findall('mixFactor=\d+\.?\d*', str0))
        strt = ''.join(re.findall('tElec=\d+\.?\d*', str0))
        # print(list(filter(str.isdigit, string2)))
        if strmixf == '':
            pass
        else:
            mixf = float(''.join(list(filter(lambda x: x in "0123456789.",
                                             strmixf))))
        print('t', strt)
        if strt == '':
            pass
        else:
            telect = float(''.join(list(filter(lambda x: x in "0123456789.",
                                               strt))))
        if 'tPeriodic=false' in datalist[4] or 'tPeriodic=F' in datalist[4]:
            periodic = 'F'
        if 'tPeriodic=true' in datalist[4] or 'tPeriodic=T' in datalist[4]:
            periodic = 'T'
        ksampling[0] = float(datalist[5][0])
        ksampling[1] = float(datalist[5][1])
        ksampling[2] = float(datalist[5][2])
        generalpara["mixf"] = mixf
        generalpara["maxi"] = maxi
        generalpara["periodic"] = periodic
        generalpara["ksampling"] = ksampling
        generalpara["telect"] = telect
        # return mixf, maxi, periodic, ksampling, telect
        return generalpara

    def read_file(self, generalpara):
        inputfile_name = generalpara["filename"]
        datalist = []
        read_file = open(inputfile_name)
        try:
            for line in read_file:
                num_line = line.strip().split()
                datalist.append(num_line)
        finally:
            read_file.close()
        # print(self.datalist)
        return datalist

    def get_latticepara(self, generalpara):
        unitlattice = np.empty((3, 3))
        datalist = self.read_in_file(GEN_PARA)
        # ###  need revise
        for row in range(12, 15):
            for col in range(0, 3):
                unitlattice[row-12][col] = float(datalist[row][col])
        # print(unitlattice)
        generalpara["latticepara"] = unitlattice/PUBPARA["BOHR"]
        return generalpara

    def get_line_coordinate(self, generalpara, GEN_PARA):
        line_coordinate = 0
        natom = 0
        for i in range(20):
            # read the fisrt several lines
            datalist = self.read_in_file(GEN_PARA)
            if '$coordinate' in datalist[i]:
                line_coordinate = i
                line_len = len(datalist)
        natom = line_len - line_coordinate - 1
        generalpara["natom"] = natom
        generalpara["line_coor"] = line_coordinate
        generalpara["line_inputfile"] = line_len
        # return line_coordinate, natom, line_len
        return generalpara

    def get_coor(self, generalpara):
        num_line = 0
        # line_coor, natom, line_len = self.get_line_coordinate(GEN_PARA)
        generalpara = self.get_line_coordinate(generalpara, GEN_PARA)
        line_len = generalpara["line_inputfile"]
        line_coor = generalpara["line_coor"]
        coor = np.zeros((line_len-line_coor-2+1, 4))
        datalist = self.read_in_file(GEN_PARA)
        for i in range(line_coor+1, line_len):
            if datalist[i] != []:
                coor[num_line, :] = np.array(datalist[i])
                num_line += 1
                # print(self.line_len,i,self.datalist[i])
        atomic_num = list(coor[:, 0])
        coor = coor/PUBPARA["BOHR"]
        atom_name = []
        atom_num = []
        atom_name_all = []
        atom_lmax = []
        natomtype = []
        atomind = np.zeros((len(atomic_num)+1))
        for i in atomic_num:
            atom_lmax.append(VAL_ORB[ATOMNAME[int(i)-1]])
        for i in range(0, len(atomic_num)):
            atomind[i+1] = atomind[i] + atom_lmax[i]*atom_lmax[i]
            # print(i,self.atomind)
        for i in atomic_num:
            atom_name_all.append(ATOMNAME[int(i)-1])
            # atom_namelist.append(list(self.coor[0,:]).count(i))
        for i in atomic_num:
            while atomic_num.count(i) > 1:
                atomic_num.remove(i)
        for i in atomic_num:
            atom_name.append(ATOMNAME[int(i)-1])
            # atom_namelist.append(list(self.coor[0,:]).count(i))
            atom_num.append(list(coor[:, 0]).count(i))
        # print(atom_num,atom_name_all,self.atom_lmax,atomic_num_all)
        # print(coor)
        dictatom = dict(zip(dict(enumerate(set(atom_name_all))).values(),
                            dict(enumerate(set(atom_name_all))).keys()))
        for atomi in atom_name_all:
            natomtype.append(dictatom[atomi])
        generalpara["coor"] = coor
        generalpara["atomind"] = atomind
        # print("coor", coor)
        generalpara["atomnameall"] = atom_name_all
        generalpara["norbs"] = atomind[-1]
        generalpara = self.get_coor_vec(generalpara)
        generalpara["natomtype"] = natomtype
        # return coor, atomind, atom_name_all, distan, d_norm, d_vec
        return generalpara

    def coortest(self, generalpara):
        # get the coordinate, atom type, atom number
        # generate the coordinate
        coor = generalpara["coor"]
        coor[:, 1:4] = coor[:, 1:4]/PUBPARA["BOHR"]
        natom = coor.shape[0]
        atomic_num = list(coor[:, 0])
        atom_name = []
        atom_num = []
        # atomic_num_all = list(coordinate[:, 0])
        atom_name_all = []
        atom_lmax = []
        natomtype = []
        atomind = np.zeros((len(atomic_num)+1))
        for i in atomic_num:
            atom_lmax.append(VAL_ORB[ATOMNAME[int(i)-1]])
        for i in range(0, len(atomic_num)):
            atomind[i + 1] = atomind[i] + atom_lmax[i] * atom_lmax[i]
            # print('i', i, atomind)
        for i in atomic_num:
            atom_name_all.append(ATOMNAME[int(i) - 1])
            # atom_namelist.append(list(self.data_coor[0,:]).count(i))
        for i in atomic_num:
            while atomic_num.count(i) > 1:
                atomic_num.remove(i)
        for i in atomic_num:
            atom_name.append(ATOMNAME[int(i)-1])
            # atom_namelist.append(list(self.data_coor[0,:]).count(i))
            atom_num.append(list(coor[:, 0]).count(i))
        dictatom = dict(zip(dict(enumerate(set(atom_name_all))).values(),
                            dict(enumerate(set(atom_name_all))).keys()))
        for atomi in atom_name_all:
            natomtype.append(dictatom[atomi])
        generalpara["natom"] = natom
        generalpara["atomind"] = atomind
        generalpara["atomnameall"] = atom_name_all
        generalpara["natomtype"] = natomtype
        generalpara = self.get_coor_vec(generalpara)
        # return coor, atomind, atom_name_all, natom, distan, d_norm, d_vec
        return generalpara

    def get_coor_vec(self, generalpara):
        # transfer coordinate to vector
        natom = generalpara["natom"]
        coor = generalpara["coor"]
        distance = np.zeros((natom, natom))
        distance_norm = np.zeros((natom, natom, 3))
        distance_vec = np.zeros((natom, natom, 3))
        # atom_atom_type = np.zeros((natom, natom))
        for i in range(0, natom):
            for j in range(0, i):
                # print(i, j, coor[j, 1], coor[i, 1])
                xx = coor[j, 1] - coor[i, 1]
                yy = coor[j, 2] - coor[i, 2]
                zz = coor[j, 3] - coor[i, 3]
                dd = np.sqrt(xx*xx + yy*yy + zz*zz)
                distance[i, j] = dd
                # print(self.distance[i,j])
                if dd == 0:
                    distance_norm[i, j, 0] = 0
                    distance_norm[i, j, 1] = 0
                    distance_norm[i, j, 2] = 0
                    distance_vec[i, j, 0] = 0
                    distance_vec[i, j, 1] = 0
                    distance_vec[i, j, 2] = 0

                else:
                    distance_norm[i, j, 0] = xx/dd
                    distance_norm[i, j, 1] = yy/dd
                    distance_norm[i, j, 2] = zz/dd
                    distance_vec[i, j, 0] = xx
                    distance_vec[i, j, 1] = yy
                    distance_vec[i, j, 2] = zz
        generalpara["dnorm"] = distance_norm
        generalpara["distance"] = distance
        generalpara["dvec"] = distance_vec
        # print("the atom_atom distance is,\n",self.distance_vec,"\n")
        # return distance, distance_norm, distance_vec
        return generalpara"""
