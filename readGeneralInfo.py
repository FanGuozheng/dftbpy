# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 17:08:59 2018
@author: fgz
This file is to get the general information from inputfile.
The information include the read_in_file(the data list from inputfile),
get_line_coordinate(line of coordinate, lines, and atom number), get_scf
(if scf will be performed), get_task(what task is), get_mixFactor(the 
value of mixFactor),get_coor(the coordinate),get_coor_vector(the vector
between atoms, and normalized)
"""
import numpy as np
import sys
import re
from general_para import *
import pandas as pd
#default value
"""inputfile_name='in.ground'
scf='F'
task='steady'
line_coordinate=0
mixFactor=0.2"""

class readGeneralInfo(object):
    def __init__(self):
        pass
    
    ######read input file, and write the input file into a list
    def read_in_file(self,general):
        inputfile_name=general["inputfile_name"]
        self.datalist=[]
        read_file=open(inputfile_name)
        try:
            for line in read_file:
                num_line=line.strip().split()
                self.datalist.append(num_line)
        finally:
            read_file.close()
        #print(self.datalist)
        return self.datalist
    
    ######get the line of coordinate(line_coordinate+1), the total lines
    ######(line_len), and atom number
    def get_line_coordinate(self,general):
        self.line_coordinate=general["line_coordinate"]
        #inputfile_name=general["inputfile_name"]
        num_atom=0
        for i in range(20):   #read the fisrt several lines
            self.read_in_file(general)
            if '$coordinate' in self.datalist[i]:
                self.line_coordinate=i
                self.line_len=len(self.datalist)
        self.num_atom=self.line_len-self.line_coordinate-1
        #print("there are", self.num_atom, "atoms,","the line of string 'coordinate' is",self.line_coordinate,"\n")
        return self.line_coordinate, self.num_atom
    
    ######get the general information of input file, scf, mixFactor
    def get_scf(self,general):
        self.scf=general["scf"]
        self.read_in_file(general)
        #print(data)
        if 'scf=.true.' in self.datalist[0] or 'scf=.T.' in self.datalist[0]:
            self.scf='T'
        else:
            self.scf='F'
            #print("the SCF is", self.scf,"\n")
        return self.scf

    def get_task(self, general):
        self.task=general["task"]
        self.read_in_file(general)
        if "task='steady'" in self.datalist[0]:
            task='steady'
        elif "task='lead'" in self.datalist[0]:
            task='lead'
        #print("the task is", self.task)
        return self.task
    
    ######read string like "mixFactor=0.2"
    def get_mixFactor(self,general):
        self.mixFactor=general["mixFactor"]
        self.read_in_file(general)
        self.get_line_coordinate(general)
        if 'mixFactor=0.2' in self.datalist[0]:
            pass
            #print(self.datalist[0])
        string=''.join(self.datalist[0])
        string2 = ''.join(re.findall('mixFactor=\d+\.?\d*',string))
        #print(list(filter(str.isdigit, string2)))
        if string2=='':
            pass
        else:
            self.mixFactor=float(''.join(list(filter(lambda x: x in "0123456789.", string2))))
        #print(self.mixFactor,self.datalist[0])
        return self.mixFactor
    
    def get_maxiter(self,general):
        self.maxiter=general["maxiter"]
        self.read_in_file(general)
        scf=self.get_scf(general)
        string=''.join(self.datalist[0])
        string2 = ''.join(re.findall('maxiter=\d+\.?\d*',string))
        #print(list(filter(str.isdigit, string2)))
        if string2=='':
            pass
        else:
            self.maxiter=float(''.join(list(filter(lambda x: x in "0123456789.", string2))))
        #print(self.mixFactor,self.datalist[0])
        if scf=="T":
            pass
        else:
            self.maxiter=1
        #print(self.maxiter,scf)
        return self.maxiter

    ######get the coordinate,atom type,atom number
    def get_coor(self):
        num_line=0
        num_atomType=0
        self.get_line_coordinate(general)
        self.data_coor=np.zeros((4,self.line_len-self.line_coordinate-2+1))
        self.read_in_file(general)
        for i in range(self.line_coordinate+1,self.line_len):
            if self.datalist[i] != []:
                self.data_coor[:,num_line]=np.array(self.datalist[i])
                num_line+=1
                #print(self.line_len,i,self.datalist[i])
        atomType=set(self.data_coor[0,:])
        self.num_atomType=len(atomType)
        atomic_num=list(self.data_coor[0,:])
        atom_name=[]
        atom_num=[]
        atomic_num_all=list(self.data_coor[0,:])
        atom_name_all=[]
        self.atom_lmax=[]
        self.atom_ind=np.zeros((len(atomic_num)+1))
        for i in atomic_num:
            self.atom_lmax.append(valence_orbitals[atomName[int(i)-1]])
        for i in range(0,len(atomic_num)):
            self.atom_ind[i+1] = self.atom_ind[i] + self.atom_lmax[i]*self.atom_lmax[i]
            #print(i,self.atom_ind)
        for i in atomic_num:
            atom_name_all.append(atomName[int(i)-1])
            #atom_namelist.append(list(self.data_coor[0,:]).count(i))
        for i in atomic_num:
            while atomic_num.count(i) > 1:
                atomic_num.remove(i)
        for i in atomic_num:
            atom_name.append(atomName[int(i)-1])
            #atom_namelist.append(list(self.data_coor[0,:]).count(i))
            atom_num.append(list(self.data_coor[0,:]).count(i))
            
        #print(atom_num,atom_name_all,self.atom_lmax,atomic_num_all)
        #print("the atomic number and coordinate is as follows:\n", self.data_coor)
        #print("there are", self.num_atomType," atome types,",atom_namelist,"\n")
        return self.data_coor, self.num_atomType, self.atom_ind,atom_name_all
    
    def get_coor_vector(self):
        self.get_line_coordinate(general)
        self.get_coor()
        self.distance=np.zeros((self.num_atom,self.num_atom))
        self.distance_norm=np.zeros((self.num_atom,self.num_atom,3))
        self.distance_vec=np.zeros((self.num_atom,self.num_atom,3))
        atom_atom_type=np.zeros((self.num_atom,self.num_atom))
        for i in range(0,self.num_atom):
            for j in range(0,i):
                x=self.data_coor[1,j]-self.data_coor[1,i]
                y=self.data_coor[2,j]-self.data_coor[2,i]
                z=self.data_coor[3,j]-self.data_coor[3,i]
                d=np.sqrt(x*x+y*y+z*z)
                self.distance[i,j]=d
                #print(self.distance[i,j])
                if(d==0):
                    self.distance_norm[i,j,0]=0
                    self.distance_norm[i,j,1]=0
                    self.distance_norm[i,j,2]=0
                    self.distance_vec[i,j,0]=0
                    self.distance_vec[i,j,1]=0
                    self.distance_vec[i,j,2]=0

                else:
                    self.distance_norm[i,j,0]=x/d
                    self.distance_norm[i,j,1]=y/d
                    self.distance_norm[i,j,2]=z/d
                    self.distance_vec[i,j,0]=x
                    self.distance_vec[i,j,1]=y
                    self.distance_vec[i,j,2]=z

        #print("the atom_atom distance is,\n",self.distance_vec,"\n")
        #print("the normalization of the coordinate_vector is,\n",self.distance_norm,"\n")
        return self.distance, self.distance_norm, self.distance_vec

readGeneralInfo().get_line_coordinate(general)
#readGeneralInfo().read_in_file(general)
readGeneralInfo().get_mixFactor(general)
#readGeneralInfo().read_in_file(general) ##important, readGeneralInfo need()
