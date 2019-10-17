# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 19:43:12 2018
@author: fgz12  
read skf file
"""
#class module()  ML: input ML features(structrue),such as coordinate,ctomic number,distance
#class keywords: SCF, mixfactor... lodesatr or DFTB+
#class Main

import numpy as np
import re
import linecache
import scipy
from scipy.linalg import * 
from readGeneralInfo import *
from general_para import *
rGI = readGeneralInfo()

class readSkf2(object):
    def __init__(self):
        pass
    
    def read_skf_file(self,general,skf_name):
        self.input_skffile_name=skf_name
        self.data_skflist=[]
        read_file=open(self.input_skffile_name)
        try:
            for line in read_file:
                num_line=line.strip().split()
                self.data_skflist.append(num_line)
        finally:
            read_file.close()
        #print("the name of skf file is:",self.input_skffile_name,"\n")
        return self.data_skflist
    
    ######get the gridDist,nGridPoints
    def get_grid(self,skf_name):
        self.read_skf_file(general,skf_name)
        self.data_skflist0=np.array(self.data_skflist[0])
        self.data_skflist1=np.array(self.data_skflist[1])
        #print(self.data_skflist0)
        self.gridDist=0
        self.nGridPoints=0
        self.gridDist=float(''.join(list(filter(lambda x: x in "0123456789.-", self.data_skflist0[0]))))
        self.nGridPoints=int(''.join(list(filter(lambda x: x in "0123456789.-", self.data_skflist0[1]))))
        skself=[]
        #print("gridDist is:",self.gridDist)
        #print("nGridPoints is:",self.nGridPoints,"\n")
        #uHubb=self.data_skflist1[6]
        #print("self.data_skflist1[6]",self.data_skflist1[6])
        #uHubb=''.join(list(filter(lambda x: x in "0123456789.",self.data_skflist1[6])))
        #uHubb=self.data_skflist1[6]
        for i in range(0,len(self.data_skflist1)):
            uHubb=''.join(list(filter(lambda x: x in "0123456789.-",self.data_skflist1[i])))
            skself.append(uHubb)
        for i in range(0,len(self.data_skflist1)):
            skself[i]=float(skself[i])
        #print("uHUbb",uHubb,len(self.data_skflist1),skself)
        return self.gridDist, self.nGridPoints, skself

    ######get the line of spline
    def get_line_Spline(self,skf_name):
        self.read_skf_file(general,skf_name)
        self.get_grid(skf_name)
        self.line_Spline=self.nGridPoints
        for i in range(self.nGridPoints,self.nGridPoints+50):######can be revised
            #print(self.nGridPoints,self.data_skflist[i])
            #print("i",i)
            if 'Spline' in self.data_skflist[i]:
                break
            else:
                self.line_Spline+=1
        #print("the line number of Spline line ",self.line_Spline)
        return self.line_Spline

    def gMatrix(self,uHubb):
        distance=rGI.get_coor_vector()[2]
        #print(distance)
        num_atom=rGI.get_line_coordinate(general)[1]
        gMat=[]
        k=0
        for i in range(0,num_atom):
            for j in range(0,i+1):
                #if range(0,i), i=j=0 will not print
                rr=np.sqrt(distance[i,j,0]*distance[i,j,0]+distance[i,j,1]*distance[i,j,1] \
                          +distance[i,j,2]*distance[i,j,2])
                rr=rr/0.5291772106712
                #print(r,k,uHubb)
                a1=3.2*uHubb[i]
                a2=3.2*uHubb[j]
                src=1/(a1+a2)
                fac=a1*a2*src
                avg=1.6*(fac+fac*fac*src)
                fhbond=1
                if rr < 1.0E-4:
                    gval= 0.3125*avg
                    #print(gval)
                else:
                    rrc= 1.0/rr
                    rrc3= rrc*rrc*rrc
                    #print(a1,a2)
                    if abs(a1-a2) < 1.0E-5:
                        fac= avg*rr
                        fac2= fac*fac
                        efac= np.exp(-fac)/48.0
                        gval= (1.0-fhbond*(48.0+33*fac+fac2*(9.0+fac))*efac)*rrc
                    else:
                        val12=self.gamsub(a1,a2,rr,rrc)
                        val21=self.gamsub(a2,a1,rr,rrc)
                        gval= rrc-fhbond*val12-fhbond*val21
                        print(rrc,fhbond,val12,val21)
                gMat.append(gval)
                #print(gval,gMat)
        return gMat
                
    def gamsub(self,a,b,rr,rrc):
        a2= a*a
        b2= b*b
        b4= b2*b2
        b6= b4*b2
        drc= 1.0/(a2-b2)
        drc2=drc*drc
        efac= np.exp(-a*rr)
        fac= (b6-3*a2*b4)*drc2*drc*rrc
        gval= efac*(0.5*a*b4*drc2-fac)
        gdrv= -a*gval+efac*fac*rrc
        return gval
        
    def shiftHamGamma(self,num_atom,qAtom,qZero,gMat):
        qdiff=[]
        shift=[]
        for i in range(0,num_atom):
            qdiff.append(qAtom[i] - qZero[i])
            #print("qAtom, qZero,qfiff in shiftHamGamma",qAtom, qZero,qdiff)
        for i in range(0,num_atom):
            shifti= 0
            for j in range(0,num_atom):
                if j > i:
                    k=j*(j+1)/2+i
                    gamma=gMat[int(k)]
                    #print(k,num_atom,i,j)
                else:
                    k=i*(i+1)/2+j
                    #print(k,num_atom,i,j)
                    gamma=gMat[int(k)]
                shifti=shifti+qdiff[j]*gamma
                #print("shifti,qdiff[j]",shifti,qdiff[j],gamma,i,j)
            shift.append(shifti)
        #print(shift)
        return shift
    
    def fermi(self,tElec,nElect,nOrbs,eigVal,occ):
        ckbol = 3.16679E-6   #original from lodestar, with revision
        degtol = 1.0E-4
        racc = 2E-16
        dacc = 4*racc
        for i in range(1,nOrbs):
            occ[i] = 0.0
        if nElect > 1.0E-5:
            if nElect > 2*nOrbs:
                print( 'too many electrons')
                ###seems break is wrong here
            elif tElec > 5.0:
                beta = 1.0/(ckbol*tElec)
                etol = ckbol*tElec*(np.log(beta)-np.log(racc))
                tzero = False
            else:
                etol = degtol
                tzero = True
                
            if nElect > int(nElect):
                nef1 = int((nElect+2)/2)
                nef2 = int((nElect+2)/2)
            else:
                nef1 = int((nElect+1)/2)
                nef2 = int((nElect+2)/2)
            #print("nef1,2", nef1,nef2)
            eBot = eigVal[0]
            #print("nef1,nef2",nef1,nef2)
            eFermi = 0.5*(eigVal[nef1-1]+eigVal[nef2-1])
            #print("eFermi",eFermi,eigVal[nef1-1],eigVal[nef2-1])
            nup = nef1
            ndown = nef1
            nup0=nup
            ndown0=ndown
            #print("nup,ndown before",nup,ndown)
            while nup0 < nOrbs:   #
                #print("eigVal[nup]-eFermi",eigVal[nup],eFermi,nup0)
                if abs(eigVal[nup0]-eFermi) < etol:
                    nup0 = nup0+1
                    #print("nup0",nup0,abs(eigVal[nup]-eFermi),etol)
                else:
                    break
            nup=nup0
            #print("nup",nup,"nef1,2  ",nef1,nef2)
            while ndown0 > 0:
                #print(ndown0,eFermi,"abs",abs(eigVal[ndown0-1]-eFermi),eigVal)
                if abs(eigVal[ndown0-1]-eFermi) < etol:
                    ndown0 = ndown0-1
                else:
                    break
            ndown=ndown0
            #print("ndown",ndown,nup,nElect)
            ndeg = nup-ndown    #####check
            nocc2 = ndown
            for i in range(0, nocc2):
                occ[i] = 2.0
                #print("occ[i]",occ[i],"i",i,nocc2)
            if ndeg == 0:
                return occ, eFermi
            
            if tzero:
                occdg = ndeg
                #print("ndeg",ndeg,nup,ndown)
                occdg = (nElect-2*nocc2)/occdg
                #print("occdg",occdg,nocc2)
                for i in range(nocc2,nocc2+ndeg):
                    occ[i] = occdg
                    #print("occdg",occdg,nElect,nocc2)
            else:
                chleft = nElect-2*nocc2
                istart = nocc2+1
                iend = istart+ndeg-1
                if ndeg == 1:
                    occ[istart] = chleft
                    return
                ef1 = eFermi-etol-degtol
                ef2 = eFermi+etol+degtol
                ceps = dacc*chleft
                eeps = dacc*max(dabs(ef1),dabs(ef2))
                eFermi = 0.5*(ef1+ef2)  ###check
                charge = 0.0
                for i in range(istart,iend):
                    occ[i] = 2.0/(1.0+np.exp(beta*(eigVal[i]-eFermi)))
                    charge = charge+occ[i]
                    if charge > chleft:
                        ef2 = eFermi
                    else:
                        ef1 = eFermi
                    if abs(charge-chleft) > ceps or abs(ef1-ef2) < eeps:
                        continue
                    else:
                        exit

                if abs(charge-chleft) < ceps:
                    return
                else:
                    fac = chleft/charge
                    for i in range(istart,iend):
                        occ[i] = occ[i]*fac
                        #print("occ[i] in fermi", occ[i],"i",i)
        else:
            print('electron number is zero!')
            return
        #print("occ",occ)
        return occ, eFermi
    
    def mulliken(self,num_atom,nOrbs,atom_ind,overMat,denMat,qAt):
        for ii in range(0,num_atom):
            qAt[ii]=0.0
            #print("int(atom_ind[ii]+1),int(atom_ind[ii+1])",int(atom_ind[ii]+1),int(atom_ind[ii+1]))
            for i in range(int(atom_ind[ii]),int(atom_ind[ii+1])):
                for j in range(0, i):
                    k = i*(i+1)/2+j
                    qAt[ii]=qAt[ii]+denMat[int(k)]*overMat[int(k)]
                    #print("ii",ii,"qAt[ii]",qAt[ii],"denMat[k]",denMat[k],"overMat[k]1",overMat[k],"k",k)
                for j in range(i, nOrbs):
                    k = j*(j+1)/2+i
                    qAt[ii]=qAt[ii]+denMat[int(k)]*overMat[int(k)]
                    #print("ii",ii,"qAt[ii]",qAt[ii],"denMat[k]",denMat[k],"overMat[k]1",overMat[k],"k",k)
        #print("qAt",qAt)
        return qAt
    
    def get_HS_from_skf(self,skf_name):
        self.get_line_Spline(skf_name)
        self.read_skf_file(general,skf_name)
        self.get_grid(skf_name)
        j=3  
        #j is the line where H and S begins, following several lines code 
        #is to make sure the beginning line of H and S
        data_HSlist_test=np.zeros((5,20))
        j_line = 0
        for line in self.data_skflist[0:4]:
            #print(skf_name)
            jj=0
            for sub_list in line:
                if '*' in sub_list:
                    abbr=''.join(list(filter(lambda x: x in "0123456789.*", sub_list)))
                    abbr_arr=abbr.split("*")
                    #print(abbr,abbr_arr[0])
                    for _ in range(0,int(abbr_arr[0])):
                        jj+=1
                else:
                    jj+=1
            j_line+=1
            if jj == 20:
                j=j_line
                break
                
        print("j",j,skf_name)           
        self.data_HSlist=np.zeros((self.line_Spline-j+1,21))    
        for line in self.data_skflist[j-1:self.line_Spline]: ###need to revise
            #print(line,self.line_Spline)
            #self.data_HSlist[:,:]
            jj=1
            self.data_HSlist[j-j_line,0]=self.gridDist*(j-j_line)
            for sub_list in line:
                #print("sub_list",sub_list)
                #if sum(map(lambda x : '*' in x , sub_list)) != 0: ####if element in each line has *
                ####if element in each line has *
                if '*' in sub_list :
                    abbr=''.join(list(filter(lambda x: x in "0123456789.*", sub_list)))
                    abbr_arr=abbr.split("*")
                    #print(abbr,abbr_arr[0])
                    for _ in range(0,int(abbr_arr[0])):
                        self.data_HSlist[j-j_line,jj]=np.array(abbr_arr[1]) 
                        #print(j,jj,abbr_arr[0],np.array(abbr_arr[1]))
                        jj+=1
                else:
                    self.data_HSlist[j-j_line,jj]=np.array(sub_list)
                    jj+=1
                #print(abbr,abbr_arr,self.data_HSlist[j,:])
                #print(self.data_HSlist[j-3,jj-1])
            j+=1
        #print("the following are corresponding r, H and S in skf file: \n",self.data_HSlist,"\n")
        return self.data_HSlist
        
    #readSkf().get_HS_from_skf()
    def get_Spline_para(self):
        self.get_line_Spline()
        self.get_grid()
        input_skffile_name=general["input_skffile_name"]
        line=linecache.getline(input_skffile_name,self.line_Spline+2)
        every_line=line.split()
        #print(every_line)
        self.nInt_Spline = int(every_line[0])
        self.cutoff_Spline = float(every_line[1])
        line=linecache.getline(input_skffile_name,self.line_Spline+3)
        every_line=line.split()
        self.a1_Spline = float(every_line[0])
        self.a2_Spline = float(every_line[1])
        self.a3_Spline = float(every_line[2])
        #print(self.nInt_Spline,self.a3_Spline)
        self.repulsive_c=np.zeros((self.nInt_Spline,8))#the last line is 8, the other line is 6 or 7
        jj=0
        for line in open(input_skffile_name).readlines()[self.line_Spline+3:self.line_Spline+4+self.nInt_Spline-1]:###need to revise
            every_line=line.strip().split()
            #print(every_line)
            j=0
            for sub_list in every_line:
                #print(sub_list)
                self.repulsive_c[jj,j]=np.array(sub_list)
                j+=1
            #print(self.repulsive_c[jj,:])
            jj+=1
        #print("the repulsive parameter: \n",self.repulsive_c)
        return self.repulsive_c
    
    def get_electron(self):
        num_atom=rGI.get_line_coordinate(general)[1]
        atom_name_all=rGI.get_coor()[3]
        uHubb=np.zeros((num_atom))
        qAtom=np.zeros((num_atom))
        num_electrons=0
        for i in range(0,num_atom):
            #print(i,num_atom)
            qAtom[i]=valence_electrons[atom_name_all[i]]
            for j in range(0,i+1):
                qAtom[j]=valence_electrons[atom_name_all[j]]
                skf_name = atom_name_all[i]+"-"+atom_name_all[j]+".skf"
                skself=self.get_grid(skf_name)[2]
                if i == j:
                    uHubb[i]=skself[6]   #check,uHubb is skself[6]
                    num_electrons += qAtom[i]
        #print("uHubb",uHubb,"qAtom in get_electron",qAtom)
        return uHubb, qAtom, num_electrons

    def SK_trans(self):
        atom_ind=rGI.get_coor()[2]
        atom_name_all=rGI.get_coor()[3]
        num_atom=rGI.get_line_coordinate(general)[1]
        distance=rGI.get_coor_vector()[0]
        distance_norm=rGI.get_coor_vector()[1]
        #print(distance,len(distance_norm))
        mn=0
        atom_ind2=int(atom_ind[num_atom]*(atom_ind[num_atom]+1)/2)
        hamMat=np.zeros((atom_ind2))
        overMat=np.zeros((atom_ind2))
        for i in range(0,num_atom):
            #print(i,num_atom)
            lmaxi=valence_orbitals[atom_name_all[i]]
            for j in range(0,i+1):
                H_SK_matrix=np.zeros((9,9))
                S_SK_matrix=np.zeros((9,9))
                lmaxj=valence_orbitals[atom_name_all[j]]
                #if j =1,i,python will ignore distance[0,0]
                #print(i,j)
                skf_name = atom_name_all[i]+"-"+atom_name_all[j]+".skf"
                #print(atom_name_all,skf_name)
                self.get_HS_from_skf(skf_name)
                self.get_line_Spline(skf_name)
                self.get_grid(skf_name)
                skself=self.get_grid(skf_name)[2]
                r_HS_cut = (self.line_Spline-4)*self.gridDist
                d=distance[i,j]
                xyz=distance_norm[i,j,:]
                if d/0.529177249 > r_HS_cut:
                    print("ERROR,",i,"and",j,"distance > cutoff distance")
                    break
                elif d/0.529177249 < 1E-4:
                    if i != j:
                        print("ERROR, distance between",i,"and",j,"is 0.0")
                        break
                    elif lmaxi==1:
                        H_SK_matrix[0,0]=skself[2]
                        S_SK_matrix[0,0]=1.0
                    elif lmaxi==2:
                        H_SK_matrix[0,0]=skself[2]
                        S_SK_matrix[0,0]=1.0
                        H_SK_matrix[1,1]=skself[1]
                        S_SK_matrix[1,1]=1.0
                        H_SK_matrix[2,2]=skself[1]
                        S_SK_matrix[2,2]=1.0
                        H_SK_matrix[3,3]=skself[1]
                        S_SK_matrix[3,3]=1.0
                    else:
                        H_SK_matrix[0,0]=skself[2]
                        S_SK_matrix[0,0]=1.0
                        H_SK_matrix[1,1]=skself[1]
                        S_SK_matrix[1,1]=1.0
                        H_SK_matrix[2,2]=skself[1]
                        S_SK_matrix[2,2]=1.0
                        H_SK_matrix[3,3]=skself[1]
                        S_SK_matrix[3,3]=1.0
                        H_SK_matrix[4,4]=skself[0]
                        S_SK_matrix[4,4]=1.0
                        H_SK_matrix[5,5]=skself[0]
                        S_SK_matrix[5,5]=1.0
                        H_SK_matrix[6,6]=skself[0]
                        S_SK_matrix[6,6]=1.0
                        H_SK_matrix[7,7]=skself[0]
                        S_SK_matrix[7,7]=1.0
                        H_SK_matrix[8,8]=skself[0]
                        S_SK_matrix[8,8]=1.0
                else:
                    H_SK_matrix=self.shpar(i,j,d,xyz)[0]
                    S_SK_matrix=self.shpar(i,j,d,xyz)[1]
                #print(i,j,"\n H\n",H_SK_matrix,"S\n",S_SK_matrix)
                #print(H_SK_matrix)
                #if lamx of i and j is equal,then the matrix will be[n*n] matrix
                #else, will be [m*n] matrix, corresponding to the following two methods
                for n in range(0,int(atom_ind[j+1]-atom_ind[j])):
                    nn=atom_ind[j]+n
                    for m in range(0,int(atom_ind[i+1]-atom_ind[i])):
                        mm=atom_ind[i]+m
                        idx = int(mm*(mm+1)/2+nn)
                        #print("idx,mm,nn,m,n",idx,mm,nn,m,n)
                        #hamMat[idx] = H_SK_matrix[m,n]
                        #overMat[idx] = S_SK_matrix[m,n]
                        if nn <= mm:
                            idx = int(mm*(mm+1)/2+nn)
                            hamMat[idx] = H_SK_matrix[m,n]
                            overMat[idx] = S_SK_matrix[m,n]
                            #print("n,m,nn,mm,idx,i,j",n,m,nn,mm,idx,"hamMat[idx]",hamMat[idx],i,j)
                
        #print("hamMat[15]\n",hamMat[15],"H_SK_matrix[0,1]\n",H_SK_matrix[0,1])
        return hamMat,overMat
    
    def skss(self,i,j,d,x,y,z,HS_data):
        H_SK_matrix=np.zeros((9,9))
        S_SK_matrix=np.zeros((9,9))
        H_SK_matrix[0,0]=self.HS_s_s(i,j,x,y,z,HS_data)[0]
        S_SK_matrix[0,0]=self.HS_s_s(i,j,x,y,z,HS_data)[1]
        return H_SK_matrix,S_SK_matrix
    def sksp(self,i,j,d,x,y,z,HS_data):
        H_SK_matrix=self.skss(i,j,d,x,y,z,HS_data)[0]
        S_SK_matrix=self.skss(i,j,d,x,y,z,HS_data)[1]
        H_SK_matrix[0,1]=self.HS_s_x(i,j,x,y,z,HS_data)[0]
        H_SK_matrix[0,2]=self.HS_s_y(i,j,x,y,z,HS_data)[0]
        H_SK_matrix[0,3]=self.HS_s_z(i,j,x,y,z,HS_data)[0]
        S_SK_matrix[0,1]=self.HS_s_x(i,j,x,y,z,HS_data)[1]
        S_SK_matrix[0,2]=self.HS_s_y(i,j,x,y,z,HS_data)[1]
        S_SK_matrix[0,3]=self.HS_s_z(i,j,x,y,z,HS_data)[1]
        for ii in range(1,3+1):
            H_SK_matrix[ii,0]=-H_SK_matrix[0,ii]
            S_SK_matrix[ii,0]=-S_SK_matrix[0,ii]
        return H_SK_matrix,S_SK_matrix
    def skpp(self,i,j,d,x,y,z,HS_data):
        H_SK_matrix=self.sksp(i,j,d,x,y,z,HS_data)[0]
        S_SK_matrix=self.sksp(i,j,d,x,y,z,HS_data)[1]
        H_SK_matrix[1,1]=self.HS_x_x(i,j,x,y,z,HS_data)[0]
        H_SK_matrix[1,2]=self.HS_x_y(i,j,x,y,z,HS_data)[0]
        H_SK_matrix[1,3]=self.HS_x_z(i,j,x,y,z,HS_data)[0]
        H_SK_matrix[2,2]=self.HS_y_y(i,j,x,y,z,HS_data)[0]
        H_SK_matrix[2,3]=self.HS_y_z(i,j,x,y,z,HS_data)[0]
        H_SK_matrix[3,3]=self.HS_z_z(i,j,x,y,z,HS_data)[0]
        S_SK_matrix[1,1]=self.HS_x_x(i,j,x,y,z,HS_data)[1]
        S_SK_matrix[1,2]=self.HS_x_y(i,j,x,y,z,HS_data)[1]
        S_SK_matrix[1,3]=self.HS_x_z(i,j,x,y,z,HS_data)[1]
        S_SK_matrix[2,2]=self.HS_y_y(i,j,x,y,z,HS_data)[1]
        S_SK_matrix[2,3]=self.HS_y_z(i,j,x,y,z,HS_data)[1]
        S_SK_matrix[3,3]=self.HS_z_z(i,j,x,y,z,HS_data)[1]
        for ii in range(1,3+1):
            for jj in range(1,ii+1):
                H_SK_matrix[ii,jj]=H_SK_matrix[jj,ii]
                S_SK_matrix[ii,jj]=S_SK_matrix[jj,ii]
        return H_SK_matrix,S_SK_matrix
    def sksd(self,i,j,d,x,y,z,HS_data):
        H_SK_matrix=self.sksp(i,j,d,x,y,z,HS_data)[0]
        S_SK_matrix=self.sksp(i,j,d,x,y,z,HS_data)[1]
        H_SK_matrix[0,4]=self.HS_s_xy(i,j,x,y,z,HS_data)[0]
        H_SK_matrix[0,5]=self.HS_s_yz(i,j,x,y,z,HS_data)[0]
        H_SK_matrix[0,6]=self.HS_s_xz(i,j,x,y,z,HS_data)[0]
        H_SK_matrix[0,7]=self.HS_s_x2y2(i,j,x,y,z,HS_data)[0]
        H_SK_matrix[0,8]=self.HS_s_3z2r2(i,j,x,y,z,HS_data)[0]
        S_SK_matrix[0,4]=self.HS_s_xy(i,j,x,y,z,HS_data)[1]
        S_SK_matrix[0,5]=self.HS_s_yz(i,j,x,y,z,HS_data)[1]
        S_SK_matrix[0,6]=self.HS_s_xz(i,j,x,y,z,HS_data)[1]
        S_SK_matrix[0,7]=self.HS_s_x2y2(i,j,x,y,z,HS_data)[1]
        S_SK_matrix[0,8]=self.HS_s_3z2r2(i,j,x,y,z,HS_data)[1]
        for ii in range(4,8+1):
            H_SK_matrix[ii,0]=H_SK_matrix[0,ii]
            S_SK_matrix[ii,0]=S_SK_matrix[0,ii]
        return H_SK_matrix,S_SK_matrix
    def skpd(self,i,j,d,x,y,z,HS_data):
        H_SK_matrix=self.skpp(i,j,d,x,y,z,HS_data)[0]
        S_SK_matrix=self.skpp(i,j,d,x,y,z,HS_data)[1]
        H_SK_matrix[1,4]=self.HS_x_xy(i,j,x,y,z,HS_data)[0]
        H_SK_matrix[1,5]=self.HS_x_yz(i,j,x,y,z,HS_data)[0]
        H_SK_matrix[1,6]=self.HS_x_xz(i,j,x,y,z,HS_data)[0]
        H_SK_matrix[1,7]=self.HS_x_x2y2(i,j,x,y,z,HS_data)[0]
        H_SK_matrix[1,8]=self.HS_x_3z2r2(i,j,x,y,z,HS_data)[0]
        H_SK_matrix[2,4]=self.HS_y_xy(i,j,x,y,z,HS_data)[0]
        H_SK_matrix[2,5]=self.HS_y_yz(i,j,x,y,z,HS_data)[0]
        H_SK_matrix[2,6]=self.HS_y_xz(i,j,x,y,z,HS_data)[0]
        H_SK_matrix[2,7]=self.HS_y_x2y2(i,j,x,y,z,HS_data)[0]
        H_SK_matrix[2,8]=self.HS_y_3z2r2(i,j,x,y,z,HS_data)[0]
        H_SK_matrix[3,4]=self.HS_z_xy(i,j,x,y,z,HS_data)[0]
        H_SK_matrix[3,5]=self.HS_z_yz(i,j,x,y,z,HS_data)[0]
        H_SK_matrix[3,6]=self.HS_z_xz(i,j,x,y,z,HS_data)[0]
        H_SK_matrix[3,7]=self.HS_z_x2y2(i,j,x,y,z,HS_data)[0]
        H_SK_matrix[3,8]=self.HS_z_3z2r2(i,j,x,y,z,HS_data)[0]
        S_SK_matrix[1,4]=self.HS_x_xy(i,j,x,y,z,HS_data)[1]
        S_SK_matrix[1,5]=self.HS_x_yz(i,j,x,y,z,HS_data)[1]
        S_SK_matrix[1,6]=self.HS_x_xz(i,j,x,y,z,HS_data)[1]
        S_SK_matrix[1,7]=self.HS_x_x2y2(i,j,x,y,z,HS_data)[1]
        S_SK_matrix[1,8]=self.HS_x_3z2r2(i,j,x,y,z,HS_data)[1]
        S_SK_matrix[2,4]=self.HS_y_xy(i,j,x,y,z,HS_data)[1]
        S_SK_matrix[2,5]=self.HS_y_yz(i,j,x,y,z,HS_data)[1]
        S_SK_matrix[2,6]=self.HS_y_xz(i,j,x,y,z,HS_data)[1]
        S_SK_matrix[2,7]=self.HS_y_x2y2(i,j,x,y,z,HS_data)[1]
        S_SK_matrix[2,8]=self.HS_y_3z2r2(i,j,x,y,z,HS_data)[1]            
        S_SK_matrix[3,4]=self.HS_z_xy(i,j,x,y,z,HS_data)[1]
        S_SK_matrix[3,5]=self.HS_z_yz(i,j,x,y,z,HS_data)[1]
        S_SK_matrix[3,6]=self.HS_z_xz(i,j,x,y,z,HS_data)[1]
        S_SK_matrix[3,7]=self.HS_z_x2y2(i,j,x,y,z,HS_data)[1]
        S_SK_matrix[3,8]=self.HS_z_3z2r2(i,j,x,y,z,HS_data)[1]
        for ii in range(1,3+1):
            for jj in range(4,8+1):
                H_SK_matrix[jj,ii]=-H_SK_matrix[ii,jj]
                S_SK_matrix[jj,ii]=-S_SK_matrix[ii,jj]
        return H_SK_matrix,S_SK_matrix
    def skdd(self,i,j,d,x,y,z,HS_data):
        H_SK_matrix=self.skpd(i,j,d,x,y,z,HS_data)[0]
        S_SK_matrix=self.skpd(i,j,d,x,y,z,HS_data)[1]
        H_SK_matrix[4,4]=self.HS_xy_xy(i,j,x,y,z,HS_data)[0]
        H_SK_matrix[4,5]=self.HS_xy_yz(i,j,x,y,z,HS_data)[0]
        H_SK_matrix[4,6]=self.HS_xy_xz(i,j,x,y,z,HS_data)[0]
        H_SK_matrix[4,7]=self.HS_xy_x2y2(i,j,x,y,z,HS_data)[0]
        H_SK_matrix[4,8]=self.HS_xy_3z2r2(i,j,x,y,z,HS_data)[0]
        H_SK_matrix[5,5]=self.HS_yz_yz(i,j,x,y,z,HS_data)[0]
        H_SK_matrix[5,6]=self.HS_yz_xz(i,j,x,y,z,HS_data)[0]
        H_SK_matrix[5,7]=self.HS_yz_x2y2(i,j,x,y,z,HS_data)[0]
        H_SK_matrix[5,8]=self.HS_yz_3z2r2(i,j,x,y,z,HS_data)[0]
        H_SK_matrix[6,6]=self.HS_xz_xz(i,j,x,y,z,HS_data)[0]
        H_SK_matrix[6,7]=self.HS_xz_x2y2(i,j,x,y,z,HS_data)[0]
        H_SK_matrix[6,8]=self.HS_xz_3z2r2(i,j,x,y,z,HS_data)[0]
        H_SK_matrix[7,7]=self.HS_x2y2_x2y2(i,j,x,y,z,HS_data)[0]
        H_SK_matrix[7,8]=self.HS_x2y2_3z2r2(i,j,x,y,z,HS_data)[0]
        H_SK_matrix[8,8]=self.HS_3z2r2_3z2r2(i,j,x,y,z,HS_data)[0]
        S_SK_matrix[4,4]=self.HS_xy_xy(i,j,x,y,z,HS_data)[1]
        S_SK_matrix[4,5]=self.HS_xy_yz(i,j,x,y,z,HS_data)[1]
        S_SK_matrix[4,6]=self.HS_xy_xz(i,j,x,y,z,HS_data)[1]
        S_SK_matrix[4,7]=self.HS_xy_x2y2(i,j,x,y,z,HS_data)[1]
        S_SK_matrix[4,8]=self.HS_xy_3z2r2(i,j,x,y,z,HS_data)[1]
        S_SK_matrix[5,5]=self.HS_yz_yz(i,j,x,y,z,HS_data)[1]
        S_SK_matrix[5,6]=self.HS_yz_xz(i,j,x,y,z,HS_data)[1]
        S_SK_matrix[5,7]=self.HS_yz_x2y2(i,j,x,y,z,HS_data)[1]
        S_SK_matrix[5,8]=self.HS_yz_3z2r2(i,j,x,y,z,HS_data)[1]
        S_SK_matrix[6,6]=self.HS_xz_xz(i,j,x,y,z,HS_data)[1]
        S_SK_matrix[6,7]=self.HS_xz_x2y2(i,j,x,y,z,HS_data)[1]
        S_SK_matrix[6,8]=self.HS_xz_3z2r2(i,j,x,y,z,HS_data)[1]
        S_SK_matrix[7,7]=self.HS_x2y2_x2y2(i,j,x,y,z,HS_data)[1]
        S_SK_matrix[7,8]=self.HS_x2y2_3z2r2(i,j,x,y,z,HS_data)[1]
        S_SK_matrix[8,8]=self.HS_3z2r2_3z2r2(i,j,x,y,z,HS_data)[1]
        for ii in range(4,8+1):
            for jj in range(4,ii+1):
                H_SK_matrix[ii,jj]=-H_SK_matrix[jj,ii]
                S_SK_matrix[ii,jj]=-S_SK_matrix[jj,ii]
        return H_SK_matrix,S_SK_matrix
    
    def shpar(self,i,j,d,xyz):
        HS_data=self.data_HSlist[np.int(np.round(d/0.529177249/self.gridDist)),:]
        #print(d,np.round(d/0.529177249/self.gridDist),'\n',HS_data)
        x=xyz[0]
        y=xyz[1]
        z=xyz[2]
        BOHR = 0.529177249
        atom_name_all=rGI.get_coor()[3]
        lmaxi=valence_orbitals[atom_name_all[i]]
        lmaxj=valence_orbitals[atom_name_all[j]]
        maxmax=max(lmaxi,lmaxj)
        minmax=min(lmaxi,lmaxj)
        if maxmax==1:
            H_SK_matrix=self.skss(i,j,d,x,y,z,HS_data)[0]
            S_SK_matrix=self.skss(i,j,d,x,y,z,HS_data)[1]
        elif maxmax==2 and minmax==1:
            H_SK_matrix=self.sksp(i,j,d,x,y,z,HS_data)[0]
            S_SK_matrix=self.sksp(i,j,d,x,y,z,HS_data)[1]
        elif maxmax==2 and minmax==2:
            H_SK_matrix=self.skpp(i,j,d,x,y,z,HS_data)[0]
            S_SK_matrix=self.skpp(i,j,d,x,y,z,HS_data)[1]
        elif maxmax==3 and minmax==1:
            H_SK_matrix=self.sksd(i,j,d,x,y,z,HS_data)[0]
            S_SK_matrix=self.sksd(i,j,d,x,y,z,HS_data)[1]
        elif maxmax==3 and minmax==2:
            H_SK_matrix=self.skpd(i,j,d,x,y,z,HS_data)[0]
            S_SK_matrix=self.skpd(i,j,d,x,y,z,HS_data)[1]
        elif maxmax==3 and minmax==3:
            H_SK_matrix=self.skdd(i,j,d,x,y,z,HS_data)[0]
            S_SK_matrix=self.skdd(i,j,d,x,y,z,HS_data)[1]
        return H_SK_matrix,S_SK_matrix
                
    def SCF(self):
        atom_ind=rGI.get_coor()[2]  #equal to sys%nOrbs
        maxiter=rGI.get_maxiter(general)
        scf=rGI.get_scf(general)
        num_atom=rGI.get_line_coordinate(general)[1]
        uHubb=self.get_electron()[0]
        qAtom=self.get_electron()[1]
        qZero=self.get_electron()[1]
        nElect=self.get_electron()[2]
        hamMat=self.SK_trans()[0]
        overMat=self.SK_trans()[1]
        atom_ind2=int(atom_ind[num_atom]*(atom_ind[num_atom]+1)/2)
        eigVec = np.zeros((int(atom_ind[num_atom]),int(atom_ind[num_atom])))
        #eigVec=self.SK_trans()[0]
        #eigVal=np.zeros((9))
        #oldOverMat=self.SK_trans()[1]
        eigVal = np.zeros((int(atom_ind[num_atom])))
        gMat=self.gMatrix(uHubb)
        print("gMat",gMat)
        fockMat = np.zeros((atom_ind2))
        occ = np.zeros((int(atom_ind[num_atom])))
        tElec=general["tElec"]
        oldEnergy=0.0
        oldOverMat = np.zeros((int(atom_ind[num_atom]),int(atom_ind[num_atom])))
        print("hamMat",hamMat,"overMat",overMat)
        for niter in range(0,maxiter):
            print("\n \n niter",niter,"\n")
            if scf=="T":
                oldQAt = qAtom
            #print("oldQAt",oldQAt)
            for i in range(0,atom_ind2):
                fockMat[i]=hamMat[i]
            #print("fockMat[i]=hamMat[i]",fockMat)
            
            shift=self.shiftHamGamma(num_atom,qAtom,qZero,gMat)
            #print("sys%shift",shift)
            for i in range(0,num_atom):
                for ii in range(int(atom_ind[i]), int(atom_ind[i+1])):
                    occ[ii] = shift[i]
                    #print("occ[ii]",occ[ii])
            k=0
            for i in range(0,int(atom_ind[num_atom])):
                for j in range(0,i+1):
                    fockMat[k]=fockMat[k]+0.5*overMat[k]*(occ[i]+ occ[j])
                    #here, using shiftHamGamma to reduce N*N*N circulations to N*N+N
                    #occ[i]=sum of delta_q(k)*gamma(ik)
                    k += 1
            #print("fockMat\n",fockMat,"occ\n",occ,"overMat\n",overMat)
            k=0
            for i in range(0,int(atom_ind[num_atom])):
                for j in range(0,i+1):
                    eigVec[j,i]=fockMat[k]
                    oldOverMat[j,i]=overMat[k]
                    #print("eigVec",eigVec[j,i],"k",k,"i",i,"j",j)
                    #print("oldOverMat",oldOverMat[j,i])
                    k += 1
                #tansfer 1D to 2D
            #print("eigVec before eigh\n",eigVec,"\n oldOverMat before eigh \n",oldOverMat)
            eigVal,eigVec=eigh(eigVec,oldOverMat,lower=False,overwrite_a=True,overwrite_b=True)  #,eigvals_only=False
            #print("eigVec after eigh\n",eigVec,"\n oldOverMat after eigh \n",oldOverMat,"eigVal\n",eigVal)
            #A=np.array([[-0.2386004,-8.385192380750001E-002],[6.919854834957075E-310,-0.2386004]])
            #B=np.array([[1.00,0.1324605728097],[6.919854834957075E-310,1.0]])
            #C,D=eigh(A,B)
            #print("eigVec after\n",eigVec)
            occ = self.fermi(tElec,nElect,int(atom_ind[num_atom]),eigVal,occ)[0]
            print("occ after fermi",occ)
            #sum of occupied eigenvalues
            energy=0
            for i in range(0, int(atom_ind[num_atom])):
                if occ[i] > 8E-16:
                    energy = energy + occ[i]*eigVal[i]
                    print("energy",energy,"occ[i]",occ[i],"eigVal[i]",eigVal[i])
                else:
                    break
            #print("int(atom_ind[num_atom])",int(atom_ind[num_atom]))
            #print(occ)
            
            for n in range(0, int(atom_ind[num_atom])):
                if occ[n] > 8E-16:
                    #print(occ[i],"n",n,int(atom_ind[num_atom]))
                    n +=1  #if only occ[0]!=0, n should be 0+1 for the follwing cycle
                else:
                    break
            #ii = min(i+5,int(atom_ind[num_atom]))
            #jj = max(i-6,1)
        
            #calculate density matrix
            denMat=np.zeros((atom_ind2))
            work=[]
            for i in range(0, n+1):  #n = no. of occupied orbitals
                work.append(np.sqrt(occ[i]))
            for j in range(0, n+1):  #n = no. of occupied orbitals
                for i in range(0, int(atom_ind[num_atom])):
                    #print("eigVec[i,j]",eigVec[i,j],"i,j",i,j, work[j])
                    eigVec[i,j] = eigVec[i,j] * work[j]
            for j in range(n+1,int(atom_ind[num_atom])): 
                eigVec[:,j] = 0.0
                    #print("eigVec[i,j]",eigVec[i,j],"i,j",i,j,"n",n,"work[j]",work[j])
            #print("eigVec[i,j] befoere dgemm\n",eigVec)
            # realize dgemm function
            #('n', 't',nOrbs,nOrbs,n,1,eigVec,nOrbs,eigVec,nOrbs,0,oldOverMat,nOrbs)
            #oldOverMat=np.dot(eigVec,eigVec.T)
            oldOverMat=scipy.linalg.blas.dgemm(alpha=1.0,a=eigVec,b=eigVec,beta=0.0,trans_b=1)
            #print("oldOverMat after dgemm\n",oldOverMat,"eigVec\n",eigVec)
            for i in range(0, int(atom_ind[num_atom])):
                for j in range(0, i+1):
                    m=int(i*(i+1)/2+j)
                    denMat[m]=denMat[m]+oldOverMat[j,i]
                    #print("denMat[m]",denMat[m],i,j,"m",m)

            #calculate mulliken charges
            if scf:
                #mulliken(sys%nAt, sys%nOrbs, sys%ind, overMat, denMat, sys%qAt)
                qAtom=self.mulliken(num_atom,int(atom_ind[num_atom]),atom_ind,overMat,denMat,qAtom)
                print(qAtom,"qAtom after mulliken")
            
            #calculate electronic energy
            eCoul = 0.0
            for i in range(0, num_atom):
                eCoul=eCoul+shift[i]*(qAtom[i]+qZero[i])
            energy = energy - 0.5*eCoul
            print("energy=energy-0.5eCoul",energy,"\n eCoul",eCoul)
            ######check convergence, broyden mixing
            #write restart data
            #a,b=np.linalg.eig(oldOverMat)
            
            if abs(oldEnergy-energy)<1E-4:
                print("convergence energy:",energy,"niter",niter)
                break
            oldEnergy=energy
            for i in range(0,num_atom):
                qAtom[i] = oldQAt[i]
    
    def HS_s_s(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        self.atomi_HS_type=rGI.get_coor()[0][0,atomi_HS_type]
        self.atomj_HS_type=rGI.get_coor()[0][0,atomj_HS_type]  #type of atom, for C, is 6
        #print(HS_data[10],x,y,z)
        return HS_data[10],HS_data[20]
    
    def HS_s_x(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return x*HS_data[9],x*HS_data[19]

    def HS_s_y(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return y*HS_data[9],y*HS_data[19]

    def HS_s_z(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return z*HS_data[9],z*HS_data[19]

    def HS_s_xy(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return np.sqrt(3)*x*y*HS_data[8],np.sqrt(3)*x*y*HS_data[18]

    def HS_s_yz(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return np.sqrt(3)*y*z*HS_data[8],np.sqrt(3)*y*z*HS_data[18]

    def HS_s_xz(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return np.sqrt(3)*x*z*HS_data[8],np.sqrt(3)*x*z*HS_data[18]

    def HS_s_x2y2(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return 0.5*np.sqrt(3)*(x**2-y**2)*HS_data[8],0.5*np.sqrt(3)*(x**2-y**2)*HS_data[18]

    def HS_s_3z2r2(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return (z**2-0.5*(x**2+y**2))*HS_data[8],(z**2-0.5*(x**2+y**2))*HS_data[18]

    def HS_x_s(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return (self.HS_s_x(atomi_HS_type,atomj_HS_type,-x,-y,-z,HS_data)[0],
                self.HS_s_x(atomi_HS_type,atomj_HS_type,-x,-y,-z,HS_data)[1])

    def HS_x_x(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return x**2*HS_data[6]+(1-x**2)*HS_data[7],x**2*HS_data[16]+(1-x**2)*HS_data[17]

    def HS_x_y(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return x*y*HS_data[6]-x*y*HS_data[7],x*y*HS_data[16]-x*y*HS_data[17]

    def HS_x_z(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return x*z*HS_data[6]-x*z*HS_data[7],x*z*HS_data[16]-x*z*HS_data[17]

    def HS_x_xy(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return (np.sqrt(3)*x**2*y*HS_data[4]+y*(1-2*x**2)*HS_data[5],
                np.sqrt(3)*x**2*y*HS_data[14]+y*(1-2*x**2)*HS_data[15])

    def HS_x_yz(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return (np.sqrt(3)*x*y*z*HS_data[4]-2*x*y*z*HS_data[5],
                np.sqrt(3)*x*y*z*HS_data[14]-2*x*y*z*HS_data[15])

    def HS_x_xz(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return (np.sqrt(3)*x**2*z*HS_data[4]+z*(1-2*x**2)*HS_data[5],
                np.sqrt(3)*x**2*z*HS_data[14]+z*(1-2*x**2)*HS_data[15])

    def HS_x_x2y2(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return (np.sqrt(3)/2*x*(x**2-y**2)*HS_data[4]+x*(1-x**2+y**2)*HS_data[5],
                np.sqrt(3)/2*x*(x**2-y**2)*HS_data[14]+x*(1-x**2+y**2)*HS_data[15])

    def HS_x_3z2r2(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return (x*(z**2-0.5*(x**2+y**2))*HS_data[4]-np.sqrt(3)*x*z**2*HS_data[5],
                x*(z**2-0.5*(x**2+y**2))*HS_data[14]-np.sqrt(3)*x*z**2*HS_data[15])

    def HS_y_s(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return (self.HS_s_y(atomi_HS_type,atomj_HS_type,-x,-y,-z,HS_data)[0],
                self.HS_s_y(atomi_HS_type,atomj_HS_type,-x,-y,-z,HS_data)[1])
    
    def HS_y_x(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return (self.HS_x_y(atomi_HS_type,atomj_HS_type,-x,-y,-z,HS_data)[0],
                self.HS_x_y(atomi_HS_type,atomj_HS_type,-x,-y,-z,HS_data)[1])
    
    def HS_y_y(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return y**2*HS_data[6]+(1-y**2)*HS_data[7],y**2*HS_data[16]+(1-y**2)*HS_data[17]
    
    def HS_y_z(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return y*z*HS_data[6]-y*z*HS_data[7],y*z*HS_data[16]-y*z*HS_data[17]

    def HS_y_xy(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return (np.sqrt(3)*y**2*x*HS_data[4]+x*(1-2*y**2)*HS_data[5],
                np.sqrt(3)*y**2*x*HS_data[14]+x*(1-2*y**2)*HS_data[15])

    def HS_y_yz(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return (np.sqrt(3)*y**2*z*HS_data[4]-z*(1-2*y**2)*HS_data[5],
                np.sqrt(3)*y**2*z*HS_data[14]-z*(1-2*y**2)*HS_data[15])

    def HS_y_xz(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return (np.sqrt(3)*x*y*z*HS_data[4]-2*x*y*z*HS_data[5],
                np.sqrt(3)*x*y*z*HS_data[14]-2*x*y*z*HS_data[15])

    def HS_y_x2y2(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return (np.sqrt(3)/2*y*(x**2-y**2)*HS_data[4]-y*(1+x**2-y**2)*HS_data[5],
                np.sqrt(3)/2*y*(x**2-y**2)*HS_data[14]-y*(1+x**2-y**2)*HS_data[15])
    
    def HS_y_3z2r2(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return (y*(z**2-0.5*(x**2+y**2))*HS_data[4]-np.sqrt(3)*y*z**2*HS_data[5],
                y*(z**2-0.5*(x**2+y**2))*HS_data[14]-np.sqrt(3)*y*z**2*HS_data[15])

    def HS_z_s(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return (self.HS_s_z(atomi_HS_type,atomj_HS_type,-x,-y,-z,HS_data)[0],
                self.HS_s_z(atomi_HS_type,atomj_HS_type,-x,-y,-z,HS_data)[1])
    
    def HS_z_x(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return (self.HS_x_z(atomi_HS_type,atomj_HS_type,-x,-y,-z,HS_data)[0],
                self.HS_x_z(atomi_HS_type,atomj_HS_type,-x,-y,-z,HS_data)[1])

    def HS_z_y(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return (self.HS_y_z(atomi_HS_type,atomj_HS_type,-x,-y,-z,HS_data)[0],
                self.HS_y_z(atomi_HS_type,atomj_HS_type,-x,-y,-z,HS_data)[1])
    
    def HS_z_z(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return z**2*HS_data[6]+(1-z**2)*HS_data[7],z**2*HS_data[16]+(1-z**2)*HS_data[17]
    
    def HS_z_xy(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return (np.sqrt(3)*x*y*z*HS_data[4]-2*x*y*z*HS_data[5],np.sqrt(3)*x*y*z*HS_data[14]-2*x*y*z*HS_data[15])
    
    def HS_z_yz(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return np.sqrt(3)*z**2*y*HS_data[4]-y*(1-2*z**2)*HS_data[5],np.sqrt(3)*z**2*y*HS_data[14]-y*(1-2*z**2)*HS_data[15]
    
    def HS_z_xz(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return np.sqrt(3)*z**2*x*HS_data[4]-x*(1-2*z**2)*HS_data[5],np.sqrt(3)*z**2*x*HS_data[14]-x*(1-2*z**2)*HS_data[15]

    def HS_z_x2y2(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return (np.sqrt(3)/2*z*(x**2-y**2)*HS_data[4]-z*(x**2-y**2)*HS_data[5],
                np.sqrt(3)/2*z*(x**2-y**2)*HS_data[14]-z*(x**2-y**2)*HS_data[15])
    
    def HS_z_3z2r2(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return (z*(z**2-0.5*(x**2+y**2))*HS_data[4]+np.sqrt(3)*z*(x**2+y**2)*HS_data[5],
                z*(z**2-0.5*(x**2+y**2))*HS_data[14]+np.sqrt(3)*z*(x**2+y**2)*HS_data[15])

    def HS_xy_s(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return (self.HS_s_xy(atomi_HS_type,atomj_HS_type,-x,-y,-z,HS_data)[0],
                self.HS_s_xy(atomi_HS_type,atomj_HS_type,-x,-y,-z,HS_data)[1])
    
    def HS_xy_x(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return (self.HS_x_xy(atomi_HS_type,atomj_HS_type,-x,-y,-z,HS_data)[0],
                self.HS_x_xy(atomi_HS_type,atomj_HS_type,-x,-y,-z,HS_data)[1])

    def HS_xy_y(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return (self.HS_y_xy(atomi_HS_type,atomj_HS_type,-x,-y,-z,HS_data)[0],
                self.HS_y_xy(atomi_HS_type,atomj_HS_type,-x,-y,-z,HS_data)[1])
    
    def HS_xy_z(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return (self.HS_z_xy(atomi_HS_type,atomj_HS_type,-x,-y,-z,HS_data)[0],
                self.HS_z_xy(atomi_HS_type,atomj_HS_type,-x,-y,-z,HS_data)[1])
    
    def HS_xy_xy(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return (3*x**2*y**2*HS_data[1]+(x**2+y**2-4*x**2*y**2)*HS_data[2]+(z**2+x**2*y**2)*HS_data[3],
                3*x**2*y**2*HS_data[11]+(x**2+y**2-4*x**2*y**2)*HS_data[12]+(z**2+x**2*y**2)*HS_data[13])

    def HS_xy_yz(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return (3*x*y**2*z*HS_data[1]+x*z*(1-4*y**2)*HS_data[2]+x*z*(y**2-1)*HS_data[3],
                3*x*y**2*z*HS_data[11]+x*z*(1-4*y**2)*HS_data[12]+x*z*(y**2-1)*HS_data[13])

    def HS_xy_xz(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return (3*x**2*y*z*HS_data[1]+y*z*(1-4*x**2)*HS_data[2]+y*z*(x**2-1)*HS_data[3],
                3*x**2*y*z*HS_data[11]+y*z*(1-4*x**2)*HS_data[12]+y*z*(x**2-1)*HS_data[13])

    def HS_xy_x2y2(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return (1.5*x*y*(x**2-y**2)*HS_data[1]-2*x*y*(x**2-y**2)*HS_data[2]+0.5*x*y*(x**2-y**2)*HS_data[3],
                  1.5*x*y*(x**2-y**2)*HS_data[11]-2*x*y*(x**2-y**2)*HS_data[12]+0.5*x*y*(x**2-y**2)*HS_data[13])
    
    def HS_xy_3z2r2(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return (np.sqrt(3)*x*y*(z**2-0.5*(x**2+y**2))*HS_data[1]-2*np.sqrt(3)*x*y*z**2*HS_data[2]+np.sqrt(3)/2*x*y*(1+z**2)*HS_data[3],
                np.sqrt(3)*x*y*(z**2-0.5*(x**2+y**2))*HS_data[11]-2*np.sqrt(3)*x*y*z**2*HS_data[12]+np.sqrt(3)/2*x*y*(1+z**2)*HS_data[13])
    
    def HS_yz_s(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return (self.HS_s_yz(atomi_HS_type,atomj_HS_type,-x,-y,-z,HS_data)[0],
                self.HS_s_yz(atomi_HS_type,atomj_HS_type,-x,-y,-z,HS_data)[1])
    
    def HS_yz_x(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return (self.HS_x_yz(atomi_HS_type,atomj_HS_type,-x,-y,-z,HS_data)[0],
                self.HS_x_yz(atomi_HS_type,atomj_HS_type,-x,-y,-z,HS_data)[1])
    
    def HS_yz_y(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return (self.HS_y_yz(atomi_HS_type,atomj_HS_type,-x,-y,-z,HS_data)[0],
                self.HS_y_yz(atomi_HS_type,atomj_HS_type,-x,-y,-z,HS_data)[1])
    
    def HS_yz_z(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return (self.HS_z_yz(atomi_HS_type,atomj_HS_type,-x,-y,-z,HS_data)[0],
                self.HS_z_yz(atomi_HS_type,atomj_HS_type,-x,-y,-z,HS_data)[1])
    
    def HS_yz_xy(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return (self.HS_xy_yz(atomi_HS_type,atomj_HS_type,-x,-y,-z,HS_data)[0],
                self.HS_xy_yz(atomi_HS_type,atomj_HS_type,-x,-y,-z,HS_data)[1])

    def HS_yz_yz(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return (3*y**2*z**2*HS_data[1]+(y**2+z**2-4*y**2*z**2)*HS_data[2]+(x**2+y**2*z**2)*HS_data[3],
                3*y**2*z**2*HS_data[11]+(y**2+z**2-4*y**2*z**2)*HS_data[12]+(x**2+y**2*z**2)*HS_data[13])

    def HS_yz_xz(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return (3*x*z**2*y*HS_data[1]+x*y*(1-4*z**2)*HS_data[2]+x*y*(z**2-1)*HS_data[3],
                3*x*z**2*y*HS_data[11]+x*y*(1-4*z**2)*HS_data[12]+x*y*(z**2-1)*HS_data[13])
   
    def HS_yz_x2y2(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return (1.5*y*z*(x**2-y**2)*HS_data[1]-y*z*(1+2*(x**2-y**2))*HS_data[2]+y*z*(1+0.5*(x**2-y**2))*HS_data[3],
                1.5*y*z*(x**2-y**2)*HS_data[11]-y*z*(1+2*(x**2-y**2))*HS_data[12]+y*z*(1+0.5*(x**2-y**2))*HS_data[13])

    def HS_yz_3z2r2(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return (np.sqrt(3)*y*z*(z**2-0.5*(x**2+y**2))*HS_data[1]+np.sqrt(3)*y*z*(x**2+y**2-z**2)*HS_data[2]-np.sqrt(3)/2*y*z*(x**2+y**2)*HS_data[3],
                np.sqrt(3)*y*z*(z**2-0.5*(x**2+y**2))*HS_data[11]+np.sqrt(3)*y*z*(x**2+y**2-z**2)*HS_data[12]-np.sqrt(3)/2*y*z*(x**2+y**2)*HS_data[13])
    
    def HS_xz_s(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return (self.HS_s_xz(atomi_HS_type,atomj_HS_type,-x,-y,-z,HS_data)[0],
                self.HS_s_xz(atomi_HS_type,atomj_HS_type,-x,-y,-z,HS_data)[1])
    
    def HS_xz_x(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return (self.HS_x_xz(atomi_HS_type,atomj_HS_type,-x,-y,-z,HS_data)[0],
                self.HS_x_xz(atomi_HS_type,atomj_HS_type,-x,-y,-z,HS_data)[1])
    def HS_xz_y(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return (self.HS_y_xz(atomi_HS_type,atomj_HS_type,-x,-y,-z,HS_data)[0],
                self.HS_y_xz(atomi_HS_type,atomj_HS_type,-x,-y,-z,HS_data)[1])
    def HS_xz_z(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return (self.HS_z_xz(atomi_HS_type,atomj_HS_type,-x,-y,-z,HS_data)[0],
                self.HS_z_xz(atomi_HS_type,atomj_HS_type,-x,-y,-z,HS_data)[1])
    def HS_xz_xy(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return (self.HS_xy_xz(atomi_HS_type,atomj_HS_type,-x,-y,-z,HS_data)[0],
                self.HS_xy_xz(atomi_HS_type,atomj_HS_type,-x,-y,-z,HS_data)[1])
    def HS_xz_yz(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return (self.HS_yz_xz(atomi_HS_type,atomj_HS_type,-x,-y,-z,HS_data)[0],
                self.HS_yz_xz(atomi_HS_type,atomj_HS_type,-x,-y,-z,HS_data)[1])

    def HS_xz_xz(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return (3*x**2*z**2*HS_data[1]+(x**2+z**2-4*x**2*z**2)*HS_data[2]+(y**2+x**2*z**2)*HS_data[3],
                3*x**2*z**2*HS_data[11]+(x**2+z**2-4*x**2*z**2)*HS_data[12]+(y**2+x**2*z**2)*HS_data[13])
   
    def HS_xz_x2y2(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return (1.5*x*z*(x**2-y**2)*HS_data[1]+x*z*(1-2*(x**2-y**2))*HS_data[2]-x*z*(1-0.5*(x**2-y**2))*HS_data[3],
                1.5*x*z*(x**2-y**2)*HS_data[11]+x*z*(1-2*(x**2-y**2))*HS_data[12]-x*z*(1-0.5*(x**2-y**2))*HS_data[13])

    def HS_xz_3z2r2(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return (np.sqrt(3)*x*z*(z**2-0.5*(x**2+y**2))*HS_data[1]+np.sqrt(3)*x*z*(x**2+y**2-z**2)*HS_data[2]-np.sqrt(3)/2*x*z*(x**2+y**2)*HS_data[3],
                np.sqrt(3)*x*z*(z**2-0.5*(x**2+y**2))*HS_data[11]+np.sqrt(3)*x*z*(x**2+y**2-z**2)*HS_data[12]-np.sqrt(3)/2*x*z*(x**2+y**2)*HS_data[13])

    def HS_x2y2_s(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return (self.HS_s_x2y2(atomi_HS_type,atomj_HS_type,-x,-y,-z,HS_data)[0],
                self.HS_s_x2y2(atomi_HS_type,atomj_HS_type,-x,-y,-z,HS_data)[1])
    def HS_x2y2_x(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return (self.HS_x_x2y2(atomi_HS_type,atomj_HS_type,-x,-y,-z,HS_data)[0],
                self.HS_x_x2y2(atomi_HS_type,atomj_HS_type,-x,-y,-z,HS_data)[1])
    def HS_x2y2_y(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return (self.HS_y_x2y2(atomi_HS_type,atomj_HS_type,-x,-y,-z,HS_data)[0],
                self.HS_y_x2y2(atomi_HS_type,atomj_HS_type,-x,-y,-z,HS_data)[1])
    def HS_x2y2_z(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return (self.HS_z_x2y2(atomi_HS_type,atomj_HS_type,-x,-y,-z,HS_data)[0],
                self.HS_z_x2y2(atomi_HS_type,atomj_HS_type,-x,-y,-z,HS_data)[1])
    def HS_x2y2_xy(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return (self.HS_xy_x2y2(atomi_HS_type,atomj_HS_type,-x,-y,-z,HS_data)[0],
                self.HS_xy_x2y2(atomi_HS_type,atomj_HS_type,-x,-y,-z,HS_data)[1])
    def HS_x2y2_yz(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return (self.HS_yz_x2y2(atomi_HS_type,atomj_HS_type,-x,-y,-z,HS_data)[0],
                self.HS_yz_x2y2(atomi_HS_type,atomj_HS_type,-x,-y,-z,HS_data)[1])
    def HS_x2y2_xz(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return (self.HS_xz_x2y2(atomi_HS_type,atomj_HS_type,-x,-y,-z,HS_data)[0],
                self.HS_xz_x2y2(atomi_HS_type,atomj_HS_type,-x,-y,-z,HS_data)[1])
   
    def HS_x2y2_x2y2(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return (3/4*(x**2-y**2)**2*HS_data[1]+(x**2+y**2-(x**2-y**2)**2)*HS_data[2]+(z**2+1/4*(x**2-y**2)**2)*HS_data[3],
                3/4*(x**2-y**2)**2*HS_data[11]+(x**2+y**2-(x**2-y**2)**2)*HS_data[12]+(z**2+1/4*(x**2-y**2)**2)*HS_data[13])

    def HS_x2y2_3z2r2(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return (np.sqrt(3)/2*(x**2-y**2)*(z**2-(x**2+y**2)/2)*HS_data[1]+np.sqrt(3)*z**2*(x**2-y**2)*HS_data[2]+np.sqrt(3)/4*(1+z**2)*(x**2-y**2)*HS_data[3],
                np.sqrt(3)/2*(x**2-y**2)*(z**2-(x**2+y**2)/2)*HS_data[11]+np.sqrt(3)*z**2*(x**2-y**2)*HS_data[12]+np.sqrt(3)/4*(1+z**2)*(x**2-y**2)*HS_data[13])

    def HS_3z2r2_s(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return (self.HS_s_3z2r2(atomi_HS_type,atomj_HS_type,-x,-y,-z,HS_data)[0],
                self.HS_s_3z2r2(atomi_HS_type,atomj_HS_type,-x,-y,-z,HS_data)[1])
    def HS_3z2r2_x(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return (self.HS_x_3z2r2(atomi_HS_type,atomj_HS_type,-x,-y,-z,HS_data)[0],
                self.HS_x_3z2r2(atomi_HS_type,atomj_HS_type,-x,-y,-z,HS_data)[1])
    def HS_3z2r2_y(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return (self.HS_y_3z2r2(atomi_HS_type,atomj_HS_type,-x,-y,-z,HS_data)[0],
                self.HS_y_3z2r2(atomi_HS_type,atomj_HS_type,-x,-y,-z,HS_data)[1])
    def HS_3z2r2_z(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return (self.HS_z_3z2r2(atomi_HS_type,atomj_HS_type,-x,-y,-z,HS_data)[0],
                self.HS_z_3z2r2(atomi_HS_type,atomj_HS_type,-x,-y,-z,HS_data)[1])
    def HS_3z2r2_xy(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return (self.HS_xy_3z2r2(atomi_HS_type,atomj_HS_type,-x,-y,-z,HS_data)[0],
                self.HS_xy_3z2r2(atomi_HS_type,atomj_HS_type,-x,-y,-z,HS_data)[1])
    def HS_3z2r2_yz(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return (self.HS_yz_3z2r2(atomi_HS_type,atomj_HS_type,-x,-y,-z,HS_data)[0],
                self.HS_yz_3z2r2(atomi_HS_type,atomj_HS_type,-x,-y,-z,HS_data)[1])
    def HS_3z2r2_xz(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return (self.HS_xz_3z2r2(atomi_HS_type,atomj_HS_type,-x,-y,-z,HS_data)[0],
                self.HS_xz_3z2r2(atomi_HS_type,atomj_HS_type,-x,-y,-z,HS_data)[1])
    def HS_3z2r2_x2y2(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return (self.HS_x2y2_3z2r2(atomi_HS_type,atomj_HS_type,-x,-y,-z,HS_data)[0],
                self.HS_x2y2_3z2r2(atomi_HS_type,atomj_HS_type,-x,-y,-z,HS_data)[1])

    def HS_3z2r2_3z2r2(self,atomi_HS_type,atomj_HS_type,x,y,z,HS_data):
        return ((z**2-0.5*(x**2+y**2))**2*HS_data[1]+3*z**2*(x**2+y**2)*HS_data[2]+3/4*(x**2+y**2)**2*HS_data[3],
                (z**2-0.5*(x**2+y**2))**2*HS_data[11]+3*z**2*(x**2+y**2)*HS_data[12]+3/4*(x**2+y**2)**2*HS_data[13])

#readSkf().SK_trans()
#readGeneralInfo().get_coor_vector()
#readSkf().get_HS_from_skf()
#readSkf().get_Spline_para()
#readSkf().SK_trans()
readSkf2().SCF()
#readSkf().get_grid("Ti-Ti.skf")


