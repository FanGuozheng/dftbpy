#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 10:59:55 2019
@author: guozheng
"""
import numpy as np
import pyanitools as pya
import write_output as write
import os
from collections import Counter
import subprocess
import lattice_cell
import matplotlib.pyplot as plt
import dftbpy_benchmark as bench
import torch
from torch import nn
from torch.autograd import Variable
import sys
sys.path.append('/home/gz_fan/Documents/ML/dftb/ml/dftbtorch')
import dftbtorch.dftb_torch as dftb_torch
from dftbtorch.readt import ReadInt, ReadSKt
import dftbtorch.slakot as slakot

directory = '/home/gz_fan/Documents/ML/dftb/ml'
hdf5filelist = ['/home/gz_fan/Documents/ML/database/an1/ani_gdb_s01.h5']
dire = '/home/gz_fan/Documents/ML/dftb/slko'
tol_mlerr = 0.003
BOHR = 0.529177210903
grid0 = 0.4
gridmesh = 0.2
ATOMIND = {'H': 1, 'HH': 2, 'HC': 3, 'C': 4, 'CH': 5, 'CC': 6}
ATOMNUM = {'H': 1, 'C': 6}


def main(task, nfile=10):
    outpara = {}
    outpara['nfile'] = nfile
    outpara['rcut'] = 3
    outpara['r_s'] = 0.8
    outpara['eta'] = 1
    outpara['tol'] = 1E-4
    outpara['zeta'] = 1
    outpara['lamda'] = 1
    outpara['ang_paraall'] = []
    outpara['rad_paraall'] = []
    nbatch = 0
    if task == 'dftbplus':
        newdire = os.path.join(directory, 'dftbplus')
        os.system('rm '+newdire+'/dip.dat')
        os.system('rm '+newdire+'/bandenergy.dat')
    elif task == 'aims':
        newdire = os.path.join(directory, 'aims')
        os.system('rm '+newdire+'/vol.dat')
        os.system('rm '+newdire+'/bandenergy.dat')
        os.system('rm '+newdire+'/pol.dat')
        os.system('rm '+newdire+'/dip.dat')
    elif task == 'dftbtorch':
        newdire = os.path.join(directory, 'dftbtorch')
        os.system('rm '+newdire+'/bandenergy.dat')
        os.system('rm '+newdire+'/dipole.dat')
        dipolemall = []
        eigvalall = []
    elif task == 'envir':
        pass
    elif task == 'dftbml':
        pass
    # ######################### load file ######################### #
    coorall, symbols, specie, speciedict = loadhdfdata(ntype=1)
    ifile = 0
    if task == 'dftbml':
        dftbml('aims', outpara, coorall, nfile, specie, speciedict, symbols,
               nbatch)
    elif task == 'dftbpy_compr':
        dftbpy_compr(coorall, symbols, specie, speciedict)
    # for convinience, all options are in for loop
    else:
        for coor in coorall[0:nfile]:
            ifile += 1
            if ifile > nfile:
                break
            if task == 'aims':
                aims(ifile, coor, specie, speciedict, symbols, newdire)
            elif task == 'dftbplus':
                dftbplus(ifile, coor, specie, speciedict, newdire)
            elif task == 'dftbtorch':
                outpara = {}
                outpara['ty'] = 1
                outpara['symbols'] = symbols
                outpara['dipolemall'] = dipolemall
                outpara['eigvalall'] = eigvalall
                outpara = dftbtorchrun(outpara, coor, newdire)
                if ifile == nfile:
                    save_dftbpy(outpara, newdire)
            elif task == 'envir':
                outpara, nbatch = envir(nbatch, outpara, ifile, coor, specie,
                                        speciedict, symbols, directory)
                if ifile == nfile:
                    save_envir(outpara, directory)
            elif task == 'get_dftbpara':
                get_dftbpara()


# ############################## load data ############################### #
def loadhdfdata(ntype):
    icount = 0
    for hdf5file in hdf5filelist:
        adl = pya.anidataloader(hdf5file)
        for data in adl:
            icount += 1
            if icount == ntype:
                coorall = data['coordinates']
                symbols = data['species']
                specie = set(symbols)
                speciedict = Counter(symbols)
    return coorall, symbols, specie, speciedict


def loadrefdata(ref, dire, nfile):
    if ref == 'aims':
        newdire = os.path.join(directory, dire)
        if os.path.exists(os.path.join(newdire, 'bandenergy.dat')):
            refenergy = Variable(torch.empty(nfile, 2), requires_grad=False)
            fpenergy = open(os.path.join(newdire, 'bandenergy.dat'), 'r')
            for ifile in range(0, nfile):
                energy = np.fromfile(fpenergy, dtype=float,
                                     count=3, sep=' ')
                refenergy[ifile, :] = torch.from_numpy(energy[1:])
    elif ref == 'VASP':
        pass
    return refenergy


def loadenv(ref, dire, nfile, natom):
    if os.path.exists(os.path.join(dire, 'rad_para.dat')):
        rad = np.zeros((nfile, natom))
        fprad = open(os.path.join(dire, 'rad_para.dat'), 'r')
        for ifile in range(0, nfile):
            irad = np.fromfile(fprad, dtype=float, count=natom, sep=' ')
            rad[ifile, :] = irad[:]
    if os.path.exists(os.path.join(dire, 'ang_para.dat')):
        ang = np.zeros((nfile, natom))
        fpang = open(os.path.join(dire, 'ang_para.dat'), 'r')
        for ifile in range(0, nfile):
            iang = np.fromfile(fpang, dtype=float, count=natom, sep=' ')
            ang[ifile, :] = iang[:]
    return rad, ang


# ########################### different type ############################# #
def aims(ifile, coor, specie, speciedict, symbols, dire):
    '''here dft means FHI-aims'''
    natom = np.shape(coor)[0]
    write.FHIaims().geo_nonpe(ifile, coor, specie, speciedict, symbols)
    os.rename('geometry.in.{}'.format(ifile), 'aims/geometry.in')
    os.system('bash '+dire+'/run.sh '+dire+' '+str(ifile)+' '+str(natom))


def dftbplus(ifile, coor, specie, speciedict, dire):
    '''use dftb+ to calculate'''
    write.dftbplus().geo_nonpe2(ifile, coor, specie, speciedict)
    os.rename('geo.gen.{}'.format(ifile), 'dftbplus/geo.gen')
    os.system('bash '+dire+'/run.sh '+dire+' '+str(ifile))


def dftbtorchrun(outpara, coor, dire):
    '''use dftb_python and read SK from whole .skf file, coor as input and
    do not have to read coor from geo.gen or other input files'''
    outpara['coor'] = torch.from_numpy(coor)
    dipolemall = outpara['dipolemall']
    eigvalall = outpara['eigvalall']
    dipolem, eigval = dftb_torch.main(outpara)
    dipolemall.append(dipolem)
    eigvalall.append(eigval)
    outpara['dipolemall'] = dipolemall
    outpara['eigvalall'] = eigvalall
    return outpara


def envir(nbatch, outpara, ifile, coor, specie, speciedict, symbols,
          directory):
    r_s = outpara['r_s']
    eta = outpara['eta']
    tol = outpara['tol']
    zeta = outpara['zeta']
    rcut = outpara['rcut']
    lamda = outpara['lamda']
    ang_paraall = outpara['ang_paraall']
    rad_paraall = outpara['rad_paraall']
    rad_para = lattice_cell.rad_molecule2(coor, rcut, r_s, eta, tol, symbols)
    ang_para = lattice_cell.ang_molecule2(coor, rcut, r_s, eta, zeta, lamda,
                                          tol, symbols)
    ang_paraall.append(ang_para)
    rad_paraall.append(rad_para)
    outpara['ang_paraall'] = ang_paraall
    outpara['rad_paraall'] = rad_paraall
    '''for ispecie in speciedict:
        nbatch += speciedict[ispecie]
        for jspecie in speciedict:
            nbatch += speciedict[ispecie]*speciedict[jspecie]'''
    natom = len(coor)
    nbatch += natom*(natom+1)
    return outpara, nbatch


# ############################# save data ############################### #
def save_dftbpy(outpara, dire):
    eigvalall = outpara['eigvalall']
    dipolemall = outpara['dipolemall']
    with open(os.path.join(dire, 'bandenergy.dat'), 'w') as fopen:
        for eigval in eigvalall:
            np.savetxt(fopen, eigval, newline=" ")
            fopen.write('\n')
    with open(os.path.join(dire, 'dipole.dat'), 'w') as fopen:
        for dipolem in dipolemall:
            np.savetxt(fopen, dipolem, newline=" ")
            fopen.write('\n')


def save_envir(outpara, directory):
    ang_paraall = outpara['ang_paraall']
    rad_paraall = outpara['rad_paraall']
    with open(os.path.join(directory, 'rad_para.dat'), 'w') as fopen:
        np.savetxt(fopen, rad_paraall, fmt="%s", newline=' ')
        fopen.write('\n')
    with open(os.path.join(directory, 'ang_para.dat'), 'w') as fopen:
        np.savetxt(fopen, ang_paraall, fmt="%s", newline=' ')
        fopen.write('\n')


# ############################### ML #################################### #
def dftbml(refname, generalpara, coorall, nfile, specie, speciedict,
           symbols, nbatch):
    # this is to get initial parameters for ML #
    refbandenergy = loadrefdata(refname, refname, nfile)
    ifile = 0
    for coor in coorall[0:nfile]:
        ifile += 1
        generalpara, nbatch = envir(nbatch, generalpara, ifile, coor, specie,
                                    speciedict, symbols, directory)
    get_calpara(generalpara, symbols)
    train(generalpara, refbandenergy, nfile, coorall)


class Net(torch.nn.Module):
    def __init__(self, D_in, H, D_out):

        super(Net, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):

        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


def train(generalpara, ref, nbatch, coorall):
    get_init_para(generalpara, coorall, nbatch)
    # --------------------------define NN layers-------------------------- #
    # in_feature = generalpara['in_feature']
    # nin_ = generalpara['natom']
    # nhid_ = 10
    # nout_ = generalpara['atomind2']
    # model = Net(nin_, nhid_, nout_)
    # criterion = torch.nn.MSELoss(reduction='sum')
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    # pred_h = generalpara['ham0']
    # optimizer = torch.optim.SGD([pred_h], lr=1e-4)
    # dftbout = Variable(torch.zeros(nbatch, 2))
    '''for it in range(0, 200):
        # pred_h = model(in_feature)
        generalpara['hstable'] = pred_h
        # Forward
    '''
    # optimizer.zero_grad()
    # loss.backward(retain_graph=True)
    # optimizer.step()
    # loss = criterion(dftbout, ref)
    for ibatch in range(0, nbatch):
        for it in range(0, 200):
            generalpara['hamtable'] = Variable(generalpara['ham0'][ibatch],
                       requires_grad=True)
            generalpara['ovrtable'] = generalpara['ovr0'][ibatch]
            slakot.sk_tranml(generalpara)
            loss = singletrain(generalpara, nbatch, ref[ibatch, :])
            if it % 100 == 99:
                print(it, loss.item())


def singletrain(generalpara, nbatch, ref):
    optimizer = torch.optim.SGD([generalpara['hamtable']], lr=1e-4)
    eigval = dftb_torch.SCF(generalpara).scf_nonpe(generalpara)
    criterion = torch.nn.MSELoss(reduction='sum')
    loss = criterion(eigval, ref)
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
    print('loss', loss)
    return loss


def get_init_para(generalpara, coorall, nbatch):
    natom = generalpara['natom']
    ind2 = int(generalpara['atomind2'])
    atomname = generalpara['atomnameall']
    in_feature = torch.zeros(nbatch, natom)
    ham0 = torch.zeros(nbatch, ind2, 3)
    ovr0 = torch.zeros(nbatch, ind2, 3)
    atom_num = torch.zeros(natom)
    iatom = 0
    for iname in atomname:
        atom_num[iatom] = ATOMNUM[iname]
        iatom += 1
    ibatch = 0
    for irad in generalpara['rad_paraall']:
        in_feature[ibatch, :] = torch.from_numpy(
                generalpara['rad_paraall'][ibatch])
        # in_feature[ibatch, :, 1] = atom_num[:]
        ibatch += 1
    ibatch = 0
    for icoor in coorall[:nbatch]:
        iham, iovr = get_hstorch(generalpara, icoor)
        ham0[ibatch, :, :] = iham
        ovr0[ibatch, :, :] = iovr
        ibatch += 1
    generalpara['in_feature'] = Variable(in_feature)
    generalpara['ham0'] = Variable(ham0)
    generalpara['ovr0'] = Variable(ovr0)
    return generalpara


def get_hstorch(generalpara, coor):
    generalpara['coor'] = torch.from_numpy(coor)
    ReadInt().get_coor5(generalpara)
    dftb_torch.read_sk(generalpara)
    dftb_torch.Initialization(generalpara).getSKTable(generalpara)
    return generalpara['hamtable'], generalpara['ovrtable']


def get_calpara(generalpara, symbols):
    generalpara['ty'] = 6
    generalpara['scf'] = True
    generalpara['scc'] = False
    generalpara['task'] = 'ground'
    generalpara['mixFactor'] = 0.2
    generalpara['tElec'] = 0
    generalpara['maxIter'] = 60
    generalpara['periodic'] = False
    generalpara['scc'] = False
    generalpara['dipole'] = True
    generalpara['coorType'] = 'C'
    generalpara['symbols'] = symbols
    generalpara['atomnameall'] = symbols
    Path0 = os.getcwd()
    generalpara['filename'] = 'dftb_in'
    generalpara['direInput'] = os.path.join(Path0, 'dftbtorch')
    generalpara['direSK'] = os.path.join(Path0, 'dftbtorch/slko')
    generalpara['atomind'] = 0
    generalpara['natom'] = 0
    for isym in symbols:
        generalpara['atomind'] += ATOMIND[isym]
        generalpara['natom'] += 1
    generalpara['atomind2'] = ((generalpara['atomind']+1) *
               generalpara['atomind'])/2
    return generalpara


def dftbpy_compr(coorall, symbols, specie, speciedict):
    ifile = 0
    nfile = 1
    niter = 2
    compress_r = np.array([2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9,
                           3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9,
                           4.0, 4.2, 4.4, 4.6, 5.0])
    mldata = {}
    # -------------------we define dftb_ml is type 5------------------------ #
    mldata['ty'] = 5
    mldata['direCH'] = '/home/gz_fan/Documents/ML/dftb/slko/C_H'
    mldata['direHH'] = '/home/gz_fan/Documents/ML/dftb/slko/H_H'
    mldata['skfallHH'], mldata['nameallHH'] = bench.readsk(
            mldata['direHH'], 'H', 'Hh', 0.4, 0.02)
    mldata['skfallCH'], mldata['nameallCH'] = bench.readsk(
            mldata['direCH'], 'C', 'H', 0.4, 0.02)
    '''for hdf5file in hdf5filelist:
        adl = pya.anidataloader(hdf5file)
        for data in adl:
            coorall = data['coordinates']
            symbols = data['species']
            specie = set(symbols)
            speciedict = Counter(symbols)'''
    # ---------get_sk will read the .skf file except H and S-------- #
    mldata = get_sk(mldata, symbols)
    for coor in coorall:
        ifile += 1
        if ifile > nfile:
            break
        mldata['coor'] = coor
        mldata['symbols'] = symbols
        natom = np.shape(coor)[0]
        compressr0 = write.dftbplus().geo_nonpe_ml(
                ifile, coor, specie, speciedict, symbols)
        compressr = np.zeros((nfile, natom, niter))
        for iiter in range(0, niter):
            for iatom in range(0, natom):
                compressr[ifile-1, :, iiter] = compressr0[:]
                for ir in compress_r:
                    compressr0[iatom] = ir
                    print('{} module {} atom, compressR:'.format(
                            ifile, iatom), compressr0)
                    mldata = get_hsml(mldata, directory,
                                      compressr0, iatom, ir)
                    compressr0[:] = compressr[ifile-1, :, iiter]
                    dipolem, eigval = dftb.main(mldata)
                    print('one {} calculation'.format(symbols))
    '''with open('compressr.dat', 'w') as fopen2:
        for ifile in range(0, nfile):
            np.savetxt(fopen2, compressr[ifile, :, :], fmt="%s", newline=' ')
            fopen2.write('\n')'''


def get_hsml(mldata, directory, compressr0, iatom, ir):
    coor = mldata['coor']
    symbols = mldata['symbols']
    natom = np.shape(coor)[0]
    hsall = np.zeros((natom, natom, 20))
    hs_skf = np.zeros(20)
    for iatom in range(0, natom):
        for jatom in range(0, natom):
            r1 = compressr0[iatom]
            r2 = compressr0[jatom]
            name1 = symbols[iatom]
            name2 = symbols[jatom]
            dist = np.sqrt(sum((coor[iatom, :]-coor[jatom, :])**2))/BOHR
            if name1 == name2 and name1 == 'H':
                skffileall = mldata['skfallHH']
                nameall = mldata['nameallHH']
                ty = nameall.index([name1, name2])
            else:
                skffileall = mldata['skfallCH']
                nameall = mldata['nameallCH']
                ty = nameall.index([name1, name2])
            if iatom != jatom:
                skffile = skffileall[ty]
                hs_skf = bench.gensk(dist, r1, r2, skffile, nameall,
                                     grid0, gridmesh)
            else:
                hs_skf[:] = 0
            hsall[iatom, jatom, :] = hs_skf[:]
    mldata['h_s_all'] = hsall
    return mldata


def get_sk(mldata, symbols):
    symbolsset = set(symbols)
    # ----nameall = mldata['nameallCH'] may have to be revised further------ #
    nameall = mldata['nameallCH']
    for iatom in range(0, len(symbolsset)):
        for jatom in range(0, len(symbolsset)):
            name1 = symbols[iatom]
            name2 = symbols[jatom]
            if name1 == name2:
                ty = nameall.index([name1, name2])
                grid = mldata['skfallCH'][ty]['gridmeshpoint'][0]
                ngrids = mldata['skfallCH'][ty]['gridmeshpoint'][1]
                mldata['grid_dist'+name1+name2] = grid
                mldata['ngridpoint'+name1+name2] = ngrids
                Espd_Uspd = mldata['skfallCH'][ty]['onsitespeu']
                mldata['Espd_Uspd'+name1+name2] = Espd_Uspd
                # mass and r_cut are read from list of .skf files, they are
                # the same, so we choose the first one
                massrcut = mldata['skfallCH'][ty]['massrcut'][0]
                mldata['mass_cd'+name1+name2] = massrcut
            else:
                ty = nameall.index([name1, name2])
                # print('type:', ty, name1, name2)
                grid = mldata['skfallCH'][ty]['gridmeshpoint'][0]
                ngrids = mldata['skfallCH'][ty]['gridmeshpoint'][1]
                mldata['grid_dist'+name1+name2] = grid
                mldata['ngridpoint'+name1+name2] = ngrids
                massrcut = mldata['skfallCH'][ty]['massrcut'][0]
                mldata['mass_cd'+name1+name2] = massrcut
    return mldata


def compare(dir_dftb, dir_ref, natom, iatom, compress_r, compressr, ifile, it):
    # n_v, n_char need revise !!!!!!!!!!!!!!!!!!!!!!!!!!
    n_v = np.array([38.37861207, 10.31539447, 10.31539447, 10.31539447, 10.31539447])
    n_char = np.array([4, 1, 1, 1, 1])
    ncompressr = len(compress_r)
    fp0 = open(os.path.join(dir_dftb, 'onsite.dat'), 'r')
    fp1 = open(os.path.join(dir_ref, 'vol.dat'), 'r')
    data1 = np.fromfile(fp1, dtype=float, count=natom, sep=' ')
    data1 = data1/n_v[iatom]
    onsite = np.zeros((ncompressr, natom))
    for ir1 in range(0, ncompressr):
        data0 = np.fromfile(fp0, dtype=float, count=natom+3, sep=' ')
        onsite[ir1, :] = data0[3:]/n_char[iatom]
    ionsite = onsite[:, iatom]
    onsite_err = list(abs(ionsite[:]-data1[iatom]))
    min_compressr = onsite_err.index(min(onsite_err))
    print('onsite_err', onsite_err, 'min_compressr', min_compressr)
    err = []
    num = []
    ii = 0
    # the No. of closet compression r in original compression r list
    nr = list(abs(compress_r-compressr[iatom])).index(min(abs(compress_r-compressr[iatom])))
    for ierr in onsite_err:
        ii += 1
        if ierr < tol_mlerr:
            err.append(ierr)
            num.append(ii-1)
    # x = np.linspace(1, ncompressr, ncompressr)
    plt.plot(compress_r, ionsite[:]-data1[iatom])
    plt.savefig("{} ifile {} iatom {}loop.eps".format(ifile, iatom, it), dpi=150)
    plt.show()
    if num != []:
        # print(num, num.index(min(num)), nr)
        # ir_end = num[num.index(min(num))+nr]
        ir_end = num[0]
        for inum in num:
            if abs(ir_end-nr) > abs(inum-nr):
                ir_end = inum
                print('inum', inum, nr, num)
        new_compressr = (compressr[iatom]+compress_r[ir_end])/2
        old_compressr = compress_r[ir_end]
    else:
        new_compressr = (2*compressr[iatom]+compress_r[min_compressr])/3
        old_compressr = compress_r[min_compressr]
    compressr[iatom] = new_compressr
    with open('ml.log', 'a') as fopen:
        fopen.write("{} ifile {} iatom {} loop".format(ifile, iatom, it)+"\n")
        fopen.write('optimized compression radius: '+str(old_compressr))
        fopen.write('\n')
        fopen.write('new compression radius \n')
        np.savetxt(fopen, compressr, fmt="%s", newline=' ')
        fopen.write('\n')
    cmd = ': > dataset1/onsite.dat'
    subprocess.run(cmd, shell=True, universal_newlines=True, check=True)
    return compressr


if __name__ == '__main__':
    task = 'dftbml'
    main(task)
