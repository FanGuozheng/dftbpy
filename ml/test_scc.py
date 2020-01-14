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
import lattice_cell
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import dftbtorch.dftb_torch as dftb_torch
from dftbtorch.readt import ReadInt
import dftbtorch.slakot as slakot
directory = '/home/gz_fan/Documents/ML/dftb/ml'
diresk = '/home/gz_fan/Documents/ML/dftb/slko'
tol_mlerr = 0.003
BOHR = 0.529177210903
gridmesh = 0.2
ATOMIND = {'H': 1, 'HH': 2, 'HC': 3, 'C': 4, 'CH': 5, 'CC': 6}
ATOMNUM = {'H': 1, 'C': 6}
HNUM = {'HH': 1, 'HC': 2, 'CC': 4, 'CH': 0}


def main(task):
    outpara = {}

    # get the default para for dftb and ML, these para will maintain unchanged
    getmlpara(outpara)

    # load dataset, here is hdf type
    LoadData(outpara).loadhdfdata(outpara)

    # run reference calculations, either dft or dftb
    RunML(outpara).mlref7(outpara)

    # run dftb in ML process
    RunML(outpara).mldftb7(outpara)

    # plot data from ML
    plot(outpara)


def getmlpara(outpara):
    '''
    ref: the reference of ML
    datasettype: type of dataset
    hdf_num (int): for hdf data type, read which kind of molecule
    nfile_dataset (int): read how many molecules of each dataset
    optim_para: the optimized parameters
    '''
    outpara['refname'] = 'dftb_torch'

    # the following is for parameters when loading hdf dataset
    outpara['datasettype'] = 'hdf'
    hdffilelist = []
    hdffilelist.append('/home/gz_fan/Documents/ML/database/an1/ani_gdb_s01.h5')
    outpara['hdffile'] = hdffilelist
    outpara['hdf_num'] = 1
    outpara['nfile_dataset'] = ['5']
    outpara['optim_para'] = ['Hamiltonian']

    # the following is for environment parameters
    outpara['rcut'] = 3
    outpara['r_s'] = 0.8
    outpara['eta'] = 1
    outpara['tol'] = 1E-4
    outpara['zeta'] = 1
    outpara['lamda'] = 1
    outpara['ang_paraall'] = []
    outpara['rad_paraall'] = []

    # the following is for ML (keep fixed all the time)
    # splinetype: Bspline, Polyspline
    outpara['ty'] = 7
    outpara['mlsteps'] = 10
    outpara['splinehs'] = True
    outpara['splinetype'] = 'Polyspline'
    outpara['splinedist'] = 0.2
    outpara['splinecutoff'] = 12

    # the following is for dftb (keep fixed all the time)
    outpara['scf'] = True
    outpara['scc'] = False
    outpara['task'] = 'ground'
    outpara['mixFactor'] = 0.2
    outpara['tElec'] = 0
    outpara['maxIter'] = 60
    outpara['periodic'] = False
    outpara['scc'] = False
    outpara['dipole'] = True
    outpara['coorType'] = 'C'
    Path0 = os.getcwd()
    outpara['filename'] = 'dftb_in'
    outpara['direInput'] = os.path.join(Path0, 'dftbtorch')
    outpara['direSK'] = os.path.join(Path0, 'dftbtorch/slko')

    # the following is for plotting
    outpara['plot_ham'] = True
    return outpara


class RunML():
    '''
    This is class for ML process (loading Ref data and dataset, running
    calculations of ref method and dftb method, saving ml data).
    len(outpara['nfile']) is to get optimize how many dataset.
    Assumption: the atom specie in each dataset maintain unchanged, otherwise
    we have to check for each new moluecle, if there is new atom specie.
    '''
    def __init__(self, outpara):
        self.outpara = outpara

    def aims_ref():
        pass

    def mlref7(self, outpara):
        if len(outpara['nfile_dataset']) == 1:
            nbatch = outpara['nfile'] = int(outpara['nfile_dataset'][0])

        # for each batch, run reference (dftb) calculations
        for ibatch in range(0, nbatch):
            # get the coor of ibatch, and start initialization
            outpara['ibatch'] = ibatch
            outpara['coor'] = torch.from_numpy(outpara['coorall'][ibatch])
            dftb_torch.Initialization(outpara)

            # in the dataset, atom specie is the same, so we will read and
            # store the SK for the first molecule
            if ibatch == 0:
                GenMLPara(outpara).get_hrownum(outpara)
                slakot.SlaKo(outpara).read_skdata(outpara)
                slakot.SlaKo(outpara).getSKSplPara(outpara)

                # run dftb calculations
                RunCalc(outpara).idftb_torchspline(outpara)

                # save data
                if outpara['splinetype'] == 'Polyspline':
                    SaveData(outpara).save2D(
                            outpara['yspline'], name='spline0.dat', ty='w')
                elif outpara['splinetype'] == 'Bspline':
                    SaveData(outpara).save2D(
                            outpara['cspline'], name='spline0.dat', ty='w')
                SaveData(outpara).save1D(
                        outpara['ref'].detach().numpy(),
                        name='eigref.dat', ty='w')
                SaveData(outpara).save1D(
                        outpara['hammat'].detach().numpy(),
                        name='ham0.dat', ty='w')
            else:
                RunCalc(outpara).idftb_torchspline(outpara)
                SaveData(outpara).save1D(
                        outpara['ref'].detach().numpy(),
                        name='eigref.dat', ty='a')
                SaveData(outpara).save1D(
                        outpara['hammat'].detach().numpy(),
                        name='ham0.dat', ty='a')

    def mldftb7(self, outpara):
        # read info for ml and dftb
        nbatch = outpara['nfile']
        if outpara['splinetype'] == 'Bspline':
            outpara['cspline'] = Variable(outpara['csplinerand'],
                                          requires_grad=True)
        elif outpara['splinetype'] == 'Polyspline':
            outpara['yspline'] = Variable(outpara['ysplinerand'],
                                          requires_grad=True)
        coorall = outpara['coorall']
        ref = outpara['ref']

        # calculate one by one to optimize para
        for ibatch in range(0, nbatch):
            # construct basic data for dftb
            outpara['coor'] = torch.from_numpy(coorall[ibatch])

            # for each molecule we will run mlsteps
            for it in range(0, outpara['mlsteps']):
                # define optim para
                if outpara['splinetype'] == 'Polyspline':
                    optimizer = torch.optim.SGD([outpara['yspline']], lr=1e-4)

                # dftb calculations
                dftb_torch.Initialization(outpara).GenSKMatrix(outpara)
                eigval = dftb_torch.SCF(outpara).scf_nonpe(outpara)

                # define loss function
                criterion = torch.nn.MSELoss(reduction='sum')
                loss = criterion(eigval, ref)

                # save data
                SaveData(outpara).save1D(
                        eigval.detach().numpy(), name='eigbp.dat', ty='a')
                SaveData(outpara).save2D(
                        outpara['yspline'].detach().numpy(),
                        name='spline.dat', ty='a')
                SaveData(outpara).save1D(
                        outpara['hammat'].detach().numpy(), name='ham.dat',
                        ty='a')

                # clear gradients and define back propagation
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                if it % 10 == 9:
                    print(it, loss.item())


class GenMLPara():
    '''
    this class aims to form parameters for ML
    genenvir: atomic environment parameters
    get_hrownum: get how many row lines of c parameters in bspline
    '''
    def __init__(self, outpara):
        self.outpara = outpara

    def genenvir(nbatch, outpara, ifile, coor, specie, speciedict, symbols,
                 directory):
        rcut = outpara['rcut']
        r_s = outpara['r_s']
        eta = outpara['eta']
        tol = outpara['tol']
        zeta = outpara['zeta']
        lamda = outpara['lambda']
        ang_paraall = outpara['ang_paraall']
        rad_paraall = outpara['rad_paraall']
        rad_para = lattice_cell.rad_molecule2(coor, rcut, r_s, eta, tol,
                                              symbols)
        ang_para = lattice_cell.ang_molecule2(coor, rcut, r_s, eta, zeta,
                                              lamda, tol, symbols)
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

    def genmlpara0(self, outpara):
        pass

    def igenmlpara(self, outpara):
        pass

    def get_hrownum(self, outpara):
        htable_num = 0
        for iatom in outpara['atomname_set']:
            for jatom in outpara['atomname_set']:
                nameij = iatom+jatom
                htable_num += HNUM[nameij]
        print('initial H-table has {} rows'.format(htable_num))
        outpara['htable_num'] = htable_num
        return outpara


def gen_mlpara(outpara, nbatch):
    natom = outpara['natom']
    coorall = outpara['coorall']
    ind2 = int(outpara['atomind2'])
    # atomname = outpara['atomnameall']
    ham0 = torch.zeros(nbatch, ind2, 3)
    ovr0 = torch.zeros(nbatch, ind2, 3)
    hamrand = torch.zeros(nbatch, ind2, 3)
    ovrrand = torch.zeros(nbatch, ind2, 3)
    ibatch = 0
    if outpara['ty'] == 6:
        for icoor in coorall[:nbatch]:
            iham, iovr = get_hs(outpara, icoor)
            ham0[ibatch, :, :] = iham
            ovr0[ibatch, :, :] = iovr
            hamrand[ibatch, :, :] = iham + torch.randn(ind2, 3)/20
            ovrrand[ibatch, :, :] = iovr + torch.randn(ind2, 3)/20
            ibatch += 1
        outpara['ham0'] = ham0
        outpara['ovr0'] = ovr0
        outpara['hamrand'] = Variable(hamrand)
        outpara['ovrrand'] = Variable(ovrrand)
    elif outpara['ty'] == 7:
        for icoor in coorall[:nbatch]:
            # this loop is to update the spline parameters (c, t)
            # there will be another for calculations
            get_hsspline(outpara, icoor, nbatch, ibatch)
            ibatch += 1
    return outpara


def get_hs(outpara, coor):
    '''This function is to read coordination and H table from .skf file
    without slater-Koster transformations, corresponding ty == 6'''
    outpara['coor'] = torch.from_numpy(coor)
    ReadInt(outpara).get_coorall(outpara)
    dftb_torch.read_sk(outpara)
    dftb_torch.Initialization(outpara)
    return outpara['hamtable'], outpara['ovrtable']


def get_hsspline(outpara, coor, nbatch, ibatch):
    '''This function is to read coordination and H table and build spline,
    H(r) and onsite, corresponding ty == 6, in this case, we assume
    calculation parameters are same, only coordinations are different'''
    outpara['coor'] = torch.from_numpy(coor)
    ReadInt(outpara).get_coor(outpara)
    dftb_torch.read_sk(outpara)
    get_hnum(outpara)
    # dftb_torch.Initialization(outpara)
    dftb_torch.Initialization(outpara).ReadInputpara(outpara)
    dftb_torch.Initialization(outpara).ReadGeo(outpara)
    dftb_torch.Initialization(outpara).ReadSK(outpara)
    dftb_torch.Initialization(outpara).formSKDataSpline(outpara)
    outpara['atomname_setold'] = outpara['atomname_set']
    outpara['onsite_num'] = len(outpara['atomname_set'])


class LoadData():
    '''
    In hdf data type, each ntype represents one type of molecule, such as for
        ntype = 1, the molecules are all CH4
    '''
    def __init__(self, outpara):
        self.outpara = outpara

    def loadhdfdata(self, outpara):
        ntype = self.outpara['hdf_num']
        hdf5filelist = outpara['hdffile']
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
                    self.outpara['coorall'] = coorall
                    self.outpara['symbols'] = symbols
                    self.outpara['specie'] = specie
                    self.outpara['speciedict'] = speciedict
        return self.outpara

    def loadrefdata(self, ref, dire, nfile):
        if ref == 'aims':
            newdire = os.path.join(directory, dire)
            if os.path.exists(os.path.join(newdire,
                                           'bandenergy.dat')):
                refenergy = Variable(torch.empty(nfile, 2),
                                     requires_grad=False)
                fpenergy = open(os.path.join(newdire, 'bandenergy.dat'), 'r')
                for ifile in range(0, nfile):
                    energy = np.fromfile(fpenergy, dtype=float,
                                         count=3, sep=' ')
                    refenergy[ifile, :] = torch.from_numpy(energy[1:])
        elif ref == 'dftbrand':
            newdire = os.path.join(directory, dire)
            if os.path.exists(os.path.join(newdire, 'bandenergy.dat')):
                refenergy = Variable(torch.empty(nfile, 2),
                                     requires_grad=False)
                fpenergy = open(os.path.join(newdire, 'bandenergy.dat'), 'r')
                for ifile in range(0, nfile):
                    energy = np.fromfile(fpenergy, dtype=float,
                                         count=3, sep=' ')
                    refenergy[ifile, :] = torch.from_numpy(energy[1:])
        elif ref == 'VASP':
            pass
        return refenergy

    def loadenv(ref, diresk, nfile, natom):
        if os.path.exists(os.path.join(diresk, 'rad_para.dat')):
            rad = np.zeros((nfile, natom))
            fprad = open(os.path.join(diresk, 'rad_para.dat'), 'r')
            for ifile in range(0, nfile):
                irad = np.fromfile(fprad, dtype=float, count=natom, sep=' ')
                rad[ifile, :] = irad[:]
        if os.path.exists(os.path.join(diresk, 'ang_para.dat')):
            ang = np.zeros((nfile, natom))
            fpang = open(os.path.join(diresk, 'ang_para.dat'), 'r')
            for ifile in range(0, nfile):
                iang = np.fromfile(fpang, dtype=float, count=natom, sep=' ')
                ang[ifile, :] = iang[:]
        return rad, ang


class RunCalc():

    def __init__(self, outpara):
        self.outpara = outpara

    def aims(self, ifile, coor, specie, speciedict, symbols, dire):
        '''here dft means FHI-aims'''
        natom = np.shape(coor)[0]
        write.FHIaims().geo_nonpe(ifile, coor, specie, speciedict, symbols)
        os.rename('geometry.in.{}'.format(ifile), 'aims/geometry.in')
        os.system('bash '+dire+'/run.sh '+dire+' '+str(ifile)+' '+str(natom))

    def dftbplus(self, ifile, coor, specie, speciedict, diresk):
        '''use dftb+ to calculate'''
        write.dftbplus().geo_nonpe2(ifile, coor, specie, speciedict)
        os.rename('geo.gen.{}'.format(ifile), 'dftbplus/geo.gen')
        os.system('bash '+diresk+'/run.sh '+diresk+' '+str(ifile))

    def dftbtorchrun(self, outpara, coor, diresk):
        '''
        use dftb_python and read SK from whole .skf file, coor as input and
        do not have to read coor from geo.gen or other input files
        '''
        outpara['coor'] = torch.from_numpy(coor)
        dipolemall = outpara['dipolemall']
        eigvalall = outpara['eigvalall']
        dipolem, eigval = dftb_torch.main(outpara)
        dipolemall.append(dipolem)
        eigvalall.append(eigval)
        outpara['dipolemall'] = dipolemall
        outpara['eigvalall'] = eigvalall
        return outpara

    def idftb_torchspline(self, outpara):
        '''
        use dftb_python and read SK from whole .skf file, coor as input and
        do not have to read coor from geo.gen or other input files
        '''
        ibatch = outpara['ibatch']
        outpara['coor'] = torch.from_numpy(outpara['coorall'][ibatch])
        dftb_torch.Initialization(outpara).GenSKMatrix(outpara)
        ref = dftb_torch.SCF(outpara).scf_nonpe(outpara)
        outpara['ref'] = ref
        return outpara


class SaveData():
    '''
    data is numpy type matrix
    blank defines where we'll write blank line
    name is the name of the saved file
    save2D will save file line by line
    savetype: 'a': appendix; 'w': save as a new file (replace the old)
    '''
    def __init__(self, outpara):
        self.outpara = outpara

    def save1D(self, data, name, blank='lower', dire=None, ty='w'):
        if dire is None:
            newdire = os.getcwd()
        else:
            newdire = dire
        with open(os.path.join(newdire, name), ty) as fopen:
            if blank == 'upper':
                fopen.write('\n')
            np.savetxt(fopen, data, newline=" ")
            fopen.write('\n')
            if blank == 'lower':
                fopen.write('\n')

    def save2D(self, data, name, blank='lower', dire='.', ty='w'):
        if dire is None:
            newdire = os.getcwd()
        else:
            newdire = dire
        with open(os.path.join(newdire, name), ty) as fopen:
            for idata in data:
                if blank == 'upper':
                    fopen.write('\n')
                np.savetxt(fopen, idata, newline=" ")
                fopen.write('\n')
                if blank == 'lower':
                    fopen.write('\n')

    def save_dftbpy(outpara, diresk):
        eigvalall = outpara['eigvalall']
        dipolemall = outpara['dipolemall']
        with open(os.path.join(diresk, 'bandenergy.dat'), 'w') as fopen:
            for eigval in eigvalall:
                np.savetxt(fopen, eigval, newline=" ")
                fopen.write('\n')
        with open(os.path.join(diresk, 'dipole.dat'), 'w') as fopen:
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


def plot(outpara, nfile):
    fpeigval = open('eigval_backp.dat', 'r')
    fpeigref = open('eigval_ref.dat', 'r')
    istep = 0
    for ifile in range(0, nfile):
        datbfpeigref = np.fromfile(fpeigref, dtype=float, count=2, sep=' ')
        for i in range(0, nsteps):
            datbfpeigval = np.fromfile(fpeigval, dtype=float, count=2, sep=' ')
            plt.plot(istep, datbfpeigref[0], 'x', color='r')
            plt.plot(istep, datbfpeigval[0], 'o', color='b')
            istep += 1
    plt.show()
    fpeigval = open('eigval_backp.dat', 'r')
    fpeigref = open('eigval_ref.dat', 'r')
    istep = 0
    datbfpeigref = np.fromfile(fpeigref, dtype=float, count=2, sep=' ')
    for i in range(0, nsteps):
        datbfpeigval = np.fromfile(fpeigval, dtype=float, count=2, sep=' ')
        # print(datbfpeigval[1], datbfpeigval)
        plt.plot(istep, datbfpeigref[1], 'x', color='r')
        # print(datbfpeigval[1], datbfpeigval)
        plt.plot(istep, datbfpeigval[1], 'o', color='b')
        istep += 1
    plt.show()
    # ------------------------------------------------------------------ #
    fpspline = open('c_spline.dat', 'r')
    fpsplineref = open('c_spline0.dat', 'r')
    istep = 0
    csplineref = np.fromfile(fpsplineref, dtype=float, count=100, sep=' ')
    for i in range(0, nsteps):
        cspline_update = np.fromfile(fpspline, dtype=float, count=100, sep=' ')
        plt.plot(istep, csplineref[50], 'x', color='r')
        plt.plot(istep, cspline_update[50], 'o', color='b')
        istep += 1
    plt.show()
    fpspline = open('c_spline.dat', 'r')
    fpsplineref = open('c_spline0.dat', 'r')
    istep = 0
    csplineref = np.fromfile(fpsplineref, dtype=float, count=100, sep=' ')
    for i in range(0, nsteps):
        cspline_update = np.fromfile(fpspline, dtype=float, count=100, sep=' ')
        plt.plot(istep, csplineref[50], 'x', color='r')
        plt.plot(istep, cspline_update[50], 'o', color='b')
        istep += 1
    plt.show()
    # ------------------------------------------------------------------- #
    fphamupdate = open('ham.dat', 'r')
    fphamref = open('ham0.dat', 'r')
    istep = 0
    for ifile in range(0, nfile):
        hamref = np.fromfile(fphamref, dtype=float, count=36, sep=' ')
        for i in range(0, nsteps):
            hamupdate = np.fromfile(fphamupdate, dtype=float, count=36, sep=' ')
            plt.plot(istep, hamref[31], 'x', color='r')
            plt.plot(istep, hamupdate[31], 'o', color='b')
            istep += 1
    plt.show()
    fphamupdate = open('ham.dat', 'r')
    fphamref = open('ham0.dat', 'r')
    istep = 0
    for ifile in range(0, nfile):
        hamref = np.fromfile(fphamref, dtype=float, count=36, sep=' ')
        for i in range(0, nsteps):
            hamupdate = np.fromfile(fphamupdate, dtype=float, count=36, sep=' ')
            plt.plot(istep, hamref[30], 'x', color='r')
            plt.plot(istep, hamupdate[30], 'o', color='b')
            istep += 1
    plt.show()



class Net(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Net, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


def train(refname, outpara, nbatch):
    coorall = outpara['coorall']
    get_init_para(outpara, coorall, nbatch)
    # --------------------------back propagation-------------------------- #
    os.system('rm eigval_backp.dat')
    os.system('rm eigval_ref.dat')
    # os.system('rm hamtable_backp.dat')
    os.system('rm c_spline0.dat')
    os.system('rm c_spline.dat')
    os.system('rm ham0.dat')
    os.system('rm ham.dat')
    if outpara['ty'] == 6:
        for ibatch in range(0, nbatch):
            # calculate initial dftbplus reference
            outpara['hamtable'] = outpara['ham0'][ibatch]
            outpara['ovrtable'] = outpara['ovr0'][ibatch]
            slakot.sk_tranml(outpara)
            ref = dftb_torch.SCF(outpara).scf_nonpe(outpara)
            with open('eigval_backp.dat', 'a') as fopen:
                fopen.write('\n')
                fopen.write('\n')
                np.savetxt(fopen, ref.detach().numpy(), newline=" ")
                fopen.write('\n')
            with open('hamtable_backp.dat', 'a') as fopen:
                fopen.write('\n')
                fopen.write('\n')
                np.savetxt(fopen, outpara['ham0'][ibatch].numpy(),
                           newline=" ")
                fopen.write('\n')
            # print('ref', ref, outpara['hamtable'])
            for it in range(0, 100):
                if refname == 'dftbrand':
                    outpara['hamtable'] = Variable(
                            outpara['hamrand'][ibatch], requires_grad=True)
                    outpara['ovrtable'] = outpara['ovr0'][ibatch]
                else:
                    outpara['hamtable'] = Variable(
                            outpara['ham0'][ibatch], requires_grad=True)
                    outpara['ovrtable'] = outpara['ovr0'][ibatch]
                slakot.sk_tranml(outpara)
                loss = singletrain(outpara, nbatch, ref, it)
                # loss = singletrain(outpara, nbatch, ref[ibatch, :])
                if it % 100 == 99:
                    print(it, loss.item())
    elif outpara['ty'] == 7:
        c_spline0 = outpara['c_spline']
        c_splinerand = outpara['c_splinerand']
        for ibatch in range(0, nbatch):
            # outpara['c_spline'] = c_spline0
            outpara['coor'] = torch.from_numpy(coorall[ibatch])
            ReadInt(outpara).get_coor(outpara)
            # dftb_torch.read_sk(outpara)
            # get_hnum(outpara)
            # dftb_torch.Initialization(outpara).ReadInput(outpara)
            # dftb_torch.Initialization(outpara).ReadGeo(outpara)
            # dftb_torch.Initialization(outpara).ReadSK(outpara)
            # dftb_torch.Initialization(outpara).getSKTable(outpara)
            dftb_torch.Initialization(outpara).GenSKMatrix(outpara)
            ref = dftb_torch.SCF(outpara).scf_nonpe(outpara)
            with open('eigval_ref.dat', 'a') as fopen:
                fopen.write('\n')
                fopen.write('\n')
                np.savetxt(fopen, ref.detach().numpy(), newline=" ")
                fopen.write('\n')
            with open('c_spline0.dat', 'a') as fopen:
                fopen.write('\n')
                fopen.write('\n')
                c_spline_temp = c_spline0[1, :].detach().numpy()
                np.savetxt(fopen, c_spline_temp, newline=" ")
                fopen.write('\n')
            with open('ham0.dat', 'a') as fopen:
                fopen.write('\n')
                fopen.write('\n')
                ham0 = outpara['hammat'].detach().numpy()
                np.savetxt(fopen, ham0, newline=" ")
                fopen.write('\n')
        outpara['c_spline'] = Variable(c_splinerand, requires_grad=True)
        for ibatch in range(0, nbatch):
            outpara['coor'] = torch.from_numpy(coorall[ibatch])
            ReadInt(outpara).get_coor(outpara)
            dftb_torch.Initialization(outpara).GenSKMatrix(outpara)
            with open('eigval_backp.dat', 'a') as fopen:
                fopen.write('\n')
            with open('c_spline.dat', 'a') as fopen:
                fopen.write('\n')
            with open('ham.dat', 'a') as fopen:
                fopen.write('\n')
            for it in range(0, nsteps):
                # loss = singletrain(outpara, nbatch, ref, it)
                optimizer = torch.optim.SGD([outpara['c_spline']], lr=1e-4)
                dftb_torch.Initialization(outpara).GenSKMatrix(outpara)
                eigval = dftb_torch.SCF(outpara).scf_nonpe(outpara)
                criterion = torch.nn.MSELoss(reduction='sum')
                loss = criterion(eigval, ref)
                with open('eigval_backp.dat', 'a') as fopen:
                    np.savetxt(fopen, eigval.detach().numpy(), newline=" ")
                    fopen.write('\n')
                with open('c_spline.dat', 'a') as fopen:
                    np.savetxt(fopen,
                               outpara['c_spline'][1, :].detach().numpy(),
                               newline=" ")
                    fopen.write('\n')
                with open('ham.dat', 'a') as fopen:
                    np.savetxt(fopen, outpara['hammat'].detach().numpy(),
                               newline=" ")
                    fopen.write('\n')
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                if it % 10 == 9:
                    print(it, loss.item())
        shape1, shape2 = outpara['c_spline'].shape
        with open('c_splineall.dat', 'a') as fopen:
            for i in range(0, shape1):
                icspline = outpara['c_spline'][i, :].detach().numpy()
                np.savetxt(fopen, icspline, newline=" ")
                fopen.write('\n')
            fopen.write('\n')
            fopen.write('\n')
            fopen.write('\n')
            for i in range(0, shape1):
                icspliner = outpara['c_splinerand'][i, :].detach().numpy()
                np.savetxt(fopen, icspliner, newline=" ")
                fopen.write('\n')


def singletrain(outpara, nbatch, ref, it):
    optimizer = torch.optim.SGD([outpara['hamtable']], lr=1e-4)
    eigval = dftb_torch.SCF(outpara).scf_nonpe(outpara)
    criterion = torch.nn.MSELoss(reduction='sum')
    loss = criterion(eigval, ref)
    with open('eigval_backp.dat', 'a') as fopen:
        np.savetxt(fopen, eigval.detach().numpy(), newline=" ")
        fopen.write('\n')
    with open('hamtable_backp.dat', 'a') as fopen:
        np.savetxt(fopen, outpara['hamtable'].detach().numpy(),
                   newline=" ")
        fopen.write('\n')
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
    # print('loss', loss)
    return loss


'''def get_hsml(mldata, directory, compressr0, iatom, ir):
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
    return mldata'''


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


if __name__ == '__main__':
    task = 'dftbml'
    main(task)
