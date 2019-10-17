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
import sys
sys.path.insert(1, '/home/gz_fan/Documents/ML/dftb/prog')
import dftb_python as dftb
directory = '/home/gz_fan/Documents/ML/dftb/ml'
hdf5filelist = ['/home/gz_fan/Documents/ML/database/an1/ani_gdb_s01.h5']
dire = '/home/gz_fan/Documents/ML/dftb/slko'
tol_mlerr = 0.003
BOHR = 0.529177210903
grid0 = 0.4
gridmesh = 0.2


def main(task):
    if task == 'dftbml':
        dftbml()
    elif task == 'dftbplus':
        dftbplus()
    elif task == 'dft':
        dft()
    elif task == 'envir':
        envir()
    elif task == 'dftbpy':
        dftbpy()


def dftbml():
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
    mldata['skfallHH'], mldata['nameallHH'] = bench.readsk(mldata['direHH'],
                                              'H', 'Hh', 0.4, 0.02)
    mldata['skfallCH'], mldata['nameallCH'] = bench.readsk(mldata['direCH'],
                                              'C', 'H', 0.4, 0.02)
    for hdf5file in hdf5filelist:
        adl = pya.anidataloader(hdf5file)
        for data in adl:
            coorall = data['coordinates']
            symbols = data['species']
            specie = set(symbols)
            speciedict = Counter(symbols)
            # ---------get_sk will read the .skf file except H and S-------- #
            mldata = get_sk(mldata, symbols)
            for coor in coorall:
                ifile += 1
                if ifile > nfile:
                    break
                mldata['coor'] = coor
                mldata['symbols'] = symbols
                natom = np.shape(coor)[0]
                compressr0 = write.dftbplus().geo_nonpe_ml(ifile, coor, specie,
                                           speciedict, symbols)
                compressr = np.zeros((nfile, natom, niter))
                for iiter in range(0, niter):
                    for iatom in range(0, natom):
                        compressr[ifile-1, :, iiter] = compressr0[:]
                        for ir in compress_r:
                            compressr0[iatom] = ir
                            print('{} module {} atom, compressR:'.format(
                                    ifile, iatom), compressr0)
                            mldata = get_hsml(mldata, directory, compressr0, iatom, ir)
                            compressr0[:] = compressr[ifile-1, :, iiter]
                            dipolem, eigval = dftb.main(mldata)
                            print('one {} calculation'.format(symbols))
    '''with open('compressr.dat', 'w') as fopen2:
        for ifile in range(0, nfile):
            np.savetxt(fopen2, compressr[ifile, :, :], fmt="%s", newline=' ')
            fopen2.write('\n')'''


def dftbplus():
    '''use dftb+ to calculate'''
    nfile = 50
    icount = 0
    diredftb = os.path.join(directory, 'dftbplus')
    os.system('rm '+diredftb+'/dipole.dat')
    os.system('rm '+diredftb+'/bandenergy1.dat')
    os.system('rm '+diredftb+'/bandenergy2.dat')
    for hdf5file in hdf5filelist:
        adl = pya.anidataloader(hdf5file)
        for data in adl:
            icount += 1
            if icount > 1:
                break
            coorall = data['coordinates']
            ifile = 0
            for coor in coorall:
                ifile += 1
                if ifile > nfile:
                    break
                symbols = data['species']
                specie = set(symbols)
                speciedict = Counter(symbols)
                write.dftbplus().geo_nonpe2(ifile, coor, specie, speciedict)
                os.rename('geo.gen.{}'.format(ifile), 'dftbplus/geo.gen')
                os.system('bash '+diredftb+'/run.sh '+diredftb)


def dftbpy():
    nfile = 50
    icount = 0
    diredftb = os.path.join(directory, 'dftbpy')
    outpara = {}
    outpara['ty'] = 1
    dipolemall = []
    eigvalall = []
    for hdf5file in hdf5filelist:
        adl = pya.anidataloader(hdf5file)
        for data in adl:
            icount += 1
            if icount > 1:
                break
            coorall = data['coordinates']
            ifile = 0
            for coor in coorall:
                ifile += 1
                if ifile > nfile:
                    break
                symbols = data['species']
                outpara['coor'] = coor
                outpara['symbols'] = symbols
                dipolem, eigval = dftb.main(outpara)
                dipolemall.append(dipolem)
                eigvalall.append(eigval)
    with open(os.path.join(diredftb, 'bandenergy.dat'), 'w') as fopen:
        for eigval in eigvalall:
            np.savetxt(fopen, eigval, newline=" ")
            fopen.write('\n')
    with open(os.path.join(diredftb, 'dipole.dat'), 'w') as fopen:
        for dipolem in dipolemall:
            np.savetxt(fopen, dipolem, newline=" ")
            fopen.write('\n')


        
def envir(coorall, nfile):
    rcut = 3
    r_s = 0.8
    eta = 1
    tol = 1E-4
    zeta = 1
    lamda = 1
    env_para = np.zeros((nfile, 10, 2))
    ang_paraall = np.zeros((nfile, 10))
    rad_paraall = np.zeros((nfile, 10))
    for coor in coorall:
        ifile += 1
        if ifile > nfile:
            break
        rad_para, dist_HH, dist_CH = lattice_cell.rad_module2(coor, rcut, r_s, eta, tol, symbols, dist_HH, dist_CH)
        ang_para = lattice_cell.ang_module2(coor, rcut, r_s, eta, zeta, lamda, tol, symbols)
        ang_paraall[ifile-1, 0] = ifile
        ang_paraall[ifile-1, 1:1+natom] = ang_para[:]
        rad_paraall[ifile-1, 0] = ifile
        rad_paraall[ifile-1, 1:1+natom] = rad_para[:]
    with open('rad_para.dat', 'w') as fopen:
        np.savetxt(fopen, rad_paraall, fmt="%s", newline=' ')
        fopen.write('\n')
    with open('ang_para.dat', 'w') as fopen:
        np.savetxt(fopen, ang_paraall, fmt="%s", newline=' ')
        fopen.write('\n')


def dft():
    nfile = 5
    for hdf5file in hdf5filelist:
        adl = pya.anidataloader(hdf5file)
        for data in adl:
            coorall = data['coordinates']
            ifile = 0
            for coor in coorall:
                ifile += 1
                if ifile > nfile:
                    break
                symbols = data['species']
                specie = set(symbols)
                speciedict = Counter(symbols)
                write.FHIaims().geo_nonpe(ifile, coor, specie, speciedict,
                                          symbols)
                os.rename('geometry.in.{}'.format(ifile), 'aims/geometry.in')
                os.system('./aims/aim < aims/control.in | tee aims/aims.out')


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
    print('data1', data1)
    onsite = np.zeros((ncompressr, natom))
    for ir1 in range(0, ncompressr):
        data0 = np.fromfile(fp0, dtype=float, count=natom+3, sep=' ')
        onsite[ir1, :] = data0[3:]/n_char[iatom]
        print('data0', onsite[ir1, :])
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
    print('new_compressr', new_compressr)
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
    task = 'dftbpy'
    main(task)
