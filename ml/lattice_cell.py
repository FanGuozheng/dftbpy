#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
general = {'HH': 0.4, 'HC': 1, 'CH': 1, 'CC': 1}


def rad_molecule(coor, rcut, r_s, eta, tol):
    row, col = np.shape(coor)
    distance = np.zeros((row, row))
    distij = np.zeros(3)
    rad_para = np.zeros(row)
    for iatom in range(0, row):
        for jatom in range(0, row):
            distij[:] = coor[iatom, 1:]-coor[jatom, 1:]
            distance[iatom, jatom] = np.sqrt(sum(distij[:] ** 2))
            print('distance[iatom, jatom]', distance[iatom, jatom])
            if distance[iatom, jatom] > rcut or distance[iatom, jatom] < tol:
                rad_para[iatom] += 0
            else:
                fc = 0.5 * (np.cos(np.pi * distance[iatom, jatom] / rcut) + 1)
                rad_para[iatom] += np.exp(-eta * (distance[iatom, jatom] - r_s)
                                          ** 2) * fc
        rad_para[iatom] = round(rad_para[iatom], 2)
    return rad_para


def rad_molecule2(coor, rcut, r_s, eta, tol, symbols):
    row, col = np.shape(coor)
    distance = np.zeros((row, row))
    distij = np.zeros(3)
    rad_para = np.zeros(row)
    for iatom in range(0, row):
        for jatom in range(0, row):
            symij = symbols[iatom]+symbols[jatom]
            symij_val = general[symij]
            distij[:] = coor[iatom, 1:] - coor[jatom,  1:]
            distance[iatom, jatom] = np.sqrt(sum(distij[:] ** 2))
            '''
            if distance[iatom, jatom] > 1.4:
                dist_HH.append(distance[iatom, jatom])
            elif distance[iatom, jatom] > 0.5:
                dist_CH.append(distance[iatom, jatom])
            else:
                pass'''
            if distance[iatom, jatom] > rcut or distance[iatom, jatom] < tol:
                rad_para[iatom] += 0
            else:
                fc = 0.5*(np.cos(np.pi*distance[iatom, jatom]/rcut)+1)
                rad_para[iatom] += symij_val*np.exp(-eta*(distance[iatom, jatom]-r_s)**2)*fc
        rad_para[iatom] = round(rad_para[iatom], 4)
    return rad_para


def ang_molecule(coor, rcut, r_s, eta, zeta, lamda, tol):
    eta = 1
    row, col = np.shape(coor)
    d_ij = np.zeros((row, row))
    d_ik = np.zeros((row, row))
    d_jk = np.zeros((row, row))
    d_ij_vec = np.zeros(3)
    d_ik_vec = np.zeros(3)
    d_jk_vec = np.zeros(3)
    ang_para = np.zeros(row)
    ang_para = np.zeros(row)
    for iatom in range(0, row):
        for jatom in range(0, row):
            for katom in range(0, row):
                d_ij_vec[:] = coor[iatom, :]-coor[jatom, :]
                d_ik_vec[:] = coor[iatom, :]-coor[katom, :]
                d_jk_vec[:] = coor[jatom, :]-coor[katom, :]
                d_ij[iatom, jatom] = np.sqrt(sum(d_ij_vec[:]**2))
                d_ik[iatom, katom] = np.sqrt(sum(d_ik_vec[:]**2))
                d_jk[jatom, katom] = np.sqrt(sum(d_jk_vec[:]**2))
                if d_ij[iatom, jatom] > rcut or d_ik[iatom, katom] > rcut or d_jk[jatom, katom] > rcut:
                    ang_para[iatom] += 0
                elif d_ij[iatom, jatom] < tol or d_ik[iatom, katom] < tol or d_jk[jatom, katom] < tol:
                    ang_para[iatom] += 0
                else:
                    fcij = 0.5*(np.cos(np.pi*d_ij[iatom, jatom]/rcut)+1)
                    fcik = 0.5*(np.cos(np.pi*d_ik[iatom, katom]/rcut)+1)
                    fcjk = 0.5*(np.cos(np.pi*d_jk[jatom, katom]/rcut)+1)
                    R_2 = d_ij[iatom, jatom]**2+d_ik[iatom, katom]**2+d_jk[jatom, katom]**2
                    cosijk = np.dot(d_ij_vec, d_ik_vec)/(d_ij[iatom, jatom]*d_ik[iatom, katom])
                    ang = 2**(1-zeta)*(1+lamda*cosijk)**zeta
                    ang_para[iatom] += ang*np.exp(-eta*R_2)*fcij*fcik*fcjk
        ang_para[iatom] = round(ang_para[iatom], 4)
    return ang_para


def ang_molecule2(coor, rcut, r_s, eta, zeta, lamda, tol, symbols):
    '''this function differ the diferent species'''
    row, col = np.shape(coor)
    d_ij = d_ik = d_jk = np.zeros((row, row))
    d_ij_vec, d_ik_vec, d_jk_vec = np.zeros(3), np.zeros(3), np.zeros(3)
    ang_para = np.zeros(row)

    for iatom in range(0, row):
        for jatom in range(0, row):
            for katom in range(0, row):
                d_ij_vec[:] = coor[iatom, 1:]-coor[jatom, 1:]
                d_ik_vec[:] = coor[iatom, 1:]-coor[katom, 1:]
                d_jk_vec[:] = coor[jatom, 1:]-coor[katom, 1:]
                d_ij[iatom, jatom] = np.sqrt(sum(d_ij_vec[:] ** 2))
                d_ik[iatom, katom] = np.sqrt(sum(d_ik_vec[:] ** 2))
                d_jk[jatom, katom] = np.sqrt(sum(d_jk_vec[:] ** 2))
                if d_ij[iatom, jatom] > rcut or d_ik[iatom, katom] > rcut or d_jk[jatom, katom] > rcut:
                    ang_para[iatom] += 0
                elif d_ij[iatom, jatom] < tol or d_ik[iatom, katom] < tol or d_jk[jatom, katom] < tol:
                    ang_para[iatom] += 0
                else:
                    fcij = 0.5*(np.cos(np.pi*d_ij[iatom, jatom]/rcut)+1)
                    fcik = 0.5*(np.cos(np.pi*d_ik[iatom, katom]/rcut)+1)
                    fcjk = 0.5*(np.cos(np.pi*d_jk[jatom, katom]/rcut)+1)
                    R_2 = d_ij[iatom, jatom]**2+d_ik[iatom, katom]**2+d_jk[jatom, katom]**2
                    cosijk = np.dot(d_ij_vec, d_ik_vec)/(d_ij[iatom, jatom]*d_ik[iatom, katom])
                    ang = 2**(1-zeta)*(1+lamda*cosijk)**zeta
                    ang_para[iatom] += ang*np.exp(-eta*R_2)*fcij*fcik*fcjk
        ang_para[iatom] = round(ang_para[iatom], 6)
        # print(ang_para[iatom])
    return ang_para


def rad_ang_molecule(coor, rcut, r_s, eta, zeta, lamda, w_ang, tol):
    row, col = np.shape(coor)
    d_ij = np.zeros((row, row))
    d_ik = np.zeros((row, row))
    d_jk = np.zeros((row, row))
    d_ij_vec = np.zeros(3)
    d_ik_vec = np.zeros(3)
    d_jk_vec = np.zeros(3)
    distance = np.zeros((row, row))
    distij = np.zeros(3)
    rad_para = np.zeros(row)
    ang_para = np.zeros(row)
    radang_para = np.zeros(row)
    for iatom in range(0, row):
        for jatom in range(0, row):
            distij[:] = coor[iatom, :]-coor[jatom, :]
            distance[iatom, jatom] = np.sqrt(sum(distij[:]**2))
            if distance[iatom, jatom] > rcut or distance[iatom, jatom] < tol:
                rad_para[iatom] += 0
            else:
                fc = 0.5*(np.cos(np.pi*distance[iatom, jatom]/rcut)+1)
                rad_para[iatom] += np.exp(-eta*(distance[iatom, jatom]-r_s)**2)*fc
            for katom in range(0, row):
                d_ij_vec[:] = coor[iatom, :]-coor[jatom, :]
                d_ik_vec[:] = coor[iatom, :]-coor[katom, :]
                d_jk_vec[:] = coor[jatom, :]-coor[katom, :]
                d_ij[iatom, jatom] = np.sqrt(sum(d_ij_vec[:]**2))
                d_ik[iatom, katom] = np.sqrt(sum(d_ik_vec[:]**2))
                d_jk[jatom, katom] = np.sqrt(sum(d_jk_vec[:]**2))
                if d_ij[iatom, jatom] > rcut or d_ik[iatom, katom] > rcut:
                    ang_para[iatom] += 0
                elif d_ij[iatom, jatom] < tol or d_ik[iatom, katom] < tol:
                    ang_para[iatom] += 0
                else:
                    fcij = 0.5*(np.cos(np.pi*d_ij[iatom, jatom]/rcut)+1)
                    fcik = 0.5*(np.cos(np.pi*d_ik[iatom, katom]/rcut)+1)
                    fcjk = 0.5*(np.cos(np.pi*d_jk[jatom, katom]/rcut)+1)
                    R_ij_2 = d_ij[iatom, jatom]**2+d_ik[iatom, katom]**2+d_jk[jatom, katom]**2
                    ang = 2**(1-zeta)*(1+lamda*np.dot(d_ij_vec, d_ik_vec)/np.linalg.norm(d_ij_vec)/np.linalg.norm(d_ik_vec))**zeta
                    ang_para[iatom] += ang*np.exp(-eta*R_ij_2)*fcij*fcik*fcjk
        radang_para[iatom] = round(rad_para[iatom]+w_ang*ang_para[iatom], 2)
    return radang_para


def main(coor, unit, rcut, r_s, eta, tol):
    # coorf = coorall[ifile]
    # lattpara = lattparaall[ifile]
    # unit = unitall[ifile]
    row, col = np.shape(coor)
    abc_x, abc_y, abc_z, coorc = abc_xyz(coor, unit)
    nsupercell = np.zeros(3)
    # decide how many supercells along a, b, c
    if abc_x > rcut:
        nsupercell[0] = 3
    else:
        nsupercell[0] = np.ceil(rcut/abc_x)*2+1
    if abc_y > rcut:
        nsupercell[1] = 3
    else:
        nsupercell[1] = np.ceil(rcut/abc_y)*2+1
    if abc_z > rcut:
        nsupercell[2] = 3
    else:
        nsupercell[2] = np.ceil(rcut/abc_z)*2+1
    nsupercell_ = int(nsupercell[0]*nsupercell[1]*nsupercell[2])
    supercoor = create_supercell(nsupercell[0], nsupercell[1],
                                 nsupercell[2], coorc, abc_x, abc_y,
                                 abc_z, type='c')
    coor_ = np.zeros((row, 3))
    # coor_ is the cell in the middle of the supercell
    coor_[:, :] = supercoor[int(nsupercell_/2)*row: int(nsupercell_/2)*row+row, :]
    rad_para = rad_func(coor_, supercoor, nsupercell_, rcut, r_s, eta, tol)
    return rad_para


def rad_func(coor_, supercoor, nsupercell_, rcut, r_s, eta, tol):
    """generate all the radical atomic environment parameters"""
    row, col = np.shape(coor_)
    distance = np.zeros((row, nsupercell_*row))
    distij = np.zeros(3)
    rad_para = np.zeros(row)
    for iatom in range(0, row):
        for jatom in range(0, row*nsupercell_):
            distij[:] = coor_[iatom, :]-supercoor[jatom, :]
            distance[iatom, jatom] = np.sqrt(sum(distij[:]**2))
            if distance[iatom, jatom] > rcut or distance[iatom, jatom] < tol:
                rad_para[iatom] += 0
            else:
                fc = 0.5*(np.cos(np.pi*distance[iatom, jatom]/rcut)+1)
                rad_para[iatom] += np.exp(-eta*(distance[iatom, jatom]-r_s)**2*fc)
        rad_para[iatom] = round(rad_para[iatom], 2)
    return rad_para


def ang_func(coor_, supercoor, nsupercell_, rcut, r_s, eta, tol):
    row, col = np.shape(coor_)
    d_ij = np.zeros((row, nsupercell_*row))
    d_ik = np.zeros((row, nsupercell_*row))
    d_jk = np.zeros((row, nsupercell_*row))
    d_ij_vec = np.zeros(3)
    d_ik_vec = np.zeros(3)
    d_jk_vec = np.zeros(3)
    ang_para = np.zeros(row)
    for iatom in range(0, row):
        for jatom in range(0, row*nsupercell_):
            for katom in range(0, row*nsupercell_):
                d_ij_vec[:] = coor_[iatom, :]-supercoor[jatom, :]
                d_ik_vec[:] = coor_[iatom, :]-supercoor[katom, :]
                d_jk_vec[:] = coor_[jatom, :]-supercoor[katom, :]
                d_ij[iatom, jatom] = np.sqrt(sum(d_ij_vec[:]**2))
                d_ik[iatom, katom] = np.sqrt(sum(d_ik_vec[:]**2))
                d_jk[jatom, katom] = np.sqrt(sum(d_jk_vec[:]**2))
                if d_ij[iatom, jatom] > rcut or d_ik[iatom, katom] > rcut:
                    ang_para[iatom] += 0
                elif d_ij[iatom, jatom] < tol or d_ik[iatom, katom] < tol:
                    ang_para[iatom] += 0
                else:
                    fcij = 0.5*(np.cos(np.pi*d_ij[iatom, jatom]/rcut)+1)
                    fcik = 0.5*(np.cos(np.pi*d_ik[iatom, katom]/rcut)+1)
                    fcjk = 0.5*(np.cos(np.pi*d_jk[jatom, katom]/rcut)+1)
                    R_ij_2 = d_ij[:]**2+d_ik[:]**2+d_jk[:]**2
                    ang_para[iatom] += np.exp(-eta*R_ij_2*fcij*fcik*fcjk)
    return ang_para


def abc_xyz(coorf, unit):
    """transfer fraction to cartesian along x, y, z"""
    row, col = np.shape(coorf)
    coorc = np.zeros((row, col))
    coorc[:, 0] = coorf[:, 0]*unit[0, 0]+coorf[:, 1]*unit[1, 0]+coorf[:, 2]*unit[2, 0]
    coorc[:, 1] = coorf[:, 1]*unit[1, 1]+coorf[:, 2]*unit[2, 1]
    coorc[:, 2] = coorf[:, 2]*unit[2, 2]
    abc_x = unit[0, 0]+unit[1, 0]+unit[2, 0]
    abc_y = unit[1, 1]+unit[2, 1]
    abc_z = unit[2, 2]
    return abc_x, abc_y, abc_z, coorc


def create_supercell(na, nb, nc, coor, abc_x, abc_y, abc_z, type):
    """create upercells
    Input:
    na, nb, nc: how many cells along a, b, c
    coor: primary cell coordination
    abc_x, abc_y, abc_z: projection of a, b, c along x, y, z
    output:
    coordination of supercell(fraction or cartesian)
    """
    row, col = np.shape(coor)
    numsupercell_ = int(na*nb*nc)
    supercoor = np.zeros((numsupercell_*row, 3))
    iline = 0
    for ia in range(0, int(na)):
        for ib in range(0, int(nb)):
            for ic in range(0, int(nc)):
                supercoor[iline*row: (iline+1)*row, :] = coor[:, :]
                supercoor[iline*row: (iline+1)*row, 0] = coor[:, 0]+ia*abc_x
                supercoor[iline*row: (iline+1)*row, 1] = coor[:, 1]+ib*abc_y
                supercoor[iline*row: (iline+1)*row, 2] = coor[:, 2]+ic*abc_z
                iline += 1
    if type == 'c':
        return supercoor
    else:
        supercoor[:, 0] = supercoor[:, 0]/abc_x/na
        supercoor[:, 1] = supercoor[:, 1]/abc_y/nb
        supercoor[:, 2] = supercoor[:, 2]/abc_z/nc
        return supercoor
