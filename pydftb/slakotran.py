#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from dftb_math import DFTBmath
import math



def sk_tran(generalpara):
    '''transfer H and S according to slater-koster rules'''
    atomind = generalpara['atomind']
    natom = generalpara['natom']
    atomname = generalpara['atomnameall']
    distance_vec = generalpara['distance_vec']
    # izp = generalpara['natomtype']
    atomind2 = int(atomind[natom]*(atomind[natom]+1)/2)
    hammat = np.zeros((atomind2))
    overmat = np.zeros((atomind2))
    rr = np.zeros(3)
    for i in range(0, natom):
        lmaxi = generalpara['lmaxall'][i]
        for j in range(0, i+1):
            lmaxj = generalpara['lmaxall'][j]
            lmax = max(lmaxi, lmaxj)
            hams = np.zeros((9, 9))
            ovrs = np.zeros((9, 9))
            generalpara['nameij'] = atomname[i]+atomname[j]
            rr[:] = distance_vec[i, j, :]
            hams, ovrs = slkode(rr, i, j, generalpara, hams,
                                ovrs, lmax, hammat, overmat)
            for n in range(0, int(atomind[j+1] - atomind[j])):
                nn = atomind[j] + n
                for m in range(0, int(atomind[i+1] - atomind[i])):
                    mm = atomind[i] + m
                    idx = int(mm * (mm+1)/2 + nn)
                    if nn <= mm:
                        idx = int(mm*(mm+1)/2 + nn)
                        hammat[idx] = hams[m, n]
                        overmat[idx] = ovrs[m, n]
    return hammat, overmat


def slkode(rr, i, j, generalpara, ham_matrix, s_matrix, lmax, hammat, overmat):
    # here we transfer i from ith atom to ith spiece
    coor = generalpara['coor']
    dd = np.sqrt(rr[0]*rr[0] + rr[1]*rr[1] + rr[2]*rr[2])
    nameij = generalpara['nameij']
    if generalpara['ty'] == 0:
        hs_data = getsk(generalpara, nameij, dd)
    elif generalpara['ty'] == 1:
        hs_data = getsk(generalpara, nameij, dd)
    elif generalpara['ty'] == 5:
        hs_data = generalpara['h_s_all'][i, j, :]
    # print("generalpara['h_s_all']", generalpara['h_s_all'])
    grid_dist = generalpara['grid_dist'+nameij]
    skself = generalpara['onsite']
    cutoff = generalpara['cutoffsk'+nameij]
    skselfnew = np.empty(3)
    xyz = rr[:]
    if dd > cutoff:
        return ham_matrix, s_matrix
    if dd < 1E-4:
        if i != j:
            print("ERROR,distancebetween", i, "atom and", j, "atom is 0")
        else:
            skselfnew[:] = skself[i, :]
        if lmax == 1:
            ham_matrix[0, 0] = skselfnew[2]
            s_matrix[0, 0] = 1.0
        elif lmax == 2:
            ham_matrix[0, 0] = skselfnew[2]
            s_matrix[0, 0] = 1.0
            ham_matrix[1, 1] = skselfnew[1]
            s_matrix[1, 1] = 1.0
            ham_matrix[2, 2] = skselfnew[1]
            s_matrix[2, 2] = 1.0
            ham_matrix[3, 3] = skselfnew[1]
            s_matrix[3, 3] = 1.0
        else:
            ham_matrix[0, 0] = skselfnew[2]
            s_matrix[0, 0] = 1.0
            ham_matrix[1, 1] = skselfnew[1]
            s_matrix[1, 1] = 1.0
            ham_matrix[2, 2] = skselfnew[1]
            s_matrix[2, 2] = 1.0
            ham_matrix[3, 3] = skselfnew[1]
            s_matrix[3, 3] = 1.0
            ham_matrix[4, 4] = skselfnew[0]
            s_matrix[4, 4] = 1.0
            ham_matrix[5, 5] = skselfnew[0]
            s_matrix[5, 5] = 1.0
            ham_matrix[6, 6] = skselfnew[0]
            s_matrix[6, 6] = 1.0
            ham_matrix[7, 7] = skselfnew[0]
            s_matrix[7, 7] = 1.0
            ham_matrix[8, 8] = skselfnew[0]
            s_matrix[8, 8] = 1.0
    else:
        ham_matrix, s_matrix = shpar(generalpara, rr, i, j, dd, xyz, coor,
                                     grid_dist, hs_data, ham_matrix, s_matrix)
    return ham_matrix, s_matrix


def getsk(generalpara, nameij, dd):
    # ninterp is the num of points for interpolation, here is 8
    ninterp = generalpara['ninterp']
    datalist = generalpara['h_s_all'+nameij]
    griddist = generalpara['grid_dist'+nameij]
    cutoff = generalpara['cutoffsk'+nameij]
    ngridpoint = generalpara['ngridpoint'+nameij]
    grid0 = generalpara['grid0']
    ind = int(dd/griddist)
    nlinesk = ngridpoint
    lensk = nlinesk*griddist
    hsdata = np.zeros(20)
    if dd < grid0:
        hsdata[:] = 0
    elif dd >= grid0 and dd < lensk:
        datainterp = np.zeros((int(ninterp), 20))
        ddinterp = np.zeros(int(ninterp))
        nlinesk = min(nlinesk, int(ind+ninterp/2+1))
        nlinesk = max(ninterp, nlinesk)
        for ii in range(0, ninterp):
            ddinterp[ii] = (nlinesk-ninterp+ii)*griddist
        datainterp[:, :] = datalist[nlinesk-ninterp-1:nlinesk-1]
        for ii in range(0, ninterp):
            ddinterp[ii] = (nlinesk-ninterp+ii)*griddist
        datainterp[:, :] = datalist[nlinesk-ninterp-1:nlinesk-1]
        hsdata = DFTBmath().polysk3thsk(datainterp, ddinterp, dd)
    elif dd >= lensk and dd <= cutoff:
        datainterp = np.zeros((ninterp, 20))
        ddinterp = np.zeros(ninterp)
        datainterp[:, :] = datalist[ngridpoint-ninterp:ngridpoint]
        ddinterp = np.linspace((nline-nup)*griddist, (nline+ndown-1)*griddist,
                               num=ninterp)
        hsdata = DFTBmath().polysk5thsk(datainterp, ddinterp, dd)
    else:
        print('Error: the {} distance > cutoff'.format(nameij))
    return hsdata


def shpar(generalpara, rr, i, j, dd, xyz, coor, grid_dist, hs_data, hams,
          ovrs):
    '''if generalpara['ty'] == 0:
        hs_data_str = datalist[np.int(np.round(dd/grid_dist))-1]
        hs_data = np.array(hs_data_str)
    elif generalpara['ty'] == 1:
        hs_data_str = datalist[np.int(np.round(dd/grid_dist))-1]
        hs_data = np.array(hs_data_str)
    elif generalpara['ty'] == 5:
        hs_data = datalist'''
    xyz2 = np.sqrt(xyz[0]*xyz[0] + xyz[1]*xyz[1] + xyz[2]*xyz[2])
    if xyz2 > 1E-4:
        xx = xyz[0]/xyz2
        yy = xyz[1]/xyz2
        zz = xyz[2]/xyz2
    lmaxi = generalpara['lmaxall'][i]
    lmaxj = generalpara['lmaxall'][j]
    maxmax = max(lmaxi, lmaxj)
    minmax = min(lmaxi, lmaxj)
    if maxmax == 1:
        hams, ovrs = skss(i, j, xx, yy, zz, hs_data, hams, ovrs)
    elif maxmax == 2 and minmax == 1:
        hams, ovrs = sksp(i, j, xx, yy, zz, hs_data, hams, ovrs)
    elif maxmax == 2 and minmax == 2:
        hams, ovrs = skpp(i, j, xx, yy, zz, hs_data, hams, ovrs)
    elif maxmax == 3 and minmax == 1:
        hams, ovrs = sksd(i, j, xx, yy, zz, hs_data, hams, ovrs)
    elif maxmax == 3 and minmax == 2:
        hams, ovrs = skpd(i, j, xx, yy, zz, hs_data, hams, ovrs)
    elif maxmax == 3 and minmax == 3:
        hams, ovrs = skdd(i, j, xx, yy, zz, hs_data, hams, ovrs)
    return hams, ovrs


def skss(ii, jj, xx, yy, zz, hs_data, ham_matrix, s_matrix):
    """slater-koster transfermaton for s orvitals"""
    ham_matrix[0, 0] = hs_s_s(ii, jj, xx, yy, zz, hs_data)[0]
    s_matrix[0, 0] = hs_s_s(ii, jj, xx, yy, zz, hs_data)[1]
    return ham_matrix, s_matrix


def sksp(ii, jj, xx, yy, zz, hs_data, ham_matrix, s_matrix):
    ham_matrix, s_matrix = skss(ii, jj, xx, yy, zz, hs_data, ham_matrix,
                                s_matrix)
    ham_matrix[1, 0] = hs_s_x(ii, jj, xx, yy, zz, hs_data)[0]
    ham_matrix[2, 0] = hs_s_y(ii, jj, xx, yy, zz, hs_data)[0]
    ham_matrix[3, 0] = hs_s_z(ii, jj, xx, yy, zz, hs_data)[0]
    s_matrix[1, 0] = hs_s_x(ii, jj, xx, yy, zz, hs_data)[1]
    s_matrix[2, 0] = hs_s_y(ii, jj, xx, yy, zz, hs_data)[1]
    s_matrix[3, 0] = hs_s_z(ii, jj, xx, yy, zz, hs_data)[1]
    for ii in range(1, 3+1):
        ham_matrix[0, ii] = -ham_matrix[ii, 0]
        s_matrix[0, ii] = -s_matrix[ii, 0]
    return ham_matrix, s_matrix


def skpp(ii, jj, xx, yy, zz, hs_data, ham_matrix, s_matrix):
        ham_matrix, s_matrix = sksp(ii, jj, xx, yy, zz, hs_data,
                                    ham_matrix, s_matrix)
        ham_matrix[1, 1] = hs_x_x(ii, jj, xx, yy, zz, hs_data)[0]
        ham_matrix[1, 2] = hs_x_y(ii, jj, xx, yy, zz, hs_data)[0]
        ham_matrix[1, 3] = hs_x_z(ii, jj, xx, yy, zz, hs_data)[0]
        ham_matrix[2, 2] = hs_y_y(ii, jj, xx, yy, zz, hs_data)[0]
        ham_matrix[2, 3] = hs_y_z(ii, jj, xx, yy, zz, hs_data)[0]
        ham_matrix[3, 3] = hs_z_z(ii, jj, xx, yy, zz, hs_data)[0]
        s_matrix[1, 1] = hs_x_x(ii, jj, xx, yy, zz, hs_data)[1]
        s_matrix[1, 2] = hs_x_y(ii, jj, xx, yy, zz, hs_data)[1]
        s_matrix[1, 3] = hs_x_z(ii, jj, xx, yy, zz, hs_data)[1]
        s_matrix[2, 2] = hs_y_y(ii, jj, xx, yy, zz, hs_data)[1]
        s_matrix[2, 3] = hs_y_z(ii, jj, xx, yy, zz, hs_data)[1]
        s_matrix[3, 3] = hs_z_z(ii, jj, xx, yy, zz, hs_data)[1]
        for ii in range(1, 3+1):
            for jj in range(1, ii+1):
                ham_matrix[ii, jj] = ham_matrix[jj, ii]
                s_matrix[ii, jj] = s_matrix[jj, ii]
        return ham_matrix, s_matrix


def sksd(ii, jj, xx, yy, zz, hs_data, ham_matrix, s_matrix):
    ham_matrix, s_matrix = sksp(ii, jj, xx, yy, zz, hs_data,
                                ham_matrix, s_matrix)
    ham_matrix[0, 4] = hs_s_xy(ii, jj, xx, yy, zz, hs_data)[0]
    ham_matrix[0, 5] = hs_s_yz(ii, jj, xx, yy, zz, hs_data)[0]
    ham_matrix[0, 6] = hs_s_xz(ii, jj, xx, yy, zz, hs_data)[0]
    ham_matrix[0, 7] = hs_s_x2y2(ii, jj, xx, yy, zz, hs_data)[0]
    ham_matrix[0, 8] = hs_s_3z2r2(ii, jj, xx, yy, zz, hs_data)[0]
    s_matrix[0, 4] = hs_s_xy(ii, jj, xx, yy, zz, hs_data)[1]
    s_matrix[0, 5] = hs_s_yz(ii, jj, xx, yy, zz, hs_data)[1]
    s_matrix[0, 6] = hs_s_xz(ii, jj, xx, yy, zz, hs_data)[1]
    s_matrix[0, 7] = hs_s_x2y2(ii, jj, xx, yy, zz, hs_data)[1]
    s_matrix[0, 8] = hs_s_3z2r2(ii, jj, xx, yy, zz, hs_data)[1]
    for ii in range(4, 8+1):
        ham_matrix[ii, 0] = ham_matrix[0, ii]
        s_matrix[ii, 0] = s_matrix[0, ii]
    return ham_matrix, s_matrix


def skpd(self, ii, jj, xx, yy, zz, hs_data, ham_matrix, s_matrix):
    ham_matrix, s_matrix = self.skpp(ii, jj, xx, yy, zz, hs_data,
                                     ham_matrix, s_matrix)
    ham_matrix[1, 4] = hs_x_xy(ii, jj, xx, yy, zz, hs_data)[0]
    ham_matrix[1, 5] = hs_x_yz(ii, jj, xx, yy, zz, hs_data)[0]
    ham_matrix[1, 6] = hs_x_xz(ii, jj, xx, yy, zz, hs_data)[0]
    ham_matrix[1, 7] = hs_x_x2y2(ii, jj, xx, yy, zz, hs_data)[0]
    ham_matrix[1, 8] = hs_x_3z2r2(ii, jj, xx, yy, zz, hs_data)[0]
    ham_matrix[2, 4] = hs_y_xy(ii, jj, xx, yy, zz, hs_data)[0]
    ham_matrix[2, 5] = hs_y_yz(ii, jj, xx, yy, zz, hs_data)[0]
    ham_matrix[2, 6] = hs_y_xz(ii, jj, xx, yy, zz, hs_data)[0]
    ham_matrix[2, 7] = hs_y_x2y2(ii, jj, xx, yy, zz, hs_data)[0]
    ham_matrix[2, 8] = hs_y_3z2r2(ii, jj, xx, yy, zz, hs_data)[0]
    ham_matrix[3, 4] = hs_z_xy(ii, jj, xx, yy, zz, hs_data)[0]
    ham_matrix[3, 5] = hs_z_yz(ii, jj, xx, yy, zz, hs_data)[0]
    ham_matrix[3, 6] = hs_z_xz(ii, jj, xx, yy, zz, hs_data)[0]
    ham_matrix[3, 7] = hs_z_x2y2(ii, jj, xx, yy, zz, hs_data)[0]
    ham_matrix[3, 8] = hs_z_3z2r2(ii, jj, xx, yy, zz, hs_data)[0]
    s_matrix[1, 4] = hs_x_xy(ii, jj, xx, yy, zz, hs_data)[1]
    s_matrix[1, 5] = hs_x_yz(ii, jj, xx, yy, zz, hs_data)[1]
    s_matrix[1, 6] = hs_x_xz(ii, jj, xx, yy, zz, hs_data)[1]
    s_matrix[1, 7] = hs_x_x2y2(ii, jj, xx, yy, zz, hs_data)[1]
    s_matrix[1, 8] = hs_x_3z2r2(ii, jj, xx, yy, zz, hs_data)[1]
    s_matrix[2, 4] = hs_y_xy(ii, jj, xx, yy, zz, hs_data)[1]
    s_matrix[2, 5] = hs_y_yz(ii, jj, xx, yy, zz, hs_data)[1]
    s_matrix[2, 6] = hs_y_xz(ii, jj, xx, yy, zz, hs_data)[1]
    s_matrix[2, 7] = hs_y_x2y2(ii, jj, xx, yy, zz, hs_data)[1]
    s_matrix[2, 8] = hs_y_3z2r2(ii, jj, xx, yy, zz, hs_data)[1]
    s_matrix[3, 4] = hs_z_xy(ii, jj, xx, yy, zz, hs_data)[1]
    s_matrix[3, 5] = hs_z_yz(ii, jj, xx, yy, zz, hs_data)[1]
    s_matrix[3, 6] = hs_z_xz(ii, jj, xx, yy, zz, hs_data)[1]
    s_matrix[3, 7] = hs_z_x2y2(ii, jj, xx, yy, zz, hs_data)[1]
    s_matrix[3, 8] = hs_z_3z2r2(ii, jj, xx, yy, zz, hs_data)[1]
    for ii in range(1, 3+1):
        for jj in range(4, 8+1):
            ham_matrix[jj, ii] = -ham_matrix[ii, jj]
            s_matrix[jj, ii] = -s_matrix[ii, jj]
    return ham_matrix, s_matrix


def skdd(self, ii, jj, xx, yy, zz, data, ham_matrix, s_matrix):
    ham_matrix, s_matrix = self.skpd(ii, jj, xx, yy, zz, data,
                                     ham_matrix, s_matrix)
    ham_matrix[4, 4] = hs_xy_xy(ii, jj, xx, yy, zz, data)[0]
    ham_matrix[4, 5] = hs_xy_yz(ii, jj, xx, yy, zz, data)[0]
    ham_matrix[4, 6] = hs_xy_xz(ii, jj, xx, yy, zz, data)[0]
    ham_matrix[4, 7] = hs_xy_x2y2(ii, jj, xx, yy, zz, data)[0]
    ham_matrix[4, 8] = hs_xy_3z2r2(ii, jj, xx, yy, zz, data)[0]
    ham_matrix[5, 5] = hs_yz_yz(ii, jj, xx, yy, zz, data)[0]
    ham_matrix[5, 6] = hs_yz_xz(ii, jj, xx, yy, zz, data)[0]
    ham_matrix[5, 7] = hs_yz_x2y2(ii, jj, xx, yy, zz, data)[0]
    ham_matrix[5, 8] = hs_yz_3z2r2(ii, jj, xx, yy, zz, data)[0]
    ham_matrix[6, 6] = hs_xz_xz(ii, jj, xx, yy, zz, data)[0]
    ham_matrix[6, 7] = hs_xz_x2y2(ii, jj, xx, yy, zz, data)[0]
    ham_matrix[6, 8] = hs_xz_3z2r2(ii, jj, xx, yy, zz, data)[0]
    ham_matrix[7, 7] = hs_x2y2_x2y2(ii, jj, xx, yy, zz, data)[0]
    ham_matrix[7, 8] = hs_x2y2_3z2r2(ii, jj, xx, yy, zz, data)[0]
    ham_matrix[8, 8] = HS_3z2r2_3z2r2(ii, jj, xx, yy, zz, data)[0]
    s_matrix[4, 4] = hs_xy_xy(ii, jj, xx, yy, zz, data)[1]
    s_matrix[4, 5] = hs_xy_yz(ii, jj, xx, yy, zz, data)[1]
    s_matrix[4, 6] = hs_xy_xz(ii, jj, xx, yy, zz, data)[1]
    s_matrix[4, 7] = hs_xy_x2y2(ii, jj, xx, yy, zz, data)[1]
    s_matrix[4, 8] = hs_xy_3z2r2(ii, jj, xx, yy, zz, data)[1]
    s_matrix[5, 5] = hs_yz_yz(ii, jj, xx, yy, zz, data)[1]
    s_matrix[5, 6] = hs_yz_xz(ii, jj, xx, yy, zz, data)[1]
    s_matrix[5, 7] = hs_yz_x2y2(ii, jj, xx, yy, zz, data)[1]
    s_matrix[5, 8] = hs_yz_3z2r2(ii, jj, xx, yy, zz, data)[1]
    s_matrix[6, 6] = hs_xz_xz(ii, jj, xx, yy, zz, data)[1]
    s_matrix[6, 7] = hs_xz_x2y2(ii, jj, xx, yy, zz, data)[1]
    s_matrix[6, 8] = hs_xz_3z2r2(ii, jj, xx, yy, zz, data)[1]
    s_matrix[7, 7] = hs_x2y2_x2y2(ii, jj, xx, yy, zz, data)[1]
    s_matrix[7, 8] = hs_x2y2_3z2r2(ii, jj, xx, yy, zz, data)[1]
    s_matrix[8, 8] = HS_3z2r2_3z2r2(ii, jj, xx, yy, zz, data)[1]
    for ii in range(4, 8+1):
        for jj in range(4, ii+1):
            ham_matrix[ii, jj] = -ham_matrix[jj, ii]
            s_matrix[ii, jj] = -s_matrix[jj, ii]
    return ham_matrix, s_matrix


def hs_s_s(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return hs_data[9], hs_data[19]


def hs_s_x(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return x*hs_data[8], x*hs_data[18]


def hs_s_y(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return y*hs_data[8], y*hs_data[18]


def hs_s_z(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return z*hs_data[8], z*hs_data[18]


def hs_s_xy(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return np.sqrt(3)*x*y*hs_data[7], np.sqrt(3)*x*y*hs_data[17]


def hs_s_yz(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return np.sqrt(3)*y*z*hs_data[7], np.sqrt(3)*y*z*hs_data[17]


def hs_s_xz(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return np.sqrt(3)*x*z*hs_data[7], np.sqrt(3)*x*z*hs_data[17]


def hs_s_x2y2(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return 0.5*np.sqrt(3)*(x**2-y**2)*hs_data[7], \
           0.5*np.sqrt(3)*(x**2-y**2)*hs_data[17]


def hs_s_3z2r2(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return (z**2-0.5*(x**2+y**2))*hs_data[7], \
           (z**2-0.5*(x**2+y**2))*hs_data[17]


def hs_x_s(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return (hs_s_x(atomi_hs_type, atomj_hs_type, -x, -y, -z, hs_data)[0],
            hs_s_x(atomi_hs_type, atomj_hs_type, -x, -y, -z, hs_data)[1])


def hs_x_x(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return x**2*hs_data[5]+(1-x**2)*hs_data[6],\
           x**2*hs_data[15]+(1-x**2)*hs_data[16]


def hs_x_y(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return x*y*hs_data[5]-x*y*hs_data[6], x*y*hs_data[15]-x*y*hs_data[16]


def hs_x_z(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return x*z*hs_data[5]-x*z*hs_data[6], x*z*hs_data[15]-x*z*hs_data[16]


def hs_x_xy(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return (np.sqrt(3)*x**2*y*hs_data[3] + y*(1-2*x**2)*hs_data[4],
            np.sqrt(3)*x**2*y*hs_data[13] + y*(1-2*x**2)*hs_data[14])


def hs_x_yz(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return (np.sqrt(3)*x*y*z*hs_data[3]-2*x*y*z*hs_data[4],
            np.sqrt(3)*x*y*z*hs_data[13]-2*x*y*z*hs_data[14])


def hs_x_xz(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return (np.sqrt(3)*x**2*z*hs_data[3]+z*(1-2*x**2)*hs_data[4],
            np.sqrt(3)*x**2*z*hs_data[13]+z*(1-2*x**2)*hs_data[14])


def hs_x_x2y2(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return (np.sqrt(3)/2*x*(x**2-y**2)*hs_data[3]+x*(1-x**2+y**2)*hs_data[4],
            np.sqrt(3)/2*x*(x**2-y**2)*hs_data[13]+x*(1-x**2+y**2)*hs_data[14])


def hs_x_3z2r2(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return (x*(z**2-0.5*(x**2+y**2))*hs_data[3]-np.sqrt(3)*x*z**2*hs_data[4],
            x*(z**2-0.5*(x**2+y**2))*hs_data[14]-np.sqrt(3)*x*z**2*hs_data[14])


def hs_y_s(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return (hs_s_y(atomi_hs_type, atomj_hs_type, -x, -y, -z, hs_data)[0],
            hs_s_y(atomi_hs_type, atomj_hs_type, -x, -y, -z, hs_data)[1])


def hs_y_x(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return (hs_x_y(atomi_hs_type, atomj_hs_type, -x, -y, -z, hs_data)[0],
            hs_x_y(atomi_hs_type, atomj_hs_type, -x, -y, -z, hs_data)[1])


def hs_y_y(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return y**2*hs_data[5]+(1-y**2)*hs_data[6], \
           y**2*hs_data[15]+(1-y**2)*hs_data[16]


def hs_y_z(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return y*z*hs_data[5]-y*z*hs_data[6], \
           y*z*hs_data[15]-y*z*hs_data[16]


def hs_y_xy(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return (np.sqrt(3)*y**2*x*hs_data[3]+x*(1-2*y**2)*hs_data[4],
            np.sqrt(3)*y**2*x*hs_data[13]+x*(1-2*y**2)*hs_data[14])


def hs_y_yz(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return (np.sqrt(3)*y**2*z*hs_data[3]-z*(1-2*y**2)*hs_data[4],
            np.sqrt(3)*y**2*z*hs_data[13]-z*(1-2*y**2)*hs_data[14])


def hs_y_xz(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return (np.sqrt(3)*x*y*z*hs_data[3]-2*x*y*z*hs_data[4],
            np.sqrt(3)*x*y*z*hs_data[13]-2*x*y*z*hs_data[14])


def hs_y_x2y2(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return (np.sqrt(3)/2*y*(x**2-y**2)*hs_data[3]-y*(1+x**2-y**2)*hs_data[4],
            np.sqrt(3)/2*y*(x**2-y**2)*hs_data[13]-y*(1+x**2-y**2)*hs_data[14])


def hs_y_3z2r2(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return (y*(z**2-0.5*(x**2+y**2))*hs_data[3]-np.sqrt(3)*y*z**2*hs_data[4],
            y*(z**2-0.5*(x**2+y**2))*hs_data[13]-np.sqrt(3)*y*z**2*hs_data[14])


def hs_z_s(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return (hs_s_z(atomi_hs_type, atomj_hs_type, -x, -y, -z, hs_data)[0],
            hs_s_z(atomi_hs_type, atomj_hs_type, -x, -y, -z, hs_data)[1])


def hs_z_x(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return (hs_x_z(atomi_hs_type, atomj_hs_type, -x, -y, -z, hs_data)[0],
            hs_x_z(atomi_hs_type, atomj_hs_type, -x, -y, -z, hs_data)[1])


def hs_z_y(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return (hs_y_z(atomi_hs_type, atomj_hs_type, -x, -y, -z, hs_data)[0],
            hs_y_z(atomi_hs_type, atomj_hs_type, -x, -y, -z, hs_data)[1])


def hs_z_z(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return z**2*hs_data[5]+(1-z**2)*hs_data[6], \
           z**2*hs_data[15]+(1-z**2)*hs_data[16]


def hs_z_xy(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return (np.sqrt(3)*x*y*z*hs_data[3] - 2*x*y*z*hs_data[4],
            np.sqrt(3)*x*y*z*hs_data[13] - 2*x*y*z*hs_data[14])


def hs_z_yz(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return (np.sqrt(3)*z**2*y*hs_data[3] - y*(1-2*z**2)*hs_data[4],
            np.sqrt(3)*z**2*y*hs_data[13] - y*(1-2*z**2)*hs_data[14])


def hs_z_xz(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return (np.sqrt(3)*z**2*x*hs_data[3] - x*(1-2*z**2)*hs_data[4],
            np.sqrt(3)*z**2*x*hs_data[13] - x*(1-2*z**2)*hs_data[14])


def hs_z_x2y2(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return (np.sqrt(3)/2*z*(x**2-y**2)*hs_data[3] - z*(x**2-y**2)*hs_data[4],
            np.sqrt(3)/2*z*(x**2-y**2)*hs_data[13] - z*(x**2-y**2)*hs_data[14])


def hs_z_3z2r2(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return (z*(z**2-0.5*(x**2+y**2))*hs_data[3]+np.sqrt(3)*z*(x**2+y**2)*hs_data[4],
            z*(z**2-0.5*(x**2+y**2))*hs_data[13]+np.sqrt(3)*z*(x**2+y**2)*hs_data[14])


def hs_xy_s(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return (hs_s_xy(atomi_hs_type, atomj_hs_type, -x, -y, -z, hs_data)[0],
            hs_s_xy(atomi_hs_type, atomj_hs_type, -x, -y, -z, hs_data)[1])

 
def hs_xy_x(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return (hs_x_xy(atomi_hs_type, atomj_hs_type, -x, -y, -z, hs_data)[0],
            hs_x_xy(atomi_hs_type, atomj_hs_type, -x, -y, -z, hs_data)[1])


def hs_xy_y(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return (hs_y_xy(atomi_hs_type, atomj_hs_type, -x, -y, -z, hs_data)[0],
            hs_y_xy(atomi_hs_type, atomj_hs_type, -x, -y, -z, hs_data)[1])


def hs_xy_z(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return (hs_z_xy(atomi_hs_type, atomj_hs_type, -x, -y, -z, hs_data)[0],
            hs_z_xy(atomi_hs_type, atomj_hs_type, -x, -y, -z, hs_data)[1])


def hs_xy_xy(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return (3*x**2*y**2*hs_data[0]+(x**2+y**2-4*x**2*y**2)*
            hs_data[1]+(z**2+x**2*y**2)*hs_data[2],
            3*x**2*y**2*hs_data[10] + (x**2+y**2-4*x**2*y**2)*
            hs_data[11]+(z**2+x**2*y**2)*hs_data[12])


def hs_xy_yz(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return (3*x*y**2*z*hs_data[0]+x*z*(1-4*y**2)*hs_data[1]+x*z*(y**2-1)*hs_data[2],
            3*x*y**2*z*hs_data[10]+x*z*(1-4*y**2)*hs_data[11]+x*z*(y**2-1)*hs_data[12])


def hs_xy_xz(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return (3*x**2*y*z*hs_data[0]+y*z*(1-4*x**2)*hs_data[1]+y*z*(x**2-1)*hs_data[2],
            3*x**2*y*z*hs_data[10]+y*z*(1-4*x**2)*hs_data[11]+y*z*(x**2-1)*hs_data[12])


def hs_xy_x2y2(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return (1.5*x*y*(x**2-y**2)*hs_data[0]-2*x*y*(x**2-y**2)*hs_data[1] +
            0.5*x*y*(x**2-y**2)*hs_data[2],
            1.5*x*y*(x**2-y**2)*hs_data[10]-2*x*y*(x**2-y**2)*hs_data[11] +
            0.5*x*y*(x**2-y**2)*hs_data[12])


def hs_xy_3z2r2(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return (np.sqrt(3)*x*y*(z**2-0.5*(x**2+y**2))*hs_data[0] -
            2*np.sqrt(3)*x*y*z**2*hs_data[1]+np.sqrt(3)/2*x*y*(1+z**2)*hs_data[2],
            np.sqrt(3)*x*y*(z**2-0.5*(x**2+y**2))*hs_data[10] -
            2*np.sqrt(3)*x*y*z**2*hs_data[11]+np.sqrt(3)/2*x*y*(1+z**2)*hs_data[12])


def hs_yz_s(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return (hs_s_yz(atomi_hs_type, atomj_hs_type, -x, -y, -z, hs_data)[0],
            hs_s_yz(atomi_hs_type, atomj_hs_type, -x, -y, -z, hs_data)[1])


def hs_yz_x(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return (hs_x_yz(atomi_hs_type, atomj_hs_type, -x, -y, -z, hs_data)[0],
            hs_x_yz(atomi_hs_type, atomj_hs_type, -x, -y, -z, hs_data)[1])


def hs_yz_y(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return (hs_y_yz(atomi_hs_type, atomj_hs_type, -x, -y, -z, hs_data)[0],
            hs_y_yz(atomi_hs_type, atomj_hs_type, -x,-y, -z, hs_data)[1])


def hs_yz_z(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return (hs_z_yz(atomi_hs_type, atomj_hs_type, -x, -y, -z, hs_data)[0],
            hs_z_yz(atomi_hs_type, atomj_hs_type, -x, -y, -z, hs_data)[1])


def hs_yz_xy(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return (hs_xy_yz(atomi_hs_type, atomj_hs_type, -x, -y, -z, hs_data)[0],
            hs_xy_yz(atomi_hs_type, atomj_hs_type, -x, -y, -z, hs_data)[1])


def hs_yz_yz(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return (3*y**2*z**2*hs_data[0] + (y**2+z**2-4*y**2*z**2) *
            hs_data[1]+(x**2+y**2*z**2)*hs_data[2],
            3*y**2*z**2*hs_data[10] + (y**2+z**2-4*y**2*z**2) *
            hs_data[11]+(x**2+y**2*z**2)*hs_data[12])


def hs_yz_xz(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return (3*x*z**2*y*hs_data[0]+x*y*(1-4*z**2)*hs_data[1]+x*y*(z**2-1)*hs_data[2],
            3*x*z**2*y*hs_data[10]+x*y*(1-4*z**2)*hs_data[11]+x*y*(z**2-1)*hs_data[12])


def hs_yz_x2y2(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return (1.5*y*z*(x**2-y**2)*hs_data[0]-y*z*(1+2*(x**2-y**2))*
            hs_data[1]+y*z*(1+0.5*(x**2-y**2))*hs_data[2],
            1.5*y*z*(x**2-y**2)*hs_data[10]-y*z*(1+2*(x**2-y**2))*
            hs_data[11]+y*z*(1+0.5*(x**2-y**2))*hs_data[12])


def hs_yz_3z2r2(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return (np.sqrt(3)*y*z*(z**2-0.5*(x**2+y**2))*hs_data[0] +
            np.sqrt(3)*y*z*(x**2+y**2-z**2)*hs_data[1] -
            np.sqrt(3)/2*y*z*(x**2+y**2)*hs_data[2],
            np.sqrt(3)*y*z*(z**2-0.5*(x**2+y**2)) *
            hs_data[10]+np.sqrt(3)*y*z*(x**2+y**2-z**2)*hs_data[11]-
            np.sqrt(3)/2*y*z*(x**2+y**2)*hs_data[12])


def hs_xz_s(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return (hs_s_xz(atomi_hs_type, atomj_hs_type, -x, -y, -z, hs_data)[0],
            hs_s_xz(atomi_hs_type, atomj_hs_type, -x, -y, -z, hs_data)[1])


def hs_xz_x(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return (hs_x_xz(atomi_hs_type, atomj_hs_type, -x, -y, -z, hs_data)[0],
            hs_x_xz(atomi_hs_type, atomj_hs_type, -x, -y, -z, hs_data)[1])


def hs_xz_y(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return (hs_y_xz(atomi_hs_type, atomj_hs_type, -x, -y, -z, hs_data)[0],
            hs_y_xz(atomi_hs_type, atomj_hs_type, -x, -y, -z, hs_data)[1])


def hs_xz_z(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return (hs_z_xz(atomi_hs_type, atomj_hs_type, -x, -y, -z, hs_data)[0],
            hs_z_xz(atomi_hs_type, atomj_hs_type, -x, -y, -z, hs_data)[1])


def hs_xz_xy(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return (hs_xy_xz(atomi_hs_type, atomj_hs_type, -x, -y, -z, hs_data)[0],
            hs_xy_xz(atomi_hs_type, atomj_hs_type, -x, -y, -z, hs_data)[1])


def hs_xz_yz(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return (hs_yz_xz(atomi_hs_type, atomj_hs_type, -x, -y, -z, hs_data)[0],
            hs_yz_xz(atomi_hs_type, atomj_hs_type, -x, -y, -z, hs_data)[1])


def hs_xz_xz(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return (3*x**2*z**2*hs_data[0] + (x**2+z**2-4*x**2*z**2)*
            hs_data[1]+(y**2+x**2*z**2)*hs_data[2],
            3*x**2*z**2*hs_data[10] + (x**2+z**2-4*x**2*z**2)*
            hs_data[11] + (y**2+x**2*z**2)*hs_data[12])


def hs_xz_x2y2(atomi_hs_type, atomj_hs_type, x,y,z,hs_data):
    return (1.5*x*z*(x**2-y**2)*hs_data[0]+x*z*(1-2*(x**2-y**2))*
            hs_data[1]-x*z*(1-0.5*(x**2-y**2))*hs_data[2],
            1.5*x*z*(x**2-y**2)*hs_data[10]+x*z*(1-2*(x**2-y**2))*
            hs_data[11]-x*z*(1-0.5*(x**2-y**2))*hs_data[12])


def hs_xz_3z2r2(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return (np.sqrt(3)*x*z*(z**2-0.5*(x**2+y**2))*hs_data[0] +
            np.sqrt(3)*x*z*(x**2+y**2-z**2)*hs_data[1] -
            np.sqrt(3)/2*x*z*(x**2+y**2)*hs_data[2],
            np.sqrt(3)*x*z*(z**2-0.5*(x**2+y**2))*hs_data[10] +
            np.sqrt(3)*x*z*(x**2+y**2-z**2)*hs_data[11] -
            np.sqrt(3)/2*x*z*(x**2+y**2)*hs_data[12])


def hs_x2y2_s(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return (hs_s_x2y2(atomi_hs_type, atomj_hs_type, -x, -y, -z, hs_data)[0],
            hs_s_x2y2(atomi_hs_type, atomj_hs_type, -x, -y, -z, hs_data)[1])


def hs_x2y2_x(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
        return (hs_x_x2y2(atomi_hs_type, atomj_hs_type, -x, -y, -z, hs_data)[0],
                hs_x_x2y2(atomi_hs_type, atomj_hs_type, -x, -y, -z, hs_data)[1])


def hs_x2y2_y(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return (hs_y_x2y2(atomi_hs_type, atomj_hs_type, -x, -y, -z, hs_data)[0],
            hs_y_x2y2(atomi_hs_type, atomj_hs_type, -x, -y, -z, hs_data)[1])


def hs_x2y2_z(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return (hs_z_x2y2(atomi_hs_type, atomj_hs_type, -x, -y, -z, hs_data)[0],
            hs_z_x2y2(atomi_hs_type, atomj_hs_type, -x, -y, -z, hs_data)[1])


def hs_x2y2_xy(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return (hs_xy_x2y2(atomi_hs_type, atomj_hs_type, -x, -y, -z, hs_data)[0],
            hs_xy_x2y2(atomi_hs_type, atomj_hs_type, -x, -y, -z, hs_data)[1])


def hs_x2y2_yz(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return (hs_yz_x2y2(atomi_hs_type, atomj_hs_type, -x, -y, -z, hs_data)[0],
            hs_yz_x2y2(atomi_hs_type, atomj_hs_type, -x, -y, -z, hs_data)[1])


def hs_x2y2_xz(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return (hs_xz_x2y2(atomi_hs_type, atomj_hs_type, -x, -y, -z, hs_data)[0],
            hs_xz_x2y2(atomi_hs_type, atomj_hs_type, -x, -y, -z, hs_data)[1])


def hs_x2y2_x2y2(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return (3/4*(x**2-y**2)**2*hs_data[1]+(x**2+y**2 -
            (x**2-y**2)**2)*hs_data[2]+(z**2+1/4*(x**2-y**2)**2)*hs_data[3],
            3/4*(x**2-y**2)**2*hs_data[11]+(x**2+y**2 -
            (x**2-y**2)**2)*hs_data[12]+(z**2+1/4*(x**2-y**2)**2)*hs_data[13])


def hs_x2y2_3z2r2(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return (np.sqrt(3)/2*(x**2-y**2)*(z**2-(x**2+y**2)/2) *
            hs_data[0]+np.sqrt(3)*z**2*(x**2-y**2)*hs_data[1] +
            np.sqrt(3)/4*(1+z**2)*(x**2-y**2)*hs_data[2],
            np.sqrt(3)/2*(x**2-y**2)*(z**2-(x**2+y**2)/2) *
            hs_data[10]+np.sqrt(3)*z**2*(x**2-y**2)*hs_data[11] +
            np.sqrt(3)/4*(1+z**2)*(x**2-y**2)*hs_data[12])


def HS_3z2r2_s(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return (hs_s_3z2r2(atomi_hs_type, atomj_hs_type, -x, -y, -z, hs_data)[0],
            hs_s_3z2r2(atomi_hs_type, atomj_hs_type, -x, -y, -z, hs_data)[1])


def HS_3z2r2_x(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return (hs_x_3z2r2(atomi_hs_type, atomj_hs_type, -x, -y, -z, hs_data)[0],
            hs_x_3z2r2(atomi_hs_type, atomj_hs_type, -x, -y, -z, hs_data)[1])


def HS_3z2r2_y(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return (hs_y_3z2r2(atomi_hs_type, atomj_hs_type, -x, -y, -z, hs_data)[0],
            hs_y_3z2r2(atomi_hs_type, atomj_hs_type, -x, -y, -z, hs_data)[1])


def HS_3z2r2_z(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return (hs_z_3z2r2(atomi_hs_type, atomj_hs_type, -x, -y, -z, hs_data)[0],
            hs_z_3z2r2(atomi_hs_type, atomj_hs_type, -x, -y, -z, hs_data)[1])


def HS_3z2r2_xy(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return (hs_xy_3z2r2(atomi_hs_type, atomj_hs_type, -x, -y, -z, hs_data)[0],
            hs_xy_3z2r2(atomi_hs_type, atomj_hs_type, -x, -y, -z, hs_data)[1])


def HS_3z2r2_yz(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return (hs_yz_3z2r2(atomi_hs_type, atomj_hs_type, -x, -y, -z, hs_data)[0],
            hs_yz_3z2r2(atomi_hs_type, atomj_hs_type, -x, -y, -z, hs_data)[1])


def HS_3z2r2_xz(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return (hs_xz_3z2r2(atomi_hs_type, atomj_hs_type, -x, -y, -z, hs_data)[0],
            hs_xz_3z2r2(atomi_hs_type, atomj_hs_type, -x, -y, -z, hs_data)[1])


def HS_3z2r2_x2y2(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return (hs_x2y2_3z2r2(atomi_hs_type, atomj_hs_type, -x, -y, -z, hs_data)[0],
            hs_x2y2_3z2r2(atomi_hs_type, atomj_hs_type, -x, -y, -z, hs_data))[1]


def HS_3z2r2_3z2r2(atomi_hs_type, atomj_hs_type, x, y, z, hs_data):
    return ((z**2-0.5*(x**2+y**2))**2*hs_data[0]+3*z**2*(x**2+y**2)*
            hs_data[1]+3/4*(x**2+y**2)**2*hs_data[2],
            (z**2-0.5*(x**2+y**2))**2*hs_data[10]+3*z**2*(x**2+y**2)*
            hs_data[11]+3/4*(x**2+y**2)**2*hs_data[12])
