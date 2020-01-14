#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from matht import Bspline, polySpline, DFTBmath
from readt import ReadSKt
import torch as t
import matplotlib.pyplot as plt
nls = 1
nlp = 3
nld = 9
VAL_ORB = {"H": 1, "C": 2, "N": 2, "O": 2, "Ti": 3}
HNUM = {'HH': 1, 'HC': 2, 'CC': 4, 'CH': 0}


class SlaKo():

    def __init__(self, para):
        self.para = para
        self.atomspecie = para['atomname_set']

        # use spline method to generate hamiltonian
        if para['splinehs']:
            self.htable_num = para['htable_num']
            self.dist = para['splinedist']
            self.cutoff = para['splinecutoff']
            self.nspecie = len(self.atomspecie)

    def read_skdata(self, para):
        '''
        read and store the SK table raw data, right now only for
        s, p and d oribitals
        '''
        atomname = para['atomnameall']
        for namei in atomname:
            for namej in atomname:
                ReadSKt(para, namei, namej)

    def getSKSplPara(self, para):
        if para['splinetype'] == 'Bspline':
            self.genbsplpara(para)
        elif para['splinetype'] == 'Polyspline':
            self.genpsplpara(para)

    def genbsplpara(self, para):
        lines = int(self.cutoff/self.dist)
        cspline = t.zeros(self.htable_num, lines)
        csplinerand = t.zeros(self.htable_num, lines)
        ihtable = 0
        for ii in range(0, self.nspecie):
            for jj in range(0, self.nspecie):
                namei = self.atomspecie[ii]
                namej = self.atomspecie[jj]
                griddist = para['grid_dist'+namei+namej]
                ngridpoint = para['ngridpoint'+namei+namej]
                t_beg = 0.0
                t_end = ngridpoint*griddist
                t_num = int((t_end - t_beg + griddist)/self.dist)
                print('ngridpoint, griddist', ngridpoint, griddist)
                tspline = t.linspace(t_beg, t_end, t_num)
                cspline = _cspline(para, namei, namej, ihtable, griddist,
                                   cspline, tspline)
                ihtable += HNUM[self.atomspecie[ii]+self.atomspecie[jj]]
                # read all hs data, store in hamname and ovrname
        shape1, shape2 = cspline.shape
        csplinerand = cspline + t.randn(shape1, shape2)/10
        para['cspline'] = cspline
        para['csplinerand'] = csplinerand
        para['tspline'] = tspline

    def genpsplpara(self, para):
        lines = int(self.cutoff/self.dist+1)
        yspline = t.zeros(self.htable_num, lines)
        ysplinerand = t.zeros(self.htable_num, lines)

        # ihtable is label of which orbital and which specie it is
        ihtable = 0
        for ii in range(0, self.nspecie):
            for jj in range(0, self.nspecie):
                namei = self.atomspecie[ii]
                namej = self.atomspecie[jj]
                nameij = namei + namej
                griddist = para['grid_dist'+nameij]
                xx = t.linspace(0, self.cutoff, lines)

                # _yspline will generate the spline para
                yspline = _yspline(para, nameij, ihtable, griddist,
                                   yspline, xx)
                ihtable += HNUM[self.atomspecie[ii]+self.atomspecie[jj]]
        shape1, shape2 = yspline.shape
        ysplinerand = yspline + t.randn(shape1, shape2)/10
        para['yspline'] = yspline
        para['ysplinerand'] = ysplinerand  # the y aixs of spline with randn
        para['yspline_rr'] = xx  # the x aixs of polyspline

    def sk_tranold(self, para):
        '''transfer H and S according to slater-koster rules'''
        atomind = para['atomind']
        natom = para['natom']
        atomname = para['atomnameall']
        distance_vec = para['distance_vec']
        atomind2 = para['atomind2']
        hammat = t.zeros((atomind2))
        overmat = t.zeros((atomind2))
        rr = t.zeros(3)
        for i in range(0, natom):
            lmaxi = para['lmaxall'][i]
            for j in range(0, i+1):
                lmaxj = para['lmaxall'][j]
                lmax = max(lmaxi, lmaxj)
                hams = t.zeros((9, 9))
                ovrs = t.zeros((9, 9))
                para['nameij'] = atomname[i]+atomname[j]
                rr[:] = distance_vec[i, j, :]
                hams, ovrs = slkode(
                        rr, i, j, para, hams, ovrs, lmax, hammat, overmat)
                for n in range(0, int(atomind[j+1] - atomind[j])):
                    nn = atomind[j] + n
                    for m in range(0, int(atomind[i+1] - atomind[i])):
                        mm = atomind[i] + m
                        idx = int(mm * (mm+1)/2 + nn)
                        if nn <= mm:
                            idx = int(mm*(mm+1)/2 + nn)
                            hammat[idx] = hams[m, n]
                            overmat[idx] = ovrs[m, n]
        para['hammat'] = hammat
        para['overmat'] = overmat
        return para


def _cspline(para, namei, namej, itable, ngridpoint, c_spline, t_spline):
    datalist = para['h_s_all'+namei+namej]
    nlinesk = para['ngridpoint'+namei+namej]
    dist = para['splinedist']
    ninterval = int(dist/ngridpoint)
    nhtable = HNUM[namei+namej]
    print('ninterval', ninterval, nlinesk)
    datalist_arr = np.asarray(datalist)
    if nhtable == 1:

        # the default distance diff in .skf is 0.02, we can set flexible
        # distance by para: splinedist, ngridpoint
        for ii in range(0, nlinesk):
            if ii % ninterval == 0:
                c_spline[itable, int(ii/ninterval)] = datalist_arr[ii, 9]

        xx = t.linspace(0, 4, 21)
        fig, ax = plt.subplots()
        ax.plot(xx, [Bspline().bspline(x, t_spline, c_spline[0, :], 2) for x in xx],
                     'r-', lw=3, label='spline')
        ax.plot(xx, c_spline[0, 0:21], 'y-', lw=3, label='spline')
        xx = t.linspace(1.0, 4, 151)
        ax.plot(xx, datalist_arr[50:201, 9], 'b-', lw=3, label='original')
        print(Bspline().bspline(1, t_spline, c_spline[0, :], 2),
                 c_spline[0, 5], datalist_arr[50, 9])
        plt.show()
        
    elif nhtable == 2:
        # s and porbital
        # datalist_arr = np.asarray(datalist)
        for ii in range(0, nlinesk):
            if ii % ninterval == 0:
                c_spline[itable, int(ii/ninterval)] = datalist_arr[ii, 9]
                c_spline[itable+1, int(ii/ninterval)] = datalist_arr[ii, 8]
    elif nhtable == 4:
        # datalist_arr = np.asarray(datalist)
        # the squeues is ss0, sp0, pp0, pp1
        for ii in range(0, nlinesk):
            if ii % ninterval == 0:
                c_spline[itable, int(ii/ninterval)] = datalist_arr[ii, 9]
                c_spline[itable+1, int(ii/ninterval)] = datalist_arr[ii, 8]
                c_spline[itable+2, int(ii/ninterval)] = datalist_arr[ii, 5]
                c_spline[itable+3, int(ii/ninterval)] = datalist_arr[ii, 6]
    elif nhtable == 0:
        pass
    return c_spline


def _yspline(para, nameij, itable, ngridpoint, yspline, rr):
    datalist = para['h_s_all'+nameij]
    nlinesk = para['ngridpoint'+nameij]
    dist = para['splinedist']
    ninterval = int(dist/ngridpoint)
    nhtable = HNUM[nameij]
    print('ninterval', ninterval, nlinesk)
    datalist_arr = np.asarray(datalist)

    # ss0 orbital (e.g. H-H)
    if nhtable == 1:
        for ii in range(0, nlinesk):
            if ii % ninterval == 0:
                yspline[itable, int(ii/ninterval)] = datalist_arr[ii, 9]

        # the following is for test (ss0)
        xx = t.linspace(0, 4, 21)
        fig, ax = plt.subplots()
        ax.plot(xx, [polySpline(rr, yspline[0, :], x).cubic() for x in xx],
                'r-', lw=3, label='spline')
        ax.plot(xx, yspline[0, 0:21], 'y-', lw=3, label='spline')
        xx = t.linspace(1.0, 4, 151)
        ax.plot(xx, datalist_arr[50:201, 9], 'b-', lw=3, label='original')
        plt.show()

    # ss0 and sp0 orbital (e.g. C-H)
    elif nhtable == 2:
        for ii in range(0, nlinesk):
            if ii % ninterval == 0:
                yspline[itable, int(ii/ninterval)] = datalist_arr[ii, 9]
                yspline[itable+1, int(ii/ninterval)] = datalist_arr[ii, 8]

    # ss0, sp0, pp0, pp1 orbital (e.g. C-H)
    elif nhtable == 4:
        for ii in range(0, nlinesk):
            if ii % ninterval == 0:
                yspline[itable, int(ii/ninterval)] = datalist_arr[ii, 9]
                yspline[itable+1, int(ii/ninterval)] = datalist_arr[ii, 8]
                yspline[itable+2, int(ii/ninterval)] = datalist_arr[ii, 5]
                yspline[itable+3, int(ii/ninterval)] = datalist_arr[ii, 6]
    return yspline


def slkode(rr, i, j, generalpara, ham, ovr, lmax, hammat, overmat):
    # here we transfer i from ith atom to ith spiece
    nameij = generalpara['nameij']
    dd = t.sqrt(rr[0]*rr[0] + rr[1]*rr[1] + rr[2]*rr[2])
    if generalpara['ty'] == 0 or generalpara['ty'] == 1 or generalpara['ty'] == 7:
        hs_data = getsk(generalpara, nameij, dd)
    elif generalpara['ty'] == 5:
        hs_data = generalpara['h_s_all'][i, j, :]
    cutoff = generalpara['cutoffsk'+nameij]
    skselfnew = t.zeros(3)
    if dd > cutoff:
        return ham, ovr
    if dd < 1E-4:
        if i != j:
            print("ERROR,distancebetween", i, "atom and", j, "atom is 0")
        else:
            skselfnew[:] = t.FloatTensor(generalpara['Espd_Uspd'+nameij][0:3])
        if lmax == 1:
            ham[0, 0] = skselfnew[2]
            ovr[0, 0] = 1.0
        elif lmax == 2:
            ham[0, 0] = skselfnew[2]
            ovr[0, 0] = 1.0
            ham[1, 1] = skselfnew[1]
            ovr[1, 1] = 1.0
            ham[2, 2] = skselfnew[1]
            ovr[2, 2] = 1.0
            ham[3, 3] = skselfnew[1]
            ovr[3, 3] = 1.0
        else:
            ham[0, 0] = skselfnew[2]
            ovr[0, 0] = 1.0
            ham[1, 1] = skselfnew[1]
            ovr[1, 1] = 1.0
            ham[2, 2] = skselfnew[1]
            ovr[2, 2] = 1.0
            ham[3, 3] = skselfnew[1]
            ovr[3, 3] = 1.0
            ham[4, 4] = skselfnew[0]
            ovr[4, 4] = 1.0
            ham[5, 5] = skselfnew[0]
            ovr[5, 5] = 1.0
            ham[6, 6] = skselfnew[0]
            ovr[6, 6] = 1.0
            ham[7, 7] = skselfnew[0]
            ovr[7, 7] = 1.0
            ham[8, 8] = skselfnew[0]
            ovr[8, 8] = 1.0
    else:
        if generalpara['ty'] == 7:
            ham, ovr = shparspline(generalpara, rr, i, j, dd, hs_data,
                                   ham, ovr)
        else:
            ham, ovr = shpar(generalpara, rr, i, j, dd, hs_data, ham, ovr)
        # print(i, j, ham)
    return ham, ovr


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
    hsdata = t.zeros(20)
    if dd < grid0:
        hsdata[:] = 0
    elif dd >= grid0 and dd < lensk:
        datainterp = t.zeros((int(ninterp), 20))
        ddinterp = t.zeros(int(ninterp))
        nlinesk = min(nlinesk, int(ind+ninterp/2+1))
        nlinesk = max(ninterp, nlinesk)
        for ii in range(0, ninterp):
            ddinterp[ii] = (nlinesk-ninterp+ii)*griddist
        datainterp[:, :] = t.from_numpy(
                np.array(datalist[nlinesk-ninterp-1:nlinesk-1]))
        hsdata = DFTBmath().polysk3thsk(datainterp, ddinterp, dd)
    elif dd >= lensk and dd <= cutoff:
        datainterp = t.zeros(ninterp, 20)
        ddinterp = t.zeros(ninterp)
        datainterp[:, :] = datalist[ngridpoint-ninterp:ngridpoint]
        ddinterp = t.linspace((nline-nup)*griddist, (nline+ndown-1)*griddist,
                               num=ninterp)
        hsdata = DFTBmath(generalpara).polysk5thsk(datainterp, ddinterp, dd)
    else:
        print('Error: the {} distance > cutoff'.format(nameij))
    return hsdata


def shpar(generalpara, xyz, i, j, dd, hs_data, hams, ovrs):
    xx = xyz[0]/dd
    yy = xyz[1]/dd
    zz = xyz[2]/dd
    lmaxi = generalpara['lmaxall'][i]
    lmaxj = generalpara['lmaxall'][j]
    maxmax = max(lmaxi, lmaxj)
    minmax = min(lmaxi, lmaxj)
    if maxmax == 1:
        skss(xx, yy, zz, hs_data, hams, ovrs)
    elif maxmax == 2 and minmax == 1:
        sksp(xx, yy, zz, hs_data, hams, ovrs)
        # print(i, j, 'hams[0, 0]', hams[0, 0])
    elif maxmax == 2 and minmax == 2:
        skpp(xx, yy, zz, hs_data, hams, ovrs)
    elif maxmax == 3 and minmax == 1:
        sksd(xx, yy, zz, hs_data, hams, ovrs)
    elif maxmax == 3 and minmax == 2:
        skpd(xx, yy, zz, hs_data, hams, ovrs)
    elif maxmax == 3 and minmax == 3:
        skdd(xx, yy, zz, hs_data, hams, ovrs)
    return hams, ovrs


def shparspline(generalpara, xyz, i, j, dd, hs_data, hams, ovrs):
    xx = xyz[0]/dd
    yy = xyz[1]/dd
    zz = xyz[2]/dd
    lmaxi = generalpara['lmaxall'][i]
    lmaxj = generalpara['lmaxall'][j]
    namei = generalpara['atomnameall'][i]
    namej = generalpara['atomnameall'][j]
    # here we need revise !!!!!!!!!!
    nameij = namei + namej
    maxmax = max(lmaxi, lmaxj)
    minmax = min(lmaxi, lmaxj)
    if generalpara['splinetype'] == 'Polyspline':
        yspline = generalpara['yspline']
        ysplinex = generalpara['yspline_rr']
        if nameij == 'HH':
            spliney = yspline[0, :]
            splinex = ysplinex
        elif nameij == 'CH' or nameij == 'HC':
            spliney = yspline[1:3, :]
            splinex = ysplinex
        elif nameij == 'CC':
            spliney = yspline[3:7, :]
            splinex = ysplinex
        if maxmax == 1:
            sksspspline(xx, yy, zz, dd, splinex, spliney, hs_data, hams, ovrs)
        elif maxmax == 2 and minmax == 1:
            sksppspline(xx, yy, zz, dd, splinex, spliney, hs_data, hams, ovrs)
        elif maxmax == 2 and minmax == 2:
            skpppspline(xx, yy, zz, dd, splinex, spliney, hs_data, hams, ovrs)
        elif maxmax == 3 and minmax == 1:
            sksd(xx, yy, zz, hs_data, hams, ovrs)
        elif maxmax == 3 and minmax == 2:
            skpd(xx, yy, zz, hs_data, hams, ovrs)
        elif maxmax == 3 and minmax == 3:
            skdd(xx, yy, zz, hs_data, hams, ovrs)
    elif generalpara['splinetype'] == 'Bspline':
        tspline = generalpara['tspline']
        cspline = generalpara['cspline']
        if nameij == 'HH':
            spline = cspline[0, :]
        elif nameij == 'CH' or nameij == 'HC':
            spline = cspline[1:3, :]
        elif nameij == 'CC':
            spline = cspline[3:7, :]
        if maxmax == 1:
            skssbspline(xx, yy, zz, dd, tspline, spline, hs_data, hams, ovrs)
        elif maxmax == 2 and minmax == 1:
            skspbspline(xx, yy, zz, dd, tspline, spline, hs_data, hams, ovrs)
        elif maxmax == 2 and minmax == 2:
            skppbspline(xx, yy, zz, dd, tspline, spline, hs_data, hams, ovrs)
        elif maxmax == 3 and minmax == 1:
            sksd(xx, yy, zz, hs_data, hams, ovrs)
        elif maxmax == 3 and minmax == 2:
            skpd(xx, yy, zz, hs_data, hams, ovrs)
        elif maxmax == 3 and minmax == 3:
            skdd(xx, yy, zz, hs_data, hams, ovrs)
    return hams, ovrs


def skss(xx, yy, zz, data, ham, ovr):
    """slater-koster transfermaton for s orvitals"""
    ham[0, 0], ovr[0, 0] = hs_s_s(xx, yy, zz, data[9], data[19])
    return ham, ovr


def skssbspline(xx, yy, zz, dd, t, c, data, ham, ovr):
    """slater-koster transfermaton for s orvitals"""
    h_data = Bspline().bspline(dd, t, c, 2)
    ham[0, 0], ovr[0, 0] = hs_s_s(xx, yy, zz, h_data, data[19])
    return ham, ovr


def sksspspline(xx, yy, zz, dd, splx, sply, data, ham, ovr):
    """slater-koster transfermaton for s orvitals"""
    h_data = polySpline(splx, sply, dd).cubic()
    ham[0, 0], ovr[0, 0] = hs_s_s(xx, yy, zz, h_data, data[19])
    return ham, ovr


def sksp(xx, yy, zz, data, ham, ovr):
    ham, ovr = skss(xx, yy, zz, data, ham, ovr)
    ham[1, 0], ovr[1, 0] = hs_s_x(xx, yy, zz, data[8], data[18])
    ham[2, 0], ovr[2, 0] = hs_s_y(xx, yy, zz, data[8], data[18])
    ham[3, 0], ovr[3, 0] = hs_s_z(xx, yy, zz, data[8], data[18])
    # print('Hsp', data[8])
    for ii in range(nls, nlp+nls):
        ham[0, ii] = -ham[ii, 0]
        ovr[0, ii] = -ovr[ii, 0]
    return ham, ovr


def skspbspline(xx, yy, zz, dd, t, c, data, ham, ovr):
    ham, ovr = skssbspline(xx, yy, zz, dd, t, c[0, :], data, ham, ovr)
    h_data = Bspline().bspline(dd, t, c[1, :], 2)
    ham[1, 0], ovr[1, 0] = hs_s_x(xx, yy, zz, h_data, data[18])
    ham[2, 0], ovr[2, 0] = hs_s_y(xx, yy, zz, h_data, data[18])
    ham[3, 0], ovr[3, 0] = hs_s_z(xx, yy, zz, h_data, data[18])
    for ii in range(nls, nlp+nls):
        ham[0, ii] = -ham[ii, 0]
        ovr[0, ii] = -ovr[ii, 0]
    return ham, ovr


def sksppspline(xx, yy, zz, dd, splx, sply, data, ham, ovr):
    ham, ovr = skssbspline(xx, yy, zz, dd, splx, sply[0, :], data, ham, ovr)
    h_data = polySpline(splx, sply[1, :], dd).cubic()
    ham[1, 0], ovr[1, 0] = hs_s_x(xx, yy, zz, h_data, data[18])
    ham[2, 0], ovr[2, 0] = hs_s_y(xx, yy, zz, h_data, data[18])
    ham[3, 0], ovr[3, 0] = hs_s_z(xx, yy, zz, h_data, data[18])
    for ii in range(nls, nlp+nls):
        ham[0, ii] = -ham[ii, 0]
        ovr[0, ii] = -ovr[ii, 0]
    return ham, ovr


def sksd(xx, yy, zz, data, ham, ovr):
    ham, ovr = sksp(xx, yy, zz, data, ham, ovr)
    ham[0, 4], ovr[0, 4] = hs_s_xy(xx, yy, zz, data[7], data[17])
    ham[0, 5], ovr[0, 5] = hs_s_yz(xx, yy, zz, data[7], data[17])
    ham[0, 6], ovr[0, 6] = hs_s_xz(xx, yy, zz, data[7], data[17])
    ham[0, 7], ovr[0, 7] = hs_s_x2y2(xx, yy, zz, data[7], data[17])
    ham[0, 8], ovr[0, 8] = hs_s_3z2r2(xx, yy, zz, data[7], data[17])
    for ii in range(nls+nlp, nld):
        ham[ii, 0] = ham[0, ii]
        ovr[ii, 0] = ovr[0, ii]
    return ham, ovr


def skpp(xx, yy, zz, dd, t, c, data, ham, ovr):
    ham, ovr = sksp(xx, yy, zz, data, ham, ovr)
    ham[1, 1], ovr[1, 1] = hs_x_x(
            xx, yy, zz, data[5], data[15], data[6], data[16])
    ham[1, 2], ovr[1, 2] = hs_x_y(
            xx, yy, zz, data[5], data[15], data[6], data[16])
    ham[1, 3], ovr[1, 3] = hs_x_z(
            xx, yy, zz, data[5], data[15], data[6], data[16])
    ham[2, 2], ovr[2, 2] = hs_y_y(
            xx, yy, zz, data[5], data[15], data[6], data[16])
    ham[2, 3], ovr[2, 3] = hs_y_z(
            xx, yy, zz, data[5], data[15], data[6], data[16])
    ham[3, 3], ovr[3, 3] = hs_z_z(
            xx, yy, zz, data[5], data[15], data[6], data[16])
    for ii in range(nls, nlp+nls):
        for jj in range(nls, ii+nls):
            ham[ii, jj] = ham[jj, ii]
            ovr[ii, jj] = ovr[jj, ii]
    return ham, ovr


def skppbspline(xx, yy, zz, dd, t, c, data, ham, ovr):
    ham, ovr = skspbspline(xx, yy, zz, dd, t, c[0:2, :], data, ham, ovr)
    h_pp0 = Bspline().bspline(dd, t, c[2, :], 2)
    h_pp1 = Bspline().bspline(dd, t, c[3, :], 2)
    ham[1, 1], ovr[1, 1] = hs_x_x(xx, yy, zz, h_pp0, data[15], h_pp1, data[16])
    ham[1, 2], ovr[1, 2] = hs_x_y(xx, yy, zz, h_pp0, data[15], h_pp1, data[16])
    ham[1, 3], ovr[1, 3] = hs_x_z(xx, yy, zz, h_pp0, data[15], h_pp1, data[16])
    ham[2, 2], ovr[2, 2] = hs_y_y(xx, yy, zz, h_pp0, data[15], h_pp1, data[16])
    ham[2, 3], ovr[2, 3] = hs_y_z(xx, yy, zz, h_pp0, data[15], h_pp1, data[16])
    ham[3, 3], ovr[3, 3] = hs_z_z(xx, yy, zz, h_pp0, data[15], h_pp1, data[16])
    for ii in range(nls, nlp+nls):
        for jj in range(nls, ii+nls):
            ham[ii, jj] = ham[jj, ii]
            ovr[ii, jj] = ovr[jj, ii]
    return ham, ovr


def skpppspline(xx, yy, zz, dd, splx, sply, data, ham, ovr):
    ham, ovr = skspbspline(xx, yy, zz, dd, splx, sply[0:2, :], data, ham, ovr)
    h_pp0 = polySpline(splx, sply[2, :], dd).cubic()
    h_pp1 = polySpline(splx, sply[3, :], dd).cubic()
    ham[1, 1], ovr[1, 1] = hs_x_x(xx, yy, zz, h_pp0, data[15], h_pp1, data[16])
    ham[1, 2], ovr[1, 2] = hs_x_y(xx, yy, zz, h_pp0, data[15], h_pp1, data[16])
    ham[1, 3], ovr[1, 3] = hs_x_z(xx, yy, zz, h_pp0, data[15], h_pp1, data[16])
    ham[2, 2], ovr[2, 2] = hs_y_y(xx, yy, zz, h_pp0, data[15], h_pp1, data[16])
    ham[2, 3], ovr[2, 3] = hs_y_z(xx, yy, zz, h_pp0, data[15], h_pp1, data[16])
    ham[3, 3], ovr[3, 3] = hs_z_z(xx, yy, zz, h_pp0, data[15], h_pp1, data[16])
    for ii in range(nls, nlp+nls):
        for jj in range(nls, ii+nls):
            ham[ii, jj] = ham[jj, ii]
            ovr[ii, jj] = ovr[jj, ii]
    return ham, ovr


def skpd(self, xx, yy, zz, data, ham, ovr):
    ham, ovr = self.skpp(xx, yy, zz, data, ham, ovr)
    ham[1, 4], ovr[1, 4] = hs_x_xy(
            xx, yy, zz, data[3], data[13], data[4], data[14])
    ham[1, 5], ovr[1, 5] = hs_x_yz(
            xx, yy, zz, data[3], data[13], data[4], data[14])
    ham[1, 6], ovr[1, 6] = hs_x_xz(
            xx, yy, zz, data[3], data[13], data[4], data[14])
    ham[1, 7], ovr[1, 7] = hs_x_x2y2(
            xx, yy, zz, data[3], data[13], data[4], data[14])
    ham[1, 8], ovr[1, 8] = hs_x_3z2r2(
            xx, yy, zz, data[3], data[13], data[4], data[14])
    ham[2, 4], ovr[2, 4] = hs_y_xy(
            xx, yy, zz, data[3], data[13], data[4], data[14])
    ham[2, 5], ovr[2, 5] = hs_y_yz(
            xx, yy, zz, data[3], data[13], data[4], data[14])
    ham[2, 6], ovr[2, 6] = hs_y_xz(
            xx, yy, zz, data[3], data[13], data[4], data[14])
    ham[2, 7], ovr[2, 7] = hs_y_x2y2(
            xx, yy, zz, data[3], data[13], data[4], data[14])
    ham[2, 8], ovr[2, 8] = hs_y_3z2r2(
            xx, yy, zz, data[3], data[13], data[4], data[14])
    ham[3, 4], ovr[3, 4] = hs_z_xy(
            xx, yy, zz, data[3], data[13], data[4], data[14])
    ham[3, 5], ovr[3, 5] = hs_z_yz(
            xx, yy, zz, data[3], data[13], data[4], data[14])
    ham[3, 6], ovr[3, 6] = hs_z_xz(
            xx, yy, zz, data[3], data[13], data[4], data[14])
    ham[3, 7], ovr[3, 7] = hs_z_x2y2(
            xx, yy, zz, data[3], data[13], data[4], data[14])
    ham[3, 8], ovr[3, 8] = hs_z_3z2r2(
            xx, yy, zz, data[3], data[13], data[4], data[14])
    for ii in range(nls, nls+nlp):
        for jj in range(nls+nlp, nld):
            ham[jj, ii] = -ham[ii, jj]
            ovr[jj, ii] = -ovr[ii, jj]
    return ham, ovr


def skdd(self, xx, yy, zz, data, ham, ovr):
    ham, ovr = self.skpd(xx, yy, zz, data, ham, ovr)
    ham[4, 4], ovr[4, 4] = hs_xy_xy(
            xx, yy, zz, data[0], data[10], data[1], data[11], data[2], data[12])
    ham[4, 5], ovr[4, 5] = hs_xy_yz(
            xx, yy, zz, data[0], data[10], data[1], data[11], data[2], data[12])
    ham[4, 6], ovr[4, 6] = hs_xy_xz(
            xx, yy, zz, data[0], data[10], data[1], data[11], data[2], data[12])
    ham[4, 7], ovr[4, 7] = hs_xy_x2y2(
            xx, yy, zz, data[0], data[10], data[1], data[11], data[2], data[12])
    ham[4, 8], ovr[4, 8] = hs_xy_3z2r2(
            xx, yy, zz, data[0], data[10], data[1], data[11], data[2], data[12])
    ham[5, 5], ovr[5, 5] = hs_yz_yz(
            xx, yy, zz, data[0], data[10], data[1], data[11], data[2], data[12])
    ham[5, 6], ovr[5, 6] = hs_yz_xz(
            xx, yy, zz, data[0], data[10], data[1], data[11], data[2], data[12])
    ham[5, 7], ovr[5, 7] = hs_yz_x2y2(
            xx, yy, zz, data[0], data[10], data[1], data[11], data[2], data[12])
    ham[5, 8], ovr[5, 8] = hs_yz_3z2r2(
            xx, yy, zz, data[0], data[10], data[1], data[11], data[2], data[12])
    ham[6, 6], ovr[6, 6] = hs_xz_xz(
            xx, yy, zz, data[0], data[10], data[1], data[11], data[2], data[12])
    ham[6, 7], ovr[6, 7] = hs_xz_x2y2(
            xx, yy, zz, data[0], data[10], data[1], data[11], data[2], data[12])
    ham[6, 8], ovr[6, 8] = hs_xz_3z2r2(
            xx, yy, zz, data[0], data[10], data[1], data[11], data[2], data[12])
    ham[7, 7], ovr[7, 7] = hs_x2y2_x2y2(
            xx, yy, zz, data[0], data[10], data[1], data[11], data[2], data[12])
    ham[7, 8], ovr[7, 8] = hs_x2y2_3z2r2(
            xx, yy, zz, data[0], data[10], data[1], data[11], data[2], data[12])
    ham[8, 8], ovr[8, 8] = hs_3z2r2_3z2r2(
            xx, yy, zz, data[0], data[10], data[1], data[11], data[2], data[12])
    for ii in range(nls+nlp, nld):
        for jj in range(nls+nlp, ii+nls):
            ham[ii, jj] = ham[jj, ii]
            ovr[ii, jj] = ovr[jj, ii]
    return ham, ovr


def getSKTable(generalpara):
    '''In hamtable, the first line is '''
    atomind = generalpara['atomind']
    natom = generalpara['natom']
    atomname = generalpara['atomnameall']
    distance_vec = generalpara['distance_vec']
    atomind2 = generalpara['atomind2']
    hamtable = t.zeros(atomind2, 3)
    ovrtable = t.zeros(atomind2, 3)
    rr_ij = t.zeros(atomind2, 3)
    dd = t.zeros(atomind2)
    haminfo = t.zeros(atomind2)
    ovrinfo = t.zeros(atomind2)
    rr = t.zeros(3)
    for i in range(0, natom):
        lmaxi = generalpara['lmaxall'][i]
        for j in range(0, i+1):
            lmaxj = generalpara['lmaxall'][j]
            lmax = max(lmaxi, lmaxj)
            ham = t.zeros(4, 9, 9)
            ovr = t.zeros(4, 9, 9)
            generalpara['nameij'] = atomname[i]+atomname[j]
            rr[:] = distance_vec[i, j, :]
            hams, ovrs = HS_dist(
                    rr, i, j, generalpara, ham, ovr, lmax)
            for n in range(0, int(atomind[j+1] - atomind[j])):
                nn = atomind[j] + n
                for m in range(0, int(atomind[i+1] - atomind[i])):
                    mm = atomind[i] + m
                    idx = int(mm * (mm+1)/2 + nn)
                    if nn <= mm:
                        idx = int(mm*(mm+1)/2 + nn)
                        hamtable[idx, :] = ham[1:, m, n]
                        ovrtable[idx, :] = ovr[1:, m, n]
                        rr_ij[idx, :] = rr[:]
                        dd[idx] = generalpara['distance'][i, j]
                        haminfo[idx] = ham[0, m, n]
                        ovrinfo[idx] = ovr[0, m, n]
    generalpara['hamtable'] = hamtable
    generalpara['ovrtable'] = ovrtable
    generalpara['haminfo'] = haminfo
    generalpara['ovrinfo'] = ovrinfo
    generalpara['rr_ij'] = rr_ij
    generalpara['dd_ij'] = dd
    return generalpara


def HS_dist(rr, i, j, generalpara, ham, ovr, lmax):
    # haminfo record the info of the corresponding hamiltonian,
    # for s, p, d onsite, haminfo = 0, 1, 2
    # for ss, sp1, sp2, sp3 integral, haminfo = 10, 11, 12, 13
    # for sd1, sd2, sd3, sd4, sd5, orbital, haminfo is 14, 15, 16, 17, 18
    # for p1s, p1p1, p1p2, p1p3, orbital, haminfo is 20, 21, 22, 23
    # for p2s, p2p1, p2p2, p2p3, orbital, haminfo is 30, 31, 32, 33
    # for p3s, p3p1, p3p2, p3p3, orbital, haminfo is 40, 41, 42, 43
    dd = t.sqrt(rr[0]*rr[0] + rr[1]*rr[1] + rr[2]*rr[2])
    nameij = generalpara['nameij']
    if generalpara['ty'] == 6:
        hs_data = getsk(generalpara, nameij, dd)
    elif generalpara['ty'] == 5:
        hs_data = generalpara['h_s_all'][i, j, :]
    skself = generalpara['onsite']
    cutoff = generalpara['cutoffsk'+nameij]
    skselfnew = t.zeros(3)
    if dd > cutoff:
        return ham, ovr
    if dd < 1E-4:
        if i != j:
            print("ERROR,distancebetween", i, "atom and", j, "atom is 0")
        else:
            skselfnew[:] = t.from_numpy(skself[i, :])
        if lmax == 1:
            ham[:2, 0, 0] = t.tensor([0, skselfnew[2]])
            ovr[:2, 0, 0] = t.tensor([0, 1.0])
        elif lmax == 2:
            ham[:2, 0, 0] = t.tensor([0, skselfnew[2]])
            ovr[:2, 0, 0] = t.tensor([0, 1.0])
            ham[:2, 1, 1] = t.tensor([1, skselfnew[1]])
            ovr[:2, 1, 1] = t.tensor([1, 1.0])
            ham[:2, 2, 2] = t.tensor([1, skselfnew[1]])
            ovr[:2, 2, 2] = t.tensor([1, 1.0])
            ham[:2, 3, 3] = t.tensor([1, skselfnew[1]])
            ovr[:2, 3, 3] = t.tensor([1, 1.0])
        else:
            ham[:2, 0, 0] = t.tensor([0, skselfnew[2]])
            ovr[:2, 0, 0] = t.tensor([0, 1.0])
            ham[:2, 1, 1] = t.tensor([1, skselfnew[1]])
            ovr[:2, 1, 1] = t.tensor([1, 1.0])
            ham[:2, 2, 2] = t.tensor([1, skselfnew[1]])
            ovr[:2, 2, 2] = t.tensor([1, 1.0])
            ham[:2, 3, 3] = t.tensor([1, skselfnew[1]])
            ovr[:2, 3, 3] = t.tensor([1, 1.0])
            ham[:2, 4, 4] = t.tensor([2, skselfnew[0]])
            ovr[:2, 4, 4] = t.tensor([2, 1.0])
            ham[:2, 5, 5] = t.tensor([2, skselfnew[0]])
            ovr[:2, 5, 5] = t.tensor([2, 1.0])
            ham[:2, 6, 6] = t.tensor([2, skselfnew[0]])
            ovr[:2, 6, 6] = t.tensor([2, 1.0])
            ham[:2, 7, 7] = t.tensor([2, skselfnew[0]])
            ovr[:2, 7, 7] = t.tensor([2, 1.0])
            ham[:2, 8, 8] = t.tensor([2, skselfnew[0]])
            ovr[:2, 8, 8] = t.tensor([2, 1.0])
    else:
        lmaxi = generalpara['lmaxall'][i]
        lmaxj = generalpara['lmaxall'][j]
        maxmax = max(lmaxi, lmaxj)
        minmax = min(lmaxi, lmaxj)
        if maxmax == 1:
            getss(hs_data, ham, ovr)
        elif maxmax == 2 and minmax == 1:
            getsp(hs_data, ham, ovr)
        elif maxmax == 2 and minmax == 2:
            getpp(hs_data, ham, ovr)
        elif maxmax == 3 and minmax == 1:
            getsd(hs_data, ham, ovr)
        elif maxmax == 3 and minmax == 2:
            getpd(hs_data, ham, ovr)
        elif maxmax == 3 and minmax == 3:
            getdd(hs_data, ham, ovr)
    return ham, ovr


def getss(hs_data, hamtable, ovrtable):
    """slater-koster transfermaton for s orvitals"""
    hamtable[0, 0, 0], ovrtable[0, 0, 0] = 10, 10
    hamtable[1, 0, 0], ovrtable[1, 0, 0] = get_s_s(hs_data)
    return hamtable, ovrtable


def getsp(hs_data, hamtable, ovrtable):
    getss(hs_data, hamtable, ovrtable)
    hamtable[0, 1:4, 0], ovrtable[0, 1:4, 0] = (t.tensor(
            [11, 12, 13]), t.tensor([11, 12, 13]))
    hamtable[1, 1, 0], ovrtable[1, 1, 0] = get_s_x(hs_data)
    hamtable[1, 2, 0], ovrtable[1, 2, 0] = get_s_y(hs_data)
    hamtable[1, 3, 0], ovrtable[1, 3, 0] = get_s_z(hs_data)
    for ii in range(nls, nlp+nls):
        hamtable[0, 0, ii] = hamtable[0, ii, 0]
        ovrtable[0, 0, ii] = ovrtable[0, ii, 0]
        hamtable[1:, 0, ii] = -hamtable[1:, ii, 0]
        ovrtable[1:, 0, ii] = -ovrtable[1:, ii, 0]
    return hamtable, ovrtable


def getsd(hs_data, hamtable, ovrtable):
    getsp(hs_data, hamtable, ovrtable)
    (hamtable[0, 4:9, 0], ovrtable[0, 4:9, 0]) = ([14, 15, 16, 17, 18],
                                                  [14, 15, 16, 17, 18])
    hamtable[1, 0, 4], ovrtable[1, 0, 4] = get_s_xy(hs_data)
    hamtable[1, 0, 5], ovrtable[1, 0, 5] = get_s_yz(hs_data)
    hamtable[1, 0, 6], ovrtable[1, 0, 6] = get_s_xz(hs_data)
    hamtable[1, 0, 7], ovrtable[1, 0, 7] = get_s_x2y2(hs_data)
    hamtable[1, 0, 8], ovrtable[1, 0, 8] = get_s_3z2r2(hs_data)
    for ii in range(nls+nlp, nld):
        hamtable[:, ii, 0] = hamtable[: 0, ii]
        ovrtable[:, ii, 0] = ovrtable[:, 0, ii]
    return hamtable, ovrtable


def getpp(hs_data, hamtable, ovrtable):
        getsp(hs_data, hamtable, ovrtable)
        hamtable[0, 1, 1:4], ovrtable[0, 1, 1:4] = [21, 22, 23], [21, 22, 23]
        (hamtable[1, 1, 1], ovrtable[1, 1, 1], hamtable[2, 1, 1],
         ovrtable[2, 1, 1]) = get_x_x(hs_data)
        (hamtable[1, 1, 2], ovrtable[1, 1, 2], hamtable[2, 1, 2],
         ovrtable[2, 1, 2]) = get_x_y(hs_data)
        (hamtable[1, 1, 3], ovrtable[1, 1, 3], hamtable[2, 1, 3],
         ovrtable[2, 1, 3]) = get_x_z(hs_data)
        hamtable[0, 2, 2:4], ovrtable[0, 2, 2:4] = [32, 33], [32, 33]
        (hamtable[1, 2, 2], ovrtable[1, 2, 2], hamtable[2, 2, 2],
         ovrtable[2, 2, 2]) = get_y_y(hs_data)
        (hamtable[1, 2, 3], ovrtable[1, 2, 3], hamtable[2, 2, 3],
         ovrtable[2, 2, 3]) = get_y_z(hs_data)
        hamtable[0, 3, 3], ovrtable[0, 3, 3] = 43, 43
        (hamtable[1, 3, 3], ovrtable[1, 3, 3], hamtable[2, 3, 3],
         ovrtable[2, 3, 3]) = get_z_z(hs_data)
        for ii in range(nls, nlp+nls):
            for jj in range(nls, ii+nls):
                hamtable[:, ii, jj] = hamtable[:, jj, ii]
                ovrtable[:, ii, jj] = ovrtable[:, jj, ii]
        return hamtable, ovrtable


def getpd(hs_data, hamtable, ovrtable):
    getpp(hs_data, hamtable, ovrtable)
    hamtable[0, 1, 4:9], ovrtable[0, 1, 4:9] = ([24, 25, 26, 27, 28],
                                                [24, 25, 26, 27, 28])
    (hamtable[1, 1, 4], ovrtable[1, 1, 4], hamtable[2, 1, 4],
     ovrtable[2, 1, 4]) = get_x_xy(hs_data)
    (hamtable[1, 1, 5], ovrtable[1, 1, 5], hamtable[2, 1, 5],
     ovrtable[2, 1, 5]) = get_x_yz(hs_data)
    (hamtable[1, 1, 6], ovrtable[1, 1, 6], hamtable[2, 1, 6],
     ovrtable[2, 1, 6]) = get_x_xz(hs_data)
    (hamtable[1, 1, 7], ovrtable[1, 1, 7], hamtable[2, 1, 7],
     ovrtable[2, 1, 7]) = get_x_x2y2(hs_data)
    (hamtable[1, 1, 8], ovrtable[1, 1, 8], hamtable[2, 1, 8],
     ovrtable[2, 1, 8]) = get_x_3z2r2(hs_data)
    hamtable[0, 2, 4:9], ovrtable[0, 2, 4:9] = ([34, 35, 36, 37, 38],
                                                [34, 35, 36, 37, 38])
    (hamtable[1, 2, 4], ovrtable[1, 2, 4], hamtable[2, 2, 4],
     ovrtable[2, 2, 4]) = get_y_xy(hs_data)
    (hamtable[1, 2, 5], ovrtable[1, 2, 5], hamtable[2, 2, 5],
     ovrtable[2, 2, 5]) = get_y_yz(hs_data)
    (hamtable[1, 2, 6], ovrtable[1, 2, 6], hamtable[2, 2, 6],
     ovrtable[2, 2, 6]) = get_y_xz(hs_data)
    (hamtable[1, 2, 7], ovrtable[1, 2, 7], hamtable[2, 2, 7],
     ovrtable[2, 2, 7]) = get_y_x2y2(hs_data)
    (hamtable[1, 2, 8], ovrtable[1, 2, 8], hamtable[2, 2, 8],
     ovrtable[2, 2, 8]) = get_y_3z2r2(hs_data)
    hamtable[0, 3, 4:9], ovrtable[0, 3, 4:9] = ([44, 45, 46, 47, 48],
                                                [44, 45, 46, 47, 48])
    (hamtable[1, 3, 4], ovrtable[1, 3, 4], hamtable[2, 3, 4],
     ovrtable[2, 3, 4]) = get_z_xy(hs_data)
    (hamtable[1, 3, 5], ovrtable[1, 3, 5], hamtable[2, 3, 5],
     ovrtable[2, 3, 5]) = get_z_yz(hs_data)
    (hamtable[1, 3, 6], ovrtable[1, 3, 6], hamtable[2, 3, 6],
     ovrtable[2, 3, 6]) = get_z_xz(hs_data)
    (hamtable[1, 3, 7], ovrtable[1, 3, 7], hamtable[2, 3, 7],
     ovrtable[2, 3, 7]) = get_z_x2y2(hs_data)
    (hamtable[1, 3, 8], ovrtable[1, 3, 8], hamtable[2, 3, 8],
     ovrtable[2, 3, 8]) = get_z_3z2r2(hs_data)
    for ii in range(nls, nls+nlp):
        for jj in range(nls+nlp, nld):
            hamtable[0, ii, jj] = hamtable[0, jj, ii]
            ovrtable[0, ii, jj] = ovrtable[0, jj, ii]
            hamtable[1:, jj, ii] = -hamtable[1:, ii, jj]
            ovrtable[1:, jj, ii] = -ovrtable[1:, ii, jj]
    return hamtable, ovrtable


def getdd(data, hamtable, ovrtable):
    getpd(data, hamtable, ovrtable)
    hamtable[4, 4], ovrtable[4, 4] = get_xy_xy(data)
    hamtable[4, 5], ovrtable[4, 5] = get_xy_yz(data)
    hamtable[4, 6], ovrtable[4, 6] = get_xy_xz(data)
    hamtable[4, 7], ovrtable[4, 7] = get_xy_x2y2(data)
    hamtable[4, 8], ovrtable[4, 8] = get_xy_3z2r2(data)
    hamtable[5, 5], ovrtable[5, 5] = get_yz_yz(data)
    hamtable[5, 6], ovrtable[5, 6] = get_yz_xz(data)
    hamtable[5, 7], ovrtable[5, 7] = get_yz_x2y2(data)
    hamtable[5, 8], ovrtable[5, 8] = get_yz_3z2r2(data)
    hamtable[6, 6], ovrtable[6, 6] = get_xz_xz(data)
    hamtable[6, 7], ovrtable[6, 7] = get_xz_x2y2(data)
    hamtable[6, 8], ovrtable[6, 8] = get_xz_3z2r2(data)
    hamtable[7, 7], ovrtable[7, 7] = get_x2y2_x2y2(data)
    hamtable[7, 8], ovrtable[7, 8] = get_x2y2_3z2r2(data)
    hamtable[8, 8], ovrtable[8, 8] = get_3z2r2_3z2r2(data)
    for ii in range(nls+nlp, nld):
        for jj in range(nls+nlp, ii+nls):
            hamtable[:, ii, jj] = hamtable[:, jj, ii]
            ovrtable[:, ii, jj] = ovrtable[:, jj, ii]
    return hamtable, ovrtable


def sk_tranml(generalpara):
    htable = generalpara['hamtable']
    stable = generalpara['ovrtable']
    hinfo = generalpara['haminfo']
    nind2 = len(htable)
    ham = t.zeros(nind2)
    ovr = t.zeros(nind2)
    for inind2 in range(0, nind2):
        ihtable = htable[inind2]
        istable = stable[inind2]
        ihinfo = hinfo[inind2]
        rr = generalpara['rr_ij'][inind2]
        dd = generalpara['dd_ij'][inind2]
        x, y, z = rr/dd
        if ihinfo < 10:
            ham[inind2], ovr[inind2] = ihtable[0], istable[0]
        elif ihinfo == 10:
            ham[inind2], ovr[inind2] = hs_s_s(x, y, z, ihtable[0], istable[0])
        elif ihinfo == 11:
            ham[inind2], ovr[inind2] = hs_s_x(x, y, z, ihtable[0], istable[0])
        elif ihinfo == 12:
            ham[inind2], ovr[inind2] = hs_s_y(x, y, z, ihtable[0], istable[0])
        elif ihinfo == 13:
            ham[inind2], ovr[inind2] = hs_s_z(x, y, z, ihtable[0], istable[0])
        elif ihinfo == 21:
            ham[inind2], ovr[inind2] = hs_x_x(x, y, z, ihtable[0], istable[0],
               ihtable[1], istable[1])
        elif ihinfo == 22:
            ham[inind2], ovr[inind2] = hs_x_y(x, y, z, ihtable[0], istable[0],
               ihtable[1], istable[1])
        elif ihinfo == 23:
            ham[inind2], ovr[inind2] = hs_x_z(x, y, z, ihtable[0], istable[0],
               ihtable[1], istable[1])
        elif ihinfo == 32:
            ham[inind2], ovr[inind2] = hs_y_y(x, y, z, ihtable[0], istable[0],
               ihtable[1], istable[1])
        elif ihinfo == 33:
            ham[inind2], ovr[inind2] = hs_y_z(x, y, z, ihtable[0], istable[0],
               ihtable[1], istable[1])
        elif ihinfo == 43:
            ham[inind2], ovr[inind2] = hs_z_z(x, y, z, ihtable[0], istable[0],
               ihtable[1], istable[1])
    generalpara['hammat'] = ham
    generalpara['overmat'] = ovr
    return generalpara


def sk_transpline(generalpara):
    pass


def get_s_s(hs_data):
    return hs_data[9], hs_data[19]


def get_s_x(hs_data):
    return hs_data[8], hs_data[18]


def get_s_y(hs_data):
    return hs_data[8], hs_data[18]


def get_s_z(hs_data):
    return hs_data[8], hs_data[18]


def get_s_xy(hs_data):
    return hs_data[7], hs_data[17]


def get_s_yz(hs_data):
    return hs_data[7], hs_data[17]


def get_s_xz(hs_data):
    return hs_data[7], hs_data[17]


def get_s_x2y2(hs_data):
    return hs_data[7], hs_data[17]


def get_s_3z2r2(hs_data):
    return hs_data[7], hs_data[17]


def get_x_s(hs_data):
    return get_s_x(hs_data)[0], get_s_x(hs_data)[1]


def get_x_x(hs_data):
    return hs_data[5], hs_data[15], hs_data[6], hs_data[16]


def get_x_y(hs_data):
    return hs_data[5], hs_data[15], hs_data[6], hs_data[16]


def get_x_z(hs_data):
    return hs_data[5], hs_data[15], hs_data[6], hs_data[16]


def get_x_xy(hs_data):
    return hs_data[3], hs_data[13], hs_data[4], hs_data[14]


def get_x_yz(hs_data):
    return hs_data[3], hs_data[13], hs_data[4], hs_data[14]


def get_x_xz(hs_data):
    return hs_data[3], hs_data[13], hs_data[4], hs_data[14]


def get_x_x2y2(hs_data):
    return hs_data[3], hs_data[13], hs_data[4], hs_data[14]


def get_x_3z2r2(hs_data):
    return hs_data[3], hs_data[13], hs_data[4], hs_data[14]


def get_y_s(hs_data):
    return get_s_y(hs_data)[0], get_s_y(hs_data)[1]


def get_y_x(hs_data):
    return (get_x_y(hs_data)[0], get_x_y(hs_data)[1],
            get_x_y(hs_data)[2], get_x_y(hs_data)[3])


def get_y_y(hs_data):
    return hs_data[5], hs_data[15], hs_data[6], hs_data[16]


def get_y_z(hs_data):
    return hs_data[5], hs_data[15], hs_data[6], hs_data[16]


def get_y_xy(hs_data):
    return hs_data[3], hs_data[13], hs_data[4], hs_data[14]


def get_y_yz(hs_data):
    return hs_data[3], hs_data[13], hs_data[4], hs_data[14]


def get_y_xz(hs_data):
    return hs_data[3], hs_data[13], hs_data[4], hs_data[14]


def get_y_x2y2(hs_data):
    return hs_data[3], hs_data[13], hs_data[4], hs_data[14]


def get_y_3z2r2(hs_data):
    return hs_data[3], hs_data[13], hs_data[4], hs_data[14]


def get_z_s(hs_data):
    return get_s_z(hs_data)[0], get_s_z(hs_data)[1]


def get_z_x(hs_data):
    return (get_x_z(hs_data)[0], get_x_z(hs_data)[1],
            get_x_z(hs_data)[2], get_x_z(hs_data)[3])


def get_z_y(hs_data):
    return (get_y_z(hs_data)[0], get_y_z(hs_data)[1],
            get_y_z(hs_data)[2], get_y_z(hs_data)[3])


def get_z_z(hs_data):
    return hs_data[5], hs_data[15], hs_data[6], hs_data[16]


def get_z_xy(hs_data):
    return hs_data[3], hs_data[13], hs_data[4], hs_data[14]


def get_z_yz(hs_data):
    return hs_data[3], hs_data[13], hs_data[4], hs_data[14]


def get_z_xz(hs_data):
    return hs_data[3], hs_data[13], hs_data[4], hs_data[14]


def get_z_x2y2(hs_data):
    return hs_data[3], hs_data[13], hs_data[4], hs_data[14]


def get_z_3z2r2(hs_data):
    return hs_data[3], hs_data[13], hs_data[4], hs_data[14]


def get_xy_s(hs_data):
    return get_s_xy(hs_data)[0], get_s_xy(hs_data)[1]


def get_xy_x(hs_data):
    return (get_x_xy(hs_data)[0], get_x_xy(hs_data)[1],
            get_x_xy(hs_data)[2], get_x_xy(hs_data)[3])


def get_xy_y(hs_data):
    return (get_y_xy(hs_data)[0], get_y_xy(hs_data)[1],
            get_y_xy(hs_data)[2], get_y_xy(hs_data)[3])


def get_xy_z(hs_data):
    return (get_z_xy(hs_data)[0], get_z_xy(hs_data)[1],
            get_z_xy(hs_data)[2], get_z_xy(hs_data)[3])


def get_xy_xy(hs_data):
    return (hs_data[0], hs_data[10], hs_data[1], hs_data[11], hs_data[2],
            hs_data[12])


def get_xy_yz(hs_data):
    return (hs_data[0], hs_data[10], hs_data[1], hs_data[11], hs_data[2],
            hs_data[12])


def get_xy_xz(hs_data):
    return (hs_data[0], hs_data[10], hs_data[1], hs_data[11], hs_data[2],
            hs_data[12])


def get_xy_x2y2(hs_data):
    return (hs_data[0], hs_data[10], hs_data[1], hs_data[11], hs_data[2],
            hs_data[12])


def get_xy_3z2r2(hs_data):
    return (hs_data[0], hs_data[10], hs_data[1], hs_data[11], hs_data[2],
            hs_data[12])


def get_yz_s(hs_data):
    return get_s_yz(hs_data)[0], get_s_yz(hs_data)[1]


def get_yz_x(hs_data):
    return (get_x_yz(hs_data)[0], get_x_yz(hs_data)[1],
            get_x_yz(hs_data)[2], get_x_yz(hs_data)[3])


def get_yz_y(hs_data):
    return (get_y_yz(hs_data)[0], get_y_yz(hs_data)[1],
            get_y_yz(hs_data)[2], get_y_yz(hs_data)[3])


def get_yz_z(hs_data):
    return (get_z_yz(hs_data)[0], get_z_yz(hs_data)[1],
            get_z_yz(hs_data)[2], get_z_yz(hs_data)[3])


def get_yz_xy(hs_data):
    return (get_xy_yz(hs_data)[0],
            get_xy_yz(hs_data)[1],
            get_xy_yz(hs_data)[2],
            get_xy_yz(hs_data)[3],
            get_xy_yz(hs_data)[4],
            get_xy_yz(hs_data)[5])


def get_yz_yz(hs_data):
    return (hs_data[0], hs_data[10], hs_data[1], hs_data[11],
            hs_data[2], hs_data[12])


def get_yz_xz(hs_data):
    return (hs_data[0], hs_data[10], hs_data[1], hs_data[11],
            hs_data[2], hs_data[12])


def get_yz_x2y2(hs_data):
    return (hs_data[0], hs_data[10], hs_data[1], hs_data[11],
            hs_data[2], hs_data[12])


def get_yz_3z2r2(hs_data):
    return (hs_data[0], hs_data[10], hs_data[1], hs_data[11],
            hs_data[2], hs_data[12])


def get_xz_s(hs_data):
    return get_s_xz(hs_data)[0], get_s_xz(hs_data)[1]


def get_xz_x(hs_data):
    return (get_x_xz(hs_data)[0],
            get_x_xz(hs_data)[1],
            get_x_xz(hs_data)[2],
            get_x_xz(hs_data)[3])


def get_xz_y(hs_data):
    return (get_y_xz(hs_data)[0],
            get_y_xz(hs_data)[1],
            get_y_xz(hs_data)[2],
            get_y_xz(hs_data)[3])


def get_xz_z(hs_data):
    return (get_z_xz(hs_data)[0],
            get_z_xz(hs_data)[1],
            get_z_xz(hs_data)[2],
            get_z_xz(hs_data)[3])


def get_xz_xy(hs_data):
    return (get_xy_xz(hs_data)[0],
            get_xy_xz(hs_data)[1],
            get_xy_xz(hs_data)[2],
            get_xy_xz(hs_data)[3],
            get_xy_xz(hs_data)[4],
            get_xy_xz(hs_data)[5])


def get_xz_yz(hs_data):
    return (get_yz_xz(hs_data)[0],
            get_yz_xz(hs_data)[1],
            get_yz_xz(hs_data)[2],
            get_yz_xz(hs_data)[3],
            get_yz_xz(hs_data)[4],
            get_yz_xz(hs_data)[5])


def get_xz_3z2r2(hs_data):
    return (hs_data[0], hs_data[10], hs_data[1], hs_data[11],
            hs_data[2], hs_data[12])


def get_xz_x2y2(hs_data):
    return (hs_data[0], hs_data[10], hs_data[1], hs_data[11],
            hs_data[2], hs_data[12])


def get_xz_xz(hs_data):
    return (hs_data[0], hs_data[10], hs_data[1], hs_data[11],
            hs_data[2], hs_data[12])


def get_x2y2_s(hs_data):
    return (get_s_x2y2(hs_data)[0],
            get_s_x2y2(hs_data)[1])


def get_x2y2_x(hs_data):
    return (get_x_x2y2(hs_data)[0],
            get_x_x2y2(hs_data)[1],
            get_x_x2y2(hs_data)[2],
            get_x_x2y2(hs_data)[3])


def get_x2y2_y(hs_data):
    return (get_y_x2y2(hs_data)[0],
            get_y_x2y2(hs_data)[1],
            get_y_x2y2(hs_data)[2],
            get_y_x2y2(hs_data)[3])


def get_x2y2_z(hs_data):
    return (get_z_x2y2(hs_data)[0],
            get_z_x2y2(hs_data)[1],
            get_z_x2y2(hs_data)[2],
            get_z_x2y2(hs_data)[3])


def get_x2y2_xy(hs_data):
    return (get_xy_x2y2(hs_data)[0],
            get_xy_x2y2(hs_data)[1],
            get_xy_x2y2(hs_data)[2],
            get_xy_x2y2(hs_data)[3],
            get_xy_x2y2(hs_data)[4],
            get_xy_x2y2(hs_data)[5])


def get_x2y2_yz(hs_data):
    return (get_yz_x2y2(hs_data)[0],
            get_yz_x2y2(hs_data)[1],
            get_yz_x2y2(hs_data)[2],
            get_yz_x2y2(hs_data)[3],
            get_yz_x2y2(hs_data)[4],
            get_yz_x2y2(hs_data)[5])


def get_x2y2_xz(hs_data):
    return (get_xz_x2y2(hs_data)[0],
            get_xz_x2y2(hs_data)[1],
            get_xz_x2y2(hs_data)[2],
            get_xz_x2y2(hs_data)[3],
            get_xz_x2y2(hs_data)[4],
            get_xz_x2y2(hs_data)[5])


def get_x2y2_x2y2(hs_data):
    return (hs_data[0], hs_data[10], hs_data[1], hs_data[11], hs_data[2],
            hs_data[12])


def get_x2y2_3z2r2(hs_data):
    return (hs_data[0], hs_data[10], hs_data[1], hs_data[11], hs_data[2],
            hs_data[12])


def get_3z2r2_s(hs_data):
    return (get_s_3z2r2(hs_data)[0],
            get_s_3z2r2(hs_data)[1])


def get_3z2r2_x(hs_data):
    return (get_x_3z2r2(hs_data)[0],
            get_x_3z2r2(hs_data)[1],
            get_x_3z2r2(hs_data)[2],
            get_x_3z2r2(hs_data)[3])


def get_3z2r2_y(hs_data):
    return (get_y_3z2r2(hs_data)[0],
            get_y_3z2r2(hs_data)[1],
            get_y_3z2r2(hs_data)[2],
            get_y_3z2r2(hs_data)[3])


def get_3z2r2_z(hs_data):
    return (get_z_3z2r2(hs_data)[0],
            get_z_3z2r2(hs_data)[1],
            get_z_3z2r2(hs_data)[2],
            get_z_3z2r2(hs_data)[3])


def get_3z2r2_xy(hs_data):
    return (get_xy_3z2r2(hs_data)[0],
            get_xy_3z2r2(hs_data)[1],
            get_xy_3z2r2(hs_data)[2],
            get_xy_3z2r2(hs_data)[3],
            get_xy_3z2r2(hs_data)[4],
            get_xy_3z2r2(hs_data)[5])


def get_3z2r2_yz(hs_data):
    return (get_yz_3z2r2(hs_data)[0],
            get_yz_3z2r2(hs_data)[1],
            get_yz_3z2r2(hs_data)[2],
            get_yz_3z2r2(hs_data)[3],
            get_yz_3z2r2(hs_data)[4],
            get_yz_3z2r2(hs_data)[5])


def get_3z2r2_xz(hs_data):
    return (get_xz_3z2r2(hs_data)[0],
            get_xz_3z2r2(hs_data)[1],
            get_xz_3z2r2(hs_data)[2],
            get_xz_3z2r2(hs_data)[3],
            get_xz_3z2r2(hs_data)[4],
            get_xz_3z2r2(hs_data)[5])


def get_3z2r2_x2y2(hs_data):
    return (get_x2y2_3z2r2(hs_data)[0],
            get_x2y2_3z2r2(hs_data)[1],
            get_x2y2_3z2r2(hs_data)[2],
            get_x2y2_3z2r2(hs_data)[3],
            get_x2y2_3z2r2(hs_data)[4],
            get_x2y2_3z2r2(-hs_data)[5])


def get_3z2r2_3z2r2(hs_data):
    return (hs_data[0], hs_data[10], hs_data[1], hs_data[11], hs_data[2],
            hs_data[12])


def hs_s_s(x, y, z, Hss0, Sss0):
    return Hss0, Sss0


def h_s_s(x, y, z, Hss0):
    return Hss0


def s_s_s(x, y, z, Sss0):
    return Sss0


def hs_s_x(x, y, z, Hsp0, Ssp0):
    return x*Hsp0, x*Ssp0


def h_s_x(x, y, z, Hsp0):
    return x*Hsp0

def s_s_x(x, y, z, Ssp0):
    return x*Ssp0


def hs_s_y(x, y, z, Hsp0, Ssp0):
    return y*Hsp0, y*Ssp0


def h_s_y(x, y, z, Hsp0):
    return y*Hsp0


def s_s_y(x, y, z, Ssp0):
    return y*Ssp0


def hs_s_z(x, y, z, Hsp0, Ssp0):
    return z*Hsp0, z*Ssp0


def h_s_z(x, y, z, Hsp0):
    return z*Hsp0


def s_s_z(x, y, z, Ssp0):
    return z*Ssp0


def hs_s_xy(x, y, z, Hsd0, Ssd0):
    return t.sqrt(t.tensor([3.]))*x*y*Hsd0, t.sqrt(t.tensor([3.]))*x*y*Ssd0


def h_s_xy(x, y, z, Hsd0):
    return t.sqrt(t.tensor([3.]))*x*y*Hsd0


def s_s_xy(x, y, z, Ssd0):
    return t.sqrt(t.tensor([3.]))*x*y*Ssd0


def hs_s_yz(x, y, z, Hsd0, Ssd0):
    return t.sqrt(t.tensor([3.]))*y*z*Hsd0, t.sqrt(t.tensor([3.]))*y*z*Ssd0


def h_s_yz(x, y, z, Hsd0):
    return t.sqrt(t.tensor([3.]))*y*z*Hsd0


def s_s_yz(x, y, z, Ssd0):
    return t.sqrt(t.tensor([3.]))*y*z*Ssd0


def hs_s_xz(x, y, z, Hsd0, Ssd0):
    return t.sqrt(t.tensor([3.]))*x*z*Hsd0, t.sqrt(t.tensor([3.]))*x*z*Ssd0


def h_s_xz(x, y, z, Hsd0):
    return t.sqrt(t.tensor([3.]))*x*z*Hsd0


def s_s_xz(x, y, z, Ssd0):
    return t.sqrt(t.tensor([3.]))*x*z*Ssd0


def hs_s_x2y2(x, y, z, Hsd0, Ssd0):
    return (0.5*t.sqrt(t.tensor([3.]))*(x**2-y**2)*Hsd0,
            0.5*t.sqrt(t.tensor([3.]))*(x**2-y**2)*Ssd0)


def hs_s_3z2r2(x, y, z, Hsd0, Ssd0):
    return (z**2-0.5*(x**2+y**2))*Hsd0, (z**2-0.5*(x**2+y**2))*Ssd0


def hs_x_s(x, y, z, Hsp0, Ssp0):
    return hs_s_x(-x, -y, -z, Hsp0, Ssp0)[0], hs_s_x(-x, -y, -z, Hsp0, Ssp0)[1]


def h_x_s(x, y, z, Hsp0):
    return h_s_x(-x, -y, -z, Hsp0)


def s_x_s(x, y, z, Ssp0):
    return hs_s_x(-x, -y, -z, Ssp0)


def hs_x_x(x, y, z, Hpp0, Spp0, Hpp1, Spp1):
    return x**2*Hpp0+(1-x**2)*Hpp1, x**2*Spp0+(1-x**2)*Spp1


def h_x_x(x, y, z, Hpp0, Hpp1):
    return x**2*Hpp0+(1-x**2)*Hpp1


def s_x_x(x, y, z, Spp0, Spp1):
    return x**2*Spp0+(1-x**2)*Spp1


def hs_x_y(x, y, z, Hpp0, Spp0, Hpp1, Spp1):
    return x*y*Hpp0-x*y*Hpp1, x*y*Spp0-x*y*Spp1


def h_x_y(x, y, z, Hpp0, Hpp1):
    return x*y*Hpp0-x*y*Hpp1


def s_x_y(x, y, z, Spp0, Spp1):
    return x*y*Spp0-x*y*Spp1


def hs_x_z(x, y, z, Hpp0, Spp0, Hpp1, Spp1):
    return x*z*Hpp0-x*z*Hpp1, x*z*Spp0-x*z*Spp1


def h_x_z(x, y, z, Hpp0, Hpp1):
    return x*z*Hpp0-x*z*Hpp1


def s_x_z(x, y, z, Spp0, Spp1):
    return x*z*Spp0-x*z*Spp1


def hs_x_xy(x, y, z, Hpd0, Spd0, Hpd1, Spd1):
    return (t.sqrt(t.tensor([3.]))*x**2*y*Hpd0+y*(1-2*x**2)*Hpd1,
            t.sqrt(t.tensor([3.]))*x**2*y*Spd0 + y*(1-2*x**2)*Spd1)


def hs_x_yz(x, y, z, Hpd0, Spd0, Hpd1, Spd1):
    return (t.sqrt(t.tensor([3.]))*x*y*z*Hpd0-2*x*y*z*Hpd1,
            t.sqrt(t.tensor([3.]))*x*y*z*Hpd0-2*x*y*z*Hpd1)


def hs_x_xz(x, y, z, Hpd0, Spd0, Hpd1, Spd1):
    return (t.sqrt(t.tensor([3.]))*x**2*z*Hpd0+z*(1-2*x**2)*Hpd1,
            t.sqrt(t.tensor([3.]))*x**2*z*Spd0+z*(1-2*x**2)*Spd1)


def hs_x_x2y2(x, y, z, Hpd0, Spd0, Hpd1, Spd1):
    return (t.sqrt(t.tensor([3.]))/2*x*(x**2-y**2)*Hpd0+x*(1-x**2+y**2)*Hpd1,
            t.sqrt(t.tensor([3.]))/2*x*(x**2-y**2)*Spd0+x*(1-x**2+y**2)*Spd1)


def hs_x_3z2r2(x, y, z, Hpd0, Spd0, Hpd1, Spd1):
    return (x*(z**2-0.5*(x**2+y**2))*Hpd0-t.sqrt(t.tensor([3.]))*x*z**2*Hpd1,
            x*(z**2-0.5*(x**2+y**2))*Spd1-t.sqrt(t.tensor([3.]))*x*z**2*Spd1)


def hs_y_s(x, y, z, Hsp0, Ssp0):
    return hs_s_y(-x, -y, -z, Hsp0, Ssp0)[0], hs_s_y(-x, -y, -z, Hsp0, Ssp0)[1]


def h_y_s(x, y, z, Hsp0):
    return h_s_y(-x, -y, -z, Hsp0)


def s_y_s(x, y, z, Ssp0):
    return hs_s_y(-x, -y, -z, Ssp0)


def hs_y_x(x, y, z, Hpp0, Spp0, Hpp1, Spp1):
    return hs_x_y(-x, -y, -z, Hpp0, Spp0, Hpp1, Spp1)[0], hs_x_y(
            -x, -y, -z, Hpp0, Spp0, Hpp1, Spp1)[1]


def h_y_x(x, y, z, Hpp0, Hpp1):
    return hs_x_y(-x, -y, -z, Hpp0, Hpp1)


def s_y_x(x, y, z, Spp0, Spp1):
    return s_x_y(-x, -y, -z, Spp0, Spp1)


def hs_y_y(x, y, z, Hpp0, Spp0, Hpp1, Spp1):
    return y**2*Hpp0+(1-y**2)*Hpp1, y**2*Spp0+(1-y**2)*Spp1


def h_y_y(x, y, z, Hpp0, Hpp1):
    return y**2*Hpp0+(1-y**2)*Hpp1


def s_y_y(x, y, z, Spp0, Spp1):
    return y**2*Spp0+(1-y**2)*Spp1


def hs_y_z(x, y, z, Hpp0, Spp0, Hpp1, Spp1):
    return y*z*Hpp0-y*z*Hpp1, y*z*Spp0-y*z*Spp1


def h_y_z(x, y, z, Hpp0, Hpp1):
    return y*z*Hpp0-y*z*Hpp1


def s_y_z(x, y, z, Spp0, Spp1):
    return y*z*Spp0-y*z*Spp1


def hs_y_xy(x, y, z, Hpd0, Spd0, Hpd1, Spd1):
    return (t.sqrt(t.tensor([3.]))*y**2*x*Hpd0+x*(1-2*y**2)*Hpd1,
            t.sqrt(t.tensor([3.]))*y**2*x*Spd0+x*(1-2*y**2)*Spd1)


def hs_y_yz(x, y, z, Hpd0, Spd0, Hpd1, Spd1):
    return (t.sqrt(t.tensor([3.]))*y**2*z*Hpd0-z*(1-2*y**2)*Hpd1,
            t.sqrt(t.tensor([3.]))*y**2*z*Spd0-z*(1-2*y**2)*Spd1)


def hs_y_xz(x, y, z, Hpd0, Spd0, Hpd1, Spd1):
    return (t.sqrt(t.tensor([3.]))*x*y*z*Hpd0-2*x*y*z*Hpd1,
            t.sqrt(t.tensor([3.]))*x*y*z*Spd0-2*x*y*z*Spd1)


def hs_y_x2y2(x, y, z, Hpd0, Spd0, Hpd1, Spd1):
    return (t.sqrt(t.tensor([3.]))/2*y*(x**2-y**2)*Hpd0-y*(1+x**2-y**2)*Hpd1,
            t.sqrt(t.tensor([3.]))/2*y*(x**2-y**2)*Spd0-y*(1+x**2-y**2)*Spd1)


def hs_y_3z2r2(x, y, z, Hpd0, Spd0, Hpd1, Spd1):
    return (y*(z**2-0.5*(x**2+y**2))*Hpd0-t.sqrt(t.tensor([3.]))*y*z**2*Hpd1,
            y*(z**2-0.5*(x**2+y**2))*Spd0-t.sqrt(t.tensor([3.]))*y*z**2*Spd1)


def hs_z_s(x, y, z, Hsp0, Ssp0):
    return hs_s_z(-x, -y, -z, Hsp0, Ssp0)[0], hs_s_z(-x, -y, -z, Hsp0, Ssp0)[1]


def h_z_s(x, y, z, Hsp0):
    return h_s_z(-x, -y, -z, Hsp0)


def s_z_s(x, y, z, Ssp0):
    return s_s_z(-x, -y, -z, Ssp0)


def hs_z_x(x, y, z, Hpp0, Spp0, Hpp1, Spp1):
    return hs_x_z(-x, -y, -z, Hpp0, Spp0, Hpp1, Spp1)[0], hs_x_z(
            -x, -y, -z, Hpp0, Spp0, Hpp1, Spp1)[1]


def h_z_x(x, y, z, Hpp0, Hpp1):
    return h_x_z(-x, -y, -z, Hpp0, Hpp1)


def s_z_x(x, y, z, Spp0, Spp1):
    return s_x_z(-x, -y, -z, Spp0, Spp1)


def hs_z_y(x, y, z, Hpp0, Spp0, Hpp1, Spp1):
    return hs_y_z(-x, -y, -z, Hpp0, Spp0, Hpp1, Spp1)[0], hs_y_z(
            -x, -y, -z, Hpp0, Spp0, Hpp1, Spp1)[1]


def h_z_y(x, y, z, Hpp0, Hpp1):
    return h_y_z(-x, -y, -z, Hpp0, Hpp1)


def s_z_y(x, y, z, Spp0, Spp1):
    return s_y_z(-x, -y, -z, Spp0, Spp1)


def hs_z_z(x, y, z, Hpp0, Spp0, Hpp1, Spp1):
    return (z**2*Hpp0+(1-z**2)*Hpp1,
            z**2*Spp0+(1-z**2)*Spp1)


def h_z_z(x, y, z, Hpp0, Hpp1):
    return z**2*Hpp0+(1-z**2)*Hpp1


def s_z_z(x, y, z, Spp0, Spp1):
    return z**2*Spp0+(1-z**2)*Spp1


def hs_z_xy(x, y, z, Hpd0, Spd0, Hpd1, Spd1):
    return (t.sqrt(t.tensor([3.]))*x*y*z*Hpd0 - 2*x*y*z*Hpd1,
            t.sqrt(t.tensor([3.]))*x*y*z*Spd0 - 2*x*y*z*Spd1)


def hs_z_yz(x, y, z, Hpd0, Spd0, Hpd1, Spd1):
    return (t.sqrt(t.tensor([3.]))*z**2*y*Hpd0 - y*(1-2*z**2)*Hpd1,
            t.sqrt(t.tensor([3.]))*z**2*y*Spd0 - y*(1-2*z**2)*Spd1)


def hs_z_xz(x, y, z, Hpd0, Spd0, Hpd1, Spd1):
    return (t.sqrt(t.tensor([3.]))*z**2*x*Hpd0 - x*(1-2*z**2)*Hpd1,
            t.sqrt(t.tensor([3.]))*z**2*x*Spd0 - x*(1-2*z**2)*Spd1)


def hs_z_x2y2(x, y, z, Hpd0, Spd0, Hpd1, Spd1):
    return (t.sqrt(t.tensor([3.]))/2*z*(x**2-y**2)*Hpd0 - z*(x**2-y**2)*Hpd1,
            t.sqrt(t.tensor([3.]))/2*z*(x**2-y**2)*Spd0 - z*(x**2-y**2)*Spd1)


def hs_z_3z2r2(x, y, z, Hpd0, Spd0, Hpd1, Spd1):
    return (z*(z**2-0.5*(x**2+y**2))*Hpd0+t.sqrt(t.tensor([3.])) *
            z*(x**2+y**2)*Hpd1,
            z*(z**2-0.5*(x**2+y**2))*Spd0+t.sqrt(t.tensor([3.])) *
            z*(x**2+y**2)*Spd1)


def hs_xy_s(x, y, z, Hsd0, Ssd0):
    return hs_s_xy(-x, -y, -z, Hsd0, Ssd0)[0], hs_s_xy(
            -x, -y, -z, Hsd0, Ssd0)[1]


def hs_xy_x(x, y, z, Hpd0, Spd0, Hpd1, Spd1):
    return hs_x_xy(-x, -y, -z, Hpd0, Spd0, Hpd1, Spd1)[0], hs_x_xy(
            -x, -y, -z, Hpd0, Spd0, Hpd1, Spd1)[1]


def hs_xy_y(x, y, z, Hpd0, Spd0, Hpd1, Spd1):
    return hs_y_xy(-x, -y, -z, Hpd0, Spd0, Hpd1, Spd1)[0], hs_y_xy(
            -x, -y, -z, Hpd0, Spd0, Hpd1, Spd1)[1]


def hs_xy_z(x, y, z, Hpd0, Spd0, Hpd1, Spd1):
    return hs_z_xy(-x, -y, -z, Hpd0, Spd0, Hpd1, Spd1)[0], hs_z_xy(
            -x, -y, -z, Hpd0, Spd0, Hpd1, Spd1)[1]


def hs_xy_xy(x, y, z, Hdd0, Sdd0, Hdd1, Sdd1, Hdd2, Sdd2):
    return (3*x**2*y**2*Hdd0+(x**2+y**2-4*x**2*y**2) *
            Hdd1+(z**2+x**2*y**2)*Hdd2,
            3*x**2*y**2*Sdd0 + (x**2+y**2-4*x**2*y**2) *
            Sdd1+(z**2+x**2*y**2)*Sdd2)


def hs_xy_yz(x, y, z, Hdd0, Sdd0, Hdd1, Sdd1, Hdd2, Sdd2):
    return (3*x*y**2*z*Hdd0+x*z*(1-4*y**2)*Hdd1 +
            x*z*(y**2-1)*Hdd2,
            3*x*y**2*z*Sdd0+x*z*(1-4*y**2)*Sdd1 +
            x*z*(y**2-1)*Sdd2)


def hs_xy_xz(x, y, z, Hdd0, Sdd0, Hdd1, Sdd1, Hdd2, Sdd2):
    return (3*x**2*y*z*Hdd0+y*z*(1-4*x**2)*Hdd1 +
            y*z*(x**2-1)*Hdd2,
            3*x**2*y*z*Sdd0+y*z*(1-4*x**2)*Sdd1 +
            y*z*(x**2-1)*Sdd2)


def hs_xy_x2y2(x, y, z, Hdd0, Sdd0, Hdd1, Sdd1, Hdd2, Sdd2):
    return (1.5*x*y*(x**2-y**2)*Hdd0-2*x*y*(x**2-y**2)*Hdd1 +
            0.5*x*y*(x**2-y**2)*Hdd2,
            1.5*x*y*(x**2-y**2)*Sdd0-2*x*y*(x**2-y**2)*Sdd1 +
            0.5*x*y*(x**2-y**2)*Sdd2)


def hs_xy_3z2r2(x, y, z, Hdd0, Sdd0, Hdd1, Sdd1, Hdd2, Sdd2):
    return (t.sqrt(t.tensor([3.]))*x*y*(z**2-0.5*(x**2+y**2))*Hdd0-2 *
            t.sqrt(t.tensor([3.])) *
            x*y*z**2*Hdd1+t.sqrt(t.tensor([3.]))/2*x*y*(1+z**2)*Hdd2,
            t.sqrt(t.tensor([3.]))*x*y*(z**2-0.5*(x**2+y**2))*Sdd0-2 *
            t.sqrt(t.tensor([3.])) *
            x*y*z**2*Sdd1+t.sqrt(t.tensor([3.]))/2*x*y*(1+z**2)*Sdd2)


def hs_yz_s(x, y, z, Hsd0, Ssd0):
    return hs_s_yz(-x, -y, -z, Hsd0, Ssd0)[0], hs_s_yz(
            -x, -y, -z, Hsd0, Ssd0)[1]


def hs_yz_x(x, y, z, Hpd0, Spd0, Hpd1, Spd1):
    return hs_x_yz(-x, -y, -z, Hpd0, Spd0, Hpd1, Spd1)[0], hs_x_yz(
            -x, -y, -z, Hpd0, Spd0, Hpd1, Spd1)[1]


def hs_yz_y(x, y, z, Hpd0, Spd0, Hpd1, Spd1):
    return hs_y_yz(-x, -y, -z, Hpd0, Spd0, Hpd1, Spd1)[0], hs_y_yz(
            -x, -y, -z, Hpd0, Spd0, Hpd1, Spd1)[1]


def hs_yz_z(x, y, z, Hpd0, Spd0, Hpd1, Spd1):
    return hs_z_yz(-x, -y, -z, Hpd0, Spd0, Hpd1, Spd1)[0], hs_z_yz(
            -x, -y, -z, Hpd0, Spd0, Hpd1, Spd1)[1]


def hs_yz_xy(x, y, z, Hdd0, Sdd0, Hdd1, Sdd1, Hdd2, Sdd2):
    return (hs_xy_yz(-x, -y, -z, Hdd0, Sdd0, Hdd1, Sdd1, Hdd2, Sdd2)[0],
            hs_xy_yz(-x, -y, -z, Hdd0, Sdd0, Hdd1, Sdd1, Hdd2, Sdd2)[1])


def hs_yz_yz(x, y, z, Hdd0, Sdd0, Hdd1, Sdd1, Hdd2, Sdd2):
    return (3*y**2*z**2*Hdd0 + (y**2+z**2-4*y**2*z**2) *
            Hdd1+(x**2+y**2*z**2)*Hdd2,
            3*y**2*z**2*Sdd0 + (y**2+z**2-4*y**2*z**2) *
            Sdd1+(x**2+y**2*z**2)*Sdd2)


def hs_yz_xz(x, y, z, Hdd0, Sdd0, Hdd1, Sdd1, Hdd2, Sdd2):
    return (3*x*z**2*y*Hdd0+x*y*(1-4*z**2)*Hdd1 +
            x*y*(z**2-1)*Hdd2,
            3*x*z**2*y*Sdd0+x*y*(1-4*z**2)*Sdd1 +
            x*y*(z**2-1)*Sdd2)


def hs_yz_x2y2(x, y, z, Hdd0, Sdd0, Hdd1, Sdd1, Hdd2, Sdd2):
    return (1.5*y*z*(x**2-y**2)*Hdd0-y*z*(1+2*(x**2-y**2)) *
            Hdd1+y*z*(1+0.5*(x**2-y**2))*Hdd2,
            1.5*y*z*(x**2-y**2)*Sdd0-y*z*(1+2*(x**2-y**2)) *
            Sdd1+y*z*(1+0.5*(x**2-y**2))*Sdd2)


def hs_yz_3z2r2(x, y, z, Hdd0, Sdd0, Hdd1, Sdd1, Hdd2, Sdd2):
    return (t.sqrt(t.tensor([3.]))*y*z*(z**2-0.5*(x**2+y**2))*Hdd0 +
            t.sqrt(t.tensor([3.]))*y*z*(x**2+y**2-z**2)*Hdd1 -
            t.sqrt(t.tensor([3.]))/2*y*z*(x**2+y**2)*Hdd2,
            t.sqrt(t.tensor([3.]))*y*z*(z**2-0.5*(x**2+y**2)) *
            Sdd0+t.sqrt(t.tensor([3.]))*y*z*(x**2+y**2-z**2)*Sdd1 -
            t.sqrt(t.tensor([3.]))/2*y*z*(x**2+y**2)*Sdd2)


def hs_xz_s(x, y, z, Hdd0, Sdd0):
    return hs_s_xz(-x, -y, -z, Hdd0, Sdd0)[0], hs_s_xz(
            -x, -y, -z, Hdd0, Sdd0)[1]


def hs_xz_x(x, y, z, Hpd0, Spd0, Hpd1, Spd1):
    return hs_x_xz(-x, -y, -z, Hpd0, Spd0, Hpd1, Spd1)[0], hs_x_xz(
            -x, -y, -z, Hpd0, Spd0, Hpd1, Spd1)[1]


def hs_xz_y(x, y, z, Hpd0, Spd0, Hpd1, Spd1):
    return hs_y_xz(-x, -y, -z, Hpd0, Spd0, Hpd1, Spd1)[0], hs_y_xz(
            -x, -y, -z, Hpd0, Spd0, Hpd1, Spd1)[1]


def hs_xz_z(x, y, z, Hpd0, Spd0, Hpd1, Spd1):
    return hs_z_xz(-x, -y, -z, Hpd0, Spd0, Hpd1, Spd1)[0], hs_z_xz(
            -x, -y, -z, Hpd0, Spd0, Hpd1, Spd1)[1]


def hs_xz_xy(x, y, z, Hdd0, Sdd0, Hdd1, Sdd1, Hdd2, Sdd2):
    return (hs_xy_xz(-x, -y, -z, Hdd0, Sdd0, Hdd1, Sdd1, Hdd2, Sdd2)[0],
            hs_xy_xz(-x, -y, -z, Hdd0, Sdd0, Hdd1, Sdd1, Hdd2, Sdd2)[1])


def hs_xz_yz(x, y, z, Hdd0, Sdd0, Hdd1, Sdd1, Hdd2, Sdd2):
    return (hs_yz_xz(-x, -y, -z, Hdd0, Sdd0, Hdd1, Sdd1, Hdd2, Sdd2)[0],
            hs_yz_xz(-x, -y, -z, Hdd0, Sdd0, Hdd1, Sdd1, Hdd2, Sdd2)[1])


def hs_xz_xz(x, y, z, Hdd0, Sdd0, Hdd1, Sdd1, Hdd2, Sdd2):
    return (3*x**2*z**2*Hdd0 + (x**2+z**2-4*x**2*z**2) *
            Hdd1+(y**2+x**2*z**2)*Hdd2,
            3*x**2*z**2*Sdd0 + (x**2+z**2-4*x**2*z**2) *
            Sdd1 + (y**2+x**2*z**2)*Sdd2)


def hs_xz_x2y2(x, y, z, Hdd0, Sdd0, Hdd1, Sdd1, Hdd2, Sdd2):
    return (1.5*x*z*(x**2-y**2)*Hdd0+x*z*(1-2*(x**2-y**2)) *
            Hdd1-x*z*(1-0.5*(x**2-y**2))*Hdd2,
            1.5*x*z*(x**2-y**2)*Sdd0+x*z*(1-2*(x**2-y**2)) *
            Sdd1-x*z*(1-0.5*(x**2-y**2))*Sdd2)


def hs_xz_3z2r2(x, y, z, Hdd0, Sdd0, Hdd1, Sdd1, Hdd2, Sdd2):
    return (t.sqrt(t.tensor([3.]))*x*z*(z**2-0.5*(x**2+y**2))*Hdd0 +
            t.sqrt(t.tensor([3.]))*x*z*(x**2+y**2-z**2)*Hdd1 -
            t.sqrt(t.tensor([3.]))/2*x*z*(x**2+y**2)*Hdd2,
            t.sqrt(t.tensor([3.]))*x*z*(z**2-0.5*(x**2+y**2))*Sdd0 +
            t.sqrt(t.tensor([3.]))*x*z*(x**2+y**2-z**2)*Sdd1 -
            t.sqrt(t.tensor([3.]))/2*x*z*(x**2+y**2)*Sdd2)


def hs_x2y2_s(x, y, z, Hsd0, Ssd0):
    return hs_s_x2y2(-x, -y, -z, Hsd0, Ssd0)[0], hs_s_x2y2(
            -x, -y, -z, Hsd0, Ssd0)[1]


def hs_x2y2_x(x, y, z, Hpd0, Spd0, Hpd1, Spd1):
    return hs_x_x2y2(-x, -y, -z, Hpd0, Spd0, Hpd1, Spd1)[0], hs_x_x2y2(
            -x, -y, -z, Hpd0, Spd0, Hpd1, Spd1)[1]


def hs_x2y2_y(x, y, z, Hpd0, Spd0, Hpd1, Spd1):
    return hs_y_x2y2(-x, -y, -z, Hpd0, Spd0, Hpd1, Spd1)[0], hs_y_x2y2(
            -x, -y, -z, Hpd0, Spd0, Hpd1, Spd1)[1]


def hs_x2y2_z(x, y, z, Hpd0, Spd0, Hpd1, Spd1):
    return hs_z_x2y2(-x, -y, -z, Hpd0, Spd0, Hpd1, Spd1)[0], hs_z_x2y2(
            -x, -y, -z, Hpd0, Spd0, Hpd1, Spd1)[1]


def hs_x2y2_xy(x, y, z, Hdd0, Sdd0, Hdd1, Sdd1, Hdd2, Sdd2):
    return (hs_xy_x2y2(-x, -y, -z, Hdd0, Sdd0, Hdd1, Sdd1, Hdd2, Sdd2)[0],
            hs_xy_x2y2(-x, -y, -z, Hdd0, Sdd0, Hdd1, Sdd1, Hdd2, Sdd2)[1])


def hs_x2y2_yz(x, y, z, Hdd0, Sdd0, Hdd1, Sdd1, Hdd2, Sdd2):
    return (hs_yz_x2y2(-x, -y, -z, Hdd0, Sdd0, Hdd1, Sdd1, Hdd2, Sdd2)[0],
            hs_yz_x2y2(-x, -y, -z, Hdd0, Sdd0, Hdd1, Sdd1, Hdd2, Sdd2)[1])


def hs_x2y2_xz(x, y, z, Hdd0, Sdd0, Hdd1, Sdd1, Hdd2, Sdd2):
    return (hs_xz_x2y2(-x, -y, -z, Hdd0, Sdd0, Hdd1, Sdd1, Hdd2, Sdd2)[0],
            hs_xz_x2y2(-x, -y, -z, Hdd0, Sdd0, Hdd1, Sdd1, Hdd2, Sdd2)[1])


def hs_x2y2_x2y2(x, y, z, Hdd0, Sdd0, Hdd1, Sdd1, Hdd2, Sdd2):
    return (3/4*(x**2-y**2)**2*Hdd0+(x**2+y**2 -
            (x**2-y**2)**2)*Hdd1+(z**2+1/4*(x**2-y**2)**2)*Hdd2,
            3/4*(x**2-y**2)**2*Sdd0+(x**2+y**2 -
            (x**2-y**2)**2)*Sdd1+(z**2+1/4*(x**2-y**2)**2)*Sdd2)


def hs_x2y2_3z2r2(x, y, z, Hdd0, Sdd0, Hdd1, Sdd1, Hdd2, Sdd2):
    return (t.sqrt(t.tensor([3.]))/2*(x**2-y**2)*(z**2-(x**2+y**2)/2) *
            Hdd0+t.sqrt(t.tensor([3.]))*z**2*(x**2-y**2)*Hdd1 +
            t.sqrt(t.tensor([3.]))/4*(1+z**2)*(x**2-y**2)*Hdd2,
            t.sqrt(t.tensor([3.]))/2*(x**2-y**2)*(z**2-(x**2+y**2)/2) *
            Sdd0+t.sqrt(t.tensor([3.]))*z**2*(x**2-y**2)*Sdd1 +
            t.sqrt(t.tensor([3.]))/4*(1+z**2)*(x**2-y**2)*Sdd2)


def hs_3z2r2_s(x, y, z, Hsd0, Ssd0):
    return (hs_s_3z2r2(-x, -y, -z, Hsd0, Ssd0)[0],
            hs_s_3z2r2(-x, -y, -z, Hsd0, Ssd0)[1])


def hs_3z2r2_x(x, y, z, Hpd0, Spd0, Hpd1, Spd1):
    return (hs_x_3z2r2(-x, -y, -z, Hpd0, Spd0, Hpd1, Spd1)[0],
            hs_x_3z2r2(-x, -y, -z, Hpd0, Spd0, Hpd1, Spd1)[1])


def hs_3z2r2_y(x, y, z, Hpd0, Spd0, Hpd1, Spd1):
    return (hs_y_3z2r2(-x, -y, -z, Hpd0, Spd0, Hpd1, Spd1)[0],
            hs_y_3z2r2(-x, -y, -z, Hpd0, Spd0, Hpd1, Spd1)[1])


def hs_3z2r2_z(x, y, z, Hpd0, Spd0, Hpd1, Spd1):
    return (hs_z_3z2r2(-x, -y, -z, Hpd0, Spd0, Hpd1, Spd1)[0],
            hs_z_3z2r2(-x, -y, -z, Hpd0, Spd0, Hpd1, Spd1)[1])


def hs_3z2r2_xy(x, y, z, Hdd0, Sdd0, Hdd1, Sdd1, Hdd2, Sdd2):
    return (hs_xy_3z2r2(-x, -y, -z, Hdd0, Sdd0, Hdd1, Sdd1, Hdd2, Sdd2)[0],
            hs_xy_3z2r2(-x, -y, -z, Hdd0, Sdd0, Hdd1, Sdd1, Hdd2, Sdd2)[1])


def hs_3z2r2_yz(x, y, z, Hdd0, Sdd0, Hdd1, Sdd1, Hdd2, Sdd2):
    return (hs_yz_3z2r2(-x, -y, -z, Hdd0, Sdd0, Hdd1, Sdd1, Hdd2, Sdd2)[0],
            hs_yz_3z2r2(-x, -y, -z, Hdd0, Sdd0, Hdd1, Sdd1, Hdd2, Sdd2)[1])


def hs_3z2r2_xz(x, y, z, Hdd0, Sdd0, Hdd1, Sdd1, Hdd2, Sdd2):
    return (hs_xz_3z2r2(-x, -y, -z, Hdd0, Sdd0, Hdd1, Sdd1, Hdd2, Sdd2)[0],
            hs_xz_3z2r2(-x, -y, -z, Hdd0, Sdd0, Hdd1, Sdd1, Hdd2, Sdd2)[1])


def hs_3z2r2_x2y2(x, y, z, Hdd0, Sdd0, Hdd1, Sdd1, Hdd2, Sdd2):
    return (hs_x2y2_3z2r2(-x, -y, -z, Hdd0, Sdd0, Hdd1, Sdd1, Hdd2, Sdd2)[0],
            hs_x2y2_3z2r2(-x, -y, -z, Hdd0, Sdd0, Hdd1, Sdd1, Hdd2, Sdd2)[1])


def hs_3z2r2_3z2r2(x, y, z, Hdd0, Sdd0, Hdd1, Sdd1, Hdd2, Sdd2):
    return ((z**2-0.5*(x**2+y**2))**2*Hdd0+3*z**2*(x**2+y**2) *
            Hdd1+3/4*(x**2+y**2)**2*Hdd2,
            (z**2-0.5*(x**2+y**2))**2*Sdd0+3*z**2*(x**2+y**2) *
            Sdd1+3/4*(x**2+y**2)**2*Sdd2)
