"""Slater-Koster integrals related."""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time
import sys
import numpy as np
import torch as t
import matplotlib.pyplot as plt
import matht
from scipy import interpolate
from matht import Bspline, DFTBmath, BicubInterp


class SKTran:
    """Slater-Koster Transformations."""

    def __init__(self, para):
        """Initialize parameters.

        Required args: integral, distance
        Returns:
            [natom, natom, 20] matrix for each calculation

        """
        self.para = para
        self.math = DFTBmath(self.para)

        # if machine learning or not
        if not self.para['Lml']:

            # directly get integrals from .skf or offer integrals
            if not self.para['LreadSKFinterp']:
                self.get_sk_all()

                # build H0 and S with full, symmetric matrices
                if self.para['HSsym'] == 'symall':
                    self.sk_tran_symall()

                # build H0 and S with only half matrices
                elif self.para['HSsym'] == 'symhalf':
                    self.sk_tran_half()

            # use interpolation to get integral with various compression radius
            if self.para['LreadSKFinterp']:

                # build H0 and S with full, symmetric matrices
                if self.para['HSsym'] == 'symall':
                    self.sk_tran_symall()

                # build H0 and S with only half matrices
                elif self.para['HSsym'] == 'symhalf':
                    self.sk_tran_half(para)

        # machine learning is True, some method only apply in this case
        if self.para['Lml']:

            # use ACSF to generate compression radius, then SK transformation
            if self.para['Lml_skf'] or self.para['Lml_acsf']:

                # build H0 and S with full, symmetric matrices
                if self.para['HSsym'] == 'symall':
                    self.sk_tran_symall()

                # build H0, S with half matrices
                elif self.para['HSsym'] == 'symhalf':
                    self.sk_tran_half()

            # directly get integrals with spline, or some other method
            elif self.para['Lml_HS']:
                self.sk_tran_symall()

    def get_sk_all(self):
        """Get integrals from .skf data with given distance."""
        # number of atom in each calculation
        natom = self.para['natom']

        # build H0 or S
        self.para['hs_all'] = t.zeros((natom, natom, 20), dtype=t.float64)

        for iat in range(natom):

            for jat in range(natom):

                # get the name of i, j atom pair
                namei = self.para['atomnameall'][iat]
                namej = self.para['atomnameall'][jat]
                nameij = namei + namej

                # the cutoff is from former step when reading skf file
                cutoff = self.para['cutoffsk' + nameij]

                # the distance is from cal_coor
                dd = self.para['distance'][iat, jat]

                # if distance larger than cutoff, return zero
                if dd > cutoff:
                    print('{} - {} distance out of range'.format(iat, jat))

                elif dd < 1E-1:

                    # two atom are too close, exit
                    if iat != jat:
                        sys.exit()
                else:

                    # get the integral by interpolation from integral table
                    self.para['hsdata'] = self.math.sk_interp(dd, nameij)

                    # make sure type is tensor
                    self.para['hs_all'][iat, jat, :] = \
                        self.data_type(self.para['hsdata'])

    def data_type(self, in_):
        """Make sure the output is tensor type."""
        if type(self.para['hsdata']) is t.Tensor:
            out_ = in_
        else:
            out_ = t.from_numpy(in_)
        return out_

    def sk_tran_half(self):
        """Transfer H and S according to slater-koster rules."""
        # index of the orbitals
        atomind = self.para['atomind']

        # number of atom
        natom = self.para['natom']

        # name of all atom in each calculation
        atomname = self.para['atomnameall']

        # vectors between different atoms (Bohr)
        dvec = self.para['dvec']

        # the sum of orbital index, equal to dimension of H0 and S
        atomind2 = self.para['atomind2']

        # build 1D, half H0, S matrices
        self.para['hammat'] = t.zeros((atomind2), dtype=t.float64)
        self.para['overmat'] = t.zeros((atomind2), dtype=t.float64)

        # temporary distance matrix
        rr = t.zeros((3), dtype=t.float64)

        for iat in range(natom):

            # l of i atom
            lmaxi = self.para['lmaxall'][iat]
            for jat in range(iat):

                # l of j atom
                lmaxj = self.para['lmaxall'][jat]
                lmax = max(lmaxi, lmaxj)

                # temporary H, S with dimension 9 (s, p, d orbitals)
                self.para['hams'] = t.zeros((9, 9), dtype=t.float64)
                self.para['ovrs'] = t.zeros((9, 9), dtype=t.float64)

                # atom name of i and j
                self.para['nameij'] = atomname[iat] + atomname[jat]

                # coordinate vector between ia
                rr[:] = dvec[iat, jat, :]

                # generate ham, over only between i, j (no f orbital)
                self.slkode(rr, iat, jat, lmax, lmaxi, lmaxj)

                # transfer temporary ham and ovr matrices to final H0, S
                for n in range(atomind[jat + 1] - atomind[jat]):
                    nn = atomind[jat] + n
                    for m in range(atomind[iat + 1] - atomind[iat]):

                        # calculate the orbital index in the 1D H0, S matrices
                        mm = atomind[iat] + m
                        idx = int(mm * (mm + 1) / 2 + nn)

                        # controls only half H0, S will be written
                        if nn <= mm:
                            idx = int(mm * (mm + 1) / 2 + nn)
                            self.para['hammat'][idx] = self.para['hams'][m, n]
                            self.para['overmat'][idx] = self.para['ovrs'][m, n]

            # build temporary on-site
            self.para['h_o'] = t.zeros((9), dtype=t.float64)
            self.para['s_o'] = t.zeros((9), dtype=t.float64)

            # get the name between atoms
            self.para['nameij'] = atomname[iat] + atomname[iat]

            # get on-site between i and j atom
            self.slkode_onsite(rr, iat, lmaxi)

            # write on-site between i and j to final on-site matrix
            for m in range(atomind[iat + 1] - atomind[iat]):
                mm = atomind[iat] + m
                idx = int(mm * (mm + 1) / 2 + mm)
                self.para['hammat'][idx] = self.para['h_o'][m]
                self.para['overmat'][idx] = self.para['s_o'][m]

    def sk_tran_symall(self):
        """Transfer H0, S according to Slater-Koster rules.

        writing the symmetric, full 2D H0, S.

        """
        # index of atom orbital
        atomind = self.para['atomind']

        # number of atom
        natom = self.para['natom']

        # total orbitals, equal to dimension of H0, S
        norb = atomind[natom]

        # atom name
        atomname = self.para['atomnameall']

        # atom coordinate vector (Bohr)
        dvec = self.para['dvec']

        # build H0, S
        self.para['hammat'] = t.zeros((norb, norb), dtype=t.float64)
        self.para['overmat'] = t.zeros((norb, norb), dtype=t.float64)

        for iat in range(natom):

            # l of i atom
            lmaxi = self.para['lmaxall'][iat]

            for jat in range(natom):

                # l of j atom
                lmaxj = self.para['lmaxall'][jat]

                # temporary H, S between i and j atom
                self.para['hams'] = t.zeros((9, 9), dtype=t.float64)
                self.para['ovrs'] = t.zeros((9, 9), dtype=t.float64)

                # temporary on-site
                self.para['h_o'] = t.zeros((9), dtype=t.float64)
                self.para['s_o'] = t.zeros((9), dtype=t.float64)

                # name of i and j atom pair
                self.para['nameij'] = atomname[iat] + atomname[jat]

                # distance vector between i and j atom
                rr = dvec[iat, jat, :]

                # for the same atom, where on-site should be construct
                if iat == jat:

                    # get on-site between i and j atom
                    self.slkode_onsite(rr, iat, lmaxi)

                    # write on-site between i and j to final on-site matrix
                    for m in range(atomind[iat + 1] - atomind[iat]):
                        mm = atomind[iat] + m
                        self.para['hammat'][mm, mm] = self.para['h_o'][m]
                        self.para['overmat'][mm, mm] = self.para['s_o'][m]

                # build H0, S with integrals for i, j atom pair
                else:

                    # get H, S with distance, initial integrals
                    self.slkode_ij(rr, iat, jat, lmaxi, lmaxj)

                    # write H0, S of i, j to final H0, S
                    for n in range(atomind[jat + 1] - atomind[jat]):
                        nn = atomind[jat] + n
                        for m in range(atomind[iat + 1] - atomind[iat]):

                            # calculate the off-diagonal orbital index
                            mm = atomind[iat] + m
                            self.para['hammat'][mm, nn] = \
                                self.para['hams'][m, n]
                            self.para['overmat'][mm, nn] = \
                                self.para['ovrs'][m, n]

    def slkode_onsite(self, rr, iat, lmax):
        """Transfer i from ith atom to ith spiece."""
        # name of i and i atom
        nameij = self.para['nameij']

        # s, p, d orbitals onsite
        do, po, so = self.para['onsite' + nameij][:]

        # max(l) is 1, only s orbitals is included in system
        if lmax == 1:
            self.para['h_o'][0] = so
            self.para['s_o'][0] = 1.0

        # max(l) is 2, including p orbitals
        elif lmax == 2:
            self.para['h_o'][0] = so
            self.para['h_o'][1: 4] = po
            self.para['s_o'][: 4] = 1.0

        # max(l) is 3, including d orbital
        else:
            self.para['h_o'][0] = so
            self.para['h_o'][1: 4] = po
            self.para['h_o'][4: 9] = do
            self.para['s_o'][:] = 1.0

    def slkode_ij(self, rr, iat, jat, li, lj):
        """Transfer integrals according to SK rules."""
        # name of i, j atom
        nameij = self.para['nameij']

        # distance between atom i, j
        dd = t.sqrt(t.sum(rr[:] ** 2))

        if not self.para['Lml']:
            if self.para['LreadSKFinterp']:
                cutoff = self.para['interpcutoff']
            else:
                cutoff = self.para['cutoffsk' + nameij]
        if self.para['Lml']:
            if self.para['Lml_skf'] or self.para['Lml_acsf']:
                cutoff = self.para['interpcutoff']  # may need revise!!!
            elif self.para['Lml_HS']:
                cutoff = self.para['interpcutoff']

        if dd > cutoff:
            print('{}atom-{}atom distance out of range'.format(iat, jat))
            print(dd, cutoff)
        elif dd < 1E-1:
            print("ERROR, distance between", iat, "and", jat, 'is too close')
            sys.exit()
        else:
            self.sk_(rr, iat, jat, dd, li, lj)

    def sk_(self, xyz, iat, jat, dd, li, lj):
        """SK transformations with defined parameters."""
        # get the temporary H, S for i, j atom pair
        hams = self.para['hams']
        ovrs = self.para['ovrs']

        # get the maximum and minimum of l
        lmax, lmin = max(li, lj), min(li, lj)

        # get distance along x, y, z
        xx, yy, zz = xyz[:] / dd

        # SK transformation according to parameter l
        if lmax == 1:
            skss(self.para, xx, yy, zz, iat, jat, hams, ovrs, li, lj)
        elif lmin == 1 and lmax == 2:
            sksp(self.para, xx, yy, zz, iat, jat, hams, ovrs, li, lj)
        elif lmin == 2 and lmax == 2:
            skpp(self.para, xx, yy, zz, iat, jat, hams, ovrs, li, lj)
        return hams, ovrs

    def slkode(self, rr, iat, jat, lmax, li, lj):
        """Transfer i from ith atom to ith spiece."""
        nameij = self.para['nameij']
        dd = t.sqrt((rr[:] ** 2).sum())
        xx, yy, zz = rr / dd

        # get the maximum and minimum of l
        lmax, lmin = max(li, lj), min(li, lj)

        # get cutoff from machine learning input
        if self.para['Lml']:
            cutoff = self.para['interpcutoff']
            if self.para['Lml_skf']:
                self.para['hsdata'] = self.para['hs_all'][iat, jat]
            else:
                getsk(self.para, nameij, dd)

        # read SKF and get cutoff from SKF for each atom pair
        else:
            getsk(self.para, nameij, dd)
            cutoff = self.para['cutoffsk' + nameij]
        skselfnew = t.zeros((3), dtype=t.float64)

        # distance between atoms is out of range
        if dd > cutoff:
            return self.para

        # distance between atoms is zero
        if dd < 1E-4:
            if iat != jat:
                print("ERROR, distance between", iat, "and", jat, "atom is 0")
            else:
                if type(self.para['onsite' + nameij]) is t.Tensor:
                    skselfnew[:] = self.para['onsite' + nameij]
                elif type(self.para['coorall'][0]) is np.ndarray:
                    skselfnew[:] = t.FloatTensor(self.para['onsite' + nameij])

            # max of l is 1, therefore only s orbital is included
            if lmax == 1:
                self.para['hams'][0, 0] = skselfnew[2]
                self.para['ovrs'][0, 0] = 1.0

            # max of l is 2, therefore p orbital is included
            elif lmax == 2:
                self.para['hams'][0, 0] = skselfnew[2]

                t.diag(self.para['hams'])[1: 4] = skselfnew[:]
                t.diag(self.para['ovrs'])[: 4] = 1.0
            # for d orbital, in to do list...

        # get integral with given distance
        else:
            # directly get integral from spline
            if self.para['Lml_HS']:
                shparspline(self.para, rr, iat, jat, dd)
            else:
                # shpar(para, para['hs_all'][i, j], rr, i, j, dd)
                if lmax == 1:
                    skss(self.para, xx, yy, zz, iat, jat,
                         self.para['hams'], self.para['ovrs'], li, lj)
                elif lmin == 1 and lmax == 2:
                    sksp(self.para, xx, yy, zz, iat, jat,
                         self.para['hams'], self.para['ovrs'], li, lj)
                elif lmin == 2 and lmax == 2:
                    skpp(self.para, xx, yy, zz, iat, jat,
                         self.para['hams'], self.para['ovrs'], li, lj)


class SKinterp:
    """Get integral from interpolation."""

    def __init__(self, para):
        """Initialize parameters."""
        self.para = para

    def genskf_interp_ij(self):
        """Generate sk integral with various compression radius.

        Args:
            atomnameall (list): all the atom name
            natom (int): number of atom
            distance (2D tensor): distance between all atoms
        Returns:
            hs_compr_all (out): [natom, natom, ncompr, ncompr, 20]

        """
        # all atom name for current calculation
        atomname = self.para['atomnameall']

        # number of atom
        natom = self.para['natom']

        # atom specie
        atomspecie = self.para['atomspecie']

        # number of compression radius grid points
        ncompr = self.para['ncompr']

        # build integral with various compression radius
        self.para['hs_compr_all'] = t.zeros(natom, natom, ncompr, ncompr, 20)

        # get i and j atom with various compression radius at certain dist
        print('build matrix: [N, N, N_R, N_R, 20]')
        print('N is number of atom, N_R is number of compression')

        for iatom in range(natom):
            for jatom in range(natom):
                dij = self.para['distance'][iatom, jatom]
                namei, namej = atomname[iatom], atomname[jatom]
                nameij = namei + namej
                compr_grid = self.para[namei + '_compr_grid']
                self.para['hs_ij'] = t.zeros(ncompr, ncompr, 20)

                if dij > 1e-2:
                    self.genskf_interp_ijd_4d(dij, nameij, compr_grid)
                self.para['hs_compr_all'][iatom, jatom, :, :, :] = \
                    self.para['hs_ij']

        for iat in atomspecie:

            # onsite is the same, therefore read [0, 0] instead
            onsite = t.zeros((3), dtype=t.float64)
            uhubb = t.zeros((3), dtype=t.float64)
            onsite[:] = self.para['onsite' + iat]
            uhubb[:] = self.para['uhubb' + iat]
            self.para['onsite' + iat + iat] = onsite
            self.para['uhubb' + iat + iat] = uhubb

    def genskf_interp_ijd_old(self, dij, nameij, rgrid):
        """Interpolate skf of i and j atom with various compression radius."""
        cutoff = self.para['interpcutoff']
        ncompr = int(np.sqrt(self.para['nfile_rall' + nameij]))
        for icompr in range(0, ncompr):
            for jcompr in range(0, ncompr):
                grid_dist = \
                    self.para['grid_dist_rall' + nameij][icompr, jcompr]
                skfijd = \
                    self.para['hs_all_rall' + nameij][icompr, jcompr, :, :]
                col = skfijd.shape[1]
                for icol in range(0, col):
                    if (max(skfijd[:, icol]), min(skfijd[:, icol])) == (0, 0):
                        self.para['hs_ij'][icompr, jcompr, icol] = 0.0
                    else:
                        nline = int((cutoff - grid_dist) / grid_dist + 1)
                        xp = t.linspace(grid_dist, nline * grid_dist, nline)
                        yp = skfijd[:, icol][:nline]
                        self.para['hs_ij'][icompr, jcompr, icol] = \
                            matht.polyInter(xp, yp, dij)

    def genskf_interp_ijd(self, dij, nameij, rgrid):
        """Interpolate skf of i and j atom with various compression radius."""
        cutoff = self.para['interpcutoff']
        ncompr = int(np.sqrt(self.para['nfile_rall' + nameij]))
        assert self.para['grid_dist_rall' + nameij][0, 0] == \
            self.para['grid_dist_rall' + nameij][-1, -1]
        grid_dist = self.para['grid_dist_rall' + nameij][0, 0]
        nline = int((cutoff - grid_dist) / grid_dist + 1)
        xp = t.linspace(grid_dist, nline * grid_dist, nline)
        # timelist = [0]

        for icompr in range(0, ncompr):
            for jcompr in range(0, ncompr):
                # timelist.append(time.time())
                # print('timeijd:', timelist[-1] - timelist[-2])
                skfijd = \
                    self.para['hs_all_rall' + nameij][icompr, jcompr, :, :]
                col = skfijd.shape[1]
                for icol in range(0, col):
                    if (max(skfijd[:, icol]), min(skfijd[:, icol])) == (0, 0):
                        self.para['hs_ij'][icompr, jcompr, icol] = 0.0
                    else:
                        yp = skfijd[:, icol][:nline]
                        func = interpolate.interp1d(xp.numpy(), yp.numpy(), kind='cubic')
                        self.para['hs_ij'][icompr, jcompr, icol] = \
                            t.from_numpy(func(dij))

    def genskf_interp_ijd_(self, dij, nameij, rgrid):
        """Interpolate skf of i and j atom with various compression radius."""
        # cutoff = self.para['interpcutoff']
        assert self.para['grid_dist_rall' + nameij][0, 0] == \
            self.para['grid_dist_rall' + nameij][-1, -1]
        self.para['grid_dist' + nameij] = \
            self.para['grid_dist_rall' + nameij][0, 0]
        self.para['ngridpoint' + nameij] = \
            self.para['ngridpoint_rall' + nameij].min()
        ncompr = int(np.sqrt(self.para['nfile_rall' + nameij]))
        for icompr in range(0, ncompr):
            for jcompr in range(0, ncompr):
                self.para['hs_all' + nameij] = \
                    self.para['hs_all_rall' + nameij][icompr, jcompr, :, :]
                # col = skfijd.shape[1]
                self.para['hs_ij'][icompr, jcompr, :] = \
                    DFTBmath(self.para).sk_interp(dij, nameij)

    def genskf_interp_ijd_4d(self, dij, nameij, rgrid):
        """Interpolate skf of i and j atom with various compression radius."""
        # cutoff = self.para['interpcutoff']
        assert self.para['grid_dist_rall' + nameij][0, 0] == \
            self.para['grid_dist_rall' + nameij][-1, -1]
        self.para['grid_dist' + nameij] = \
            self.para['grid_dist_rall' + nameij][0, 0]
        self.para['ngridpoint' + nameij] = \
            self.para['ngridpoint_rall' + nameij].min()
        ncompr = int(np.sqrt(self.para['nfile_rall' + nameij]))
        self.para['hs_all' + nameij] = \
            self.para['hs_all_rall' + nameij][:, :, :, :]
        self.para['hs_ij'][:, :, :] = \
            DFTBmath(self.para).sk_interp_4d(dij, nameij, ncompr)

    def genskf_interp_r(self, para):
        """Generate interpolation of SKF with given compression radius.

        Args:
            compression R
            H and S between all atoms ([ncompr, ncompr, 20] * natom * natom)
        Return:
            H and S matrice ([natom, natom, 20])

        """
        natom = para['natom']
        atomname = para['atomnameall']
        bicubic = BicubInterp()
        hs_ij = t.zeros(natom, natom, 20)

        print('Getting HS table according to compression R and build matrix:',
              '[N_atom1, N_atom2, 20], also for onsite and uhubb')

        icount = 0
        for iatom in range(natom):
            iname = atomname[iatom]
            xmesh = para[iname + '_compr_grid']
            for jatom in range(natom):
                jname = atomname[jatom]
                ymesh = para[jname + '_compr_grid']
                icompr = para['compr_ml'][iatom]
                jcompr = para['compr_ml'][jatom]
                zmeshall = para['hs_compr_all'][icount]
                for icol in range(0, 20):
                    hs_ij[iatom, jatom, icol] = \
                        bicubic.bicubic_2d(xmesh, ymesh, zmeshall[:, :, icol],
                                           icompr, jcompr)
                icount += 1

            onsite = t.zeros(3)
            uhubb = t.zeros(3)
            for icol in range(0, 3):
                zmesh_onsite = para['onsite_rall' + iname + iname]
                zmesh_uhubb = para['uhubb_rall' + iname + iname]
                onsite[icol] = \
                    bicubic.bicubic_2d(xmesh, ymesh, zmesh_onsite[:, :, icol],
                                       icompr, jcompr)
                uhubb[icol] = \
                    bicubic.bicubic_2d(xmesh, ymesh, zmesh_uhubb[:, :, icol],
                                       icompr, jcompr)
                para['onsite' + iname + iname] = onsite
                para['uhubb' + iname + iname] = uhubb
        para['hs_all'] = hs_ij

    def genskf_interp_compr(self):
        """Generate interpolation of SKF with given compression radius.

        Args:
            compression R
            H and S between all atoms ([ncompr, ncompr, 20] * natom * natom)
        Return:
            H and S matrice ([natom, natom, 20])

        """
        natom = self.para['natom']
        atomname = self.para['atomnameall']
        bicubic = BicubInterp()
        hs_ij = t.zeros(natom, natom, 20)
        timelist = [0]
        timelist.append(time.time())
        print('Getting HS table according to compression R and build matrix:',
              '[N_atom1, N_atom2, 20], also for onsite and uhubb')

        icount = 0
        for iatom in range(natom):
            iname = atomname[iatom]
            icompr = self.para['compr_ml'][iatom]
            xmesh = self.para[iname + '_compr_grid']
            for jatom in range(natom):
                jname = atomname[jatom]
                ymesh = self.para[jname + '_compr_grid']
                jcompr = self.para['compr_ml'][jatom]
                zmeshall = self.para['hs_compr_all'][iatom, jatom]
                if iatom != jatom:
                    for icol in range(0, 20):
                        hs_ij[iatom, jatom, icol] = \
                            bicubic.bicubic_2d(
                                    xmesh, ymesh, zmeshall[:, :, icol],
                                    icompr, jcompr)
                    '''hs_ij[iatom, jatom, :] = bicubic.bicubic_3d(
                            xmesh, ymesh, zmeshall[:, :, :], icompr, jcompr)'''
                icount += 1
                timelist.append(time.time())
        self.para['hs_all'] = hs_ij
        timelist.append(time.time())
        print('total time genskf_interp_compr:', timelist[-1] - timelist[1])

class SKspline:
    """Get integral from spline."""

    def __init__(self, para):
        """Initialize parameters."""
        self.para = para

    def get_sk_spldata(self):
        """According to the type of interpolation, call different function."""
        print('-' * 35, 'Generating H or S spline data', '-' * 35)
        if self.para['interptype'] == 'Bspline':
            self.gen_bsplpara()
        elif self.para['interptype'] == 'Polyspline':
            self.gen_psplpara()

    def gen_bsplpara(self):
        """Generate B-spline parameters."""
        h_spl_num = self.para['h_spl_num']
        lines = int(self.cutoff / self.dist)
        cspline = t.zeros(h_spl_num, lines)
        cspl_rand = t.zeros(h_spl_num, lines)
        ihtable = 0
        for ii in range(0, self.nspecie):
            for jj in range(0, self.nspecie):
                nameij = self.atomspecie[ii] + self.atomspecie[jj]
                griddist = self.para['grid_dist'+nameij]
                ngridpoint = self.para['ngridpoint'+nameij]
                t_beg = 0.0
                t_end = ngridpoint*griddist
                t_num = int((t_end - t_beg + griddist)/self.dist)
                tspline = t.linspace(t_beg, t_end, t_num)
                cspline = _cspline(self.para, nameij, ihtable, griddist,
                                   cspline, tspline)
                ihtable += HNUM[self.atomspecie[ii]+self.atomspecie[jj]]
        shape1, shape2 = cspline.shape
        cspl_rand = cspline + t.randn(shape1, shape2)/10
        self.para['cspline'] = cspline
        self.para['cspl_rand'] = cspl_rand
        self.para['tspline'] = tspline

    def gen_psplpara(self):
        """Generate spline parameters."""
        atomspecie = self.para['atomspecie']
        cutoff = self.para['interpcutoff']
        dist = self.para['interpdist']
        h_spl_num = self.para['h_spl_num']
        nspecie = len(atomspecie)
        xp_start = t.zeros(nspecie)
        for ii in range(0, nspecie):
            nameij = atomspecie[ii] + atomspecie[ii]
            xp_start[ii] = self.para['grid_dist' + nameij]
            if ii > 0:
                assert xp_start[ii] == xp_start[ii - 1]
        lines = int((cutoff - xp_start[0]) / dist + 1)
        self.para['splyall'] = t.zeros(h_spl_num, lines)
        self.para['splyall_rand'] = t.zeros(h_spl_num, lines)

        # ihtable is label of which orbital and which specie it is
        for ii in range(0, nspecie):
            for jj in range(0, nspecie):
                nameij = atomspecie[ii] + atomspecie[jj]
                griddist = self.para['grid_dist' + nameij]
                self.para['interp_xall'] = t.linspace(
                        xp_start[0], cutoff, lines)
                if HNUM[nameij] > 0:
                    spl_ypara(self.para, nameij, griddist, xp_start[0], lines)

        # build rand interpspline data (add randn number)
        self.para['splyall_rand'][:, :] = self.para['splyall'][:, :]
        row, col = self.para['splyall'].shape
        self.para['splyall_rand'] = self.para['splyall_rand'] + \
            t.randn(row, col) * self.para['rand_threshold']

    def add_rand(self, tensor_init, tensor_rand, threshold, multi_para):
        """Add random values."""
        if len(tensor_init.shape) == 1:
            tensor_temp = t.zeros(len(tensor_init))
            tensor_temp[:] = tensor_init[:]
            tensor_temp[tensor_temp > threshold] += t.randn(1) * multi_para
            tensor_rand[:] == tensor_temp[:]
        elif len(tensor_init.shape) == 2:
            tensor_temp = t.zeros(tensor_init.shape[0], tensor_init.shape[1])
            tensor_temp[:, :] = tensor_init[:, :]
            tensor_temp[tensor_temp > threshold] += t.randn(1) * multi_para
            tensor_rand[:, :] == tensor_temp[:, :]


def call_spline(xp, yp, rr, ty, order=2):
    if ty == 'Polyspline':
        y_rr = matht.polyInter(xp, yp, rr)
    elif ty == 'Bspline':
        y_rr = Bspline().bspline(rr, xp, yp, order)
    return y_rr


def _cspline(para, nameij, itable, ngridpoint, c_spline, t_spline):
    '''
    according to the griddist (distance between two pints) in .skf file and
    griddist in new B-spline interpolation, read data from SK table, then build
    the x and y for B-spline interpolation in matht.py file
    what you need: start point of x and grid distance of x; SK table data;
    define the interpolation type;
    '''
    datalist = para['hs_all' + nameij]
    nlinesk = para['ngridpoint' + nameij]
    dist = para['interpdist']
    ninterval = int(dist / ngridpoint)
    nhtable = HNUM[nameij]
    datalist_arr = np.asarray(datalist)
    if nhtable == 1:

        # the default distance diff in .skf is 0.02, we can set flexible
        # distance by para: splinedist, ngridpoint
        for ii in range(0, nlinesk):
            if ii % ninterval == 0:
                c_spline[itable, int(ii/ninterval)] = datalist_arr[ii, 9]

    # s and p orbital
    elif nhtable == 2:
        for ii in range(0, nlinesk):
            if ii % ninterval == 0:
                c_spline[itable, int(ii/ninterval)] = datalist_arr[ii, 9]
                c_spline[itable + 1, int(ii/ninterval)] = datalist_arr[ii, 8]

    # the squeues is ss0, sp0, pp0, pp1
    elif nhtable == 4:
        for ii in range(0, nlinesk):
            if ii % ninterval == 0:
                c_spline[itable, int(ii/ninterval)] = datalist_arr[ii, 9]
                c_spline[itable + 1, int(ii/ninterval)] = datalist_arr[ii, 8]
                c_spline[itable + 2, int(ii/ninterval)] = datalist_arr[ii, 5]
                c_spline[itable + 3, int(ii/ninterval)] = datalist_arr[ii, 6]
    elif nhtable == 0:
        pass
    return c_spline


def spl_ypara(para, nameij, ngridpoint, xp0, lines):
    '''
    according to the griddist (distance between two pints) in .skf file and
    griddist in new spline interpolation, read data from SK table, then build
    the x and y for spline interpolation in matht.py file
    what you need: start point of x and grid distance of x; SK table data;
    define the interpolation type; how many lines (points) in x or y.
    '''
    xp = para['interp_xall']
    datalist = para['hs_all' + nameij]
    dist = para['interpdist']
    ty = para['interptype']
    if int(dist / ngridpoint) != dist / ngridpoint:
        raise ValueError('interpdist must be int multiple of ngridpoint')
    else:
        intv = int(dist / ngridpoint)
        line_beg = int(xp0 / ngridpoint)
    indx = para['spl_label'].index(nameij)
    nhtable = int(para['spl_label'][indx + 2])
    itab = int(para['spl_label'][indx + 1]) - int(para['spl_label'][indx + 2])
    datalist_arr = np.asarray(datalist)

    # ss0 orbital (e.g. H-H)
    if nhtable == 1:
        for ii in range(0, lines):
            iline = int(ii * intv + line_beg - 1)
            para['splyall'][itab, ii] = datalist_arr[iline, 9]

        # the following is for test (ss0)
        yspline = para['splyall']
        fig, ax = plt.subplots()
        ax.plot(xp, [call_spline(xp, yspline[itab, :], rr, ty) for rr in xp],
                'r-', lw=3, label='spline')
        ax.plot(xp, yspline[itab, :lines], 'y-', lw=3, label='spline')
        xx = t.linspace(0, xp[-1], len(xp))
        print('plot the spline data')
        ax.plot(xx, datalist_arr[:len(xp), 9], 'b-', lw=3, label='original')
        plt.show()

    # ss0 and sp0 orbital (e.g. C-H)
    elif nhtable == 2:
        for ii in range(0, lines):
            iline = int(ii * intv + line_beg - 1)
            para['splyall'][itab, ii] = datalist_arr[iline, 9]
            para['splyall'][itab + 1, ii] = datalist_arr[iline, 8]

    # ss0, sp0, pp0, pp1 orbital (e.g. C-H)
    elif nhtable == 4:
        for ii in range(0, lines):
            iline = int(ii * intv + line_beg - 1)
            para['splyall'][itab, ii] = datalist_arr[iline, 9]
            para['splyall'][itab + 1, ii] = datalist_arr[iline, 8]
            para['splyall'][itab + 2, ii] = datalist_arr[iline, 5]
            para['splyall'][itab + 3, ii] = datalist_arr[iline, 6]
    return para


def getsk_(para, rr, i, j, li, lj):
    """Read .skf data."""
    dd = t.sqrt(t.sum(rr[:] ** 2))
    namei, namej = para['atomnameall'][i], para['atomnameall'][j]
    nameij, nameji = namei + namej, namej + namei
    lmax, lmin = max(li, lj), min(li, lj)
    if lmax == 1:
        getsk(para, nameij, dd)
        para['hsdataij'] = para['hsdata']
    elif lmin == 1 and lmax == 2:
        getsk(para, nameij, dd)
        para['hsdataij'] = para['hsdata']
        getsk(para, nameji, dd)
        para['hsdataji'] = para['hsdata']
    elif lmin == 2 and lmax == 2:
        getsk(para, nameij, dd)
        para['hsdataij'] = para['hsdata']
        getsk(para, nameji, dd)
        para['hsdataji'] = para['hsdata']


def getsk(para, nameij, dd):
    # ninterp is the num of points for interpolation, here is 8
    ninterp = para['ninterp']
    datalist = para['hs_all' + nameij]
    griddist = para['grid_dist' + nameij]
    cutoff = para['cutoffsk' + nameij]
    ngridpoint = para['ngridpoint' + nameij]
    grid0 = para['grid_dist' + nameij]
    ind = int(dd / griddist)
    ilast = ngridpoint
    lensk = ilast * griddist
    para['hsdata'] = t.zeros(20)
    if dd < grid0:
        para['hsdata'][:] = 0
    elif grid0 <= dd < lensk:  # need revise!!!
        datainterp = t.zeros(int(ninterp), 20)
        ddinterp = t.zeros(int(ninterp))
        ilast = min(ilast, int(ind + ninterp / 2 + 1))
        ilast = max(ninterp, ilast)
        for ii in range(0, ninterp):
            ddinterp[ii] = (ilast - ninterp + ii) * griddist
        datainterp[:, :] = t.from_numpy(
                np.array(datalist[ilast - ninterp - 1:ilast - 1]))
        para['hsdata'] = DFTBmath(para).polysk3thsk(datainterp, ddinterp, dd)
    elif dd >= lensk and dd <= cutoff:
        datainterp = t.zeros(ninterp, 20)
        ddinterp = t.zeros(ninterp)
        datainterp[:, :] = datalist[ngridpoint - ninterp:ngridpoint]
        ddinterp = t.linspace((nline - nup) * griddist, (nline + ndown - 1) * \
                              griddist, num=ninterp)
        para['hsdata'] = DFTBmath(para).polysk5thsk(datainterp, ddinterp, dd)
    else:
        print('Error: the {} distance > cutoff'.format(nameij))
    return para


def shparspline(para, xyz, i, j, dd):
    '''
    this function is to update para['hams'], para['ovrs']
    '''
    xx = xyz[0]/dd
    yy = xyz[1]/dd
    zz = xyz[2]/dd
    lmaxi = para['lmaxall'][i]
    lmaxj = para['lmaxall'][j]
    nameij = para['atomnameall'][i] + para['atomnameall'][j]
    nameji = para['atomnameall'][j] + para['atomnameall'][i]
    # here we need revise !!!!!!!!!
    if HNUM[nameij] == 0:
        nameij = nameji
    indx = para['spl_label'].index(nameij)
    nhtable = int(para['spl_label'][indx + 2])
    itab = int(para['spl_label'][indx + 1]) - int(para['spl_label'][indx + 2])
    maxmax = max(lmaxi, lmaxj)
    minmax = min(lmaxi, lmaxj)
    if para['interptype'] == 'Polyspline':
        if para['cal_ref']:
            yspline = para['splyall']
        elif not para['cal_ref']:
            yspline = para['splyall_rand']
        # if it is 1D, we should write this way to avoid produce 2D tensor
        # and the row == 1
        if nhtable == 1:
            para['interp_y'] = yspline[itab, :]
        else:
            para['interp_y'] = yspline[itab:itab + nhtable, :]
        if maxmax == 1:
            skss_spline(para, xx, yy, zz, dd)
        elif maxmax == 2 and minmax == 1:
            sksp_spline(para, xx, yy, zz, dd)
        elif maxmax == 2 and minmax == 2:
            skpp_spline(para, xx, yy, zz, dd)
        elif maxmax == 3 and minmax == 1:
            sksd(xx, yy, zz)
        elif maxmax == 3 and minmax == 2:
            skpd(xx, yy, zz)
        elif maxmax == 3 and minmax == 3:
            skdd(xx, yy, zz)
    elif para['splinetype'] == 'Bspline':
        tspline = para['tspline']
        cspline = para['cspline']
        if nameij == 'HH':
            spline = cspline[0, :]
        elif nameij == 'CH' or nameij == 'HC':
            spline = cspline[1:3, :]
        elif nameij == 'CC':
            spline = cspline[3:7, :]
        if maxmax == 1:
            skssbspline(xx, yy, zz, dd, tspline, spline)
        elif maxmax == 2 and minmax == 1:
            skspbspline(xx, yy, zz, dd, tspline, spline)
        elif maxmax == 2 and minmax == 2:
            skppbspline(xx, yy, zz, dd, tspline, spline)
        elif maxmax == 3 and minmax == 1:
            sksd(xx, yy, zz, hs_data, hams, ovrs)
        elif maxmax == 3 and minmax == 2:
            skpd(xx, yy, zz, hs_data, hams, ovrs)
        elif maxmax == 3 and minmax == 3:
            skdd(xx, yy, zz, hs_data, hams, ovrs)
    return para


def skss(para, xx, yy, zz, i, j, ham, ovr, li, lj):
    """slater-koster transfermaton for s orvitals"""
    hs_all = para['hs_all']
    ham[0, 0], ovr[0, 0] = hs_s_s(
            xx, yy, zz, hs_all[i, j, 9], hs_all[i, j, 19])
    return ham, ovr



def skssbspline(xx, yy, zz, dd, t, c, data, ham, ovr):
    """slater-koster transfermaton for s orvitals"""
    h_data = Bspline().bspline(dd, t, c, 2)
    ham[0, 0], ovr[0, 0] = hs_s_s(xx, yy, zz, h_data, data[19])
    return ham, ovr


def skss_spline(para, xx, yy, zz, dd):
    """slater-koster transfermaton for s orvitals"""
    data = para['hsdata']
    splx = para['interp_xall']
    sply = para['interp_y']
    h_data = call_spline(splx, sply, dd, para['interptype'])
    para['hams'][0, 0], para['ovrs'][0, 0] = hs_s_s(
        xx, yy, zz, h_data, data[19])
    return para


def sksp(para, xx, yy, zz, i, j, ham, ovr, li, lj):
    """SK tranformation of s and p orbitals."""
    hs_all = para['hs_all']
    ham, ovr = skss(para, xx, yy, zz, i, j, ham, ovr, li, lj)
    if li == lj:
        ham[0, 1], ovr[0, 1] = hs_s_x(
                xx, yy, zz, hs_all[i, j, 8], hs_all[i, j, 18])
        ham[0, 2], ovr[0, 2] = hs_s_y(
                xx, yy, zz, hs_all[i, j, 8], hs_all[i, j, 18])
        ham[0, 3], ovr[0, 3] = hs_s_z(
                xx, yy, zz, hs_all[i, j, 8], hs_all[i, j, 18])
        ham[1, 0], ovr[1, 0] = hs_s_x(
                xx, yy, zz, -hs_all[j, i, 8], -hs_all[j, i, 18])
        ham[2, 0], ovr[2, 0] = hs_s_y(
                xx, yy, zz, -hs_all[j, i, 8], -hs_all[j, i, 18])
        ham[3, 0], ovr[3, 0] = hs_s_z(
                xx, yy, zz, -hs_all[j, i, 8], -hs_all[j, i, 18])
    elif li < lj:
        ham[0, 1], ovr[0, 1] = hs_s_x(
                xx, yy, zz, hs_all[i, j, 8], hs_all[i, j, 18])
        ham[0, 2], ovr[0, 2] = hs_s_y(
                xx, yy, zz, hs_all[i, j, 8], hs_all[i, j, 18])
        ham[0, 3], ovr[0, 3] = hs_s_z(
                xx, yy, zz, hs_all[i, j, 8], hs_all[i, j, 18])
        ham[1, 0], ovr[1, 0] = hs_s_x(
                xx, yy, zz, hs_all[i, j, 8], hs_all[i, j, 18])
        ham[2, 0], ovr[2, 0] = hs_s_y(
                xx, yy, zz, hs_all[i, j, 8], hs_all[i, j, 18])
        ham[3, 0], ovr[3, 0] = hs_s_z(
                xx, yy, zz, hs_all[i, j, 8], hs_all[i, j, 18])
    elif li > lj:
        ham[0, 1], ovr[0, 1] = hs_s_x(
                xx, yy, zz, -hs_all[j, i, 8], -hs_all[j, i, 18])
        ham[0, 2], ovr[0, 2] = hs_s_y(
                xx, yy, zz, -hs_all[j, i, 8], -hs_all[j, i, 18])
        ham[0, 3], ovr[0, 3] = hs_s_z(
                xx, yy, zz, -hs_all[j, i, 8], -hs_all[j, i, 18])
        ham[1, 0], ovr[1, 0] = hs_s_x(
                xx, yy, zz, -hs_all[j, i, 8], -hs_all[j, i, 18])
        ham[2, 0], ovr[2, 0] = hs_s_y(
                xx, yy, zz, -hs_all[j, i, 8], -hs_all[j, i, 18])
        ham[3, 0], ovr[3, 0] = hs_s_z(
                xx, yy, zz, -hs_all[j, i, 8], -hs_all[j, i, 18])
    return ham, ovr


def skspbspline(xx, yy, zz, dd, t, c, data, ham, ovr):
    ham, ovr = skssbspline(xx, yy, zz, dd, t, c[0, :], data, ham, ovr)
    h_data = Bspline().bspline(dd, t, c[1, :], 2)
    ham[0, 1], ovr[0, 1] = hs_s_x(xx, yy, zz, h_data, data[18])
    ham[0, 2], ovr[0, 2] = hs_s_y(xx, yy, zz, h_data, data[18])
    ham[0, 3], ovr[0, 3] = hs_s_z(xx, yy, zz, h_data, data[18])
    for ii in range(nls, nlp + nls):
        ham[ii, 0] = -ham[0, ii]
        ovr[ii, 0] = -ovr[0, ii]
    return ham, ovr


def sksp_spline(para, xx, yy, zz, dd):
    data = para['hsdata']
    splx = para['interp_xall']
    sply = para['interp_y']
    h_data = call_spline(splx, sply[1, :], dd, para['interptype'])
    para['interp_y'] = para['interp_y'][0, :]
    skss_spline(para, xx, yy, zz, dd)
    para['hams'][0, 1], para['ovrs'][0, 1] = hs_s_x(
        xx, yy, zz, h_data, data[18])
    para['hams'][0, 2], para['ovrs'][0, 2] = hs_s_y(
        xx, yy, zz, h_data, data[18])
    para['hams'][0, 3], para['ovrs'][0, 3] = hs_s_z(
        xx, yy, zz, h_data, data[18])
    for ii in range(nls, nlp+nls):
        para['hams'][0, ii] = -para['hams'][ii, 0]
        para['ovrs'][0, ii] = -para['ovrs'][ii, 0]
    return para


def sksd(xx, yy, zz, data, ham, ovr):
    ham, ovr = sksp(xx, yy, zz, data, ham, ovr)
    ham[0, 4], ovr[0, 4] = hs_s_xy(xx, yy, zz, data[7], data[17])
    ham[0, 5], ovr[0, 5] = hs_s_yz(xx, yy, zz, data[7], data[17])
    ham[0, 6], ovr[0, 6] = hs_s_xz(xx, yy, zz, data[7], data[17])
    ham[0, 7], ovr[0, 7] = hs_s_x2y2(xx, yy, zz, data[7], data[17])
    ham[0, 8], ovr[0, 8] = hs_s_3z2r2(xx, yy, zz, data[7], data[17])
    for ii in range(nls + nlp, nld):
        ham[ii, 0] = ham[0, ii]
        ovr[ii, 0] = ovr[0, ii]
    return ham, ovr


def skpp(para, xx, yy, zz, i, j, ham, ovr, li, lj):
    """SK tranformation of p and p orbitals."""
    # hs_all is a matrix with demension [natom, natom, 20]
    hs_all = para['hs_all']

    # parameter control the orbital number
    nls = para['nls']
    nlp = para['nlp']

    # call sksp_ to build sp, ss orbital integral matrix
    ham, ovr = sksp(para, xx, yy, zz, i, j, ham, ovr, li, lj)

    ham[1, 1], ovr[1, 1] = hs_x_x(
            xx, yy, zz, hs_all[i, j, 5], hs_all[i, j, 15],
            hs_all[i, j, 6], hs_all[i, j, 16])
    ham[1, 2], ovr[1, 2] = hs_x_y(
            xx, yy, zz, hs_all[i, j, 5], hs_all[i, j, 15],
            hs_all[i, j, 6], hs_all[i, j, 16])
    ham[1, 3], ovr[1, 3] = hs_x_z(
            xx, yy, zz, hs_all[i, j, 5], hs_all[i, j, 15],
            hs_all[i, j, 6], hs_all[i, j, 16])
    ham[2, 2], ovr[2, 2] = hs_y_y(
            xx, yy, zz, hs_all[i, j, 5], hs_all[i, j, 15],
            hs_all[i, j, 6], hs_all[i, j, 16])
    ham[2, 3], ovr[2, 3] = hs_y_z(
            xx, yy, zz, hs_all[i, j, 5], hs_all[i, j, 15],
            hs_all[i, j, 6], hs_all[i, j, 16])
    ham[3, 3], ovr[3, 3] = hs_z_z(
            xx, yy, zz, hs_all[i, j, 5], hs_all[i, j, 15],
            hs_all[i, j, 6], hs_all[i, j, 16])

    # the pp orbital, the transpose is the same
    for ii in range(nls, nlp + nls):
        for jj in range(nls, ii + nls):
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
        for jj in range(nls, ii + nls):
            ham[ii, jj] = ham[jj, ii]
            ovr[ii, jj] = ovr[jj, ii]
    return ham, ovr


def skpp_spline(para, xx, yy, zz, dd):
    data = para['hsdata']
    splx = para['interp_xall']
    sply = para['interp_y']
    h_pp0 = call_spline(splx, sply[2, :], dd), para['interptype']
    h_pp1 = matht.polyInter(splx, sply[3, :], dd, para['interptype'])
    para['interp_y'] = para['interp_y'][0:2, :]
    sksp_spline(para, xx, yy, zz, dd)
    para['ham'][1, 1], para['ovr'][1, 1] = hs_x_x(
        xx, yy, zz, h_pp0, data[15], h_pp1, data[16])
    para['ham'][1, 2], para['ovr'][1, 2] = hs_x_y(
        xx, yy, zz, h_pp0, data[15], h_pp1, data[16])
    para['ham'][1, 3], para['ovr'][1, 3] = hs_x_z(
        xx, yy, zz, h_pp0, data[15], h_pp1, data[16])
    para['ham'][2, 2], para['ovr'][2, 2] = hs_y_y(
        xx, yy, zz, h_pp0, data[15], h_pp1, data[16])
    para['ham'][2, 3], para['ovr'][2, 3] = hs_y_z(
        xx, yy, zz, h_pp0, data[15], h_pp1, data[16])
    para['ham'][3, 3], para['ovr'][3, 3] = hs_z_z(
        xx, yy, zz, h_pp0, data[15], h_pp1, data[16])
    for ii in range(nls, nlp + nls):
        for jj in range(nls, ii + nls):
            para['ham'][ii, jj] = para['ham'][jj, ii]
            para['ovr'][ii, jj] = para['ovr'][jj, ii]
    return para


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
    for ii in range(nls, nls + nlp):
        for jj in range(nls + nlp, nld):
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


def getSKTable(para):
    '''In hamtable, the first line is '''
    atomind = para['atomind']
    natom = para['natom']
    atomname = para['atomnameall']
    dvec = para['dvec']
    atomind2 = para['atomind2']
    hamtable = t.zeros(atomind2, 3)
    ovrtable = t.zeros(atomind2, 3)
    rr_ij = t.zeros(atomind2, 3)
    dd = t.zeros(atomind2)
    haminfo = t.zeros(atomind2)
    ovrinfo = t.zeros(atomind2)
    rr = t.zeros(3)
    for i in range(0, natom):
        lmaxi = para['lmaxall'][i]
        for j in range(0, i+1):
            lmaxj = para['lmaxall'][j]
            lmax = max(lmaxi, lmaxj)
            ham = t.zeros(4, 9, 9)
            ovr = t.zeros(4, 9, 9)
            para['nameij'] = atomname[i]+atomname[j]
            rr[:] = dvec[i, j, :]
            hams, ovrs = HS_dist(
                    rr, i, j, para, ham, ovr, lmax)
            for n in range(0, atomind[j+1] - atomind[j]):
                nn = atomind[j] + n
                for m in range(0, atomind[i+1] - atomind[i]):
                    mm = atomind[i] + m
                    idx = int(mm * (mm + 1)/2 + nn)
                    if nn <= mm:
                        idx = int(mm * (mm + 1)/2 + nn)
                        hamtable[idx, :] = ham[1:, m, n]
                        ovrtable[idx, :] = ovr[1:, m, n]
                        rr_ij[idx, :] = rr[:]
                        dd[idx] = para['distance'][i, j]
                        haminfo[idx] = ham[0, m, n]
                        ovrinfo[idx] = ovr[0, m, n]
    para['hamtable'] = hamtable
    para['ovrtable'] = ovrtable
    para['haminfo'] = haminfo
    para['ovrinfo'] = ovrinfo
    para['rr_ij'] = rr_ij
    para['dd_ij'] = dd
    return para


def HS_dist(rr, i, j, para, ham, ovr, lmax):
    # haminfo record the info of the corresponding hamiltonian,
    # for s, p, d onsite, haminfo = 0, 1, 2
    # for ss, sp1, sp2, sp3 integral, haminfo = 10, 11, 12, 13
    # for sd1, sd2, sd3, sd4, sd5, orbital, haminfo is 14, 15, 16, 17, 18
    # for p1s, p1p1, p1p2, p1p3, orbital, haminfo is 20, 21, 22, 23
    # for p2s, p2p1, p2p2, p2p3, orbital, haminfo is 30, 31, 32, 33
    # for p3s, p3p1, p3p2, p3p3, orbital, haminfo is 40, 41, 42, 43
    dd = t.sqrt(rr[0]*rr[0] + rr[1]*rr[1] + rr[2]*rr[2])
    nameij = para['nameij']
    if para['ty'] == 6:
        hs_data = getsk(para, nameij, dd)
    elif para['ty'] == 5:
        hs_data = para['hs_all'][i, j, :]
    skself = para['onsite']
    cutoff = para['cutoffsk'+nameij]
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
        lmaxi = para['lmaxall'][i]
        lmaxj = para['lmaxall'][j]
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


def sk_tranml(para):
    htable = para['hamtable']
    stable = para['ovrtable']
    hinfo = para['haminfo']
    nind2 = len(htable)
    ham = t.zeros(nind2)
    ovr = t.zeros(nind2)
    for inind2 in range(0, nind2):
        ihtable = htable[inind2]
        istable = stable[inind2]
        ihinfo = hinfo[inind2]
        rr = para['rr_ij'][inind2]
        dd = para['dd_ij'][inind2]
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
    para['hammat'] = ham
    para['overmat'] = ovr
    return para


def sk_transpline(para):
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


def hs_s_s(x, y, z, hss0, sss0):
    return hss0, sss0


def h_s_s(x, y, z, hss0):
    return h_ss0


def s_s_s(x, y, z, sss0):
    return sss0


def hs_s_x(x, y, z, hsp0, ssp0):
    return x * hsp0, x * ssp0


def h_s_x(x, y, z, hsp0):
    return x*hsp0


def s_s_x(x, y, z, ssp0):
    return x*ssp0


def hs_s_y(x, y, z, hsp0, ssp0):
    return y*hsp0, y*ssp0


def h_s_y(x, y, z, hsp0):
    return y*hsp0


def s_s_y(x, y, z, ssp0):
    return y*ssp0


def hs_s_z(x, y, z, hsp0, ssp0):
    return z*hsp0, z*ssp0


def h_s_z(x, y, z, hsp0):
    return z*hsp0


def s_s_z(x, y, z, ssp0):
    return z*ssp0


def hs_s_xy(x, y, z, hsd0, ssd0):
    return t.sqrt(t.tensor([3.]))*x*y*hsd0, t.sqrt(t.tensor([3.]))*x*y*ssd0


def h_s_xy(x, y, z, hsd0):
    return t.sqrt(t.tensor([3.]))*x*y*hsd0


def s_s_xy(x, y, z, ssd0):
    return t.sqrt(t.tensor([3.]))*x*y*ssd0


def hs_s_yz(x, y, z, hsd0, ssd0):
    return t.sqrt(t.tensor([3.]))*y*z*hsd0, t.sqrt(t.tensor([3.]))*y*z*ssd0


def h_s_yz(x, y, z, hsd0):
    return t.sqrt(t.tensor([3.]))*y*z*hsd0


def s_s_yz(x, y, z, ssd0):
    return t.sqrt(t.tensor([3.]))*y*z*ssd0


def hs_s_xz(x, y, z, hsd0, ssd0):
    return t.sqrt(t.tensor([3.]))*x*z*hsd0, t.sqrt(t.tensor([3.]))*x*z*ssd0


def h_s_xz(x, y, z, hsd0):
    return t.sqrt(t.tensor([3.]))*x*z*hsd0


def s_s_xz(x, y, z, ssd0):
    return t.sqrt(t.tensor([3.]))*x*z*ssd0


def hs_s_x2y2(x, y, z, hsd0, ssd0):
    return (0.5*t.sqrt(t.tensor([3.]))*(x**2-y**2)*hsd0,
            0.5*t.sqrt(t.tensor([3.]))*(x**2-y**2)*ssd0)


def hs_s_3z2r2(x, y, z, hsd0, ssd0):
    return (z**2-0.5*(x**2+y**2))*hsd0, (z**2-0.5*(x**2+y**2))*ssd0


def hs_x_s(x, y, z, hsp0, ssp0):
    return hs_s_x(-x, -y, -z, hsp0, ssp0)[0], hs_s_x(-x, -y, -z, hsp0, ssp0)[1]


def h_x_s(x, y, z, hsp0):
    return h_s_x(-x, -y, -z, hsp0)


def s_x_s(x, y, z, ssp0):
    return hs_s_x(-x, -y, -z, ssp0)


def hs_x_x(x, y, z, hpp0, spp0, hpp1, spp1):
    return x**2*hpp0+(1-x**2)*hpp1, x**2*spp0+(1-x**2)*spp1


def h_x_x(x, y, z, hpp0, hpp1):
    return x**2*hpp0+(1-x**2)*hpp1


def s_x_x(x, y, z, spp0, spp1):
    return x**2*spp0+(1-x**2)*spp1


def hs_x_y(x, y, z, hpp0, spp0, hpp1, spp1):
    return x*y*hpp0-x*y*hpp1, x*y*spp0-x*y*spp1


def h_x_y(x, y, z, hpp0, hpp1):
    return x*y*hpp0-x*y*hpp1


def s_x_y(x, y, z, spp0, spp1):
    return x*y*spp0-x*y*spp1


def hs_x_z(x, y, z, hpp0, spp0, hpp1, spp1):
    return x*z*hpp0-x*z*hpp1, x*z*spp0-x*z*spp1


def h_x_z(x, y, z, hpp0, hpp1):
    return x*z*hpp0-x*z*hpp1


def s_x_z(x, y, z, spp0, spp1):
    return x*z*spp0-x*z*spp1


def hs_x_xy(x, y, z, hpd0, spd0, hpd1, spd1):
    return (t.sqrt(t.tensor([3.]))*x**2*y*hpd0+y*(1-2*x**2)*hpd1,
            t.sqrt(t.tensor([3.]))*x**2*y*spd0 + y*(1-2*x**2)*spd1)


def hs_x_yz(x, y, z, hpd0, spd0, hpd1, spd1):
    return (t.sqrt(t.tensor([3.]))*x*y*z*hpd0-2*x*y*z*hpd1,
            t.sqrt(t.tensor([3.]))*x*y*z*hpd0-2*x*y*z*hpd1)


def hs_x_xz(x, y, z, hpd0, spd0, hpd1, spd1):
    return (t.sqrt(t.tensor([3.]))*x**2*z*hpd0+z*(1-2*x**2)*hpd1,
            t.sqrt(t.tensor([3.]))*x**2*z*spd0+z*(1-2*x**2)*spd1)


def hs_x_x2y2(x, y, z, hpd0, spd0, hpd1, spd1):
    return (t.sqrt(t.tensor([3.]))/2*x*(x**2-y**2)*hpd0+x*(1-x**2+y**2)*hpd1,
            t.sqrt(t.tensor([3.]))/2*x*(x**2-y**2)*spd0+x*(1-x**2+y**2)*spd1)


def hs_x_3z2r2(x, y, z, hpd0, spd0, hpd1, spd1):
    return (x*(z**2-0.5*(x**2+y**2))*hpd0-t.sqrt(t.tensor([3.]))*x*z**2*hpd1,
            x*(z**2-0.5*(x**2+y**2))*spd1-t.sqrt(t.tensor([3.]))*x*z**2*spd1)


def hs_y_s(x, y, z, hsp0, ssp0):
    return hs_s_y(-x, -y, -z, hsp0, ssp0)[0], hs_s_y(-x, -y, -z, hsp0, ssp0)[1]


def h_y_s(x, y, z, hsp0):
    return h_s_y(-x, -y, -z, hsp0)


def s_y_s(x, y, z, ssp0):
    return hs_s_y(-x, -y, -z, ssp0)


def hs_y_x(x, y, z, hpp0, spp0, hpp1, spp1):
    return hs_x_y(-x, -y, -z, hpp0, spp0, hpp1, spp1)[0], hs_x_y(
            -x, -y, -z, hpp0, spp0, hpp1, spp1)[1]


def h_y_x(x, y, z, hpp0, hpp1):
    return hs_x_y(-x, -y, -z, hpp0, hpp1)


def s_y_x(x, y, z, spp0, spp1):
    return s_x_y(-x, -y, -z, spp0, spp1)


def hs_y_y(x, y, z, hpp0, spp0, hpp1, spp1):
    return y**2*hpp0+(1-y**2)*hpp1, y**2*spp0+(1-y**2)*spp1


def h_y_y(x, y, z, hpp0, hpp1):
    return y**2*hpp0+(1-y**2)*hpp1


def s_y_y(x, y, z, spp0, spp1):
    return y**2*spp0+(1-y**2)*spp1


def hs_y_z(x, y, z, hpp0, spp0, hpp1, spp1):
    return y*z*hpp0-y*z*hpp1, y*z*spp0-y*z*spp1


def h_y_z(x, y, z, hpp0, hpp1):
    return y*z*hpp0-y*z*hpp1


def s_y_z(x, y, z, spp0, spp1):
    return y*z*spp0-y*z*spp1


def hs_y_xy(x, y, z, hpd0, spd0, hpd1, spd1):
    return (t.sqrt(t.tensor([3.]))*y**2*x*hpd0+x*(1-2*y**2)*hpd1,
            t.sqrt(t.tensor([3.]))*y**2*x*spd0+x*(1-2*y**2)*spd1)


def hs_y_yz(x, y, z, hpd0, spd0, hpd1, spd1):
    return (t.sqrt(t.tensor([3.]))*y**2*z*hpd0-z*(1-2*y**2)*hpd1,
            t.sqrt(t.tensor([3.]))*y**2*z*spd0-z*(1-2*y**2)*spd1)


def hs_y_xz(x, y, z, hpd0, spd0, hpd1, spd1):
    return (t.sqrt(t.tensor([3.]))*x*y*z*hpd0-2*x*y*z*hpd1,
            t.sqrt(t.tensor([3.]))*x*y*z*spd0-2*x*y*z*spd1)


def hs_y_x2y2(x, y, z, hpd0, spd0, hpd1, spd1):
    return (t.sqrt(t.tensor([3.]))/2*y*(x**2-y**2)*hpd0-y*(1+x**2-y**2)*hpd1,
            t.sqrt(t.tensor([3.]))/2*y*(x**2-y**2)*spd0-y*(1+x**2-y**2)*spd1)


def hs_y_3z2r2(x, y, z, hpd0, spd0, hpd1, spd1):
    return (y*(z**2-0.5*(x**2+y**2))*hpd0-t.sqrt(t.tensor([3.]))*y*z**2*hpd1,
            y*(z**2-0.5*(x**2+y**2))*spd0-t.sqrt(t.tensor([3.]))*y*z**2*spd1)


def hs_z_s(x, y, z, hsp0, ssp0):
    return hs_s_z(-x, -y, -z, hsp0, ssp0)[0], hs_s_z(-x, -y, -z, hsp0, ssp0)[1]


def h_z_s(x, y, z, hsp0):
    return h_s_z(-x, -y, -z, hsp0)


def s_z_s(x, y, z, ssp0):
    return s_s_z(-x, -y, -z, ssp0)


def hs_z_x(x, y, z, hpp0, spp0, hpp1, spp1):
    return hs_x_z(-x, -y, -z, hpp0, spp0, hpp1, spp1)[0], hs_x_z(
            -x, -y, -z, hpp0, spp0, hpp1, spp1)[1]


def h_z_x(x, y, z, hpp0, hpp1):
    return h_x_z(-x, -y, -z, hpp0, hpp1)


def s_z_x(x, y, z, spp0, spp1):
    return s_x_z(-x, -y, -z, spp0, spp1)


def hs_z_y(x, y, z, hpp0, spp0, hpp1, spp1):
    return hs_y_z(-x, -y, -z, hpp0, spp0, hpp1, spp1)[0], hs_y_z(
            -x, -y, -z, hpp0, spp0, hpp1, spp1)[1]


def h_z_y(x, y, z, hpp0, hpp1):
    return h_y_z(-x, -y, -z, hpp0, hpp1)


def s_z_y(x, y, z, spp0, spp1):
    return s_y_z(-x, -y, -z, spp0, spp1)


def hs_z_z(x, y, z, hpp0, spp0, hpp1, spp1):
    return (z**2*hpp0+(1-z**2)*hpp1,
            z**2*spp0+(1-z**2)*spp1)


def h_z_z(x, y, z, hpp0, hpp1):
    return z**2*hpp0+(1-z**2)*hpp1


def s_z_z(x, y, z, spp0, spp1):
    return z**2*spp0+(1-z**2)*spp1


def hs_z_xy(x, y, z, hpd0, spd0, hpd1, spd1):
    return (t.sqrt(t.tensor([3.]))*x*y*z*hpd0 - 2*x*y*z*hpd1,
            t.sqrt(t.tensor([3.]))*x*y*z*spd0 - 2*x*y*z*spd1)


def hs_z_yz(x, y, z, hpd0, spd0, hpd1, spd1):
    return (t.sqrt(t.tensor([3.]))*z**2*y*hpd0 - y*(1-2*z**2)*hpd1,
            t.sqrt(t.tensor([3.]))*z**2*y*spd0 - y*(1-2*z**2)*spd1)


def hs_z_xz(x, y, z, hpd0, spd0, hpd1, spd1):
    return (t.sqrt(t.tensor([3.]))*z**2*x*hpd0 - x*(1-2*z**2)*hpd1,
            t.sqrt(t.tensor([3.]))*z**2*x*spd0 - x*(1-2*z**2)*spd1)


def hs_z_x2y2(x, y, z, hpd0, spd0, hpd1, spd1):
    return (t.sqrt(t.tensor([3.]))/2*z*(x**2-y**2)*hpd0 - z*(x**2-y**2)*hpd1,
            t.sqrt(t.tensor([3.]))/2*z*(x**2-y**2)*spd0 - z*(x**2-y**2)*spd1)


def hs_z_3z2r2(x, y, z, hpd0, spd0, hpd1, spd1):
    return (z*(z**2-0.5*(x**2+y**2))*hpd0+t.sqrt(t.tensor([3.])) *
            z*(x**2+y**2)*hpd1,
            z*(z**2-0.5*(x**2+y**2))*spd0+t.sqrt(t.tensor([3.])) *
            z*(x**2+y**2)*spd1)


def hs_xy_s(x, y, z, hsd0, ssd0):
    return hs_s_xy(-x, -y, -z, hsd0, ssd0)[0], hs_s_xy(
            -x, -y, -z, hsd0, ssd0)[1]


def hs_xy_x(x, y, z, hpd0, spd0, hpd1, spd1):
    return hs_x_xy(-x, -y, -z, hpd0, spd0, hpd1, spd1)[0], hs_x_xy(
            -x, -y, -z, hpd0, spd0, hpd1, spd1)[1]


def hs_xy_y(x, y, z, hpd0, spd0, hpd1, spd1):
    return hs_y_xy(-x, -y, -z, hpd0, spd0, hpd1, spd1)[0], hs_y_xy(
            -x, -y, -z, hpd0, spd0, hpd1, spd1)[1]


def hs_xy_z(x, y, z, hpd0, spd0, hpd1, spd1):
    return hs_z_xy(-x, -y, -z, hpd0, spd0, hpd1, spd1)[0], hs_z_xy(
            -x, -y, -z, hpd0, spd0, hpd1, spd1)[1]


def hs_xy_xy(x, y, z, hdd0, sdd0, hdd1, sdd1, hdd2, sdd2):
    return (3*x**2*y**2*hdd0+(x**2+y**2-4*x**2*y**2) *
            hdd1+(z**2+x**2*y**2)*hdd2,
            3*x**2*y**2*sdd0 + (x**2+y**2-4*x**2*y**2) *
            sdd1+(z**2+x**2*y**2)*sdd2)


def hs_xy_yz(x, y, z, hdd0, sdd0, hdd1, sdd1, hdd2, sdd2):
    return (3*x*y**2*z*hdd0+x*z*(1-4*y**2)*hdd1 +
            x*z*(y**2-1)*hdd2,
            3*x*y**2*z*sdd0+x*z*(1-4*y**2)*sdd1 +
            x*z*(y**2-1)*sdd2)


def hs_xy_xz(x, y, z, hdd0, sdd0, hdd1, sdd1, hdd2, sdd2):
    return (3*x**2*y*z*hdd0+y*z*(1-4*x**2)*hdd1 +
            y*z*(x**2-1)*hdd2,
            3*x**2*y*z*sdd0+y*z*(1-4*x**2)*sdd1 +
            y*z*(x**2-1)*sdd2)


def hs_xy_x2y2(x, y, z, hdd0, sdd0, hdd1, sdd1, hdd2, sdd2):
    return (1.5*x*y*(x**2-y**2)*hdd0-2*x*y*(x**2-y**2)*hdd1 +
            0.5*x*y*(x**2-y**2)*hdd2,
            1.5*x*y*(x**2-y**2)*sdd0-2*x*y*(x**2-y**2)*sdd1 +
            0.5*x*y*(x**2-y**2)*sdd2)


def hs_xy_3z2r2(x, y, z, hdd0, sdd0, hdd1, sdd1, hdd2, sdd2):
    return (t.sqrt(t.tensor([3.]))*x*y*(z**2-0.5*(x**2+y**2))*hdd0-2 *
            t.sqrt(t.tensor([3.])) *
            x*y*z**2*hdd1+t.sqrt(t.tensor([3.]))/2*x*y*(1+z**2)*hdd2,
            t.sqrt(t.tensor([3.]))*x*y*(z**2-0.5*(x**2+y**2))*sdd0-2 *
            t.sqrt(t.tensor([3.])) *
            x*y*z**2*sdd1+t.sqrt(t.tensor([3.]))/2*x*y*(1+z**2)*sdd2)


def hs_yz_s(x, y, z, hsd0, ssd0):
    return hs_s_yz(-x, -y, -z, hsd0, ssd0)[0], hs_s_yz(
            -x, -y, -z, hsd0, ssd0)[1]


def hs_yz_x(x, y, z, hpd0, spd0, hpd1, spd1):
    return hs_x_yz(-x, -y, -z, hpd0, spd0, hpd1, spd1)[0], hs_x_yz(
            -x, -y, -z, hpd0, spd0, hpd1, spd1)[1]


def hs_yz_y(x, y, z, hpd0, spd0, hpd1, spd1):
    return hs_y_yz(-x, -y, -z, hpd0, spd0, hpd1, spd1)[0], hs_y_yz(
            -x, -y, -z, hpd0, spd0, hpd1, spd1)[1]


def hs_yz_z(x, y, z, hpd0, spd0, hpd1, spd1):
    return hs_z_yz(-x, -y, -z, hpd0, spd0, hpd1, spd1)[0], hs_z_yz(
            -x, -y, -z, hpd0, spd0, hpd1, spd1)[1]


def hs_yz_xy(x, y, z, hdd0, sdd0, hdd1, sdd1, hdd2, sdd2):
    return (hs_xy_yz(-x, -y, -z, hdd0, sdd0, hdd1, sdd1, hdd2, sdd2)[0],
            hs_xy_yz(-x, -y, -z, hdd0, sdd0, hdd1, sdd1, hdd2, sdd2)[1])


def hs_yz_yz(x, y, z, hdd0, sdd0, hdd1, sdd1, hdd2, sdd2):
    return (3*y**2*z**2*hdd0 + (y**2+z**2-4*y**2*z**2) *
            hdd1+(x**2+y**2*z**2)*hdd2,
            3*y**2*z**2*sdd0 + (y**2+z**2-4*y**2*z**2) *
            sdd1+(x**2+y**2*z**2)*sdd2)


def hs_yz_xz(x, y, z, hdd0, sdd0, hdd1, sdd1, hdd2, sdd2):
    return (3*x*z**2*y*hdd0+x*y*(1-4*z**2)*hdd1 +
            x*y*(z**2-1)*hdd2,
            3*x*z**2*y*sdd0+x*y*(1-4*z**2)*sdd1 +
            x*y*(z**2-1)*sdd2)


def hs_yz_x2y2(x, y, z, hdd0, sdd0, hdd1, sdd1, hdd2, sdd2):
    return (1.5*y*z*(x**2-y**2)*hdd0-y*z*(1+2*(x**2-y**2)) *
            hdd1+y*z*(1+0.5*(x**2-y**2))*hdd2,
            1.5*y*z*(x**2-y**2)*sdd0-y*z*(1+2*(x**2-y**2)) *
            sdd1+y*z*(1+0.5*(x**2-y**2))*sdd2)


def hs_yz_3z2r2(x, y, z, hdd0, sdd0, hdd1, sdd1, hdd2, sdd2):
    return (t.sqrt(t.tensor([3.]))*y*z*(z**2-0.5*(x**2+y**2))*hdd0 +
            t.sqrt(t.tensor([3.]))*y*z*(x**2+y**2-z**2)*hdd1 -
            t.sqrt(t.tensor([3.]))/2*y*z*(x**2+y**2)*hdd2,
            t.sqrt(t.tensor([3.]))*y*z*(z**2-0.5*(x**2+y**2)) *
            sdd0+t.sqrt(t.tensor([3.]))*y*z*(x**2+y**2-z**2)*sdd1 -
            t.sqrt(t.tensor([3.]))/2*y*z*(x**2+y**2)*sdd2)


def hs_xz_s(x, y, z, hdd0, sdd0):
    return hs_s_xz(-x, -y, -z, hdd0, sdd0)[0], hs_s_xz(
            -x, -y, -z, hdd0, sdd0)[1]


def hs_xz_x(x, y, z, hpd0, spd0, hpd1, spd1):
    return hs_x_xz(-x, -y, -z, hpd0, spd0, hpd1, spd1)[0], hs_x_xz(
            -x, -y, -z, hpd0, spd0, hpd1, spd1)[1]


def hs_xz_y(x, y, z, hpd0, spd0, hpd1, spd1):
    return hs_y_xz(-x, -y, -z, hpd0, spd0, hpd1, spd1)[0], hs_y_xz(
            -x, -y, -z, hpd0, spd0, hpd1, spd1)[1]


def hs_xz_z(x, y, z, hpd0, spd0, hpd1, spd1):
    return hs_z_xz(-x, -y, -z, hpd0, spd0, hpd1, spd1)[0], hs_z_xz(
            -x, -y, -z, hpd0, spd0, hpd1, spd1)[1]


def hs_xz_xy(x, y, z, hdd0, sdd0, hdd1, sdd1, hdd2, sdd2):
    return (hs_xy_xz(-x, -y, -z, hdd0, sdd0, hdd1, sdd1, hdd2, sdd2)[0],
            hs_xy_xz(-x, -y, -z, hdd0, sdd0, hdd1, sdd1, hdd2, sdd2)[1])


def hs_xz_yz(x, y, z, hdd0, sdd0, hdd1, sdd1, hdd2, sdd2):
    return (hs_yz_xz(-x, -y, -z, hdd0, sdd0, hdd1, sdd1, hdd2, sdd2)[0],
            hs_yz_xz(-x, -y, -z, hdd0, sdd0, hdd1, sdd1, hdd2, sdd2)[1])


def hs_xz_xz(x, y, z, hdd0, sdd0, hdd1, sdd1, hdd2, sdd2):
    return (3*x**2*z**2*hdd0 + (x**2+z**2-4*x**2*z**2) *
            hdd1+(y**2+x**2*z**2)*hdd2,
            3*x**2*z**2*sdd0 + (x**2+z**2-4*x**2*z**2) *
            sdd1 + (y**2+x**2*z**2)*sdd2)


def hs_xz_x2y2(x, y, z, hdd0, sdd0, hdd1, sdd1, hdd2, sdd2):
    return (1.5*x*z*(x**2-y**2)*hdd0+x*z*(1-2*(x**2-y**2)) *
            hdd1-x*z*(1-0.5*(x**2-y**2))*hdd2,
            1.5*x*z*(x**2-y**2)*sdd0+x*z*(1-2*(x**2-y**2)) *
            sdd1-x*z*(1-0.5*(x**2-y**2))*sdd2)


def hs_xz_3z2r2(x, y, z, hdd0, sdd0, hdd1, sdd1, hdd2, sdd2):
    return (t.sqrt(t.tensor([3.]))*x*z*(z**2-0.5*(x**2+y**2))*hdd0 +
            t.sqrt(t.tensor([3.]))*x*z*(x**2+y**2-z**2)*hdd1 -
            t.sqrt(t.tensor([3.]))/2*x*z*(x**2+y**2)*hdd2,
            t.sqrt(t.tensor([3.]))*x*z*(z**2-0.5*(x**2+y**2))*sdd0 +
            t.sqrt(t.tensor([3.]))*x*z*(x**2+y**2-z**2)*sdd1 -
            t.sqrt(t.tensor([3.]))/2*x*z*(x**2+y**2)*sdd2)


def hs_x2y2_s(x, y, z, hsd0, ssd0):
    return hs_s_x2y2(-x, -y, -z, hsd0, ssd0)[0], hs_s_x2y2(
            -x, -y, -z, hsd0, ssd0)[1]


def hs_x2y2_x(x, y, z, hpd0, spd0, hpd1, spd1):
    return hs_x_x2y2(-x, -y, -z, hpd0, spd0, hpd1, spd1)[0], hs_x_x2y2(
            -x, -y, -z, hpd0, spd0, hpd1, spd1)[1]


def hs_x2y2_y(x, y, z, hpd0, spd0, hpd1, spd1):
    return hs_y_x2y2(-x, -y, -z, hpd0, spd0, hpd1, spd1)[0], hs_y_x2y2(
            -x, -y, -z, hpd0, spd0, hpd1, spd1)[1]


def hs_x2y2_z(x, y, z, hpd0, spd0, hpd1, spd1):
    return hs_z_x2y2(-x, -y, -z, hpd0, spd0, hpd1, spd1)[0], hs_z_x2y2(
            -x, -y, -z, hpd0, spd0, hpd1, spd1)[1]


def hs_x2y2_xy(x, y, z, hdd0, sdd0, hdd1, sdd1, hdd2, sdd2):
    return (hs_xy_x2y2(-x, -y, -z, hdd0, sdd0, hdd1, sdd1, hdd2, sdd2)[0],
            hs_xy_x2y2(-x, -y, -z, hdd0, sdd0, hdd1, sdd1, hdd2, sdd2)[1])


def hs_x2y2_yz(x, y, z, hdd0, sdd0, hdd1, sdd1, hdd2, sdd2):
    return (hs_yz_x2y2(-x, -y, -z, hdd0, sdd0, hdd1, sdd1, hdd2, sdd2)[0],
            hs_yz_x2y2(-x, -y, -z, hdd0, sdd0, hdd1, sdd1, hdd2, sdd2)[1])


def hs_x2y2_xz(x, y, z, hdd0, sdd0, hdd1, sdd1, hdd2, sdd2):
    return (hs_xz_x2y2(-x, -y, -z, hdd0, sdd0, hdd1, sdd1, hdd2, sdd2)[0],
            hs_xz_x2y2(-x, -y, -z, hdd0, sdd0, hdd1, sdd1, hdd2, sdd2)[1])


def hs_x2y2_x2y2(x, y, z, hdd0, sdd0, hdd1, sdd1, hdd2, sdd2):
    return (3/4*(x**2-y**2)**2*hdd0+(x**2+y**2 -
            (x**2-y**2)**2)*hdd1+(z**2+1/4*(x**2-y**2)**2)*hdd2,
            3/4*(x**2-y**2)**2*sdd0+(x**2+y**2 -
            (x**2-y**2)**2)*sdd1+(z**2+1/4*(x**2-y**2)**2)*sdd2)


def hs_x2y2_3z2r2(x, y, z, hdd0, sdd0, hdd1, sdd1, hdd2, sdd2):
    return (t.sqrt(t.tensor([3.]))/2*(x**2-y**2)*(z**2-(x**2+y**2)/2) *
            hdd0+t.sqrt(t.tensor([3.]))*z**2*(x**2-y**2)*hdd1 +
            t.sqrt(t.tensor([3.]))/4*(1+z**2)*(x**2-y**2)*hdd2,
            t.sqrt(t.tensor([3.]))/2*(x**2-y**2)*(z**2-(x**2+y**2)/2) *
            sdd0+t.sqrt(t.tensor([3.]))*z**2*(x**2-y**2)*sdd1 +
            t.sqrt(t.tensor([3.]))/4*(1+z**2)*(x**2-y**2)*sdd2)


def hs_3z2r2_s(x, y, z, hsd0, ssd0):
    return (hs_s_3z2r2(-x, -y, -z, hsd0, ssd0)[0],
            hs_s_3z2r2(-x, -y, -z, hsd0, ssd0)[1])


def hs_3z2r2_x(x, y, z, hpd0, spd0, hpd1, spd1):
    return (hs_x_3z2r2(-x, -y, -z, hpd0, spd0, hpd1, spd1)[0],
            hs_x_3z2r2(-x, -y, -z, hpd0, spd0, hpd1, spd1)[1])


def hs_3z2r2_y(x, y, z, hpd0, spd0, hpd1, spd1):
    return (hs_y_3z2r2(-x, -y, -z, hpd0, spd0, hpd1, spd1)[0],
            hs_y_3z2r2(-x, -y, -z, hpd0, spd0, hpd1, spd1)[1])


def hs_3z2r2_z(x, y, z, hpd0, spd0, hpd1, spd1):
    return (hs_z_3z2r2(-x, -y, -z, hpd0, spd0, hpd1, spd1)[0],
            hs_z_3z2r2(-x, -y, -z, hpd0, spd0, hpd1, spd1)[1])


def hs_3z2r2_xy(x, y, z, hdd0, sdd0, hdd1, sdd1, hdd2, sdd2):
    return (hs_xy_3z2r2(-x, -y, -z, hdd0, sdd0, hdd1, sdd1, hdd2, sdd2)[0],
            hs_xy_3z2r2(-x, -y, -z, hdd0, sdd0, hdd1, sdd1, hdd2, sdd2)[1])


def hs_3z2r2_yz(x, y, z, hdd0, sdd0, hdd1, sdd1, hdd2, sdd2):
    return (hs_yz_3z2r2(-x, -y, -z, hdd0, sdd0, hdd1, sdd1, hdd2, sdd2)[0],
            hs_yz_3z2r2(-x, -y, -z, hdd0, sdd0, hdd1, sdd1, hdd2, sdd2)[1])


def hs_3z2r2_xz(x, y, z, hdd0, sdd0, hdd1, sdd1, hdd2, sdd2):
    return (hs_xz_3z2r2(-x, -y, -z, hdd0, sdd0, hdd1, sdd1, hdd2, sdd2)[0],
            hs_xz_3z2r2(-x, -y, -z, hdd0, sdd0, hdd1, sdd1, hdd2, sdd2)[1])


def hs_3z2r2_x2y2(x, y, z, hdd0, sdd0, hdd1, sdd1, hdd2, sdd2):
    return (hs_x2y2_3z2r2(-x, -y, -z, hdd0, sdd0, hdd1, sdd1, hdd2, sdd2)[0],
            hs_x2y2_3z2r2(-x, -y, -z, hdd0, sdd0, hdd1, sdd1, hdd2, sdd2)[1])


def hs_3z2r2_3z2r2(x, y, z, hdd0, sdd0, hdd1, sdd1, hdd2, sdd2):
    return ((z**2-0.5*(x**2+y**2))**2*hdd0+3*z**2*(x**2+y**2) *
            hdd1+3/4*(x**2+y**2)**2*hdd2,
            (z**2-0.5*(x**2+y**2))**2*sdd0+3*z**2*(x**2+y**2) *
            sdd1+3/4*(x**2+y**2)**2*sdd2)
