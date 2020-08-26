"""Slater-Koster integrals related."""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time
import sys
import numpy as np
import torch as t
import matplotlib.pyplot as plt
import h5py
import matht
from scipy import interpolate
from matht import Bspline, DFTBmath, BicubInterp
ATOMNAME = {1: 'H', 6: 'C', 7: 'N', 8: 'O'}


class SKTran:
    """Slater-Koster Transformations."""

    def __init__(self, para):
        """Initialize parameters.

        Args:
            integral
            geometry (distance)
        Returns:
            [natom, natom, 20] matrix for each calculation
        """
        self.para = para
        self.math = DFTBmath(self.para)

        # if machine learning or not
        if not self.para['Lml']:

            # read integrals from .skf with various compression radius
            if not self.para['LreadSKFinterp']:
                self.get_sk_all()

            # build H0 and S with full, symmetric matrices
            if self.para['HSsym'] == 'symall':
                self.sk_tran_symall()

            # build H0 and S with only half matrices
            elif self.para['HSsym'] == 'symhalf':
                self.sk_tran_half()

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
                    if self.para['sk_tran'] == 'new':
                        self.slkode_vec(rr, iat, jat, lmaxi, lmaxj)
                    elif self.para['sk_tran'] == 'old':
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

    def slkode_vec(self, rr, iat, jat, li, lj):
        """Generate H0, S by vectorized method."""
        lmax, lmin = max(li, lj), min(li, lj)
        xx, yy, zz = rr[:] / t.sqrt(t.sum(rr[:] ** 2))
        hsall = self.para['hs_all']

        if lmax == 1:
            self.para['hams'][0, 0], self.para['ovrs'][0, 0] = \
                self.skss_vec(hsall, xx, yy, zz, iat, jat)
        if lmin == 1 and lmax == 2:
            self.para['hams'][:4, :4], self.para['ovrs'][:4, :4] = \
                self.sksp_vec(hsall, xx, yy, zz, iat, jat, li, lj)
        if lmin == 2 and lmax == 2:
            self.para['hams'][:4, :4], self.para['ovrs'][:4, :4] = \
                self.skpp_vec(hsall, xx, yy, zz, iat, jat, li, lj)

    def skss_vec(self, hs, x, y, z, i, j):
        """Return H0, S of ss after sk transformations.

        Parameters:
            hs: H, S tables with dimension [natom, natom, 20]

        """
        return hs[i, j, 9], hs[i, j, 19]

    def sksp_vec(self, hs, x, y, z, i, j, li, lj):
        """Return H0, S of ss, sp after sk transformations.

        Parameters:
            hs: H, S tables with dimension [natom, natom, 20]

        For sp orbitals here, such as for CH4 system, if we want to get H_s
        and C_p integral, we can only read from H-C.skf, therefore for the
        first loop layer, if the atom specie is C and second is H, the sp0 in
        C-H.skf is 0 and instead we will read sp0 from [j, i, 8], which is
        from H-C.skf

        """
        # read sp0 from <namei-namej>.skf
        if li < lj:
            H = t.tensor([[
                # SS, SP_y, SP_z, SP_x
                hs[i, j, 9],
                y * hs[i, j, 8],
                z * hs[i, j, 8],
                x * hs[i, j, 8]],
                # P_yS
                [y * hs[i, j, 8], 0.0, 0.0, 0.0],
                # P_zS
                [z * hs[i, j, 8], 0.0, 0.0, 0.0],
                # P_xS
                [x * hs[i, j, 8], 0.0, 0.0, 0.0]])
            S = t.tensor([[
                hs[i, j, 19],
                y * hs[i, j, 18],
                z * hs[i, j, 18],
                x * hs[i, j, 18]],
                [y * hs[i, j, 18], 0.0, 0.0, 0.0],
                [z * hs[i, j, 18], 0.0, 0.0, 0.0],
                [x * hs[i, j, 18], 0.0, 0.0, 0.0]])

        # read sp0 from <namej-namei>.skf
        if li > lj:
            H = t.tensor([[
                hs[j, i, 9],
                -y * hs[j, i, 8],
                -z * hs[j, i, 8],
                -x * hs[j, i, 8]],
                [-y * hs[j, i, 8], 0.0, 0.0, 0.0],
                [-z * hs[j, i, 8], 0.0, 0.0, 0.0],
                [-x * hs[j, i, 8], 0.0, 0.0, 0.0]])
            S = t.tensor([[
                hs[j, i, 19],
                -y * hs[j, i, 18],
                -z * hs[j, i, 18],
                -x * hs[j, i, 18]],
                [-y * hs[j, i, 18], 0.0, 0.0, 0.0],
                [-z * hs[j, i, 18], 0.0, 0.0, 0.0],
                [-x * hs[j, i, 18], 0.0, 0.0, 0.0]])
        return H, S

    def skpp_vec(self, hs, x, y, z, i, j, li, lj):
        """Return H0, S of ss, sp, pp after sk transformations."""
        H = t.tensor([[
            # SS, SP_y, SP_z, SP_x
            hs[i, j, 9],
            y * hs[i, j, 8],
            z * hs[i, j, 8],
            x * hs[i, j, 8]],
            # P_yS, P_yP_y, P_yP_z, P_yP_x
            [-y * hs[j, i, 8],
             y * y * hs[i, j, 5] + (1 - y * y) * hs[i, j, 6],
             y * z * hs[i, j, 5] - y * z * hs[i, j, 6],
             y * x * hs[i, j, 5] - y * x * hs[i, j, 6]],
            # P_zS, P_zP_y, P_zP_z, P_zP_x
            [-z * hs[j, i, 8],
             z * y * hs[i, j, 5] - z * y * hs[i, j, 6],
             z * z * hs[i, j, 5] + (1 - z * z) * hs[i, j, 6],
             z * x * hs[i, j, 5] - z * x * hs[i, j, 6]],
            [-x * hs[j, i, 8],
             x * y * hs[i, j, 5] - x * y * hs[i, j, 6],
             x * z * hs[i, j, 5] - x * z * hs[i, j, 6],
             x * x * hs[i, j, 5] + (1 - x * x) * hs[i, j, 6]]])
        S = t.tensor([[
            hs[i, j, 19],
            y * hs[i, j, 18],
            z * hs[i, j, 18],
            x * hs[i, j, 18]],
            [-y * hs[j, i, 18],
             y * y * hs[i, j, 15] + (1 - y * y) * hs[i, j, 16],
             y * z * hs[i, j, 15] - y * z * hs[i, j, 16],
             y * x * hs[i, j, 15] - y * x * hs[i, j, 16]],
            [-z * hs[j, i, 18],
             z * y * hs[i, j, 15] - z * y * hs[i, j, 16],
             z * z * hs[i, j, 15] + (1 - z * z) * hs[i, j, 16],
             z * x * hs[i, j, 15] - z * x * hs[i, j, 16]],
            [-x * hs[j, i, 18],
             x * y * hs[i, j, 15] - x * y * hs[i, j, 16],
             x * z * hs[i, j, 15] - x * z * hs[i, j, 16],
             x * x * hs[i, j, 15] + (1 - x * x) * hs[i, j, 16]]])
        return H, S

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
        self.math = DFTBmath(self.para)

    def genskf_interp_dist_vec(self):
        """Generate integral along distance dimension."""
        time0 = time.time()
        self.para['hs_compr_all'] = []
        distance = self.para['distance']

        # index of row, column of distance matrix, no digonal
        ind = t.triu_indices(distance.shape[0], distance.shape[0], 1)
        dist_1d = distance[ind[0], ind[1]]

        # get the skf with hdf type
        hdfsk = os.path.join(self.para['dire_hdfSK'], self.para['name_hdfSK'])

        # read all skf according to atom number (species) and indices and add
        # these skf to a list, attention: hdf only store numpy type data
        skf = []
        with h5py.File(hdfsk, 'r') as f:
            [skf.append(f[ATOMNAME[int(self.para['coor'][i, 0])] +
                          ATOMNAME[int(self.para['coor'][j, 0])] +
                          '/hs_all_rall'][:]) for i, j in ind.T]

            # get the grid sidtance, which should be the same
            grid_dist = f['globalgroup'].attrs['grid_dist']

        # get the distance according to indices (upper triangle elements)
        ind_ = dist_1d / grid_dist

        # index of distance in each skf
        indd = []
        [indd.append(int(i + self.para['ninterp'] / 2 + 1)) for i in ind_]

        # get integrals with ninterp (normally 8) line for interpolation
        yy = []
        [yy.append(iskf[:, :, j - self.para['ninterp'] - 1:j - 1, :])
         for iskf, j in zip(skf, indd)]

        # get the distances corresponding to the integrals
        xx = []
        [xx.append(t.linspace(i - self.para['ninterp'], i,
                              self.para['ninterp']) * grid_dist) for i in indd]
        time2 = time.time()
        [self.para['hs_compr_all'].append(
            self.math.poly_interp_4d(ixx, t.from_numpy(iyy), idd))
            for ixx, iyy, idd in zip(xx, yy, dist_1d)]

        timeend = time.time()
        print('time of distance interpolation: ', timeend - time2)
        print('total time of distance interpolation in skf: ', timeend - time0)

    def genskf_interp_dist(self):
        """Generate sk integral with various compression radius along distance.

        Args:
            atomnameall (list): all the atom name
            natom (int): number of atom
            distance (2D tensor): distance between all atoms
        Returns:
            hs_compr_all (out): [natom, natom, ncompr, ncompr, 20]

        """
        time0 = time.time()
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
        print('N is number of atom in molecule, N_R is number of compression')

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

        # get the time after interpolation
        time2 = time.time()
        for iat in atomspecie:

            # onsite is the same, therefore read [0, 0] instead
            onsite = t.zeros((3), dtype=t.float64)
            uhubb = t.zeros((3), dtype=t.float64)
            onsite[:] = self.para['onsite' + iat + iat]
            uhubb[:] = self.para['uhubb' + iat + iat]
            self.para['onsite' + iat + iat] = onsite
            self.para['uhubb' + iat + iat] = uhubb
        timeend = time.time()
        print('time of distance interpolation: ', time2 - time0)
        print('total time of distance interpolation in skf: ', timeend - time0)

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
                    self.math.sk_interp(dij, nameij)

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
            self.math.sk_interp_4d(dij, nameij, ncompr)

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

    def genskf_interp_compr_vec(self):
        """Generate interpolation of SKF with given compression radius."""
        pass


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


def hs_s_s(x, y, z, hss0, sss0):
    return hss0, sss0


def hs_s_x(x, y, z, hsp0, ssp0):
    return x * hsp0, x * ssp0


def hs_s_y(x, y, z, hsp0, ssp0):
    return y*hsp0, y*ssp0


def hs_s_z(x, y, z, hsp0, ssp0):
    return z*hsp0, z*ssp0


def hs_s_xy(x, y, z, hsd0, ssd0):
    return t.sqrt(t.tensor([3.]))*x*y*hsd0, t.sqrt(t.tensor([3.]))*x*y*ssd0


def hs_s_yz(x, y, z, hsd0, ssd0):
    return t.sqrt(t.tensor([3.]))*y*z*hsd0, t.sqrt(t.tensor([3.]))*y*z*ssd0


def hs_s_xz(x, y, z, hsd0, ssd0):
    return t.sqrt(t.tensor([3.]))*x*z*hsd0, t.sqrt(t.tensor([3.]))*x*z*ssd0


def hs_s_x2y2(x, y, z, hsd0, ssd0):
    return (0.5*t.sqrt(t.tensor([3.]))*(x**2-y**2)*hsd0,
            0.5*t.sqrt(t.tensor([3.]))*(x**2-y**2)*ssd0)


def hs_s_3z2r2(x, y, z, hsd0, ssd0):
    return (z**2-0.5*(x**2+y**2))*hsd0, (z**2-0.5*(x**2+y**2))*ssd0


def hs_x_s(x, y, z, hsp0, ssp0):
    return hs_s_x(-x, -y, -z, hsp0, ssp0)[0], hs_s_x(-x, -y, -z, hsp0, ssp0)[1]


def hs_x_x(x, y, z, hpp0, spp0, hpp1, spp1):
    return x**2*hpp0+(1-x**2)*hpp1, x**2*spp0+(1-x**2)*spp1


def hs_x_y(x, y, z, hpp0, spp0, hpp1, spp1):
    return x*y*hpp0-x*y*hpp1, x*y*spp0-x*y*spp1


def hs_x_z(x, y, z, hpp0, spp0, hpp1, spp1):
    return x*z*hpp0-x*z*hpp1, x*z*spp0-x*z*spp1


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


def hs_y_x(x, y, z, hpp0, spp0, hpp1, spp1):
    return hs_x_y(-x, -y, -z, hpp0, spp0, hpp1, spp1)[0], hs_x_y(
            -x, -y, -z, hpp0, spp0, hpp1, spp1)[1]


def hs_y_y(x, y, z, hpp0, spp0, hpp1, spp1):
    return y**2*hpp0+(1-y**2)*hpp1, y**2*spp0+(1-y**2)*spp1


def hs_y_z(x, y, z, hpp0, spp0, hpp1, spp1):
    return y*z*hpp0-y*z*hpp1, y*z*spp0-y*z*spp1



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



def hs_z_x(x, y, z, hpp0, spp0, hpp1, spp1):
    return hs_x_z(-x, -y, -z, hpp0, spp0, hpp1, spp1)[0], hs_x_z(
            -x, -y, -z, hpp0, spp0, hpp1, spp1)[1]


def hs_z_y(x, y, z, hpp0, spp0, hpp1, spp1):
    return hs_y_z(-x, -y, -z, hpp0, spp0, hpp1, spp1)[0], hs_y_z(
            -x, -y, -z, hpp0, spp0, hpp1, spp1)[1]


def hs_z_z(x, y, z, hpp0, spp0, hpp1, spp1):
    return (z**2*hpp0+(1-z**2)*hpp1,
            z**2*spp0+(1-z**2)*spp1)


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
