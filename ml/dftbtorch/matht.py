#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import scipy.interpolate


class DFTBmath(object):

    def __init__(self):
        pass

    def polysk3thsk(self, allarr, darr, dd):
        row, col = np.shape(allarr)
        skftable = np.zeros(row)
        hs = np.zeros(col)
        for ii in range(0, col):
            skftable[:] = allarr[:, ii]
            fcubic = scipy.interpolate.interp1d(darr, skftable, kind='cubic')
            hs[ii] = fcubic(dd)
        return hs

    def polysk5thsk(self, allarr, darr, dd):
        ni = ninterpline
        dx = self.gridmesh*5
        ytail = hs_skf[ni-1, :]
        ytailp = (hs_skf[ni-1, :]-hs_skf[ni-2, :])/self.gridmesh
        ytailp2 = (hs_skf[ni-2, :]-hs_skf[ni-3, :])/self.gridmesh
        ytailpp = (ytailp-ytailp2)/self.gridmesh
        xx = np.array([self.gridmesh*4, self.gridmesh*3, self.gridmesh*2,
                       self.gridmesh, 0.0])
        nline = ninterpline
        for xxi in xx:
            dx1 = ytailp * dx
            dx2 = ytailpp * dx * dx
            dd = 10.0 * ytail - 4.0 * dx1 + 0.5 * dx2
            ee = -15.0 * ytail + 7.0 * dx1 - 1.0 * dx2
            ff = 6.0 * ytail - 3.0 * dx1 + 0.5 * dx2
            xr = xxi / dx
            yy = ((ff*xr + ee)*xr + dd)*xr*xr*xr
            hs_skf[nline, :] = yy
            nline += 1
        return hs_skf
