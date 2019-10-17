# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import os.path
import numpy as np
import scipy.interpolate

intergraltyperef = {'[2, 2, 0, 0]': 0, '[2, 2, 1, 0]': 1, '[2, 2, 2, 0]': 2,
                    '[1, 2, 0, 0]': 3, '[1, 2, 1, 0]': 4, '[1, 1, 0, 0]': 5,
                    '[1, 1, 1, 0]': 6, '[0, 2, 0, 0]': 7, '[0, 1, 0, 0]': 8,
                    '[0, 0, 0, 0]': 9, '[2, 2, 0, 1]': 10, '[2, 2, 1, 1]': 11,
                    '[2, 2, 2, 1]': 12, '[1, 2, 0, 1]': 13, '[1, 2, 1, 1]': 14,
                    '[1, 1, 0, 1]': 15, '[1, 1, 1, 1]': 16, '[0, 2, 0, 1]': 17,
                    '[0, 1, 0, 1]': 18, '[0, 0, 0, 1]': 19}


class SkInterpolator:
    """This code aims to generate integrals by interpolation method.
    Therefore, the inputs include the grid points of the integrals,
    the compression radius of atom1 (r1) and atom2 (r2)
    """
    def __init__(self, grid0, gridmesh):
        self.grid0 = grid0
        self.gridmesh = gridmesh

    def readskffile(self, ninterpfile, atomnameall, directory):
        """gridmesh_points, onsite_spe_u, mass_rcut, integrals are parameters
        from skf file"""
        atomname1 = atomnameall[0]
        atomname2 = atomnameall[1]
        filenamelist = SkInterpolator.getfilenamelist(self, atomname1,
                                                      atomname2, directory)
        nfile = len(filenamelist)
        ncompressr = int(np.sqrt(nfile))
        skffile = {}
        gridmesh_points = np.empty((nfile, 2))
        mass_rcut = np.empty((nfile, 20))
        integrals = []
        loopnum = 0
        for filename in filenamelist:
            fp = open(os.path.join(directory, filename), "r")
            words = fp.readline().split()
            gridmesh_points[loopnum, 0] = float(words[0])
            gridmesh_points[loopnum, 1] = int(words[1])
            nrow = int(words[1])
            if ninterpfile in (0, 3):
                spe = fp.readline().split()
                skffile["onsitespeu"] = spe
                data = np.fromfile(fp, dtype=float, count=20, sep=" ")
                mass_rcut[loopnum, :] = data
                nitem = nrow*20
                data = np.fromfile(fp, dtype=float, count=nitem, sep=" ")
                data.shape = (nrow, 20)
                integrals.append(data)
                skffile["rest"] = fp.read()
            elif ninterpfile in (1, 2):
                data = np.fromfile(fp, dtype=float, count=20, sep=" ")
                mass_rcut[loopnum, :] = data
                nitem = nrow*20
                data = np.fromfile(fp, dtype=float, count=nitem, sep=" ")
                data.shape = (nrow, 20)
                integrals.append(data)
                skffile["rest"] = fp.read()
            loopnum += 1
        # here extra 5 lines are for polytozero
        skflinemax = int(gridmesh_points[:, 1].max())+5
        skffile["skflinemax"] = skflinemax
        superskf = np.empty((ncompressr, ncompressr, skflinemax, 20))
        numk = 0
        print('nfile', nfile, 'ncompressr', ncompressr)
        for skfi in range(0, nfile):
            rowi = int(numk // ncompressr)
            colj = int(numk % ncompressr)
            numk += 1
            nrow = int(gridmesh_points[skfi, 1])
            ncol = 20
            nitem = nrow * ncol
            superskf[rowi, colj, :nrow, :] = integrals[skfi]
        skffile["gridmeshpoint"] = gridmesh_points
        skffile["massrcut"] = mass_rcut
        skffile["intergrals"] = superskf
        skffile["nfilenamelist"] = nfile
        return skffile

    def getfilenamelist(self, atomname1, atomname2, directory):
        """read all the skf files and return lists of skf files according to
        the types of skf """
        filename = atomname1+'-'+atomname2+'.skf.'
        filenamelist = []
        filenames = os.listdir(directory)
        filenames.sort()
        for name in filenames:
            if name.startswith(filename):
                filenamelist.append(name)
        return filenamelist

    def getallgenintegral(self, ninterpfile, skffile, r1, r2, gridarr1,
                          gridarr2):
        """this function is to generate the whole integrals"""
        superskf = skffile["intergrals"]
        nfile = skffile["nfilenamelist"]
        row = int(np.sqrt(nfile))
        xneigh = (np.abs(gridarr1 - r1)).argmin()
        yneigh = (np.abs(gridarr2 - r2)).argmin()
        ninterp = round(xneigh*row + yneigh)
        ninterpline = int(skffile["gridmeshpoint"][ninterp, 1])
        print("ninterpline", ninterpline)
        hs_skf = np.empty((ninterpline+5, 20))
        for lineskf in range(0, ninterpline):
            distance = lineskf*self.gridmesh + self.grid0
            counti = 0
            for intergrali in intergraltyperef:
                znew3 = SkInterpolator.getintegral(self, r1, r2, intergrali,
                                                   distance, gridarr1,
                                                   gridarr2, superskf)
                hs_skf[lineskf, counti] = znew3
                counti += 1
        return hs_skf, ninterpline

    def getintegral(self, interpr1, interpr2, integraltype, distance,
                    gridarr1, gridarr2, superskf):
        """this function is to generate interpolation at given distance and
        given compression radius"""
        numgridpoints = len(gridarr1)
        numgridpoints2 = len(gridarr2)
        if numgridpoints != numgridpoints2:
            print('Error: the dimension is not equal')
        skftable = np.empty((numgridpoints, numgridpoints))
        numline = int((distance - self.grid0)/self.gridmesh)
        numtypeline = intergraltyperef[integraltype]
        skftable = superskf[:, :, numline, numtypeline]
        funcubic = scipy.interpolate.interp2d(gridarr2, gridarr1, skftable,
                                              kind='cubic')
        interporbital = funcubic(interpr2, interpr1)
        return interporbital

    def polytozero(self, hs_skf, ninterpline):
        """Here, we fit the tail of skf file (5lines, 5th order)"""
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

    def saveskffile(self, ninterpfile, atomnameall, skffile, hs_skf,
                    ninterpline):
        """this function is to save all parts in skf file"""
        atomname1 = atomnameall[0]
        atomname2 = atomnameall[1]
        nfile = skffile["nfilenamelist"]
        if ninterpfile in (0, 3):
            print('generate {}-{}.skf'.format(atomname1, atomname2))
            with open('{}-{}.skf'.format(atomname1, atomname1), 'w') as fopen:
                fopen.write(str(skffile["gridmeshpoint"][nfile-1][0])+" ")
                fopen.write(str(int(ninterpline)))
                fopen.write('\n')
                np.savetxt(fopen, skffile["onsitespeu"], fmt="%s", newline=" ")
                fopen.write('\n')
                np.savetxt(fopen, skffile["massrcut"][nfile-1], newline=" ")
                fopen.write('\n')
                np.savetxt(fopen, hs_skf)
                fopen.write('\n')
                fopen.write(skffile["rest"])
        elif ninterpfile in (1, 2):
            print('generate {}-{}.skf'.format(atomname1, atomname2))
            with open('{}-{}.skf'.format(atomname1, atomname2), 'w') as fopen:
                fopen.write(str(skffile["gridmeshpoint"][nfile-1][0])+" ")
                fopen.write(str(int(ninterpline)))
                fopen.write('\n')
                np.savetxt(fopen, skffile["massrcut"][nfile-1], newline=" ")
                fopen.write('\n')
                np.savetxt(fopen, hs_skf)
                fopen.write('\n')
                fopen.write(skffile["rest"])
