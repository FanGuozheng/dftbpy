#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np


class DFTB_elect:

    def __init__(self, para):
        self.para = para

    def fermi(self, eigval, occ):
        nelect = self.para['nelectrons']
        natom = self.para['natom']
        atomind = self.para['atomind']
        norbs = int(atomind[natom])
        telec = self.para['tElec']
        ckbol = 3.16679E-6   # original from lodestar, with revision
        degtol = 1.0E-4
        racc = 2E-16
        dacc = 4 * racc
        for i in range(1, norbs):
            occ[i] = 0.0
        if nelect > 1.0E-5:
            if nelect > 2*norbs:
                print('too many electrons')
            elif telec > 5.0:
                beta = 1.0/(ckbol*telec)
                etol = ckbol*telec*(np.log(beta)-np.log(racc))
                tzero = False
            else:
                etol = degtol
                tzero = True
            if nelect > int(nelect):
                nef1 = int((nelect + 2) / 2)
                nef2 = int((nelect + 2) / 2)
            else:
                nef1 = int((nelect + 1) / 2)
                nef2 = int((nelect + 2) / 2)
            # eBot = eigval[0]
            efermi = 0.5 * (eigval[nef1 - 1] + eigval[nef2 - 1])
            nup = nef1
            ndown = nef1
            nup0 = nup
            ndown0 = ndown
            while nup0 < norbs:   #
                if abs(eigval[nup0] - efermi) < etol:
                    nup0 = nup0+1
                else:
                    break
            nup = nup0
            while ndown0 > 0:
                if abs(eigval[ndown0 - 1] - efermi) < etol:
                    ndown0 = ndown0 - 1
                else:
                    break
            ndown = ndown0
            ndeg = nup - ndown    # check
            nocc2 = ndown
            for i in range(0, nocc2):
                occ[i] = 2.0
            if ndeg == 0:
                return occ, efermi
            if tzero:
                occdg = ndeg
                occdg = (nelect-2*nocc2)/occdg
                for i in range(nocc2, nocc2+ndeg):
                    occ[i] = occdg
            else:
                chleft = nelect - 2 * nocc2
                istart = nocc2 + 1
                iend = istart + ndeg - 1
                if ndeg == 1:
                    occ[istart] = chleft
                    return
                ef1 = efermi - etol - degtol
                ef2 = efermi + etol + degtol
                ceps = dacc * chleft
                eeps = dacc * max(abs(ef1), abs(ef2))
                efermi = 0.5 * (ef1 + ef2)  # check
                charge = 0.0
                for i in range(istart, iend):
                    occ[i] = 2.0/(1.0 + np.exp(beta * (eigval[i] - efermi)))
                    charge = charge + occ[i]
                    if charge > chleft:
                        ef2 = efermi
                    else:
                        ef1 = efermi
                    if abs(charge - chleft) > ceps or abs(ef1 - ef2) < eeps:
                        continue
                    else:
                        exit
                if abs(charge-chleft) < ceps:
                    return
                else:
                    fac = chleft / charge
                    for i in range(istart, iend):
                        occ[i] = occ[i] * fac

    def gmatrix(self):
        distance = self.para['distance']
        uhubb = self.para['uhubb']
        natom = self.para['natom']
        gmat = []
        for iatom in range(0, natom):
            namei = self.para['atomnameall'][iatom]
            ii = self.para['atomname_set'].index(namei)
            for jatom in range(0, iatom+1):
                namej = self.para['atomnameall'][jatom]
                jj = self.para['atomname_set'].index(namej)
                rr = distance[iatom, jatom]
                a1 = 3.2 * uhubb[ii, 2]
                a2 = 3.2 * uhubb[jj, 2]
                src = 1 / (a1 + a2)
                fac = a1 * a2 * src
                avg = 1.6 * (fac + fac * fac * src)
                fhbond = 1
                if rr < 1.0E-4:
                    gval = 0.3125*avg
                else:
                    rrc = 1.0/rr
                    if abs(a1 - a2) < 1.0E-5:
                        fac = avg*rr
                        fac2 = fac * fac
                        efac = np.exp(-fac) / 48.0
                        gval = (1.0 - fhbond * (48.0 + 33 * fac + fac2 *
                                                (9.0 + fac)) * efac) * rrc
                    else:
                        val12 = self.gamsub(a1, a2, rr, rrc)
                        val21 = self.gamsub(a2, a1, rr, rrc)
                        gval = rrc - fhbond * val12 - fhbond * val21
                gmat.append(gval)
        return gmat

    def gamsub(self, a, b, rr, rrc):
        a2 = a * a
        b2 = b * b
        b4 = b2 * b2
        b6 = b4 * b2
        drc = 1.0 / (a2 - b2)
        drc2 = drc * drc
        efac = np.exp(-a * rr)
        fac = (b6 - 3 * a2 * b4) * drc2 * drc * rrc
        gval = efac * (0.5 * a * b4 * drc2 - fac)
        return gval

    def shifthamgam(self, natom, qatom, qzero, gmat):
        qdiff = np.zeros(natom)
        shift = []
        qdiff[:] = qatom[:] - qzero[:]
        for i in range(0, natom):
            shifti = 0
            for j in range(0, natom):
                if j > i:
                    k = j * (j + 1) / 2 + i
                    gamma = gmat[int(k)]
                else:
                    k = i * (i + 1) / 2 + j
                    gamma = gmat[int(k)]
                shifti += qdiff[j] * gamma
            shift.append(shifti)
        shift = np.array(shift)
        return shift

    def mulliken(self, overmat, denmat):
        '''calculate Mulliken charge'''
        natom = self.para['natom']
        atomind = self.para['atomind']
        norbs = int(atomind[natom])
        qat = np.zeros((natom), dtype=float)
        for ii in range(natom):
            qat[ii] = 0.0
            for i in range(int(atomind[ii]), int(atomind[ii+1])):
                for j in range(0, i):
                    k = i*(i+1)/2+j
                    qat[ii] = qat[ii]+denmat[int(k)]*overmat[int(k)]
                for j in range(i, norbs):
                    k = j*(j+1)/2+i
                    qat[ii] = qat[ii]+denmat[int(k)]*overmat[int(k)]
        return qat
