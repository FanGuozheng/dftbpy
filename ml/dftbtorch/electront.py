#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch as t


class DFTBelect:
    '''this class is electron part'''
    def __init__(self, para):
        self.para = para
        self.nat = self.para['natom']
        self.atind = self.para['atomind']

    def fermi(self, eigval):
        '''
        fermi distribution
        there is no gradient here right now
        '''
        occ = t.zeros(self.atind[self.nat])
        nelect = self.para['nelectrons']
        natom = self.para['natom']
        norbs = int(self.atind[natom])
        telec = self.para['tElec']
        if nelect > 1e-4:
            if nelect > 2 * norbs:
                print('Warning: too many electrons')
            nef = int(nelect / 2)

            if telec < self.para['t_zero_max']:
                if nelect % 2 == 0:
                    occ[:nef] = 2
                    self.para['nocc'] = nef
                elif nelect % 2 == 1:
                    occ[:nef] = 2
                    occ[nef] = 1
                    self.para['nocc'] = nef + 1
<<<<<<< HEAD
        self.para['occ'] = occ
=======
>>>>>>> ad0a72eac1ab68c13c8d6d1ed31874c22bb490c3
        return occ

    def fermiold(self, eigval, occ):
        '''fermi-dirac distributions'''
        nelect = self.para['nelectrons']
        natom = self.para['natom']
        atomind = self.para['atomind']
        norbs = int(atomind[natom])
        telec = self.para['tElec']
        kb_hartree = self.para['boltzmann_constant_H']
        degtol = 1.0E-4
        racc = 2E-16
        dacc = 4 * racc
        for i in range(1, norbs):
            occ[i] = 0.0
        if nelect > 1.0E-5:
            if nelect > 2 * norbs:
                print('too many electrons')
            elif telec > 5.0:
                beta = 1.0 / (kb_hartree * telec)
                etol = kb_hartree * telec * (t.log(beta) - t.log(racc))
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
            while nup0 < norbs:
                if abs(eigval[nup0] - efermi) < etol:
                    nup0 = nup0 + 1
                else:
                    break
            nup = nup0
            while ndown0 > 0:
                if abs(eigval[ndown0 - 1] - efermi) < etol:
                    ndown0 = ndown0 - 1
                else:
                    break
            ndown = ndown0
            ndeg = nup - ndown
            nocc2 = ndown
            for i in range(0, nocc2):
                occ[i] = 2.0
            if ndeg == 0:
                return occ, efermi
            if tzero:
                occdg = ndeg
                occdg = (nelect - 2 * nocc2) / occdg
                for i in range(nocc2, nocc2 + ndeg):
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
                    occ[i] = 2.0/(1.0 + t.exp(beta * (eigval[i] - efermi)))
                    charge = charge + occ[i]
                    if charge > chleft:
                        ef2 = efermi
                    else:
                        ef1 = efermi
                    if abs(charge - chleft) > ceps or abs(ef1 - ef2) < eeps:
                        continue
                    else:
                        exit
                if abs(charge - chleft) < ceps:
                    return
                else:
                    fac = chleft / charge
                    for i in range(istart, iend):
                        occ[i] = occ[i] * fac

    def gmatrix(self):
        '''
        this function is to build the gamma in second-order term
        see: Self-consistent-charge density-functional tight-binding method
        for simulations of complex materials properties
        '''
        distance = self.para['distance']
        # uhubb = self.para['uhubb']
        natom = self.para['natom']
<<<<<<< HEAD
        if self.para['HSsym'] == 'symhalf':
            nameall = self.para['atomnameall']
            gmat = t.empty(int((natom + 1) * natom / 2))
            icount = 0
            for iatom in range(0, natom):
                namei = nameall[iatom] + nameall[iatom]
                for jatom in range(0, iatom + 1):
                    rr = distance[iatom, jatom]
                    namej = nameall[jatom] + nameall[jatom]
                    # a1 = 3.2 * uhubb[iatom, 2]
                    # a2 = 3.2 * uhubb[jatom, 2]
                    # need rev!!!
                    a1 = 3.2 * self.para['uhubb' + namei][2]
                    a2 = 3.2 * self.para['uhubb' + namej][2]
                    src = 1 / (a1 + a2)
                    fac = a1 * a2 * src
                    avg = 1.6 * (fac + fac * fac * src)
                    fhbond = 1
                    if rr < 1.0E-4:
                        gval = 0.3125 * avg
                    else:
                        rrc = 1.0 / rr
                        if abs(a1 - a2) < 1.0E-5:
                            fac = avg * rr
                            fac2 = fac * fac
                            efac = t.exp(-fac) / 48.0
                            gval = (1.0 - fhbond * (48.0 + 33 * fac + fac2 *
                                                    (9.0 + fac)) * efac) * rrc
                        else:
                            val12 = self.gamsub(a1, a2, rr, rrc)
                            val21 = self.gamsub(a2, a1, rr, rrc)
                            gval = rrc - fhbond * val12 - fhbond * val21
                    gmat[icount] = gval
                    icount += 1
        elif self.para['HSsym'] in ['symall', 'symall_chol']:
            nameall = self.para['atomnameall']
            gmat = t.empty(natom, natom)
            for iatom in range(0, natom):
                namei = nameall[iatom] + nameall[iatom]
                for jatom in range(0, natom):
                    rr = distance[iatom, jatom]
                    namej = nameall[jatom] + nameall[jatom]
                    # a1 = 3.2 * uhubb[iatom, 2]
                    # a2 = 3.2 * uhubb[jatom, 2]
                    # need rev!!!
                    a1 = 3.2 * self.para['uhubb' + namei][2]
                    a2 = 3.2 * self.para['uhubb' + namej][2]
                    src = 1 / (a1 + a2)
                    fac = a1 * a2 * src
                    avg = 1.6 * (fac + fac * fac * src)
                    fhbond = 1
                    if rr < 1.0E-4:
                        gval = 0.3125 * avg
                    else:
                        rrc = 1.0 / rr
                        if abs(a1 - a2) < 1.0E-5:
                            fac = avg * rr
                            fac2 = fac * fac
                            efac = t.exp(-fac) / 48.0
                            gval = (1.0 - fhbond * (48.0 + 33 * fac + fac2 *
                                                    (9.0 + fac)) * efac) * rrc
                        else:
                            val12 = self.gamsub(a1, a2, rr, rrc)
                            val21 = self.gamsub(a2, a1, rr, rrc)
                            gval = rrc - fhbond * val12 - fhbond * val21
                    gmat[iatom, jatom] = gval
=======
        nameall = self.para['atomnameall']
        gmat = t.empty(int((natom + 1) * natom / 2))
        icount = 0
        for iatom in range(0, natom):
            namei = nameall[iatom] + nameall[iatom]
            for jatom in range(0, iatom + 1):
                rr = distance[iatom, jatom]
                namej = nameall[jatom] + nameall[jatom]
                # a1 = 3.2 * uhubb[iatom, 2]
                # a2 = 3.2 * uhubb[jatom, 2]
                # need rev!!!
                a1 = 3.2 * self.para['uhubb' + namei][2]
                a2 = 3.2 * self.para['uhubb' + namej][2]
                src = 1 / (a1 + a2)
                fac = a1 * a2 * src
                avg = 1.6 * (fac + fac * fac * src)
                fhbond = 1
                if rr < 1.0E-4:
                    gval = 0.3125 * avg
                else:
                    rrc = 1.0 / rr
                    if abs(a1 - a2) < 1.0E-5:
                        fac = avg * rr
                        fac2 = fac * fac
                        efac = t.exp(-fac) / 48.0
                        gval = (1.0 - fhbond * (48.0 + 33 * fac + fac2 * (9.0
                                + fac)) * efac) * rrc
                    else:
                        val12 = self.gamsub(a1, a2, rr, rrc)
                        val21 = self.gamsub(a2, a1, rr, rrc)
                        gval = rrc - fhbond * val12 - fhbond * val21
                gmat[icount] = gval
                icount += 1
>>>>>>> ad0a72eac1ab68c13c8d6d1ed31874c22bb490c3
        return gmat

    def gamsub(self, a, b, rr, rrc):
        a2 = a * a
        b2 = b * b
        b4 = b2 * b2
        b6 = b4 * b2
        drc = 1.0 / (a2 - b2)
        drc2 = drc * drc
        efac = t.exp(-a * rr)
        fac = (b6 - 3 * a2 * b4) * drc2 * drc * rrc
        gval = efac * (0.5 * a * b4 * drc2 - fac)
        return gval

    def shifthamgam(self, para, qatom, qzero, gmat):
        '''this function is to realize: sum_K(gamma_IK * Delta_q_K)'''
        natom = para['natom']
        shift = t.zeros(natom)
        qdiff = t.zeros(natom)
        qdiff[:] = qatom[:] - qzero[:]
<<<<<<< HEAD
        if self.para['HSsym'] == 'symhalf':
            for iat in range(0, natom):
                shifti = 0
                for jat in range(0, natom):
                    if jat > iat:
                        k = jat * (jat + 1) / 2 + iat
                        gamma = gmat[int(k)]
                    else:
                        k = iat * (iat + 1) / 2 + jat
                        gamma = gmat[int(k)]
                    shifti = shifti + qdiff[jat] * gamma
                shift[iat] = shifti
        elif self.para['HSsym'] in ['symall', 'symall_chol']:
            for iat in range(0, natom):
                shifti = 0
                for jat in range(0, natom):
                    gamma = gmat[iat, jat]
                    shifti = shifti + qdiff[jat] * gamma
                shift[iat] = shifti
        return shift

    def mulliken(self, sym, overmat, denmat):
        '''calculate Mulliken charge'''
        norbs = int(self.atind[self.nat])
        qatom = t.zeros(self.nat)
        if sym == 'symhalf':
            for iat in range(0, self.nat):
                for iind in range(int(self.atind[iat]), int(self.atind[iat + 1])):
                    for jind in range(0, iind):
                        k = iind * (iind + 1) / 2 + jind
                        qatom[iat] = qatom[iat] + denmat[int(k)] * overmat[int(k)]
                    for jind in range(iind, norbs):
                        k = jind * (jind + 1) / 2 + iind
                        qatom[iat] = qatom[iat] + denmat[int(k)] * overmat[int(k)]
        elif sym in ['symall', 'symall_chol']:
            for iat in range(0, self.nat):
                for iind in range(int(self.atind[iat]), int(self.atind[iat + 1])):
                    for jind in range(0, norbs):
                        denmatij = denmat[jind, iind]
                        overmatij = overmat[jind, iind]
                        qatom[iat] = qatom[iat] + denmatij * overmatij
=======
        for iat in range(0, natom):
            shifti = 0
            for jat in range(0, natom):
                if jat > iat:
                    k = jat * (jat + 1) / 2 + iat
                    gamma = gmat[int(k)]
                else:
                    k = iat * (iat + 1) / 2 + jat
                    gamma = gmat[int(k)]
                shifti = shifti + qdiff[jat] * gamma
            shift[iat] = shifti
        return shift

    def mulliken(self, overmat, denmat):
        '''calculate Mulliken charge'''
        norbs = int(self.atind[self.nat])
        qatom = t.zeros(self.nat)
        for iat in range(0, self.nat):
            for iind in range(int(self.atind[iat]), int(self.atind[iat + 1])):
                for jind in range(0, iind):
                    k = iind * (iind + 1) / 2 + jind
                    qatom[iat] = qatom[iat] + denmat[int(k)] * overmat[int(k)]
                for jind in range(iind, norbs):
                    k = jind * (jind + 1) / 2 + iind
                    qatom[iat] = qatom[iat] + denmat[int(k)] * overmat[int(k)]
>>>>>>> ad0a72eac1ab68c13c8d6d1ed31874c22bb490c3
        return qatom
