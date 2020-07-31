"""Electronic calculations."""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch as t


class DFTBelect:
    """Deal with electronic calculation."""

    def __init__(self, para):
        """Initialize parameters."""
        self.para = para

        # number of atom
        self.nat = self.para['natom']

        # atom orbital index
        self.atind = self.para['atomind']

    def fermi(self, eigval):
        """Fermi-Dirac distributions."""
        # define occupied electron matrix
        occ = t.zeros((self.atind[self.nat]), dtype=t.float64)

        # total electron number
        nelect = self.para['nelectrons']

        # total orbital number
        norbs = int(self.atind[self.nat])

        # system temperature
        telec = self.para['tElec']

        # make sure the electron number is positive integer
        assert nelect >= 1

        # electron number consistency
        assert nelect == norbs

        # the occupied state
        nef = int(nelect / 2)

        # zero temperature
        if telec < self.para['t_zero_max']:

            # full occupied state
            occ[: nef] = 2

            # no unpaired electrons
            if nelect % 2 == 0:
                self.para['nocc'] = nef

            # exist unpaired electrons
            elif nelect % 2 == 1:
                occ[nef] = 1
                self.para['nocc'] = nef + 1

        # return occupied electron in each state
        self.para['occ'] = occ

    def gmatrix(self):
        """Build the gamma (2D) in second-order term.

        Args:
            distance
            Uhubbert
        Returns:
            Gamma matrix in second order

        """
        # get distance
        distance = self.para['distance']

        nameall = self.para['atomnameall']
        gmat = t.empty((self.nat, self.nat), dtype=t.float64)
        for iatom in range(self.nat):
            namei = nameall[iatom] + nameall[iatom]
            for jatom in range(self.nat):
                rr = distance[iatom, jatom]
                namej = nameall[jatom] + nameall[jatom]
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
        """Calculate: sum_K(gamma_IK * Delta_q_K)."""
        # get the shift gamma, gmat should be 2D
        shift = (qatom - qzero) @ gmat
        return shift

    def mulliken(self, sym, overmat, denmat):
        """Calculate Mulliken charge with 2D density, overlap matrices."""
        # sum overlap and density by hadamard product, get charge by orbital
        qatom_orbital = (denmat * overmat).sum(dim=1)

        # define charge by atom
        qatom = t.zeros((self.nat), dtype=t.float64)

        # transfer charge from orbital to atom
        for iat in range(self.nat):

            # get the sum of orbital of ith atom
            init, end = int(self.atind[iat]), int(self.atind[iat + 1])
            qatom[iat] = sum(qatom_orbital[init: end])

        return qatom
