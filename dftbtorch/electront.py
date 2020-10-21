"""Electronic calculations."""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch as t
from ml.padding import pad1d
import torch.nn.functional as F


class DFTBelect:
    """Deal with electronic calculation."""

    def __init__(self, para, geometry, skf):
        """Initialize parameters."""
        self.para = para
        self.geo = geometry
        self.skf = skf

    def fermi(self, eigval, nelectron, telec=0.):
        """Fermi-Dirac distributions without smearing.

        Parameters
        ----------
        eigval: `torch.tensor` [`float`]
            eigenvalues, 2D tensor
        nelectron: `torch.tensor` [`float`]
            number of electrons, 2D tensor
        telec: [`float`]
            temperature

        Returns
        -------
        occ: `torch.tensor` [`float']
            occupancies of electrons

        """
        # make sure each system has at least one electron
        assert False not in t.ge(nelectron, 1)

        # the number of full occupied state
        electron_pair = t.true_divide(nelectron.clone().detach(), 2).int()

        # the left single electron
        electron_single = (nelectron.clone().detach() % 2).unsqueeze(1)

        # zero temperature
        if telec == 0.:

            # occupied state for batch, if full occupied, occupied will be 2
            # with unpaired electron, return 1
            occ_ = pad1d([
                t.cat((t.ones(electron_pair[i]) * 2, electron_single[i]), 0)
                for i in range(nelectron.shape[0])])

            # pad the rest unoccupied states with 0, the size of occ is
            # the largest size in batch
            occ = F.pad(input=occ_,
                        pad=(0, eigval.shape[-1] - occ_.shape[-1]), value=0)

            # all occupied states (include full and not full occupied)
            nocc = (nelectron.clone().detach() / 2).ceil()

        # return occupation of electrons
        return occ, nocc

    def gmatrix(self, distance, natom, atomname):
        """Build the gamma (2D) in second-order term.
        Args:
            distance
            Uhubbert
        Returns:
            Gamma matrix in second order
        """
        gmat = t.zeros((natom, natom), dtype=t.float64)
        for iatom in range(natom):
            namei = atomname[iatom] + atomname[iatom]
            for jatom in range(natom):
                rr = distance[iatom, jatom]
                namej = atomname[jatom] + atomname[jatom]
                a1 = 3.2 * self.skf['uhubb' + namei][2]
                a2 = 3.2 * self.skf['uhubb' + namej][2]
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

    def _gamma_gaussian(self, U, position, **kwargs):
        """Construct the gamma matrix via the gaussian method.
        Parameters
        ----------
        U : `torch.tensor` [`float`]
            Vector specifying, in order, the Hubbard U values for each atom/orbital.
        position : `torch.tensor` [`float`], `np.array` [`float`]
            Position of each atom/orbital.
        **kwargs
            Additional keyword arguments:
                ``FWHM``:
                    Override default full width half max values. (`torch.tensor`)
                    [DEFAULT=None]
                ``low_mem_mode``:
                    Low memory mode reduces the amount of memory required to build
                    the gamma matrix by utilising for loops rather than matrix and
                    vector operations. However, this results in slower excitation.
                    (`float`) [DEFAULT=False]
        Returns
        -------
        gamma : `torch.tensor`
            Gamma matrix.
        Notes
        -----
        This follows the implementation described by Koskinen in reference [1].
        If `U` & `positions` are specified on a per-atom basis then the resulting
        matrix will most likely need be expand via "torch.repeat_interleave".
        .. [1] Koskinen, P., & Mäkinen, V. (2009). Density-functional tight-binding
           for beginners. Computational Materials Science, 47(1), 237–253.
        Todo:
        - Implement low memory mode.
        """

        # If FWHM values have been specified, use them; else calculate them
        FWHM = (kwargs['FWHM'] if 'FWHM' in kwargs
                else np.sqrt(8. * np.log(2) / np.pi) / U)

        # Catch for low memory mode
        if kwargs.get('low_mem_mode', False):
            # This is not implemented yet so raise an exception
            raise NotImplementedError('Low memory mode is not currently available')
        else:
            # This operates on the flattened upper triangular components of the
            # matrices only. This saves memory & avoids having to deal with the
            # diagonal elements.

            # An ellipsis slicer is needed for multi-system mode, this reverts to
            # "None" for single system mode.
            s = ... if U.dim() == 2 else None

            # Construct index list for upper triangle gather operation
            ut = t.unbind(t.triu_indices(U.shape[-1], U.shape[-1], 1))
            print("ut", ut)

            # Calculate "c" values using equation 27 [1]
            c = t.sqrt((4. * np.log(2.)) / (FWHM[s, ut[0]]**2 + FWHM[s, ut[1]]**2))

            # Construct the upper triangle of the euclidean distance matrix
            distances = t.sqrt(t.sum((position[s, ut[0], :] - position[s, ut[1], :])**2, -1))

            # Construct a blank gamma matrix
            gamma = t.zeros(*U.shape, U.shape[-1], dtype=U.dtype)

            # Calculate off diagonal gamma terms, via equation 26, and assign them
            gamma[s, ut[0], ut[1]] = t.erf(c * distances) / distances

            # Symmetries the upper triangle to the lower triangle
            gamma[s, ut[1], ut[0]] = gamma[s, ut[0], ut[1]]

            # Apply the diagonal terms via and ugly, but concise, indexing opp
            gamma.diagonal(0, -(U.dim() - 1))[:] = U

        # Return the gamma matrix
        return gamma


    def _gamma_slater(self, U, position, **kwargs):
        """Construct the gamma matrix via the slater method.
        Parameters
        ----------
        U : `torch.tensor` [`float`]
            Vector specifying, in order, the Hubbard U values for each atom/orbital.
        position : `torch.tensor` [`float`], `np.array` [`float`]
            Position of each atom/orbital.
        **kwargs
            Additional keyword arguments:
                ``low_mem_mode``:
                    Low memory mode reduces the amount of memory required to build
                    the gamma matrix by utilising for loops rather than matrix and
                    vector operations. However, this results in slower excitation.
                    (`float`) [DEFAULT=False]
        Returns
        -------
        gamma : `torch.tensor`
            Gamma matrix.
        Notes
        -----
        If `U` & `positions` are specified on a per-atom basis then the resulting
        matrix will most likely need be expand via "torch.repeat_interleave".
        Todo:
        - Multi-system vectorisation. [Priority: Low]
        - Add reference to a paper. [Priority: Low]
        """

        # Catch for low memory mode
        if kwargs.get('low_mem_mode', False):
            # This is not implemented yet so raise an exception
            raise NotImplementedError('Low memory mode is not currently available.')
        else:
            raise NotImplementedError('Slater based gamma construction unavailable.')
        pass


    def __gamma(U_dict, orbital_ids, posiiton, method, **kwargs):
        """Placeholder for the function that is to wrap the various gamma matrix
        construciton methods.
        Parameters
        ----------
        U_dict : `dict` [`tuple` [`int`]: `torch.tensor` [`float`]]
            Dictionary of Hubbard Us values keyed by (z , l) tuples where z & l are
            the atomic number & the azimuthal quantum number respectively. If this
            is a non-orbital-resolved calculation then different "l" values for the
            same atom should reference the same Hubbard U tensor value. The actual
            U value should be a zero dimensional tensor.
        orbital_ids : `list` [`tuple` [`int`]]
            Ordered list of tuples specifying the identities of each orbital in the
            form (n, z, l, m), where "n" is the atom number (i.e the atom the orbital
            belongs to), "z" is the atomic number, "l" the azimuthal quantum number
            and "m" the magnetic quantum number (which is technically not required).
        **kwargs
            Additional keyword arguments:
                ``low_mem_mode``:
                    Low memory mode reduces the amount of memory required to build
                    the gamma matrix by utilising for loops rather than matrix and
                    vector operations. However, this results in slower excitation.
                    (`float`) [DEFAULT=False]
        Todo:
        - Multi-system vectorisation. Note that the U vector construction will have
          to be reworked to enable it to handel multiple systems at once.
          [Priority: Low]
        - Identify more elegant way to handel construction of U vector, perhaps it
          should be done externally. [Priority: Low]
        """

        # Catch for low memory mode
        if kwargs.get('low_mem_mode', False):
            # This is not implemented yet so raise an exception
            raise NotImplementedError('Low memory mode is not currently available')

        # Build a Hubbard U vector, i.e. a vector with one U for each orbital
        U = t.stack([U_dict[orb[1:3]] for orb in orbital_ids])

    def shifthamgam(self, para, qatom, qzero, gmat):
        """Calculate: sum_K(gamma_IK * Delta_q_K)."""
        # get the shift gamma, gmat should be 2D
        shift = (qatom - qzero) @ gmat
        return shift

    def mulliken(self, overmat, denmat, atomindex, natom):
        """Calculate Mulliken charge for both batch and single system.

        Parameters
        ----------
        overmat: `torch.tensor` [`float`]
            overlap, 3D tesnor
        denmat: `torch.tensor` [`float`]
            density matrix, 3D tensor
        atomindex: `list` [`tuple` [`int`]]
            index of orbital information for each atom
        """
        # sum overlap and density by hadamard product, get charge by orbital
        qatom_orbital = (denmat * overmat).sum(dim=1)

        # define charge by atom
        qatom = t.zeros((natom), dtype=t.float64)

        # transfer charge from orbital to atom
        # qatom = qatom_orbital[]
        for iat in range(natom):

            # get the sum of orbital of ith atom
            init, end = atomindex[iat], atomindex[iat + 1]
            qatom[iat] = qatom_orbital[init: end].sum()

        return qatom
