"""
Look at using strides to further reduce memory usage.
A warning should be issued if the SHELL_RESOLVED
setting conflicts with parameter set. For example
shell resolved may be true but the additional on site terms
may be truncated and the first just repeated.
"""
import torch as t
from dftbmalt.utils.exceptions import MutualExclusivityError
from dftbmalt.utils.batch import pack
import ml.batch as batch


class Basis:
    """Contains data relating to the basis set. This is of most use when
    converting from orbital to atom resolved data; i.e. orbital resolved
    mulliken charges to atom resolve.

    Parameters
    ----------
    atomic_numbers : `torch.tensor` [`int`]
        The atomic numbers of the atoms present in the system.
    max_ls : Any
        A callable object, such as a dictionary, that yields the maximum
        permitted angular momentum associated with a given atomic number.
        Note that the keys should be standard integers not torch tensors.


    Properties
    ----------
    max_l_atom : `torch.tensor` [`int`]
        pass
    n_orbitals : `torch.tensor` [`int`], `torch.int`
        The number of orbitals orbitals present in the specified system.

    Class Properties
    ----------------
    SHELL_RESOLVED : `bool`
        Indicates if shell or atom resolved mode is active. Note; shell
        resolved is commonly referred to as orbital resolved. This is used
        to automatically return the correct resolution information. This
        can be overridden locally by specifing it as a kwarg.

    Notes
    -----
    Atomic numbers are not saved within the class instance in an effort
    to reduce redundant code.

    Cached Properties
    -----------------
    Class instances have two "cached properties" named ``_basis_list`` &
    ``_basis_blocks``, which are derived from the two class properties
    ``_look_up`` & ``_blocks`` respectively. These help speed up calls
    to the ``azimuthal_matrix`` function without needing to store the
    underling result. The ``_look_up`` list stores a vector for each
    angular momenta:
        _look_up = [
            [0],  # <- for an S-subshell
            [1, 1, 1],  # <- for a P-subshell
            [2, 2, 2, 2, 2],  # <- for a D-subshell
            [3, 3, 3, 3, 3, 3, 3],  # <- for an F-subshell
            ...  # So on and so forth
        ]
    This is used to generate atomic azimuthal identity blocks. For
    example a carbon atom has an S & P subshells and thus it would
    add one S & 3 P orbitals, i.e.:
        Carbon_atom = [0, 1, 1, 1,]
        or
        Carbon_atom = torch.cat([_look_up[0], _look_up[1]])
    These would then be used to build up the azimuthal identity for the
    whole system ``_basis_list``. To save memory, ``_basis_list`` stores
    the vectors rather than concatenating them into a single list. The
    ``_blocks`` property is a list of truthy boolean tensors of size
    NxN where N = 2ℓ + 1, ℓ being the azimuthal quantum number. These
    are used in constructing the mask which blanks out the on site terms
    from the ``azimuthal_matrix``. By storing both ``_basis_list`` and
    ``_basis_blocks`` as composites of ``_look_up`` & ``_blocks`` the
    amount of memory per system can be significantly reduced.

    Todo
    ----
    - Combine vector / matrix / tensor functions i.e. have an argument
      that controls whether on_atoms, for example, returns a vector or
      a matrix form. [Priority: Moderate]
    - Cached properties should eventually all be made private.
      [Priority: QOL]
    - Update the docstring to include the new cached and uncached
      properties. [Priority: Moderate]
    - Docstring should be completely rewritten to be clearer & more
      concise. [Priority: High]
    """

    SHELL_RESOLVED = False
    __max_l = 5
    # Cached properties (see docstring for more information).
    _look_up = [t.arange(o + 1, dtype=t.int8).repeat_interleave(2 * t.arange(o + 1) + 1) for o in range(__max_l)]
    _blocks = [t.full((i, i), True) for i in (t.arange(__max_l) + 1) ** 2]
    _sub_look_up = [t.arange(i + 1, dtype=t.int8) for i in range(__max_l)]
    _sub_blocks = [t.full((i + 1, i + 1), True) for i in range(__max_l)]

    def __init__(self, atomic_numbers, max_ls, **kwargs):
        """Get atomic, orbital information."""
        self.atomic_numbers = atomic_numbers

        # get max(l) of each atom with given atomic number, such as for
        # H, max_l_atom is 0, for C, N and O, max_l_atom is 1
        self.max_l_atom = t.tensor([max_ls[int(z)] for z in atomic_numbers])

        # get the number of orbitals (H/S) of all atoms, such as for
        # H, n_orbitals is 1, for C, N and O, n_orbitals is 4
        self.n_orbitals = t.sum(self.orbs_per_atom())
        self.n_subshells = t.sum(self.max_l_atom + 1)

        # get the number of atoms in each system
        self.n_atoms = t.tensor(len(atomic_numbers))

        # Override class level SHELL_RESOLVED variable locally if instructed
        if 'SHELL_RESOLVED' in kwargs:
            self.SHELL_RESOLVED = kwargs['SHELL_RESOLVED']

        # Cached properties see class's docstring for more information.
        cls = self.__class__

        # get orbital information for each atom, H: [0], for C: [0, 1, 1, 1]
        self._basis_list = [cls._look_up[o] for o in self.max_l_atom]

        # get H/S block of each atom itself, for H: [[1]],
        # C: [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
        self._basis_blocks = [cls._blocks[i] for i in self.max_l_atom]

        self._sub_basis_list = [cls._sub_look_up[o] for o in self.max_l_atom]
        self._sub_basis_blocks = [cls._sub_blocks[i] for i in self.max_l_atom]

        self.shape = t.Size([self.n_orbitals, self.n_orbitals])
        self.subshape = t.Size([self.n_subshells, self.n_subshells])

    def orbs_per_atom(self):
        """Returns the number of orbitals associated with each atom.

        Returns
        -------
        orbs_per_atom : `torch.tensor` [`int`]
            Number of orbitals that each atom possesses.
        """
        return (self.max_l_atom + 1) ** 2

    def orbs_per_shell(self):
        """Returns the number of orbitals associated with each shell.

        Returns
        -------
        orbs_per_shell : `torch.tensor` [`int`]
            Number of orbitals that each shell possesses.
        """
        # Calculate the number of orbitals in each shell
        orbs_per_shell = [(2 * t.arange(s + 1)) + 1 for s in self.max_l_atom]

        # Flatten and return the list
        return t.tensor([i for j in orbs_per_shell for i in j])

    def orbs_per_res(self):
        """Selectively calls ``orbs_per_atom`` or ``orbs_per_shell``
        depending on the ``SHELL_RESOLVED`` setting. This allows for
        resolution agnostic coding

         Returns
         -------
         orbs_per_res : `torch.tensor` [`int`]
            Returns orbs_per_atom if ``SHELL_RESOLVED`` is False else
            orbs_per_shell is returned.
         """
        # Identify which function should be called & return the result
        if self.SHELL_RESOLVED:
            return self.orbs_per_shell()
        else:
            return self.orbs_per_atom()

    def on_atoms(self):
        """Identifies which atom each orbital belongs to.

        Returns
        -------
        on_atom : `torch.tensor` [`int`]
            Tensor indicating to which atom each orbital belongs.
        """
        opa = self.orbs_per_atom()
        return t.arange(len(opa)).repeat_interleave(opa)

    def on_shells(self):
        """Identifies which shell each orbital belongs to.

        Returns
        -------
        on_shell : `torch.tensor` [`int`]
            Tensor indicating to which shell each orbital belongs.
        """
        ops = self.orbs_per_shell()
        return t.arange(len(ops)).repeat_interleave(ops)

    def on_res(self):
        """Selectively calls ``on_atoms`` or ``on_shells`` based on the
        status of ``SHELL_RESOLVED``; allowing for resolution agnostic
        programming.

         Returns
         -------
         on_res : `torch.tensor` [`int`]
            Returns on_atoms if ``SHELL_RESOLVED`` is False else
            on_shells is returned.
         """
        # Identify which function should be called & return the result
        if self.SHELL_RESOLVED:
            return self.on_shells()
        else:
            return self.on_atoms()

    def azimuthal_matrix(self, block=False, sort=False, mask=False,
                         char=False, mask_on_site=True, mask_lower=True,
                         mask_diag=True):
        """Tensor defining the azimuthal quantum numbers (ℓ) associated
        with each orbital-orbital interaction element. For example, the
        azimuthal matrix of an N-orbital system will have a NxNx2 shape
        where the i'th, j'th vector lists the azimuthal quantum numbers
        of the i'th & j'th orbitals respectively; these being the same
        orbitals that are associated with the i'th, j'th element of the
        overlap and Hamiltonian matrices. As is the standard convention
        ℓ values are specified by integers; e.g. 0, 1, 2, & 3 for s, p,
        d, & f orbitals respectively. Alternately, a block form of the
        azimuthal matrix can be returned that defines only one element
        per subshell-subshell interaction block.

        The azimuthal matrix is primarily intended to be used as an aid
        during large masking, get & set operations. Where for-loops may
        cause an undesirable degree of CPU-overhead.

        Optionally, segments of the matrix can be masked out with -1's
        by setting ``mask``=True. Exactly which parts get masked can be
        controlled via the keyword parameters mask_*. However, it should
        be noted that all masking parameters have no effect when ``mask``
        is set to False. The masking keyword arguments have been moved
        outside of the traditional **kwargs declaration to help reduce
        computational overhead as this is time critical function.

        Parameters
        ----------
        block : `bool`, optional
            Reduces the full azimuthal matrix, where all elements of a
            given subshell-subshell interaction are explicitly defined,
            into a block-form where only a single element is returned
            for said interaction. For example; a dd interaction block
            would be represented by a 5x5x2 sub-matrix in full matrix
            mode (FMM) but would be represented by only a single 1x1x2
            element in block matrix mode (BMM). [DEFAULT=False]
        sort : `bool`, optional
            Sorts along the last dimension so the lowest ℓ value in each
            ℓ-pair comes first. Enables all interactions of a given type
            to be gathered with a single call, e.g. [0, 1] rather than
            two separate calls [0, 1] & [1, 0]. [DEFAULT=False]
        mask : `bool`, optional
            Masks segments of the matrix, with -1's, based on the mask
            options. [DEFAULT=False]
        char : `bool`, optional
            If True; a dtype=torch.int8 version of the azimuthal matrix
            is returned, rather than the standard torch.int64 form. This
            helps reduce memory overhead when dealing with large numbers
            of non-trivially sized systems. [DEFAULT=False]
        mask_on_site : `bool`, optional
            Masks on-site block elements in FMM & on-site blocks in BMM.
            [DEFAULT=True]
        mask_lower : `bool`, optional
            Masks lower triangle of the matrix in both FMM & BMM.
            [DEFAULT=True]
        mask_diag : `bool`, optional
            Masks diagonal elements in FMM but is ignored when in BMM.
            If False, this will unmask diagonals that were masked by
            the ``mask_on_site`` option. [DEFAULT=True]

        Returns
        -------
        azimuthal_matrix : `torch.tensor` [int]
            A NxNx2 tensor, where N is the number of orbitals, which
            identities the azimuthal quantum numbers of the orbitals
            involved in the various orbital-orbital interactions.

        Notes
        -----
        For a H4C2Au2 molecule, the full azimuthal matrix with diagonal,
        on-site & lower-triangle masking will take the form:

        .. image:: ../documentation/images/Azimuthal_matrix.png

        Where all elements in the ss blocks equal [0, 0], sp: [0, 1],
        ps: [1, 0], etc. Note that if ``sort`` is True then the ps
        elements will converted from [1, 0] to [0, 1] during sorting.
        Black & grey indicate areas of the matrix that have been masked
        out by setting their values to -1. A masking value of -1 is
        chosen as it is a invalid azimuthal quantum number, & is thus
        less likely to result in collision. If ``batch`` active then
        matrix be reduced to:

        .. image:: ../documentation/images/Azimuthal_matrix_block.png

        As the diagonal elements are contained within the on-site blocks
        it is not possible to selectively mask & unmask them.

        A dense matrix implementation is used by this function as the
        sparse matrix code in pytorch, as of version-1.7, is not stable
        or mature enough to support the types of operations that the
        returned tensor is intended to perform.

        Note that an atomic reduction of the azimuthal matrix is not
        possible. Hence there is no "atomic" option like that found in
        ``atomic_number_matrix``.

        """
        # Code is full/block mode agnostic to reduce code duplication. however
        # some properties must be set at the start to make this work.
        if not block:  # If returning the full matrix
            shape = self.shape
            basis_list = self._basis_list
            basis_blocks = self._basis_blocks

        # If returning the reduced block matrix
        else:
            shape = self.subshape
            basis_list = self._sub_basis_list
            basis_blocks = self._sub_basis_blocks

        # Repeat basis list to get ℓ-values for the 1'st orbital in each
        # interaction. Expand function is used as it is faster than repeat.
        l_mat = t.cat(basis_list).expand(shape)
        # Convert from an NxNx1 matrix into the NxNx2 azimuthal matrix
        l_mat = t.stack((l_mat.T, l_mat), -1)  # <-- Unmasked result

        if mask:  # < -- If masking out parts of the matrix
            # Initialise the base mask from the on-site blocks, if relevant
            # otherwise create a blank mask.
            if mask_on_site:  # <-- If masking on site blocks
                mask = batch.block_diag(basis_blocks)
            else:  # <-- Else initialise to a blank mask
                mask = t.full_like(l_mat, False)

            # Add lower triangle of the matrix to the mask
            if mask_lower:  # <-- But only if told to do so
                mask[tuple(t.tril_indices(*shape))] = True
                print("mask0", mask, tuple(t.tril_indices(*shape)))

            # If not in block mode mask/unmask the diagonals as instructed
            if not block:  # <-- Only valid for non block matrices
                # If mask_diag True; the diagonal will be masked, if False it
                # will be unmasked.
                mask.diagonal()[:] = mask_diag

            # Apply the mask and set all the masked values to -1
            print("mask", l_mat, mask)
            l_mat[mask.long(), :] = -1

        if sort:  # <-- Sort the angular momenta terms if requested
            l_mat = l_mat.sort(-1)[0]

        if not char:  # <-- If asked not to use dtype.int8
            l_mat = l_mat.long()  # < -- Convert back to dtype.int64

        # Finally return the azimuthal_matrix tensor
        return l_mat

    def atomic_number_matrix(self, block=False, atomic=False):
        """Tensor of atomic numbers associated with each orbital-orbital
        pairing. Analogous to the ``azimuthal_matrix`` tensor but with
        atomic rather than azimuthal quantum numbers. Note that a block
        or atomic form can be returned if requested. For more information
        see the ``azimuthal_matrix`` function's documentation.

        Parameters
        ----------
        block : `bool`, optional
            Reduce matrix to its block form where there is one entry per
            subshell-pair. See ``azimuthal_matrix`` for more detail. If
            ``block`` & ``atomic`` are both False then the full matrix
            is returned. Mutually exclusive with ``atomic``.
            [DEFAULT=False]
        atomic : `bool`, optional
            Reduce matrix to its atomic form where there is one entry
            per atom-pair. This is a greater reduction than ``block``
            & results in an NxNx2 matrix where N is the number of atoms.
            Mutually exclusive with ``block``. [DEFAULT=False]

        Returns
        -------
        atomic_number_matrix : `torch.tensor` [`int`]
            A NxNx2 tensor specifying the atomic numbers associated with
            each interaction. N can be the number of orbitals, subshells
            or atoms depending on the ``block`` & ``atomic`` options.

        Raises
        ------
        MutualExclusivityError
            If both ``block`` and ``atomic`` are True.

        Todo
        ----
        - This code will need to be optimised to reduce the time taken
          to execute. [Priority: Moderate]
        """
        if block and atomic:  # Raise exception if block & atomic are True
            raise MutualExclusivityError('block', 'atomic')

        # Construct the first NxN slice of the matrix
        if not block and not atomic:  # If generating the full matrix
            an_mat = self.atomic_numbers[self.on_atoms()].expand(self.shape)

        elif block:  # If constructing the block-wise matrix
            an_mat = self.atomic_numbers.repeat_interleave(
                self.max_l_atom + 1).expand(self.subshape)

        else:  # Otherwise construct the atomic-wise matrix
            an_mat = self.atomic_numbers.expand(self.atomic_numbers.shape * 2)

        # Form the NxN slice into the full NxNx2 tensor and return it.
        return t.stack((an_mat.T, an_mat), -1)

    def index_matrix(self, block=False, atomic=False):
        """Tensor specifying the indices of the atoms associated with
        each orbital-paring. This is identical in functionality and
        operation to ``atomic_number_matrix`` differing only in that it
        returns atom index instead of atomic number. For more information
        see the ``atomic_number_matrix`` function's documentation.

        Parameters
        ----------
        block : `bool`, optional
            Reduce matrix to its block form where there is one entry per
            subshell-pair.  If ``block`` & ``atomic`` are both False then
            the full matrix is returned. Mutually exclusive with ``atomic``.
            [DEFAULT=False]
        atomic : `bool`, optional
            Reduce matrix to its atomic form where there is one entry per
            atom-pair. Mutually exclusive with ``block``. [DEFAULT=False]

        Returns
        -------
        index_matrix : `torch.tensor` [`int`]
            A NxNx2 tensor specifying the indices of the atoms associated
            with each interaction. N can be the number of orbitals, sub-
            shells or atoms depending on the ``block`` & ``atomic`` options.

        Raises
        ------
        MutualExclusivityError
            If both ``block`` and ``atomic`` are True.
        """
        if block and atomic:  # Raise exception if block & atomic are True
            raise MutualExclusivityError('block', 'atomic')

        # Construct the first NxN slice of the matrix
        if not block and not atomic:  # If generating the full matrix
            i_mat = self.on_atoms().expand(self.shape)

        elif block:  # If constructing the block-wise matrix
            i_mat = t.arange(
                len(self.max_l_atom)
            ).repeat_interleave(self.max_l_atom + 1).expand(self.subshape)

        else:  # Otherwise construct the atomic-wise matrix
            n_atoms = len(self.max_l_atom)
            i_mat = t.arange(n_atoms).expand((n_atoms, n_atoms))

        # Form the NxN slice into the full NxNx2 tensor and return it.
        return t.stack((i_mat.T, i_mat), -1)


class Bases:
    """Functions identically to the ``Basis`` class but operates on a
    collection of ``Basis`` instances. Every function in ``Basis`` is
    mirrored in this class. However, these functions return a packed
    tensor that contains information on all contained systems.

    Todo
    ----
    - Rewrite doc string.
    - Add doc-strings to functions.
    - Add methods to fetch system properties
    - Consider a more effective way than looping over each system &
      packing its result.

    """
    def __init__(self, basis_list, max_l_dict):
        self.basis_list = basis_list

        self.max_l_dict = max_l_dict


        self.n_atoms = max([basis.atomic_numbers.nelement() for basis in basis_list])
        self.n_subshells = max([basis.n_subshells.nelement() for basis in basis_list])
        self.n_orbitals = max([basis.n_orbitals for basis in basis_list])


        self.n_atoms = max([basis.atomic_numbers.nelement() for basis in basis_list])
        self.n_subshells = max([basis.n_subshells.nelement() for basis in basis_list])
        self.n_orbitals = max([basis.n_orbitals for basis in basis_list])



        self.batch_size = len(basis_list)

        self.shape = t.Size([self.batch_size, self.n_orbitals, self.n_orbitals])



    def orbs_per_atom(self):
        return pack([basis.orbs_per_atom() for basis in self.basis_list])

    def orbs_per_shell(self):
        return pack([basis.orbs_per_shell() for basis in self.basis_list])

    def orbs_per_res(self):
        if Basis.SHELL_RESOLVED:
            return self.orbs_per_shell()
        else:
            return self.orbs_per_atom()

    def on_atoms(self):
        return pack(
            [basis.on_atoms() for basis in self.basis_list],
            value=self.max_atom - 1)

    def on_shells(self):
        return pack(
            [basis.on_shells() for basis in self.basis_list],
            value=self.max_subshells - 1)

    def on_res(self):
        if Basis.SHELL_RESOLVED:
            return self.on_shells()
        else:
            return self.on_atoms()

    def azimuthal_matrix(self, block=False, sort=False, mask=False,
                         char=False, mask_on_site=True, mask_lower=True,
                         mask_diag=True):
        args = (block, sort, mask, char, mask_on_site, mask_lower, mask_diag)
        return pack(
            [basis.azimuthal_matrix(*args) for basis in self.basis_list],
            value=-1)

    def atomic_number_matrix(self, block=False, atomic=False):
        args = (block, atomic)
        return pack(
            [basis.atomic_number_matrix(*args) for basis in self.basis_list],
            value=0)

    def index_matrix(self, block=False, atomic=False):
        args = (block, atomic)
        return pack(
            [basis.index_matrix(*args) for basis in self.basis_list],
            value=-1)

