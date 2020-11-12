import torch as t
import numpy as np
from dftbmalt.data.elements import symbol
from ml.padding import pad1d, pad2d
from dftbmalt.utils.exceptions import MutualExclusivityError
from dftbmalt.utils.batch import pack
from ml.padding import pad2d
_bohr = 0.529177249
_atom_name = ["H", "He",
              "Li", "Be", "B", "C", "N", "O", "F", "Ne",
              "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
              "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu",
              "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
              "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag",
              "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
              "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb",
              "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W ", "Re", "Os",
              "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At"]
VAL_ORB = {"H": 1, "C": 2, "N": 2, "O": 2, "Ti": 3}
_l_num = {"H": 0, "He": 0,
          "Li": 0, "Be": 0,
          "B": 1, "C": 1, "N": 1, "O": 1, "F": 1, "Ne": 1,
          "Na": 0, "Mg": 0,
          "Al": 1, "Si": 1, "P": 1, "S": 1, "Cl": 1, "Ar": 1,
          "K": 0, "Ca": 0,
          "Sc": 2, "Ti": 2, "V": 2, "Cr": 2, "Mn": 2, "Fe": 2, "Co": 2,
          "Ni": 2, "Cu": 2, "Zn": 2,
          "Ga": 1, "Ge": 1, "As": 1, "Se": 1, "Br": 1, "Kr": 1,
          "Rb": 0, "Sr": 0,
          "Y": 2, "Zr": 2, "Nb": 2, "Mo": 2, "Tc": 2, "Ru": 2, "Rh": 2,
          "Pd": 2, "Ag": 2, "Cd": 2,
          "In": 1, "Sn": 1, "Sb": 1, "Te": 1, "I": 1, "Xe": 1,
          "Cs": 0, "Ba": 0,
          "La": 3, "Ce": 3, "Pr": 3, "Nd": 3, "Pm": 3, "Sm": 3, "Eu": 3,
          "Gd": 3, "Tb": 3, "Dy": 3, "Ho": 3, "Er": 3, "Tm": 3, "Yb": 3, "Lu": 3,
          "Hf": 2, "Ta": 2, "W": 2, "Re": 2, "Os": 2, "Ir": 2, "Pt": 2,
          "Au": 2, "Hg": 2,
          "Tl": 1, "Pb": 1, "Bi": 1, "Po": 1, "At": 1}


class System_:
    def __init__(self, para, dataset, skf, ml=None):
        self.para = para
        self.dataset = dataset
        self.skf = skf
        self.ml = ml
        self.cal_coor_batch()

    def cal_coor_batch(self):
        """Generate vector, distance ... according to input geometry.

        Args:
            coor: [natom, 4], the first column is atom number

        Returns:
            natomtype: the type of atom, the 1st is 0, the 2nd different is 1
            atomind: how many orbitals of each atom in DFTB calculations

        """
        # check if cordinate is defined
        if 'positions' not in self.dataset.keys():
            raise ValueError('positions is not found')

        # check coordinate type
        if type(self.dataset['positions']) is np.ndarray:
            self.dataset['positions'] = t.from_numpy(self.dataset['positions'])

        # check dimension of coordinate and transfer to batch calculations
        if not self.para['Lbatch']:
            nfile = 1
            if self.dataset['positions'].dim() == 2:
                self.dataset['positions'].unsqueeze_(0)
            if 'natomAll' not in self.dataset.keys():
                self.dataset['natomAll'] = \
                    [len(icoor) for icoor in self.dataset['positions']]
        else:
            nfile = self.dataset['positions'].shape[0]
            if 'natomAll' not in self.dataset.keys():
                self.dataset['natomAll'] = \
                    [len(icoor) for icoor in self.dataset['positions']]
        nmax = max(self.dataset['natomAll'])

        # check atom number
        if type(self.dataset['atomNumber']) is list:

            # check atomnumber dimension
            if type(self.dataset['atomNumber'][0]) is not list:
                self.dataset['atomNumber'] = [self.dataset['atomNumber']]
        elif type(self.dataset['atomNumber']) is np.ndarray:

            # check atomnumber dimension
            if self.dataset['atomNumber'].ndim == 1:
                self.dataset['atomNumber'] = np.expand_dims(
                    self.dataset['atomNumber'], axis=0)

        elif type(self.dataset['atomNumber']) == t.Tensor:

            # check atomnumber dimension
            if self.dataset['atomNumber'].dim() == 1:
                self.dataset['atomNumber'].unsqueeze_(0).numpy()

        # if generate the atomname, if atomname exist, pass
        if 'atomNameAll' in self.dataset.keys():
            latomname = False
        else:
            self.dataset['atomNameAll'] = []
            latomname = True

        # distance matrix
        self.dataset['distance'] = t.zeros((nfile, nmax, nmax), dtype=t.float64)

        # normalized distance matrix
        self.dataset['dnorm'] = t.zeros((nfile, nmax, nmax, 3), dtype=t.float64)

        # coordinate vector
        self.dataset['dvec'] = t.zeros((nfile, nmax, nmax, 3), dtype=t.float64)

        self.dataset['natomtype'], self.dataset['norbital'] = [], []
        self.dataset['atomind2'] = []
        self.dataset['atomspecie'] = []
        self.dataset['lmaxall'] = []
        self.dataset['atomind'] = []
        self.dataset['positions'] /= self.para['BOHR']

        for ib in range(nfile):
            # define list for name of atom and index of orbital
            atomind = []

            # total number of atom
            natom = self.dataset['natomAll'][ib]
            atomnumber = self.dataset['atomNumber'][ib]
            coor = self.dataset['positions'][ib]

            # get index of orbitals atom by atom
            atomind.append(0)
            atomnamelist = [_atom_name[int(num) - 1] for num in atomnumber]

            # get l parameter of each atom
            atom_lmax = [VAL_ORB[_atom_name[int(atomnumber[iat] - 1)]]
                         for iat in range(natom)]

            for iat in range(natom):
                atomind.append(int(atomind[iat] + atom_lmax[iat] ** 2))
                for jat in range(natom):

                    # coordinate vector between atom pair
                    [xx, yy, zz] = coor[jat] - coor[iat]

                    # distance between atom and atom
                    dd = t.sqrt(xx * xx + yy * yy + zz * zz)
                    self.dataset['distance'][ib, iat, jat] = dd

                    if dd > 1E-1:

                        # get normalized distance, coordinate vector matrices
                        self.dataset['dnorm'][ib, iat, jat, :] = t.tensor([xx, yy, zz], dtype=t.float64) / dd
                        self.dataset['dvec'][ib, iat, jat, :] = t.tensor([xx, yy, zz], dtype=t.float64)

            dictat = dict(zip(dict(enumerate(set(atomnamelist))).values(),
                              dict(enumerate(set(atomnamelist))).keys()))

            # the type of atom, e.g, [0, 1, 1, 1, 1] for CH4 molecule
            self.dataset['natomtype'].append([dictat[ati] for ati in atomnamelist])

            # number of orbitals (dimension of H or S)
            self.dataset['norbital'].append(atomind[-1])

            # total orbitals in each calculation if flatten to 1D
            self.dataset['atomind2'].append(
                int(atomind[natom] * (atomind[natom] + 1) / 2))

            # atom specie
            self.dataset['atomspecie'].append(list(set(atomnamelist)))

            # l parameter and index of orbital of each atom
            self.dataset['lmaxall'].append(atom_lmax)
            self.dataset['atomind'].append(atomind)

            # the name of all the atoms
            # if latomname:
            self.dataset['atomNameAll'].append(atomnamelist)

        # return dataset
        return self.dataset


class System:
    """System object.

    This object will generate single system (molecule, unit cell) information,
    or a list of systems information from dataset. The difference of single and
    multi systems input will be the dimension.
    In general, the output information will include symbols, max quantum
    number, angular moments, magnetic moments, masses and atomic charges.

    Args:
        numbers (LongTensor, sequences): Atomic number of each atom in single or
            multi system. For multi systems, if system sizes is not same, then
            use sequences tensor input.
        positions (FloatTensor): :math:`(N, 3)` where `N = number of atoms`
            in single system, or :math:`(M, N, 3)` where `M = number of
            systems, N = number of atoms` in multi systems.
        cell : `torch.tensor` [`float`], optional
            The lattice vectors of the cell, if applicable. This should be
            a 3x3 matrix. While a 1x3 vector can be specified it will be
            auto parsed into a 3x3 vector.

        **kwargs
            Additional keyword arguments:
                ``pbc``
            True will enact periodic boundary conditions (PBC) along all
            cell dimensions (buck systems), False will fully disable PBC
            (isolated molecules), & an array of booleans will only enact
            PBC on a subset of cell dimensions (slab). The first two can
            be auto-inferred from the presence of the ``cell`` parameter.
            (`bool`, `array_like` [`bool`], optional)


    Properties:
        pbc : `bool`, `array_like` [`bool`]
            See **kwarg discussion on ``pbc`` for a description.


    Notes:
        Units, such as distance, should be given in atomic units. PyTorch
        tensors will be instantiated using the default dtypes. These can be
        changed using torch.set_default_dtype.

    Todo:
        Add in option to automatically convert units. [Priority: Low]
        Create unit tests. [Priority: Low]
        Add unit metadata to HDF5 output. [Priority: Low]
    """

    def __init__(self, numbers, positions, lattice=None, unit='angstrom', **kwargs):
        # sequences of tensor
        if type(numbers) is list and type(numbers[0]) is t.Tensor:
            self.numbers = pad1d(numbers)

        # tensor
        elif type(numbers) is t.Tensor:
            self.numbers = numbers
        else:
            raise ValueError("numbers should be LongTensor or sequences")

        # tensor
        if type(positions) is list and type(positions[0]) is t.Tensor:
            self.positions = pad2d(positions)
        elif type(positions) is t.Tensor:
            self.positions = positions
        else:
            raise ValueError("positions should be torch.Tensor or list")

        # transfer positions from angstrom to bohr
        self.positions = self.positions / _bohr if unit == 'angstrom' else self.positions

        # add one dimension for single system to satisfy batch calculations
        if self.numbers.dim() == 1:
            self.numbers.unsqueeze_(0)
        if self.positions.dim() == 2:
            self.positions.unsqueeze_(0)

        # get distance
        self.distances = self.get_distances()

        # get symbols
        self.symbols = self.get_symbols()

        # size of batch size, size of each system (number of atoms)
        self.size_batch = len(self.numbers)
        self.size_system = self.get_size()

        # get max l of each atom
        self.l_max = self.get_l_numbers()

        # orbital_index is the orbital index of each atom in each system
        # orbital_index_cumsum is the acculated orbital_index
        self.orbital_index, self.orbital_index_cumsum, self.number_orbital = \
            self.get_accumulated_orbital_numbers()

        # get Hamiltonian, overlap shape in batch
        self.hs_shape = self.get_hs_shape()

        # Parse the lattice
        self.lattice = System.__resolve_lattice(lattice)

        # Get pbc value from **kwargs if present; otherwise set pbc to True if
        # the lattice parameter was passed & False if not.
        self.pbc = kwargs['pbc'] if 'pbc' in kwargs else lattice is not None

    @property
    def z(self):
        """Short-hand alias for numbers."""
        return self.numbers

    def get_atomic_numbers(self):
        """Alias to improve ase compatibility."""
        return self.numbers

    def get_distances(self):
        """Return distances between a list of atoms for each system."""
        sum_square = [(ipostion ** 2).sum(1).view(-1, 1)
                      for ipostion in self.positions]
        return pad2d([t.sqrt(abs((isq + isq.view(1, -1)) - 2. * ipos @ ipos.t()))
                      for ipos, isq in zip(self.positions, sum_square)])

    def get_symbols(self):
        """Get atom name for each system in batch."""
        return [[_atom_name[ii - 1] for ii in inu[inu.nonzero()]] for inu in self.numbers]

    def get_positions_vec(self):
        """Return positions vector between atoms."""
        return pad2d([ipo.unsqueeze(-3) - ipo.unsqueeze(-2)
                      for ipo in self.positions])

    def get_size(self):
        """Get each system size (number of atoms) in batch."""
        return [len(inum[inum.nonzero()]) for inum in self.numbers]

    def get_l_numbers(self):
        """Return the number of orbitals associated with each atom."""
        return pad1d([t.tensor([_l_num[ii] for ii in isym])
                      for isym in self.symbols])

    def get_atom_orbital_numbers(self):
        """Return the number of orbitals associated with each atom.

        The atom orbital numbers is from (l + 1) ** 2.
        """
        return pad1d([t.tensor([(_l_num[ii] + 1) ** 2 for ii in isym])
                      for isym in self.symbols])

    def get_resolved_orbital(self):
        """Return l parameter and realted atom specie of each obital."""
        resolved_orbital_specie = \
            [sum([[ii] * int(jj) for ii, jj in zip(isym, iind)], [])
             for isym, iind in zip(self.symbols, self.orbital_index)]
        _l_orbital_res = [[0], [0, 1, 1, 1], [0, 1, 1, 1, 2, 2, 2, 2, 2]]
        l_orbital_res = [sum([_l_orbital_res[iil] for iil in il], [])
                             for il in self.l_max]
        return resolved_orbital_specie, l_orbital_res

    def get_accumulated_orbital_numbers(self):
        """Return accumulated number of orbitals associated with each atom.

        For instance, for CH4, get_atom_orbital_numbers return [[4, 1, 1, 1,
        1]], this function will return [[0, 4, 5, 6, 7, 8]], max_orbital is 8.
        """
        atom_index = self.get_atom_orbital_numbers()
        index_cumsum = [t.cat((t.zeros(1), t.cumsum(iind, -1)))
                        for iind in atom_index]
        number_orbital = [int(ind[-1]) for ind in index_cumsum]
        return atom_index, index_cumsum, number_orbital

    def get_hs_shape(self):
        """Return shapes of Hamiltonian and overlap."""
        maxorb = max([iorb[-1] for iorb in self.orbital_index_cumsum]).int()

        # 3D shape: size of batch, total orbital number, total orbital number
        return t.Size([len(self.orbital_index_cumsum), maxorb, maxorb])

    def get_global_species(self):
        """Get species for single or multi systems according to numbers."""
        numbers_ = t.unique(self.numbers)
        numbers_nonzero = numbers_[numbers_.nonzero().squeeze()]
        if numbers_nonzero.dim() == 0:
            return _atom_name[numbers_nonzero - 1]
        else:
            return [_atom_name[ii - 1] for ii in numbers_nonzero]

    def azimuthal_matrix(self, block=False, sort=False, mask=False,
                         char=False, mask_on_site=True, mask_lower=True,
                         mask_diag=True, ibatch=0):
        """Tensor defining the azimuthal quantum numbers (ℓ).

        Azimuthal quantum numbers are associated with each orbital-orbital
        interaction element. For example, the azimuthal matrix of an N-orbital
        system will have a NxNx2 shape where the i'th, j'th vector lists the
        azimuthal quantum numbers of the i'th & j'th orbitals respectively;
        these being the same orbitals that are associated with the i'th, j'th
        element of the overlap and Hamiltonian matrices. As is the standard
        convention ℓ values are specified by integers; e.g. 0, 1, 2, & 3 for
        s, p, d, & f orbitals respectively. Alternately, a block form of the
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

        Args:
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

        Returns:
            azimuthal_matrix : `torch.tensor` [int]
                A NxNx2 tensor, where N is the number of orbitals, which
                identities the azimuthal quantum numbers of the orbitals
                involved in the various orbital-orbital interactions.

        Notes:
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
            shape = self.shape  # <-- shape of the output matrix
            basis_list = self._basis_list  # <-- see class docstring
            basis_blocks = self._basis_blocks # <-- see class docstring
        else:  # If returning the reduced block matrix
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
                mask = t.block_diag(*basis_blocks)
            else:  # <-- Else initialise to a blank mask
                mask = t.full_like(l_mat, False)

            # Add lower triangle of the matrix to the mask
            if mask_lower:  # <-- But only if told to do so
                mask[tuple(t.tril_indices(*shape))] = True

            # If not in block mode mask/unmask the diagonals as instructed
            if not block:  # <-- Only valid for non block matrices
                # If mask_diag True; the diagonal will be masked, if False it
                # will be unmasked.
                mask.diagonal()[:] = mask_diag

            # Apply the mask and set all the masked values to -1
            l_mat[mask, :] = -1

        if sort:  # <-- Sort the angular momenta terms if requested
            l_mat = l_mat.sort(-1)[0]

        if not char:  # <-- If asked not to use dtype.int8
            l_mat = l_mat.long()  # < -- Convert back to dtype.int64

        # Finally return the azimuthal_matrix tensor
        return l_mat

    def azimuthal_matrix_batch(self, block=False, sort=False, mask=False,
                               char=False, mask_on_site=True, mask_lower=True,
                               mask_diag=True):
        args = (block, sort, mask, char, mask_on_site, mask_lower, mask_diag)
        return pad2d([self.azimuthal_matrix(*args, ibatch=ibatch)
                      for ibatch in range(self.size_batch)])


    @classmethod
    def from_ase_atoms(cls, atoms):
        """Instantiate a System instance from an ase.Atoms object.

        Args:
            atoms: ASE Atoms object(s) to be converted into System instance(s).

        Returns:
            System : System object.

        Notes:
            Tensors will not inherit their dtype from the np.arrays, but
            will rather use the pytorch default.
        """
        if isinstance(atoms, list):  # If multiple atoms objects supplied:
            # Recursively call from_ase_atoms and return the result
            return [cls.from_ase_atoms(i) for i in atoms]

        # For a single atoms system: 1) Get default dtype, to prevent inheriting
        # the  numpy array's dtype, then 2) build and return the Systems object.
        dtype = t.get_default_dtype()
        return System(t.tensor(atoms.get_atomic_numbers()),
                      t.tensor(atoms.positions, dtype=dtype),
                      t.tensor(atoms.cell, dtype=dtype),
                      pbc=t.tensor(atoms.pbc))

    def to_hd5(self, target):
        """Converts the System instance to a set of hdf5 datasets which
        are then written into the specified hdf5.
        Parameters
        ----------
        target : `h5py.Group`, `h5py.File`
            The hdf5 entity to which the set of h5py.Dataset instances
            representing the system should be written.

        Notes
        -----
        This function does not create its own group as it expects that
        ``target`` is the group into which data should be writen.
        """
        # Short had for dataset creation
        add_data = target.create_dataset

        # Add datasets for numbers, positions, lattice, and pbc
        add_data('numbers', data=self.numbers)
        add_data('positions', data=self.positions.numpy())
        add_data('lattice', data=self.lattice.numpy())
        add_data('pbc', data=self.pbc.numpy())

    @staticmethod
    def from_hd5(source):
        """Converts an hdf5.Groups entity to a Systems instance.

        Parameters
        ----------
        source : `h5py.group`, `h5py.File`
            hdf5 File/Group containing the system's data.

        Returns
        -------
        system : `System`:
            A systems instance representing the data stored.

        Notes
        -----
        It should be noted that dtype will not be inherited from the
        database. Instead the default PyTorch dtype will be used.
        """
        # Get default dtype
        dtype = t.get_default_dtype()

        # Read & parse datasets from the database into a System instance
        # & return the result.
        return System(
            t.tensor(source['numbers']),
            t.tensor(source['positions'], dtype=dtype),
            t.tensor(source['lattice'], dtype=dtype),
            pbc=t.tensor(source['pbc']))

    # Helper Functions
    # ----------------
    # These mostly help abstract messy operations from the __init__ etc.
    @staticmethod
    def __resolve_lattice(lattice):
        """This ensure the specified lattice is in the correct format &
        initialises the default value safely.

        Parameters
        ----------
        lattice : `torch.tensor`, `None`
            The lattice parameter passed to the __init__ function.
        """

        # If "lattice" is specified but is a 1x3 vector; then convert to a 3x3
        if lattice is not None and lattice.dim() == 1:
            lattice = lattice.diag()

        # If lattice is None then use a blank array of zeros
        lattice = lattice if lattice is not None else t.zeros(3, 3)

        # Return the lattice term. All self. assigment operations should be
        # kept to the __init__ function if possible.
        return lattice

    def __str__(self):
        """Create a printable representation of the System."""
        # Return the reduced chemical formula by:
        #   1) Get unique numbers & the counts associated with them
        #   2) Look up elemental symbol & append the count to the end
        #   3) Join all element-count pairs together.
        return ''.join([f'{symbol[int(z)]}{n}'for z, n in
                        zip(*self.atomic_numbers.unique(return_counts=True))])

    def __repr__(self):
        """Create a string representation of the System object."""
        # Returns the reduced chemical formula with the class name attached
        return f'{self.__class__.__name__}({str(self)})'



'''skf_path = '/home/gz_fan/Documents/ML/dftb/DFTBMaLT/auorg-1-1'
# Ase atoms object for the molecule we want to look at
from ase.build import molecule as molecule_database
molecule = molecule_database('CH4')
molecule = System.from_ase_atoms(molecule)
# Static property section
# Max angular momentum associated with each element type
max_l_key = {1: 0, 6: 1, 7: 1, 8: 1, 79: 2}
#sk_integral_generator = SKIntegralGenerator.from_dir(skf_path)
# Section 3: Creating a basis object
molecule.get_atomic_numbers()

mole = System(t.tensor([6, 1, 1, 1, 1]),
       t.tensor([[ 0.000000000000000,  0.000000000000000,  0.000000000000000],
                 [ 0.629118000000000,  0.629118000000000,  0.629118000000000],
                 [-0.629118000000000, -0.629118000000000,  0.629118000000000],
                 [ 0.629118000000000, -0.629118000000000, -0.629118000000000],
                 [-0.629118000000000,  0.629118000000000, -0.629118000000000]]))
print(mole.get_distances(), mole.get_symbols())'''


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
    max_l_on_atom : `torch.tensor` [`int`]
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
    _blocks = [t.full((i, i), True, dtype=t.bool) for i in (t.arange(__max_l) + 1) ** 2]
    _sub_look_up = [t.arange(i + 1, dtype=t.int8) for i in range(__max_l)]
    _sub_blocks = [t.full((i + 1, i + 1), True, dtype=t.bool) for i in range(__max_l)]

    def __init__(self, atomic_numbers, max_ls, **kwargs):
        self.atomic_numbers = atomic_numbers

        self.max_l_on_atom = t.tensor([max_ls[int(z)] for z in atomic_numbers])
        self.n_orbitals = t.sum(self.orbs_per_atom())
        self.n_subshells = t.sum(self.max_l_on_atom + 1)
        self.n_atoms = t.tensor(len(atomic_numbers))

        # Override class level SHELL_RESOLVED variable locally if instructed
        if 'SHELL_RESOLVED' in kwargs:
            self.SHELL_RESOLVED = kwargs['SHELL_RESOLVED']

        # Cached properties see class's docstring for more information.
        cls = self.__class__
        self._basis_list = [cls._look_up[o] for o in self.max_l_on_atom]
        self._basis_blocks = [cls._blocks[i] for i in self.max_l_on_atom]

        self._sub_basis_list = [cls._sub_look_up[o] for o in self.max_l_on_atom]
        self._sub_basis_blocks = [cls._sub_blocks[i] for i in self.max_l_on_atom]

        self.shape = t.Size([self.n_orbitals, self.n_orbitals])
        self.subshape = t.Size([self.n_subshells, self.n_subshells])

    def orbs_per_atom(self):
        """Returns the number of orbitals associated with each atom.

        Returns
        -------
        orbs_per_atom : `torch.tensor` [`int`]
            Number of orbitals that each atom possesses.
        """
        return (self.max_l_on_atom + 1) ** 2

    def orbs_per_shell(self):
        """Returns the number of orbitals associated with each shell.

        Returns
        -------
        orbs_per_shell : `torch.tensor` [`int`]
            Number of orbitals that each shell possesses.
        """
        # Calculate the number of orbitals in each shell
        orbs_per_shell = [(2 * t.arange(s + 1)) + 1 for s in self.max_l_on_atom]

        # Flatten and return the list
        return t.tensor([i for j in orbs_per_shell for i in j])

    def orbs_per_res(self):
        """Selectively calls ``orbs_per_atom`` or ``orbs_per_shell````
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
        """Tensor defining the azimuthal quantum numbers (ℓ).

        Azimuthal quantum numbers are associated with each orbital-orbital
        interaction element. For example, the azimuthal matrix of an N-orbital
        system will have a NxNx2 shape where the i'th, j'th vector lists the
        azimuthal quantum numbers of the i'th & j'th orbitals respectively;
        these being the same orbitals that are associated with the i'th, j'th
        element of the overlap and Hamiltonian matrices. As is the standard
        convention ℓ values are specified by integers; e.g. 0, 1, 2, & 3 for
        s, p, d, & f orbitals respectively. Alternately, a block form of the
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

        Args:
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

        Returns:
            azimuthal_matrix : `torch.tensor` [int]
                A NxNx2 tensor, where N is the number of orbitals, which
                identities the azimuthal quantum numbers of the orbitals
                involved in the various orbital-orbital interactions.

        Notes:
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
            shape = self.shape  # <-- shape of the output matrix
            basis_list = self._basis_list  # <-- see class docstring
            basis_blocks = self._basis_blocks # <-- see class docstring
        else:  # If returning the reduced block matrix
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
                mask = t.block_diag(*basis_blocks)
            else:  # <-- Else initialise to a blank mask
                mask = t.full_like(l_mat, False)

            # Add lower triangle of the matrix to the mask
            if mask_lower:  # <-- But only if told to do so
                mask[tuple(t.tril_indices(*shape))] = True

            # If not in block mode mask/unmask the diagonals as instructed
            if not block:  # <-- Only valid for non block matrices
                # If mask_diag True; the diagonal will be masked, if False it
                # will be unmasked.
                mask.diagonal()[:] = mask_diag

            # Apply the mask and set all the masked values to -1
            l_mat[mask, :] = -1

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
                self.max_l_on_atom + 1).expand(self.subshape)

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
                len(self.max_l_on_atom)
            ).repeat_interleave(self.max_l_on_atom + 1).expand(self.subshape)

        else:  # Otherwise construct the atomic-wise matrix
            n_atoms = len(self.max_l_on_atom)
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
    - Check why self.max_l_dict is needed

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
