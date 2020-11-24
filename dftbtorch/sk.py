"""Slater-Koster integrals related."""
import os
import time
import sys
import numpy as np
import torch as t
import h5py
from IO.save import Save1D, Save2D
from dftbtorch.matht import DFTBmath, BicubInterp
from dftbtorch.interpolator import PolySpline, BicubInterpVec
from dftbmalt.utils.utilities import split_by_size
ATOMNAME = {1: 'H', 6: 'C', 7: 'N', 8: 'O'}
_SQR3 = np.sqrt(3.)
_HSQR3 = 0.5 * np.sqrt(3.)


class GetSKTable:
    """Read skf type files.

    Files marked with the "skf" extension are primarily used by DFTB+ to hold
    Slater-Koster integral tables & their associated repulsive components.

    Args:
        elements : `torch.tensor` [`int`]
            Atomic numbers of the elements involved. A single atomic number
            can also be specified, however it will be resolve to a tensor.
        H : `torch.tensor` [`float`]
            Hamilton_grid_pointsan integral table, interaction type varies over columns
            and distance over rows, see notes for more information.
        S : `torch.tensor` [`float`]
            Overlap integral table, see notes for more information.
        HS_grid : `torch.tensor` [`float`]
            Grid points at which the H & S integrals are evaluated.
        on_site : `torch.tensor` [`float`]
            On-site energies for angular momenta (f), d, p & s. "f" is only
            present for the extended file format.
        U : `torch.tensor` [`float`]
            Hubbard Us for the angular momenta present in ``on_site``.
        occupations: `torch.tensor` [`float`]
            The occupancies of the angular momenta present in ``on_site``.

    Optional Parameters:
        R : `torch.tensor` [`float`], optional
            Coefficients for the main part of the repulsive spline. The
            Notes section of this doc-string should be consulted for
            more information about the repulsive component.
        R_short : `torch.tensor` [`float`], optional
            Coefficients for the exponential region before the main spline.
        R_long : `torch.tensor` [`float`], optional
            Coefficients for distances beyond the main spline.
        R_grid : `torch.tensor` [`float`], optional
            Grid points at which the main repulsive spline is evaluated
            note that one additional point indicating the termination
            point of the last spline region is needed.
        R_cutoff : `torch.tensor` [`float`]
            The cutoff value for the repulsive interaction, beyond which
            it is taken to be zero. (zero dimensional tensor)
        R_poly : `torch.tensor` [`float`], optional
            Polynomial coefficients for the repulsive component, but only
            if ``repulsion_by_spline`` is False.
        mass: `torch.tensor` [`float`]
            Mass of the atom in atomic un_grid_pointsts. Only used when ``homo`` is True.


    Properties:
        repulsion_by_spline : `bool`
            True if repulsive interaction is described via a spline & False
            if described by polynomial coefficients.
        version : `str` # KWARG***
            Identifies the file version number upon read, and controls the
            format used during writing. [DEFAULT = 1.0]
        homo : `bool`
            Indicates if this is the HOMO-nuclear case.

    Notes:
        This class is currently only capable of reading skf files. This is
        due to the way in which this class currently handel's the repulsive
        spline.

        The electron_grid_pointsc section of the skf file is more or less always the
        same. The only variation in its defin_grid_pointstion is in the version 1.0
        extended file format, which introduces f orbitals. In the 0.9 and
        1.0 (standard) versions there is a hamilton_grid_pointsan and overlap element
        for each of the following interactions:
            ddσ, ddπ, ddδ, pdσ, pdπ, ppσ, ppπ, sdσ, spσ, ssσ

    Note that this order is maintained in the columns of ``H`` & ``S``.
    For the extended (1.0e) file format the available interactions are:
        ffσ, ffπ ffδ ffγ, dfσ, dfπ, dfδ, ddσ, ddπ, ddδ, pfσ, pfπ,
        pdσ, pdπ, ppσ, ppπ, sfσ, sdσ, spσ, ssσ

    The rows in ``H`` and ``S`` correspond to the each distances found
    in ``HS_grid``.

    Within skf file versions 0.9 & 1.0 the repulsive component can be
    specified by either 1) a set of polynomial coefficients & a cutoff
    or 2) a spline with a short-range exponential region & a long range
    tail. To define repulsive interaction via the first method only two
    parameter is required:
     - ``R_poly``: [c2, c3, c5, c5, c6, c7, c8, c9]
     - ``R_cutoff``: Point beyond which repulsion is pinned to zero.
    For the spline method the following parameters are required:
     - ``R``: [c0, c1, c2, c3] coefficient sets for each spline region.
     - ``R_short``: [a1, a2, a3]
     - ``R_long``: [c0, c1, c2, c3]
     - ``R_grid``: regions over which the spline segments act.
     - ``R_cutoff``: Point beyond which repulsion is pinned to zero.
    Here; ``R_short`` acts on any distance below the first ``R_grid``
    point, ``R_long`` upon distances between the last ``R_grid`` point
    and ``R_cutoff``, & the n'th region of the main spline acts over the
    distance spanned between the n'th & n'th + 1 point of ``R_grid``.
    It should be noted the spline and polynomial forms are mutually
    exclusive. A more detailed description of the repulsive term can be
    found in the skf file specification.

    This does not yet support the 2.0 skf file specification. It is worth
    noting that the non-extended 1.0 version and the 0.9 version are one
    in the same, with the exception of a comment, hence they are treated
    identically.

    More in-depth file specificationn information can be found at:
        github.com/bhourahine/slako-doc
        dftb.org/parameters/introduction

    Todo:
    - Refactor the code completely to neaten it up, and make it more
      legible. [Priority: Moderate]
    - Add in functionality to automatically construct repulsive splines
      from list of values over a specified range of distances.
      [Priority: Moderate]
    """

    def __init__(self, elements, H, S, HS_grid, R_cutoff, **kwargs):
        # If a single element was specified, resolve it to a tensor
        if isinstance(elements, int):
            self.elements = t.tensor([elements]*2)
        elif isinstance(elements, list):
            self.elements = t.tensor(elements)
        else:
            self.elements = elements

        self.H = H
        self.S = S
        self.HS_grid = HS_grid

        if self.homo:
            try:  # Assign homo-nuclear related properties
                self.on_site = kwargs['on_site']
                self.U = kwargs['U']
                self.occupations = kwargs['occupations']
                self.mass = kwargs['mass']
            except KeyError as e:  # Catch missing arguments
                raise KeyError(
                    f'HOMO-nuclear case missing "{e.args[0]}" keyword argument.')

        # Repulsion properties
        self.R_cutoff = R_cutoff
        if 'R' in kwargs:  # For spline representation
            try:
                self.R = kwargs['R']
                self.R_short = kwargs['R_short']
                self.R_long = kwargs['R_long']
                self.R_grid = kwargs['R_grid']
            except KeyError as e:  # Catch missing arguments
                raise KeyError(
                    f'Repulsive spline missing "{e.args[0]}" keyword argument.')

        # For polynomial representation
        self.R_poly = kwargs.get('R_poly', None)

        self.__repulsion_by_spline = self.R_poly is None

        # Identify the skf specification version
        if 'version' in kwargs:
            # If this information is present in the kwargs use it.
            self.version = kwargs['version']
        elif self.H.shape[1] == 20:  # Default to 1.0e if there are f orbitals
            self.version = '1.0e'
        else:  # Or 1.0 if there are no f orbitals
            self.version = '1.0'

        # Perform safety checks on the
        self.__check_electron_grid_points()  # Electron_grid_points parameters
        self.__class__.__check_repulsion(**kwargs)  # Repulsive parameters

    @property
    def homo(self):
        """Returns True if this is for the homo-nuclear case"""
        return bool(self.elements[0] == self.elements[1])

    @property
    def repulsion_by_spline(self):
        """True if a repulsive spline present & False if polynomial used."""
        # Getter used to prevent manual alteration
        return self.__repulsion_by_spline

    @classmethod
    def read_normal(cls, path):
        """Read in a skf file and returns an SKF_File instance.

        File names should follow the naming convention X-Y.skf where X and
        Y are the chemical symbol's of the elements involved.

        Args:
            path (str): Path to the target skf file.

        Todo:
        - Batch read a director of skf files. [Priority: Moderate]
        - Rewrite this spaghetti code. [Priority: Moderate]
        """
        # Helper Function Setup
        def get_version_number():
            """Returns skf version number"""
            if file.startswith('@'):  # If 1'st char is @, the version is 1.0e
                v = '1.0e'
            elif len(lines[0].split()) == 2 and lines[0].split()[1].isnumeric():
                v = '0.9'  # If no comment line; this must be version 0.9
            else:  # Otherwise version 1.0
                v = '1.0'
            return v

        # Alias for common code structure; convert str to list of floats
        lmf = lambda x: list(map(float, x.split()))

        # In_grid_pointstial IO & Setup
        if not os.path.exists(path):  # Check that the path specified exists
            raise FileNotFoundError('Target path does not exist')

        file = open(path, 'r').read()  # <- Read file to a string
        lines = file.split('\n')  # <- and as a list of lines

        ver = get_version_number()  # Identify the version number
        if ver in ['1.0', '1.0e']:  # Strip comment line if present
            lines = lines[1:]

        # HOMO & Element Data
        # -------------------
        # Get atomic numbers & identify if this is the homo case
        elements = [atomic_number[s] for s in os.path.splitext(os.path.basename(path))[0].split('-')]
        homo = elements[0] == elements[1]
        if homo:  # Fetch any homo specific data; on-site energies, Hubbard Us,
            # & occupations
            homo_ln = t.tensor(lmf(lines[1]))
            n = int((len(homo_ln) - 1) / 3)  # <- Number of shells specified
            on_site, _, U, occupations = homo_ln.split_with_sizes([n, 1, n, n])

        # H & S Tables
        g_step, n_g_points = lines[0].split()  # Number & spacing of grid points
        g_step, n_g_points = float(g_step), int(n_g_points)  # Convert to float/int

        # Construct distance list. Note; distance start at 1 * g_step not zero
        HS_grid = t.arange(1, n_g_points + 1) * g_step

        # Fetch the H and S sk tables
        H, S = t.tensor(  # Read the table (homo files have extra line)
            [lmf(i) for i in lines[2 + homo: 2 + n_g_points + homo]]
            ).chunk(2, 1)  # Split table in to two parts (H & S)

        # Repulsive Data
        # Read the polynomial repulsive representation line & get the mass
        mass, *R_poly, r_cutoff = t.tensor(lmf(lines[2 + homo]))[:10]

        # Check if there is a spline representation
        has_r_spline = 'Spline' in file

        if has_r_spline:  # If there is
            start = lines.index('Spline') + 1  # Identify spline section start

            # Read number of spline sections & overwrite the r_cutoff previously
            # fetched from the polynomial line.
            n, r_cutoff = lines[start].split()
            n, r_cutoff = int(n), float(r_cutoff)

            r_tab = t.tensor(  # Read the main repulsive spline section
                [lmf(line) for line in lines[start + 2: start + 1 + n]])

            R = r_tab[:, 2:]  # From the table extract: the spline coefficients
            R_grid = t.tensor([*r_tab[:, 0], r_tab[-1, 1]])  # & spline ranges

            # Get the short and long range terms
            R_short = t.tensor(lmf(lines[start + 1]))
            R_long = t.tensor(lmf(lines[start + 1 + n])[3:])

        # Build the parameter lists to pass to the in_grid_pointst method
        # Those that are passed by position:
        pos = (elements, H, S, HS_grid, r_cutoff)

        # Those that are passed by keyword
        kwd = {'version': ver}  # Always passed

        if homo:  # Passed only if homo case
            kwd.update({'mass': mass, 'on_site': on_site, 'U': U,
                        'occupations': occupations})

        if has_r_spline:  # Passed only if there is a repulsive spline
            kwd.update({'R': R, 'R_grid': R_grid, 'R_short': R_short,
                'R_long': R_long})

        else:  # Passed if there is no repulsive spline
            kwd.update({'R_poly': R_poly})

        # Parse data into __in_grid_pointst__ function & return the resulting instance
        return cls(*pos, **kwd)

    @classmethod
    def read(cls, path, specie, orbresolve, skf=None):
        """Read in a skf file and returns an SKF_File instance.

        File names should follow the naming convention X-Y.skf where X and
        Y are the chemical symbol's of the elements involved.

        Args:
            path (str): Path to the target skf file.

        Todo:
        - Batch read a director of skf files. [Priority: Moderate]
        - Rewrite this spaghetti code. [Priority: Moderate]
        """
        # number of specie
        nspecie = len(specie)
        skf = {} if skf is None else skf

        for iat in range(nspecie):
            for jat in range(nspecie):
                # atom name
                nameij = specie[iat] + specie[jat]

                # name of skf file
                skname = specie[iat] + '-' + specie[jat] + '.skf'
                fp = open(os.path.join(path, skname), "r")

                # get the first line information
                words = fp.readline().split()

                # distance of grid points and number of grid points
                skf['grid_dist' + nameij] = float(words[0])
                skf['ngridpoint' + nameij] = int(words[1])

                # total integral number
                nitem = int(words[1]) * 20

                # if the atom specie is the same
                if specie[iat] == specie[jat]:

                    # read the second line: onsite, U...
                    fp_line = [float(ii) for ii in fp.readline().split()]
                    fp_line_ = t.from_numpy(np.asarray(fp_line))
                    skf['onsite' + nameij] = fp_line_[0:3]
                    skf['spe' + nameij] = fp_line_[3]
                    skf['uhubb' + nameij] = fp_line_[4:7]
                    skf['occ_skf' + nameij] = fp_line_[7:10]

                    # if orbital resolved
                    if not orbresolve:
                        skf['uhubb' + nameij][:] = fp_line_[6]

                    # read third line: mass...
                    data = np.fromfile(fp, count=20, sep=' ')
                    skf['mass_cd' + nameij] = t.from_numpy(data)

                    # read all the integral and reshape
                    hs_all = np.fromfile(fp, count=nitem, sep=' ')
                    hs_all.shape = (int(words[1]), 20)
                    skf['hs_all' + nameij] = hs_all

                # atom specie is different
                else:

                    # read the second line: mass...
                    data = np.fromfile(fp, count=20, sep=' ')
                    skf['mass_cd' + nameij] = t.from_numpy(data)

                    # read all the integral and reshape
                    hs_all = np.fromfile(fp, count=nitem, sep=' ')
                    hs_all.shape = (int(words[1]), 20)
                    skf['hs_all' + nameij] = hs_all

                # read spline part
                spline = fp.readline().split()
                if 'Spline' in spline:

                    # read first line of spline
                    nint_cutoff = fp.readline().split()
                    nint_ = int(nint_cutoff[0])
                    skf['nint_rep' + nameij] = nint_
                    skf['cutoff_rep' + nameij] = float(nint_cutoff[1])

                    # read second line of spline
                    a123 = fp.readline().split()
                    skf['a1_rep' + nameij] = float(a123[0])
                    skf['a2_rep' + nameij] = float(a123[1])
                    skf['a3_rep' + nameij] = float(a123[2])

                    # read the rest of spline but not the last
                    datarep = np.fromfile(fp, dtype=float,
                                          count=(nint_ - 1) * 6, sep=' ')
                    datarep.shape = (nint_ - 1, 6)
                    skf['rep' + nameij] = t.from_numpy(datarep)

                    # raed the end line: start end c0 c1 c2 c3 c4 c5
                    datarepend = np.fromfile(fp, dtype=float,
                                             count=8, sep=' ')
                    skf['repend' + nameij] = t.from_numpy(datarepend)
        return skf

    @classmethod
    def read_compression_radii(cls, path):
        """Read in a skf file and returns an SKF_File instance.

        File names should follow the naming convention X-Y.skf where X and
        Y are the chemical symbol's of the elements involved.

        Args:
            path (str): Path to the target skf file.

        Todo:
        - Batch read a director of skf files. [Priority: Moderate]
        - Rewrite this spaghetti code. [Priority: Moderate]
        """
        # Helper Function Setup

    def write(self, path):
        """Writes a ``SKF_File`` instance to the specified path."""
        raise NotImplementedError()

    @classmethod
    def from_splines(cls):
        """Instantiates an ``SKF_File`` entity from a set of splines."""
        raise NotImplementedError()

    # Helper Functions
    # ----------------
    @staticmethod
    def __check_repulsion(**kwargs):
        """Checks that all the parameters needed to construct the
        repulsive component have been specified.

        Todo
        ----
        - This is a stale function and so should be removed and its
          contents moved to the in_grid_pointst. [Priority: Low]
        """
        # If a repulsive spline is used make sure that there are n+1 values in
        # R_grid; where n is the number of entries in R. While requiring the
        # user to provide an additional point is somewhat unorthodox, hence the
        # explicit check, the only other alternatives would be have a second
        # distance argument, which would not be great given the already vast
        # number or arguments required; or to require the distance be placed
        # in the R_long array, which would be unexpected.
        if 'R' in kwargs:
            if kwargs['R'].shape[0] + 1 != kwargs['R_grid'].shape[0]:
                raise Exception(
                    'R_grid represents the ranges over which elements in R act\n'
                    'Thus R_grid must have one element more than R_grid')

    def __check_electron_grid_points(self):
        """Makes sure electron_grid_points parameter specifications are correct
         and consistent.
        """
        # Check size of H and S are the same
        if self.H.shape != self.S.shape:
            raise Exception('H/S matrices must have matching shapes')

        # Check number grid points is correct
        if self.HS_grid.shape[0] != self.H.shape[0]:
            raise Exception(
                'Number of rows in H/S matrices must match the number '
                'of entries in HS_grid')

        # Ensure the number of on_site terms have been give for the
        # requested version type. This is only valid for homo files
        if self.homo:
            if self.H.shape[1] == 10 and len(self.on_site) != 3:
                raise Exception(
                    'Non-extended file versions expect 3 on-site terms.')
            elif self.H.shape[1] == 20 and len(self.on_site) != 4:
                raise Exception(
                    'Extended file versions expect 4 on-site terms.')


class GetSK_:
    """Get integral from interpolation."""

    def __init__(self, para, dataset, skf, ml=None):
        """Initialize parameters."""
        self.para = para
        self.dataset = dataset
        self.skf = skf
        self.ml = ml
        self.math = DFTBmath(self.para, self.skf)

    def integral_spline_parameter(self):
        """Get integral from hdf binary according to atom species."""
        time0 = time.time()

        # ML variables
        ml_variable = []

        # get the skf with hdf type
        hdfsk = self.para['datasetSK']

        # check if skf dataset exists
        if not os.path.isfile(hdfsk):
            raise FileNotFoundError('%s not found' % hdfsk)

        self.skf['hs_compr_all'] = []
        with h5py.File(hdfsk, 'r') as f:
            for ispecie in self.dataset['specieGlobal']:
                for jspecie in self.dataset['specieGlobal']:
                    nameij = ispecie + jspecie
                    grid_distance = f[nameij + '/grid_dist'][()]
                    ngrid = f[nameij + '/ngridpoint'][()]
                    yy = t.from_numpy(f[nameij + '/hs_all'][()])
                    xx = t.arange(0., ngrid * grid_distance, grid_distance, dtype=yy.dtype)
                    self.skf['polySplinex' + nameij] = xx
                    self.skf['polySplinea' + nameij], \
                    self.skf['polySplineb' + nameij], \
                    self.skf['polySplinec' + nameij], \
                    self.skf['polySplined' + nameij] = \
                        PolySpline(xx, yy).get_abcd()[:]
                    ml_variable.append(self.skf['polySplinea' + nameij].requires_grad_(True))
                    ml_variable.append(self.skf['polySplineb' + nameij].requires_grad_(True))
                    ml_variable.append(self.skf['polySplinec' + nameij].requires_grad_(True))
                    ml_variable.append(self.skf['polySplined' + nameij].requires_grad_(True))

        timeend = time.time()
        print('time of get spline parameter: ', timeend - time0)
        return ml_variable

    def genskf_interp_dist_hdf(self, ibatch, natom):
        """Generate integral along distance dimension."""
        time0 = time.time()
        ninterp = self.skf['sizeInterpolationPoints']
        self.skf['hs_compr_all'] = []
        atomnumber = self.dataset['numbers'][ibatch]
        distance = self.dataset['distances'][ibatch]

        # index of row, column of distance matrix, no digonal
        # ind = t.triu_indices(distance.shape[0], distance.shape[0], 1)
        # dist_1d = distance[ind[0], ind[1]]
        # get the skf with hdf type
        hdfsk = self.para['datasetSK']
        if not os.path.isfile(hdfsk):
            raise FileExistsError('dataset %s do not exist' % hdfsk)

        # read all skf according to atom number (species) and indices and add
        # these skf to a list, attention: hdf only store numpy type data
        with h5py.File(hdfsk, 'r') as f:
            # get the grid sidtance, which should be the same
            grid_dist = f['globalgroup'].attrs['grid_dist']

        # get the distance according to indices (upper triangle elements
        ind_ = (distance / grid_dist).int()
        indd = (ind_ + ninterp / 2 + 1).int()

        # get integrals with ninterp (normally 8) line for interpolation
        with h5py.File(hdfsk, 'r') as f:
            yy = [[f[ATOMNAME[int(atomnumber[i])] + ATOMNAME[int(atomnumber[j])] +
                     '/hs_all_rall'][:, :, indd[i, j] - ninterp - 1: indd[i, j] - 1, :]
                   for j in range(natom)] for i in range(natom)]

        # get the distances corresponding to the integrals
        xx = [[(t.arange(ninterp) + indd[i, j] - ninterp) * grid_dist
               for j in range(len(distance))] for i in range(len(distance))]

        self.skf['hs_compr_all'] = t.stack([t.stack([self.math.poly_check(
            xx[i][j], t.from_numpy(yy[i][j]).type(self.para['precision']), distance[i, j], i==j)
            for j in range(natom)]) for i in range(natom)])

        print('distance interpolation of skf time:', time.time() - time0)

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
        atomname = self.dataset['atomNameAll']

        # number of atom
        natom = self.dataset['natomall']

        # atom specie
        atomspecie = self.dataset['atomspecie']

        # number of compression radius grid points
        ncompr = self.para['ncompr']

        # build integral with various compression radius
        self.para['hs_compr_all'] = t.zeros(natom, natom, ncompr, ncompr, 20)

        # get i and j atom with various compression radius at certain dist
        print('build matrix: [N, N, N_R, N_R, 20]')
        print('N is number of atom in molecule, N_R is number of compression')

        for iatom in range(natom):
            for jatom in range(natom):
                dij = self.dataset['distance'][iatom, jatom]
                namei, namej = atomname[iatom], atomname[jatom]
                nameij = namei + namej
                compr_grid = self.para[namei + '_compr_grid']
                self.skf['hs_ij'] = t.zeros(ncompr, ncompr, 20)

                if dij > 1e-2:
                    self.genskf_interp_ijd_4d(dij, nameij, compr_grid)
                self.skf['hs_compr_all'][iatom, jatom, :, :, :] = \
                    self.skf['hs_ij']

        # get the time after interpolation
        time2 = time.time()
        for iat in atomspecie:

            # onsite is the same, therefore read [0, 0] instead
            onsite = t.zeros(3)
            uhubb = t.zeros(3)
            onsite[:] = self.para['onsite' + iat + iat]
            uhubb[:] = self.para['uhubb' + iat + iat]
            self.skf['onsite' + iat + iat] = onsite
            self.skf['uhubb' + iat + iat] = uhubb
        timeend = time.time()
        print('time of distance interpolation: ', time2 - time0)
        print('total time of distance interpolation in skf: ', timeend - time0)

    def genskf_interp_ijd_(self, dij, nameij, rgrid):
        """Interpolate skf of i and j atom with various compression radius."""
        # cutoff = self.para['interpcutoff']
        assert self.skf['grid_dist_rall' + nameij][0, 0] == \
            self.skf['grid_dist_rall' + nameij][-1, -1]
        self.skf['grid_dist' + nameij] = \
            self.skf['grid_dist_rall' + nameij][0, 0]
        self.skf['ngridpoint' + nameij] = \
            self.skf['ngridpoint_rall' + nameij].min()
        ncompr = int(np.sqrt(self.para['nfile_rall' + nameij]))
        for icompr in range(0, ncompr):
            for jcompr in range(0, ncompr):
                self.skf['hs_all' + nameij] = \
                    self.skf['hs_all_rall' + nameij][icompr, jcompr, :, :]
                # col = skfijd.shape[1]
                self.skf['hs_ij'][icompr, jcompr, :] = \
                    self.math.sk_interp(dij, nameij)

    def genskf_interp_ijd_4d(self, dij, nameij, rgrid):
        """Interpolate skf of i and j atom with various compression radius."""
        # cutoff = self.para['interpcutoff']
        assert self.skf['grid_dist_rall' + nameij][0, 0] == \
            self.skf['grid_dist_rall' + nameij][-1, -1]
        self.skf['grid_dist' + nameij] = \
            self.skf['grid_dist_rall' + nameij][0, 0]
        self.skf['ngridpoint' + nameij] = \
            self.skf['ngridpoint_rall' + nameij].min()
        ncompr = int(np.sqrt(self.skf['nfile_rall' + nameij]))
        self.skf['hs_all' + nameij] = \
            self.skf['hs_all_rall' + nameij][:, :, :, :]
        self.skf['hs_ij'][:, :, :] = \
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
        atomname = para['atomNameAll']
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
                zmeshall = self.skf['hs_compr_all'][icount]
                for icol in range(0, 20):
                    hs_ij[iatom, jatom, icol] = \
                        bicubic.bicubic_2d(xmesh, ymesh, zmeshall[:, :, icol],
                                           icompr, jcompr)
                icount += 1

            onsite = t.zeros(3)
            uhubb = t.zeros(3)
            for icol in range(0, 3):
                zmesh_onsite = self.skf['onsite_rall' + iname + iname]
                zmesh_uhubb = self.skf['uhubb_rall' + iname + iname]
                onsite[icol] = \
                    bicubic.bicubic_2d(xmesh, ymesh, zmesh_onsite[:, :, icol],
                                       icompr, jcompr)
                uhubb[icol] = \
                    bicubic.bicubic_2d(xmesh, ymesh, zmesh_uhubb[:, :, icol],
                                       icompr, jcompr)
                self.skf['onsite' + iname + iname] = onsite
                self.skf['uhubb' + iname + iname] = uhubb
        self.skf['hs_all'] = hs_ij

    def genskf_interp_compr(self, ibatch):
        """Generate interpolation of SKF with given compression radius.

        Args:
            compression R
            H and S between all atoms ([ncompr, ncompr, 20] * natom * natom)
        Return:
            H and S matrice ([natom, natom, 20])

        """
        natom = self.dataset['natomAll'][ibatch]
        atomname = self.dataset['symbols'][ibatch]
        time0 = time.time()
        print('Get HS table according to compression R: [N_atom1, N_atom2, 20]')

        if self.ml['interpolationType'] == 'BiCubVec':
            bicubic = BicubInterpVec(self.para, self.ml)
            zmesh = self.skf['hs_compr_all']
            if self.para['compr_ml'].dim() == 2:
                compr = self.para['compr_ml'][ibatch][:natom]
            else:
                compr = self.para['compr_ml']
            mesh = t.stack([self.ml[iname + '_compr_grid'] for iname in atomname])
            hs_ij = bicubic.bicubic_2d(mesh, zmesh, compr, compr)

        # elif self.ml['interp_compr_type'] == 'BiCub':
        #     icount = 0
        #     bicubic = BicubInterp()
        #     hs_ij = t.zeros(natom, natom, 20)
        #     for iatom in range(natom):
        #         iname = atomname[iatom]
        #         icompr = self.para['compr_ml'][ibatch][iatom]
        #         xmesh = self.ml[iname + '_compr_grid']
        #         for jatom in range(natom):
        #             jname = atomname[jatom]
        #             ymesh = self.ml[jname + '_compr_grid']
        #             jcompr = self.para['compr_ml'][ibatch][jatom]
        #             zmeshall = self.skf['hs_compr_all'][iatom, jatom]
        #             if iatom != jatom:
        #                 for icol in range(0, 20):
        #                     hs_ij[iatom, jatom, icol] = \
        #                         bicubic.bicubic_2d(
        #                                 xmesh, ymesh, zmeshall[:, :, icol],
        #                                 icompr, jcompr)
        #             icount += 1

        self.skf['hs_all'] = hs_ij
        timeend = time.time()
        print('total time genskf_interp_compr:', timeend - time0)


class SKTran:
    """Slater-Koster Transformations."""

    def __init__(self, para, dataset, skf, ml, ibatch):
        """Initialize parameters.

        Args:
            integral
            geometry (distance)
        Returns:
            [natom, natom, 20] matrix for each calculation
        """
        self.para = para
        self.skf = skf
        self.dataset = dataset
        self.ml = ml
        self.math = DFTBmath(self.para, self.skf)
        self.ibatch = ibatch

        # if machine learning or not
        if not self.para['Lml']:

            # read integrals from .skf with various compression radius
            if not self.dataset['LSKFinterpolation']:
                self.get_sk_all(self.ibatch)

            # build H0 and S with full, symmetric matrices
            if self.para['HSSymmetry'] == 'all':
                self.sk_tran_symall(self.ibatch)

            # build H0 and S with only half matrices
            elif self.para['HSSymmetry'] == 'half':
                self.sk_tran_half(self.ibatch)

        # machine learning is True, some method only apply in this case
        if self.para['Lml']:

            # use ACSF to generate compression radius, then SK transformation
            if self.para['task'] in ('mlCompressionR', 'testCompressionR'):

                # build H0 and S with full, symmetric matrices
                if self.para['HSSymmetry'] == 'all':
                    self.sk_tran_symall(self.ibatch)

                # build H0, S with half matrices
                elif self.para['HSSymmetry'] == 'half':
                    self.sk_tran_half(self.ibatch)

            # directly get integrals with spline, or some other method
            elif self.para['task'] in ('mlIntegral', 'testIntegral'):
                self.get_hs_spline(self.ibatch)
                self.sk_tran_symall(self.ibatch)

    def get_hs_spline(self, ibatch):
        """Get integrals from .skf data with given distance."""
        # number of atom in each calculation
        natom = self.dataset['natomAll'][self.ibatch]

        # build H0 or S
        self.skf['hs_all'] = t.zeros(natom, natom, 20)

        for iat in range(natom):
            for jat in range(natom):
                if iat != jat:

                    # get the name of i, j atom pair
                    namei = self.dataset['symbols'][ibatch][iat]
                    namej = self.dataset['symbols'][ibatch][jat]
                    nameij = namei + namej

                    # get spline parameters
                    xx = self.skf['polySplinex' + nameij]
                    abcd = [self.skf['polySplinea' + nameij],
                            self.skf['polySplineb' + nameij],
                            self.skf['polySplinec' + nameij],
                            self.skf['polySplined' + nameij]]

                    if abcd[0].device.type == 'cuda':
                        Save1D(abcd[0].detach().cpu().numpy(),
                               name=nameij+'spl_a.dat', dire='.', ty='a')
                        Save1D(abcd[1].detach().cpu().numpy(),
                               name=nameij+'spl_b.dat', dire='.', ty='a')
                        Save1D(abcd[2].detach().cpu().numpy(),
                               name=nameij+'spl_c.dat', dire='.', ty='a')
                        Save1D(abcd[3].detach().cpu().numpy(),
                               name=nameij+'spl_d.dat', dire='.', ty='a')
                    elif abcd[0].device.type == 'cpu':
                        Save1D(abcd[0].detach().numpy(),
                               name=nameij+'spl_a.dat', dire='.', ty='a')
                        Save1D(abcd[1].detach().numpy(),
                               name=nameij+'spl_b.dat', dire='.', ty='a')
                        Save1D(abcd[2].detach().numpy(),
                               name=nameij+'spl_c.dat', dire='.', ty='a')
                        Save1D(abcd[3].detach().numpy(),
                               name=nameij+'spl_d.dat', dire='.', ty='a')
                    # the distance is from cal_coor
                    dd = self.dataset['distances'][ibatch][iat, jat]
                    poly = PolySpline(x=xx, abcd=abcd)
                    self.skf['hs_all'][iat, jat] = poly(dd)

    def get_sk_all(self, ibatch):
        """Get integrals from .skf data with given distance."""
        # number of atom in each calculation
        natom = self.dataset['natomAll'][self.ibatch]

        # build H0 or S
        self.skf['hs_all'] = t.zeros(natom, natom, 20)

        for iat in range(natom):
            for jat in range(natom):
                # get the name of i, j atom pair
                namei = self.dataset['symbols'][ibatch][iat]
                namej = self.dataset['symbols'][ibatch][jat]
                nameij = namei + namej

                # the distance is from cal_coor
                dd = self.dataset['distances'][ibatch][iat, jat]

                # two atom are too close, exit
                if dd < 1E-1 and iat != jat:
                    sys.exit()
                elif iat != jat:
                    # get the integral by interpolation from integral table
                    self.skf['hs_all'][iat, jat, :] = self.math.sk_interp(dd, nameij)

    def sk_tran_half(self):
        """Transfer H and S according to slater-koster rules."""
        # index of the orbitals
        atomind = self.para['atomind']

        # number of atom
        natom = self.para['natom']

        # name of all atom in each calculation
        atomname = self.para['atomNameAll']

        # vectors between different atoms (Bohr)
        dvec = self.para['dvec']

        # the sum of orbital index, equal to dimension of H0 and S
        atomind2 = self.para['atomind2']

        # build 1D, half H0, S matrices
        self.skf['hammat'] = t.zeros(atomind2)
        self.skf['overmat'] = t.zeros(atomind2)

        # temporary distance matrix
        rr = t.zeros(3)

        for iat in range(natom):

            # l of i atom
            lmaxi = self.para['lmaxall'][iat]
            for jat in range(iat):

                # l of j atom
                lmaxj = self.para['lmaxall'][jat]
                lmax = max(lmaxi, lmaxj)

                # temporary H, S with dimension 9 (s, p, d orbitals)
                self.para['hams'] = t.zeros(9, 9)
                self.para['ovrs'] = t.zeros(9, 9)

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
                            self.skf['hammat'][idx] = self.skf['hams'][m, n]
                            self.skf['overmat'][idx] = self.skf['ovrs'][m, n]

            # build temporary on-site
            self.para['h_o'] = t.zeros(9)
            self.para['s_o'] = t.zeros(9)

            # get the name between atoms
            self.para['nameij'] = atomname[iat] + atomname[iat]

            # get on-site between i and j atom
            self.slkode_onsite(rr, iat, lmaxi)

            # write on-site between i and j to final on-site matrix
            for m in range(atomind[iat + 1] - atomind[iat]):
                mm = atomind[iat] + m
                idx = int(mm * (mm + 1) / 2 + mm)
                self.skf['hammat'][idx] = self.skf['h_o'][m]
                self.skf['overmat'][idx] = self.skf['s_o'][m]

    def sk_tran_symall(self, ibatch):
        """Transfer H0, S according to Slater-Koster rules.

        writing the symmetric, full 2D H0, S.

        """
        # index of atom orbital
        atomind = self.dataset['atomind'][ibatch]

        # number of atom
        natom = self.dataset['natomAll'][ibatch]

        # total orbitals, equal to dimension of H0, S
        norb = sum(atomind[:natom])

        # atom name
        atomname = self.dataset['symbols'][ibatch]

        # atom coordinate vector (Bohr)
        # dvec = self.dataset['dvec']

        # build H0, S
        self.skf['hammat'] = t.zeros(norb, norb)
        self.skf['overmat'] = t.zeros(norb, norb)

        for iat in range(natom):

            # l of i atom
            lmaxi = self.dataset['lmaxall'][ibatch][iat]

            for jat in range(natom):

                # l of j atom
                lmaxj = self.dataset['lmaxall'][ibatch][jat]

                # temporary H, S between i and j atom
                self.skf['hams'] = t.zeros(9, 9)
                self.skf['ovrs'] = t.zeros(9, 9)

                # temporary on-site
                self.skf['h_o'] = t.zeros(9)
                self.skf['s_o'] = t.zeros(9)

                # name of i and j atom pair
                self.para['nameij'] = atomname[iat] + atomname[jat]

                # distance vector between i and j atom
                rr = self.dataset['positions_vec'][ibatch][iat, jat]  # dvec[ibatch][iat, jat, :]

                # for the same atom, where on-site should be construct
                if iat == jat:

                    # get on-site between i and j atom
                    self.slkode_onsite(rr, iat, lmaxi)

                    # write on-site between i and j to final on-site matrix
                    for m in range(sum(atomind[:iat + 1]) - sum(atomind[:iat])):
                        mm = sum(atomind[:iat]) + m
                        self.skf['hammat'][mm, mm] = self.skf['h_o'][m]
                        self.skf['overmat'][mm, mm] = self.skf['s_o'][m]

                # build H0, S with integrals for i, j atom pair
                else:

                    # get H, S with distance, initial integrals
                    if self.skf['transformationSK'] == 'new':
                        self.slkode_vec(rr, iat, jat, lmaxi, lmaxj)
                    elif self.skf['transformationSK'] == 'old':
                        self.slkode_ij(rr, iat, jat, lmaxi, lmaxj)

                    # write H0, S of i, j to final H0, S
                    for n in range(sum(atomind[:jat + 1]) - sum(atomind[:jat])):
                        nn = sum(atomind[:jat]) + n
                        for m in range(sum(atomind[:iat + 1]) - sum(atomind[:iat])):

                            # calculate the off-diagonal orbital index
                            mm = sum(atomind[:iat]) + m
                            self.skf['hammat'][mm, nn] = self.skf['hams'][m, n]
                            self.skf['overmat'][mm, nn] = self.skf['ovrs'][m, n]

    def slkode_onsite(self, rr, iat, lmax):
        """Transfer i from ith atom to ith spiece."""
        # name of i and i atom
        nameij = self.para['nameij']

        # s, p, d orbitals onsite
        do, po, so = self.skf['onsite' + nameij][:]

        # max(l) is 0, only s orbitals is included in system
        if lmax == 0:
            self.skf['h_o'][0] = so
            self.skf['s_o'][0] = 1.0

        # max(l) is 1, including p orbitals
        elif lmax == 1:
            self.skf['h_o'][0] = so
            self.skf['h_o'][1: 4] = po
            self.skf['s_o'][: 4] = 1.0

        # max(l) is 2, including d orbital
        elif lmax == 2:
            self.skf['h_o'][0] = so
            self.skf['h_o'][1: 4] = po
            self.skf['h_o'][4: 9] = do
            self.skf['s_o'][:] = 1.0

    def slkode_ij(self, rr, iat, jat, li, lj):
        """Transfer integrals according to SK rules."""
        # name of i, j atom
        nameij = self.para['nameij']

        # distance between atom i, j
        dd = t.sqrt(t.sum(rr[:] ** 2))

        if dd < 1E-1:
            print("ERROR, distance between", iat, "and", jat, 'is too close')
            sys.exit()
        else:
            self.sk_(rr, iat, jat, dd, li, lj)

    def slkode_vec(self, rr, iat, jat, li, lj):
        """Generate H0, S by vectorized method."""
        lmax, lmin = max(li, lj), min(li, lj)
        xx, yy, zz = rr[:] / t.sqrt(t.sum(rr[:] ** 2))
        hsall = self.skf['hs_all']

        if lmax == 0:
            self.skf['hams'][0, 0], self.skf['ovrs'][0, 0] = \
                self.skss_vec(hsall, xx, yy, zz, iat, jat)
        if lmin == 0 and lmax == 1:
            self.skf['hams'][:4, :4], self.skf['ovrs'][:4, :4] = \
                self.sksp_vec(hsall, xx, yy, zz, iat, jat, li, lj)
        if lmin == 1 and lmax == 1:
            self.skf['hams'][:4, :4], self.skf['ovrs'][:4, :4] = \
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
            H = t.stack([t.stack([
                # SS, SP_y, SP_z, SP_x
                hs[i, j, 9],
                y * hs[i, j, 8],
                z * hs[i, j, 8],
                x * hs[i, j, 8]]),
                # P_yS
                t.cat((y * hs[i, j, 8].unsqueeze(0), t.zeros(3))),
                # P_zS
                t.cat((z * hs[i, j, 8].unsqueeze(0), t.zeros(3))),
                # P_xS
                t.cat((x * hs[i, j, 8].unsqueeze(0), t.zeros(3)))])
            S = t.stack([t.stack([
                hs[i, j, 19],
                y * hs[i, j, 18],
                z * hs[i, j, 18],
                x * hs[i, j, 18]]),
                t.cat((y * hs[i, j, 18].unsqueeze(0), t.zeros(3))),
                t.cat((z * hs[i, j, 18].unsqueeze(0), t.zeros(3))),
                t.cat((x * hs[i, j, 18].unsqueeze(0), t.zeros(3)))])

        # read sp0 from <namej-namei>.skf
        if li > lj:
            H = t.stack([t.stack([
                hs[j, i, 9],
                -y * hs[j, i, 8],
                -z * hs[j, i, 8],
                -x * hs[j, i, 8]]),
                t.cat((-y * hs[j, i, 8].unsqueeze(0), t.zeros(3))),
                t.cat((-z * hs[j, i, 8].unsqueeze(0), t.zeros(3))),
                t.cat((-x * hs[j, i, 8].unsqueeze(0), t.zeros(3)))])
            S = t.stack([t.stack([
                hs[j, i, 19],
                -y * hs[j, i, 18],
                -z * hs[j, i, 18],
                -x * hs[j, i, 18]]),
                t.cat((-y * hs[j, i, 18].unsqueeze(0), t.zeros(3))),
                t.cat((-z * hs[j, i, 18].unsqueeze(0), t.zeros(3))),
                t.cat((-x * hs[j, i, 18].unsqueeze(0), t.zeros(3)))])
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
        skselfnew = t.zeros(3)

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
            if lmax == 1:
                skss(self.para, xx, yy, zz, iat, jat,
                     self.para['hams'], self.para['ovrs'], li, lj)
            elif lmin == 1 and lmax == 2:
                sksp(self.para, xx, yy, zz, iat, jat,
                     self.para['hams'], self.para['ovrs'], li, lj)
            elif lmin == 2 and lmax == 2:
                skpp(self.para, xx, yy, zz, iat, jat,
                     self.para['hams'], self.para['ovrs'], li, lj)


class SKIntegralGenerator:
    """Example Slater-Koster integral generator.

    This class is an example showing the type of object that the sfk transformer
    is designed to interact with. During the SK transformation, the function
    `slaterkoster.skt` will ask this class to evaluate SK integrals at specific
    distances.

    It should be noted that this is only an example & therefore uses scipy/numpy
    to do the actual interpolation.


    Parameters
    ----------
    spline_dict : dict
        This should be a dictionary of scipy splines keyed by a tuple of strings
        of the form: (z_1, z_2, ℓ_1, ℓ_2, b, O), where the ℓ's are the azimuthal
        quantum numbers, the z's are the atomic numbers, b is the bond type, &
        O is the operator which the spline represents i.e. S or H.

    on_site_dict : dict
        A dictionary of arrays keyed by a tuple of strings. Note that this is
        currently not implemented.
    """

    def __init__(self, spline_dict, on_site_dict=None):
        self.spline_dict = spline_dict
        self.on_site_dict = on_site_dict

    @classmethod
    def from_dir(cls, directory):
        """Read all skf files in a directory & return an SKIG instance.

        Parameters
        ----------
        directory : str
            path to the directory in which skf files can be found
        """
        import glob
        from dftbmalt.io.skf import SKF_File
        from scipy.interpolate import CubicSpline

        # The interactions: ddσ, ddπ, ddδ, ...
        interactions = [(2, 2, 0), (2, 2, 1), (2, 2, 2), (1, 2, 0), (1, 2, 1),
                        (1, 1, 0), (1, 1, 1), (0, 2, 0), (0, 1, 0), (0, 0, 0)]

        # Find all the skf files
        skf_files = glob.glob(os.path.join(directory, '*.skf'))

        # Create a blank spline_dict
        spline_dict = {}

        # Loop over the files
        for skf_file in skf_files:
            # Read the file
            skf = SKF_File.read(skf_file)

            # Get the distance values
            distances = skf.HS_grid

            # Pull out the hamiltonian interactions and form a set of splines
            for i, name in zip(skf.H.T, interactions):
                spline = CubicSpline(distances, i)
                # Add to the spline dict
                spline_dict[(*skf.elements.tolist(), *name, 'H')] = spline

            # Repeat for the Overlap
            for i, name in zip(skf.S.T, interactions):
                spline = CubicSpline(distances, i)
                # Add to the spline dict
                spline_dict[(*skf.elements.tolist(), *name, 'S')] = spline

        return cls(spline_dict)


    def __call__(self, distances, atom_pair, l_pair, mat_type='H', **kwargs):

        # Convert distances from a torch tensor to a numpy array
        if type(distances) == t.Tensor:
            distances = distances.numpy()

        # Retrieve the appropriate splines
        splines = [self.spline_dict[(*atom_pair.tolist(), *l_pair.tolist(), b, mat_type)]
                   for b in range(min(l_pair)+1)]

        # Evaluate the splines at the requested distances, convert the result
        # into a torch tensor and return it.
        # print('ssssss \n', distances, atom_pair, l_pair, splines)
        return t.tensor([spline(distances) for spline in splines]).T


def skt(systems, orbital_id=None, integral_feed=None, **kwargs):
    """Construct a Hamiltonian or overlap matrix from integral values
    retrieved from ``integral_feed`` using batch-wise operable Slater-
    Koster transformations.

    SEE THE WARNING SECTION NEAR THE END OF THE DOCSTRING!


    Parameters
    ----------
    systems : Any
        Any object that has a ".positions" attribute that will return the
        positions of atoms in the associated molecule. This is more of a
        placeholder than anything else.
    orbital_id : `dftbmalt.structures.Basis`, `dftbmalt.structures.Bases`
        The molecule's Basis instance. This stores the data needed to
        perform the necessary masking operations.
    integral_feed : Any
        An object whose __call__ function takes as its arguments 1) a
        distances list, 2) an atomic number pair, 3) an azimuthal number
        pair. Note that this function's kwargs will be passed on to this
        call function as well.

    Returns
    -------
    HS : `torch.tensor` [`float`]
        A tensor holding Hamiltonian or Overlap matrices associated.

    Notes
    -----
    To control whether a Hamiltonian or overlap matrix is constructed
    one can either 1) pass in a specific ``integral_feed`` which does
    only one or the other, or 2) have an ``integral_feed`` which does
    both but takes a keyword argument telling it which it should do. The
    keyword argument can then be passed through via **kwargs.


    Warnings
    --------
    This function is subject to change as the ``systems`` object is
    somewhat overkill here as only the positions are ever extracted
    from it, thus it will be removed. Furthermore, the ``orbital_id``
    entity is not quite set up for multi system mode so it will likely
    be replaced with a collection object. In addition to this, the
    structures.Basis object that this function requires to operate
    will need to be rewritten. Finally, this function does not currently
    assign the diagonals.

    Todo
    ----
    - Add in multi-system support mode. [Priority: High]
    - Create multiple unit tests. [Priority: High]
    """
    # Hamiltonian/overlap matrix data will be placed into HS. For single-system
    # mode this is a NxN tensor where N is number of orbitals. For multi-system
    # mode the tensor will be BxMxM where B = batch size & M the largest number
    # or orbitals found on any system in the batch.
    HS = t.zeros(systems.hs_shape)

    # Detect if this is a batch mode operation
    batch = HS.dim() == 3

    # The "*_mat_*" variables hold data that is; used to build masks, gathered
    # by other masks, or both. The _f, _b & _a suffixes indicate whether the
    # tensor is full, block-wise or atom-wise resolved. See the System.systems
    # class for more information.
    # Matrices used in building getter & setter masks are initialised here.
    # These are the full (f) & block-wise (b) azimuthal identity matrices.
    l_mat_f = orbital_id.azimuthal_matrix(mask=True, mask_lower=False)
    l_mat_b = orbital_id.azimuthal_matrix(block=True, mask=True, mask_lower=False)

    # The masks will then gather data from the matrices initialised below to
    # 1) pass on to various functions & 2) create new "secondary" masks. These
    # matrices are similar in structure to the azimuthal identity matrices.
    i_mat_b = orbital_id.index_matrix(block=True)  # <- atom indices
    an_mat_a = orbital_id.atomic_number_matrix(atomic=True)  # <- atomic numbers
    # print("i_mat_b, an_mat_a", i_mat_b, an_mat_a)
    # Atomic distances:
    dist_mat_a = t.cdist(systems.positions, systems.positions, p=2)
    # directional cosigns:
    vec_mat_a = systems.positions.unsqueeze(-3) - systems.positions.unsqueeze(-2)
    vec_mat_a = t.nn.functional.normalize(vec_mat_a, p=2, dim=-1)

    # Loop over each type of azimuthal-pair interaction & its associated
    # SK-function
    for l_pair, f in zip(_SK_interactions, _SK_functions):
        # Build a bool mask that is True wherever l_mat_b = l_pair & convert
        # it to a set of index positions (aka an index mask).
        index_mask_b = t.where((l_mat_b == l_pair).all(dim=-1))

        if len(index_mask_b[0]) == 0: # Skip if no blocks of this type are found
            continue

        # Use the mask to gather the equivalent elements in the i_mat_b tensor.
        # As i_mat_b is a list of atom indices, the result (when transposed)
        # is the atomic index mask.
        index_mask_a = tuple(i_mat_b[index_mask_b].T)

        # Add system indices back in to inded_mask_a when in multi-system mode
        if batch:
            index_mask_a = (index_mask_b[0],) + index_mask_a

        # Use the atomic index mask to gather the corresponding atomic numbers,
        # distances, and directional cosigns.
        # print("dist_mat_a", an_mat_a, "\n index_mask_a", index_mask_a)
        gathered_an = an_mat_a[index_mask_a]
        gathered_dists = dist_mat_a[index_mask_a]
        gathered_vecs = vec_mat_a[index_mask_a]

        # Request integrals from the integral_feed & pass on any additional
        # keyword arguments; this may be data for adaptive models or a kwarg
        # indicating if it is an H or S matrix element that is to be evaluated.
        gathered_integrals = _batch_integral_retrieve(
            gathered_dists, gathered_an, integral_feed, l_pair, **kwargs)

        # Make a call to the relevant Slater-Koster function to get the sk-block
        sk_data = f(gathered_vecs, gathered_integrals)

        # Multidimensional assigment operations assign their data row-by-row.
        # While this does not pose a problem when dealing with SK data that
        # spans only a single row (e.g ss, sp, sd) it causes multi-row SK data
        # (e.g. ps, sd, pp) to be incorrectly signed, e.g, when attempting to
        # assign two 3x3 blocks [a-i & j-r] to a tensor the desired outcome
        # may be:
        #       ┌                           ┐
        #       │ .  .  .  .  .  .  .  .  . │
        #       │ a  b  c  .  .  .  j  k  l │
        #       │ d  e  f  .  .  .  m  n  o │
        #       │ g  h  i  .  .  .  p  q  r │
        #       │ .  .  .  .  .  .  .  .  . │
        #       └                           ┘
        # However, a more likely outcome is:
        #       ┌                           ┐
        #       │ .  .  .  .  .  .  .  .  . │
        #       │ a  b  c  .  .  .  d  e  f │
        #       │ g  h  i  .  .  .  j  k  l │
        #       │ m  n  o  .  .  .  p  q  r │
        #       │ .  .  .  .  .  .  .  .  . │
        #       └                           ┘
        # To prevent this; the SK blocks are rearranged & flatted by the
        # following code. Note, to avoid the issues associated with partial
        # row overlap only sk-blocks that are azimuthal minor, e.g. ss, sp,
        # sd etc. (lowest l value first), are considered. The azimuthal major
        # blocks, ps, ds, dp, etc. (highest l value first) are dealt
        # automatically by symmetrisation.

        # Group SK blocks by the row, or system & row for batch mode
        groupings = t.stack(index_mask_b[:-1]).unique_consecutive(dim=1, return_counts=True)[1]
        groups = split_by_size(sk_data, groupings)

        # Concatenate each group of SK blocks & flatten the result. Then
        # concatenate each of the now flatted block groups.
        sk_data_shaped = t.cat([group.transpose(1, 0).flatten() for group in groups])

        # Create the full size index mask which will assign the results
        index_mask_f = t.where((l_mat_f == l_pair).all(dim=-1))

        # Assign the data, flip the mask's indices & assign a 2'nd time to
        # ensure symmetry (this sets the azimuthal major SK blocks too).
        HS[index_mask_f] = sk_data_shaped
        if batch:  # Catch to prevent transposition of the system index number
            HS[index_mask_f[0], index_mask_f[2], index_mask_f[1]] = sk_data_shaped
        else:
            HS[index_mask_f[1], index_mask_f[0]] = sk_data_shaped

    return HS


def _batch_integral_retrieve(distances, atom_pairs, integral_feed, l_pair, **kwargs):
    """Mediate integral retrieval operation by splitting requests
    into batches of like type. integral_feed


    Notes
    -----
    This function is still subject to changes as different sk-integral
    generation methods will require different types information. This
    can currently be facilitated by passing additional information via
    the args & kwargs.

    """
    # Create an NxM tensor to hold the resulting integrals; where N is the
    # the number of integral sets & M is the number of elements per-set
    integrals = t.zeros((len(atom_pairs), l_pair.min() + 1),
                        dtype=distances.dtype)

    # Identify all unique element-element pairs
    unique_atom_pairs = atom_pairs.unique(dim=0)

    # Loop over each of the unique atom_pairs
    for atom_pair in unique_atom_pairs:
        # Construct an index mask for gather & scatter operations
        index_mask = t.where((atom_pairs == atom_pair).all(1))

        # Retrieve the integrals & assign them to the "integrals" tensor
        integrals[index_mask] = integral_feed(
            distances[index_mask], atom_pair, l_pair, **kwargs)

    # Return the resulting integrals
    return integrals

def _apply_on_sites(an_mat_a, integral_feed, orbital_id):
    # Get the diagonals of the atomic identity matrices
    on_site_element_blocks = an_mat_a.diagonal(dim1=-2, dim2=-1)
    # Identify all unique elements
    elements = on_site_element_blocks.flatten().unique()
    # Get the onsite blocks for said elements
    on_site_blocks = integral_feed.on_site(elements)

    [t.cat(list(map(on_site_blocks.__getitem__, i))) for i in on_site_element_blocks]


def _skt_ss(r, integrals):
    """Perform Slater-Koster transformations for ss interactions.

    Parameters
    ----------
    r : `torch.tensor` [`float`]
        The unit difference vector between a pair of orbitals. Or an
        array representing a set of such differences.
    integrals : `torch.tensor` [`float`]
        ss0 integral evaluated at the inter-atomic associated with
        the specified distance(s).

    Returns
    -------
    sub_block: `torch.tensor`:
        The ss  matrix sub-block, or a set thereof.

    Notes
    -----
    This function is capable of carrying out multiple transformations
    simultaneously. As ss interactions require no transformations; this
    function acts as a dummy subroutine to maintain functional
    consistency.
    """

    # No transformation is actually required so just return the integrals
    return integrals.unsqueeze(1)


def _skt_sp(r, integrals):
    """Perform Slater-Koster transformations for sp interactions.

    Parameters
    ----------
    r : `torch.tensor` [`float`]
        The unit difference vector between a pair of orbitals. Or an
        array representing a set of such differences.
    integrals : `torch.tensor` [`float`]
        sp0 integral evaluated at the inter-atomic associated with
        the specified distance(s).

    Returns
    -------
    sub_block: `torch.tensor`:
        The sp  matrix sub-block, or a set thereof.

    Notes
    -----
    This function is capable of carrying out multiple transformations
    simultaneously.

    Todo
    ----
    - Add maths describing the rotation matrix to the doc-string.
    [Priority: Low]
    """
    # Code for single evaluation operation is kept in comment form here to give
    # a clearer indication of what is being done here.

    # Unpack unit vector into its components. The transpose is needed when
    # `r` contains multiple vectors.
    # x, y, z = r.T

    # Construct the rotation matrix:
    #       ┌        ┐
    #       │ σs-p_y │
    # rot = │ σs-p_z │
    #       │ σs-p_z │
    #       └        ┘
    # rot = torch.tensor([
    #     [y],
    #     [z],
    #     [x]
    # ])

    # Apply transformation & return the results. "unsqueeze" ensures a shape
    # amenable to the transpose operation used by the skt function. The roll
    # ensures the correct azimuthal ordering.
    return (integrals * r).unsqueeze(1).roll(-1, -1)


def _skt_sd(r, integrals):
    """Perform Slater-Koster transformations for sd interactions.

    Parameters
    ----------
    r : `torch.tensor` [`float`]
        The unit difference vector between a pair of orbitals. Or an
        array representing a set of such differences.
    integrals : `torch.tensor` [`float`]
        sd0 integral evaluated at the inter-atomic associated with
        the specified distance(s).

    Returns
    -------
    sub_block: `torch.tensor`:
        The sd  matrix sub-block, or a set thereof.

    Notes
    -----
    This function is capable of carrying out multiple transformations
    simultaneously.
    """
    # Unpack unit vector into its components. The transpose is needed when
    # `r` contains multiple vectors.
    x, y, z = r.T
    # Pre calculate squares, square routes, etc
    x2, y2, z2 = r.T ** 2

    # Construct the rotation matrix:
    #       ┌              ┐
    #       │ σs-d_xy      │
    #       │ σs-d_yz      │
    # rot = │ σs-d_z^2     │
    #       │ σs-d_xz      │
    #       │ σs-d_x^2-y^2 │
    #       └              ┘
    #
    # For a single instance the operation would be:
    # rot = np.array([
    #     [_SQR3 * x * y],
    #     [_SQR3 * y * z],
    #     [z2 - 0.5 * (x2 + y2)],
    #     [_SQR3 * x * z],
    #     [0.5 * _SQR3 * (x2 - y2)]
    # ])
    rot = t.stack([
        _SQR3 * x * y,
        _SQR3 * y * z,
        z2 - 0.5 * (x2 + y2),
        _SQR3 * x * z,
        0.5 * _SQR3 * (x2 - y2)
    ])
    # Apply the transformation and return the result
    return (rot.T * integrals).unsqueeze(1)


def _skt_sf(r, integrals):
    """Perform Slater-Koster transformations for sf interactions.

    Parameters
    ----------
    r : `torch.tensor` [`float`]
        The unit difference vector between a pair of orbitals. Or an
        array representing a set of such differences.
    integrals : `torch.tensor` [`float`]
        sf0 integral evaluated at the inter-atomic associated with
        the specified distance(s).

    Returns
    -------
    sub_block: `torch.tensor`:
        The sf  matrix sub-block, or a set thereof.

    Notes
    -----
    This function is capable of carrying out multiple transformations
    simultaneously.
    """
    raise NotImplementedError()


def _skt_pp(r, integrals):
    """Perform Slater-Koster transformations for pp interactions.

    Parameters
    ----------
    r : `torch.tensor` [`float`]
        The unit difference vector between a pair of orbitals. Or an
        array representing a set of such differences.
    integrals : `torch.tensor` [`float`]
        pp0 & pp1 integrals evaluated at the inter-atomic associated
        with the specified distance(s).

    Returns
    -------
    sub_block: `torch.tensor`:
        The pp  matrix sub-block, or a set thereof.

    Notes
    -----
    This function is capable of carrying out multiple transformations
    simultaneously.
    """

    # Construct the rotation matrix:
    #       ┌                    ┐
    #       │ σp_y-p_y, πp_y-p_y │
    #       │ σp_y-p_z, πp_y-p_z │
    #       │ σp_y-p_z, πp_y-p_z │
    #       │ σp_z-p_y, πp_z-p_y │
    # rot = │ σp_z-p_z, πp_z-p_z │
    #       │ σp_z-p_x, πp_z-p_x │
    #       │ σp_x-p_y, πp_x-p_y │
    #       │ σp_x-p_z, πp_x-p_z │
    #       │ σp_x-p_x, πp_x-p_x │
    #       └                    ┘
    #
    # For a single instance the operation would be:
    # rot = np.array([
    #     [y2, 1 - y2],
    #     [yz, -yz],
    #     [xy, -xy],
    #
    #     [yz, -yz],
    #     [z2, 1 - z2],
    #     [xz, -xz],
    #
    #     [xy, -xy],
    #     [xz, -xz],
    #     [x2, 1 - x2]])
    # However this can be done much faster via the following
    r = r.T[[1, 2, 0]].T  # Reorder positions to mach physics notation
    outer = t.bmm(r.unsqueeze(2), r.unsqueeze(1)).view(-1, 9)  # Outer product
    tmp = t.tensor([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=r.dtype)  # Signs
    rot = t.stack((outer, tmp-outer), 2) # Reshape

    # Calculate & reshape the final sk blocks
    return t.bmm(rot, integrals.unsqueeze(2)).view(r.shape[0], 3, 3)


def _skt_pd(r, integrals):
    """Perform Slater-Koster transformations for pd interactions.

    Parameters
    ----------
    r : `torch.tensor` [`float`]
        The unit difference vector between a pair of orbitals. Or an
        array representing a set of such differences.
    integrals : `torch.tensor` [`float`]
        pd0 & pd1 integrals evaluated at the inter-atomic associated
        with the specified distance(s).

    Returns
    -------
    sub_block: `torch.tensor`:
        The pd  matrix sub-block, or a set thereof.

    Notes
    -----
    This function is capable of carrying out multiple transformations
    simultaneously.
    """
    # Unpack unit vector into its components. The transpose is needed when
    # `r` contains multiple vectors.
    x, y, z = r.T

    # Pre calculate squares, square routes, etc
    x2, y2, z2 = r.T ** 2
    alpha, beta = x2 + y2, x2 - y2
    z2_h_a = z2 - 0.5 * alpha
    xyz = r.prod(-1)

    # Construct the rotation matrix:
    #       ┌                                    ┐
    #       │ σp_y-d_xy,        πp_y-d_xy        │
    #       │ σp_y-d_yz,        πp_y-d_yz        │
    #       │ σp_y-d_z^2,       πp_y-d_z^2       │
    #       │ σp_y-d_xz,        πp_y-d_xz        │
    #       │ σp_y-d_(x^2-y^2), πp_y-d_(x^2-y^2) │
    #       │ σp_z-d_xy,        πp_z-d_xy        │
    #       │ σp_z-d_yz,        πp_z-d_yz        │
    # rot = │ σp_z-d_z^2,       πp_z-d_z^2       │
    #       │ σp_z-d_xz,        πp_z-d_xz        │
    #       │ σp_z-d_(x^2-y^2), πp_z-d_(x^2-y^2) │
    #       │ σp_x-d_xy,        πp_x-d_xy        │
    #       │ σp_x-d_yz,        πp_x-d_yz        │
    #       │ σp_x-d_z^2,       πp_x-d_z^2       │
    #       │ σp_x-d_xz,        πp_x-d_xz        │
    #       │ σp_x-d_(x^2-y^2), πp_x-d_(x^2-y^2) │
    #       └                                    ┘
    #
    # For a single instance the operation would be:
    # rot = np.array([
    #     [_SQR3 * y2 * x, x * (1 - 2 * y2)],
    #     [_SQR3 * y2 * z, z * (1 - 2 * y2)],
    #     [y * z2_h_a, -_SQR3 * y * z2],
    #     [_SQR3 * xyz, -2 * xyz],
    #     [_HSQR3 * y * beta, -y * (1 + beta)],
    #
    #     [_SQR3 * xyz, -2 * xyz],
    #     [_SQR3 * z2 * y, y * (1 - 2 * z2)],
    #     [z * z2_h_a, _SQR3 * z * alpha],
    #     [_SQR3 * z2 * x, x * (1 - 2 * z2)],
    #     [_HSQR3 * z * beta, -z * beta],
    #
    #     [_SQR3 * x2 * y, y * (1 - 2 * x2)],
    #     [_SQR3 * xyz, -2 * xyz],
    #     [x * z2_h_a, -_SQR3 * x * z2],
    #     [_SQR3 * x2 * z, z * (1 - 2 * x2)],
    #     [_HSQR3 * x * beta, x * (1 - beta)]
    # ])
    # There must be a nicer, vectorised and more elegant way to do this
    column_1 = t.stack((
        _SQR3 * y2 * x,
        _SQR3 * y2 * z,
        y * z2_h_a,
        _SQR3 * xyz,
        _HSQR3 * y * beta,

        _SQR3 * xyz,
        _SQR3 * z2 * y,
        z * z2_h_a,
        _SQR3 * z2 * x,
        _HSQR3 * z * beta,

        _SQR3 * x2 * y,
        _SQR3 * xyz,
        x * z2_h_a,
        _SQR3 * x2 * z,
        _HSQR3 * x * beta
    ), -1)

    column_2 = t.stack((
        x * (1 - 2 * y2),
        z * (1 - 2 * y2),
        -_SQR3 * y * z2,
        -2 * xyz,
        -y * (1 + beta),

        -2 * xyz,
        y * (1 - 2 * z2),
        _SQR3 * z * alpha,
        x * (1 - 2 * z2),
        -z * beta,

        y * (1 - 2 * x2),
        -2 * xyz,
        -_SQR3 * x * z2,
        z * (1 - 2 * x2),
        x * (1 - beta)
    ), -1)

    # Combine the two columns to create the final rotation matrix
    rot = t.stack((column_1, column_2), -1)

    # Calculate, reshape and return the SK blocks
    return (rot @ integrals.unsqueeze(2)).view(-1, 3, 5)


def _skt_pf(r, integrals):
    """Perform Slater-Koster transformations for pf interactions.

    Parameters
    ----------
    r : `torch.tensor` [`float`]
        The unit difference vector between a pair of orbitals. Or an
        array representing a set of such differences.
    integrals : `torch.tensor` [`float`]
        sf0 & sf1 integrals evaluated at the inter-atomic associated with
        the specified distance(s).

    Returns
    -------
    sub_block: `torch.tensor`:
        The pf  matrix sub-block, or a set thereof.

    Notes
    -----
    This function is capable of carrying out multiple transformations
    simultaneously.
    """
    raise NotImplementedError()


def _skt_dd(r, integrals):
    """Perform Slater-Koster transformations for dd interactions.

    Parameters
    ----------
    r : `torch.tensor` [`float`]
        The unit difference vector between a pair of orbitals. Or an
        array representing a set of such differences.
    integrals : `torch.tensor` [`float`]
        dd0, dd1 & dd2 integrals evaluated at the inter-atomic
        associated with the specified distance(s).

    Returns
    -------
    sub_block: `torch.tensor`:
        The dd  matrix sub-block, or a set thereof.

    Notes
    -----
    This function is capable of carrying out multiple transformations
    simultaneously.
    """
    # There are some tricks that could be used to reduce the size, complexity
    # and overhead of this monster. (this should be done at some point)

    # Unpack unit vector into its components. The transpose is needed when
    # `r` contains multiple vectors.
    x, y, z = r.T

    # Pre calculate squares, square routes, etc
    x2, xy, xz, y2, yz, z2 = r.T[[0, 0, 0, 1, 1, 2]] * r.T[[0, 1, 2, 1, 2, 2]]
    x2y2, y2z2, x2z2 = xy ** 2, yz ** 2, xz ** 2
    alpha, beta = x2 + y2, x2 - y2
    xyz = r.prod(-1)
    a_m_z2 = alpha - z2
    beta2 = beta ** 2
    sqr3_beta = _SQR3 * beta
    z2_h_a = z2 - 0.5 * alpha

    # Construct the rotation matrix:
    #       ┌                                                                        ┐
    #       │ σd_xy-d_xy,             πd_xy-d_xy,             δd_xy-σd_xy            │
    #       │ σd_xy-d_yz,             πd_xy-d_yz,             δd_xy-σd_yz            │
    #       │ σd_xy-d_z^2,            πd_xy-d_z^2,            δd_xy-d_z^2            │
    #       │ σd_xy-d_xz,             πd_xy-d_xz,             δd_xy-d_xz             │
    #       │ σd_xy-(x^2-y^2),        πd_xy-(x^2-y^2),        δd_xy-(x^2-y^2)        │
    #       │ σd_yz-d_xy,             πd_yz-d_xy,             δd_yz-σd_xy            │
    #       │ σd_yz-d_yz,             πd_yz-d_yz,             δd_yz-σd_yz            │
    #       │ σd_yz-d_z^2,            πd_yz-d_z^2,            δd_yz-d_z^2            │
    #       │ σd_yz-d_xz,             πd_yz-d_xz,             δd_yz-d_xz             │
    #       │ σd_yz-(x^2-y^2),        πd_yz-(x^2-y^2),        δd_yz-(x^2-y^2)        │
    #       │ σd_z^2-d_xy,            πd_z^2-d_xy,            δd_z^2-σd_xy           │
    #       │ σd_z^2-d_yz,            πd_z^2-d_yz,            δd_z^2-σd_yz           │
    # rot = │ σd_z^2-d_z^2,           πd_z^2-d_z^2,           δd_z^2-d_z^2           │
    #       │ σd_z^2-d_xz,            πd_z^2-d_xz,            δd_z^2-d_xz            │
    #       │ σd_z^2-(x^2-y^2),       πd_z^2-(x^2-y^2),       δd_z^2-(x^2-y^2)       │
    #       │ σd_xz-d_xy,             πd_xz-d_xy,             δd_xz-d_xy             │
    #       │ σd_xz-d_yz,             πd_xz-d_yz,             δd_xz-d_yz             │
    #       │ σd_xz-d_z^2,            πd_xz-d_z^2,            δd_xz-d_z^2            │
    #       │ σd_xz-d_xz,             πd_xz-d_xz,             δd_xz-d_xz             │
    #       │ σd_xz-d_xy-(x^2-y^2),   πd_xz-d_xy-(x^2-y^2),   δd_xz-d_xy(x^2-y^2)    │
    #       │ σd_(x^2-y^2)-d_xy,      πd_(x^2-y^2)-d_xy,      δd_(x^2-y^2)-d_xy      │
    #       │ σd_(x^2-y^2)-d_yz,      πd_(x^2-y^2)-d_yz,      δd_(x^2-y^2)-d_yz      │
    #       │ σd_(x^2-y^2)-d_z^2,     πd_(x^2-y^2)-d_z^2,     δd_(x^2-y^2)-d_z^2     │
    #       │ σd_(x^2-y^2)-d_xz,      πd_(x^2-y^2)-d_xz,      δd_(x^2-y^2)-d_xz      │
    #       │ σd_(x^2-y^2)-(x^2-y^2), πd_(x^2-y^2)-(x^2-y^2), δd_(x^2-y^2)-(x^2-y^2) │
    #       └                                                                        ┘
    # For a single instance the operation would be:
    # rot = np.array(
    #     [
    #         [3 * x2y2,                  alpha - 4 * x2y2,           z2 + x2y2],
    #         [3 * xyz * y,               xz * (1 - 4 * y2),          xz * (y2 - 1)],
    #         [_SQR3 * xy * z2_h_a,       -2 * _SQR3 * xyz * z,       _HSQR3 * xy * (1 + z2)],
    #         [3 * xyz * x,               yz * (1 - 4 * x2),          yz * (x2 - 1)],
    #         [1.5 * xy * beta,           -2 * xy * beta,             0.5 * xy * beta],
    #
    #         [3 * xyz * x,               zx * (1 - 4 * y2),          xz * (y2 - 1)],
    #         [3 * y2z2,                  y2 + z2 - 4 * y2z2,         x2 + y2z2],
    #         [_SQR3 * yz * z2_h_a,       _SQR3 * yz * a_m_z2,        -_HSQR3 * yz * alpha],
    #         [3 * xyz * z,               xy * (1 - 4 * z2),          xy * (z2 - 1)],
    #         [1.5 * yz * beta,           -yz * (1 + 2 * beta),       yz * (1 + 0.5 * beta)],
    #
    #         [_SQR3 * xy * z2_h_a,       -2 * _SQR3 * xyz * z,       _HSQR3 * xy * (1 + z2)],
    #         [_SQR3 * yz * z2_h_a,       _SQR3 * yz * a_m_z2,        -_HSQR3 * yz * alpha],
    #         [z2_h_a ** 2,               3 * z2 * alpha,             0.75 * alpha ** 2],
    #         [_SQR3 * xz * z2_h_a,       _SQR3 * xz * a_m_z2,        -_HSQR3 * xz * alpha],
    #         [0.5 * sqr3_beta * z2_h_a,  -z2 * sqr3_beta,            0.25 * sqr3_beta * (1 + z2)],
    #
    #         [3 * xyz * x,               yz * (1 - 4 * x2),          yz * (x2 - 1)],
    #         [3 * xyz * z,               xy * (1 - 4 * z2),          xy * (z2 - 1)],
    #         [_SQR3 * xz * z2_h_a,       _SQR3 * xz * a_m_z2,        -_HSQR3 * xz * alpha],
    #         [3 * x2z2,                  z2 + x2 - 4 * x2z2,         y2 + x2z2],
    #         [1.5 * xz * beta,           xz * (1 - 2 * beta),        -xz * (1 - 0.5 * beta)],
    #
    #         [1.5 * xy * beta,           -2 * xy * beta,             0.5 * xy * beta],
    #         [1.5 * yz * beta,           -yz * (1 + 2 * beta),       yz * (1 + 0.5 * beta)],
    #         [0.5 * sqr3_beta * z2_h_a,  -z2 * sqr3_beta,            0.25 * sqr3_beta * (1 + z2)],
    #         [1.5 * xz * beta,           xz * (1 - 2 * beta),        -xz * (1 - 0.5 * beta)],
    #         [0.75 * beta2,              alpha - beta2,              z2 + 0.25 * beta2]
    #     ]
    # )
    # Ths is starting to get a little out of hand
    column_1 = t.stack([
        3 * x2y2,
        3 * xyz * y,
        _SQR3 * xy * z2_h_a,
        3 * xyz * x,
        1.5 * xy * beta,

        3 * xyz * y,
        3 * y2z2,
        _SQR3 * yz * z2_h_a,
        3 * xyz * z,
        1.5 * yz * beta,

        _SQR3 * xy * z2_h_a,
        _SQR3 * yz * z2_h_a,
        z2_h_a ** 2,
        _SQR3 * xz * z2_h_a,
        0.5 * sqr3_beta * z2_h_a,

        3 * xyz * x,
        3 * xyz * z,
        _SQR3 * xz * z2_h_a,
        3 * x2z2,
        1.5 * xz * beta,

        1.5 * xy * beta,
        1.5 * yz * beta,
        0.5 * sqr3_beta * z2_h_a,
        1.5 * xz * beta,
        0.75 * beta2
    ], -1)

    column_2 = t.stack([
        alpha - 4 * x2y2,
        xz * (1 - 4 * y2),
        -2 * _SQR3 * xyz * z,
        yz * (1 - 4 * x2),
        -2 * xy * beta,

        xz * (1 - 4 * y2),
        y2 + z2 - 4 * y2z2,
        _SQR3 * yz * a_m_z2,
        xy * (1 - 4 * z2),
        -yz * (1 + 2 * beta),

        -2 * _SQR3 * xyz * z,
        _SQR3 * yz * a_m_z2,
        3 * z2 * alpha,
        _SQR3 * xz * a_m_z2,
        -z2 * sqr3_beta,

        yz * (1 - 4 * x2),
        xy * (1 - 4 * z2),
        _SQR3 * xz * a_m_z2,
        z2 + x2 - 4 * x2z2,
        xz * (1 - 2 * beta),

        -2 * xy * beta,
        -yz * (1 + 2 * beta),
        -z2 * sqr3_beta,
        xz * (1 - 2 * beta),
        alpha - beta2
    ], -1)

    column_3 = t.stack([
        z2 + x2y2,
        xz * (y2 - 1),
        _HSQR3 * xy * (1 + z2),
        yz * (x2 - 1),
        0.5 * xy * beta,

        xz * (y2 - 1),
        x2 + y2z2,
        -_HSQR3 * yz * alpha,
        xy * (z2 - 1),
        yz * (1 + 0.5 * beta),

        _HSQR3 * xy * (1 + z2),
        -_HSQR3 * yz * alpha,
        0.75 * alpha ** 2,
        -_HSQR3 * xz * alpha,
        0.25 * sqr3_beta * (1 + z2),

        yz * (x2 - 1),
        xy * (z2 - 1),
        -_HSQR3 * xz * alpha,
        y2 + x2z2,
        -xz * (1 - 0.5 * beta),

        0.5 * xy * beta,
        yz * (1 + 0.5 * beta),
        0.25 * sqr3_beta * (1 + z2),
        -xz * (1 - 0.5 * beta),
        z2 + 0.25 * beta2
    ], -1)

    rot = t.stack((column_1, column_2, column_3), -1)
    return (rot @ integrals.unsqueeze(2)).view(-1, 5, 5)


def _skt_df(r, integrals):
    """Performs Slater-Koster transformations for df interactions.

    Parameters
    ----------
    r : `torch.tensor` [`float`]
        The unit difference vector between a pair of orbitals. Or an
        array representing a set of such differences.
    integrals : `torch.tensor` [`float`]
        df0, df1 & df2 integrals evaluated at the inter-atomic
        associated with the specified distance(s).

    Returns
    -------
    sub_block: `torch.tensor`:
        The df  matrix sub-block, or a set thereof.

    Notes
    -----
    This function is capable of carrying out multiple transformations
    simultaneously.
    """
    raise NotImplementedError()


def _skt_ff(r, integrals):
    """Performs Slater-Koster transformations for ff interactions.

    Parameters
    ----------
    r : `torch.tensor` [`float`]
        The unit difference vector between a pair of orbitals. Or an
        array representing a set of such differences.
    integrals : `torch.tensor` [`float`]
        ff0, ff1, ff2 & ff3 integrals evaluated at the inter-atomic
        associated with the specified distance(s).

    Returns
    -------
    sub_block: `torch.tensor`:
        The ff  matrix sub-block, or a set thereof.

    Notes
    -----
    This function is capable of carrying out multiple transformations
    simultaneously.
    """
    # ... Perhaps another time.
    raise NotImplementedError()

# Known sk interactions and their associated functions
_SK_interactions = t.tensor([[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]])
_SK_functions = [_skt_ss, _skt_sp, _skt_sd, _skt_pp, _skt_pd, _skt_dd]




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
    pass

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


def hs_s_s(x, y, z, hss0, sss0):
    return hss0, sss0


def hs_s_x(x, y, z, hsp0, ssp0):
    return x * hsp0, x * ssp0


def hs_s_y(x, y, z, hsp0, ssp0):
    return y*hsp0, y*ssp0


def hs_s_z(x, y, z, hsp0, ssp0):
    return z*hsp0, z*ssp0


def hs_x_s(x, y, z, hsp0, ssp0):
    return hs_s_x(-x, -y, -z, hsp0, ssp0)[0], hs_s_x(-x, -y, -z, hsp0, ssp0)[1]


def hs_x_x(x, y, z, hpp0, spp0, hpp1, spp1):
    return x**2*hpp0+(1-x**2)*hpp1, x**2*spp0+(1-x**2)*spp1


def hs_x_y(x, y, z, hpp0, spp0, hpp1, spp1):
    return x*y*hpp0-x*y*hpp1, x*y*spp0-x*y*spp1


def hs_x_z(x, y, z, hpp0, spp0, hpp1, spp1):
    return x*z*hpp0-x*z*hpp1, x*z*spp0-x*z*spp1


def hs_y_s(x, y, z, hsp0, ssp0):
    return hs_s_y(-x, -y, -z, hsp0, ssp0)[0], hs_s_y(-x, -y, -z, hsp0, ssp0)[1]


def hs_y_x(x, y, z, hpp0, spp0, hpp1, spp1):
    return hs_x_y(-x, -y, -z, hpp0, spp0, hpp1, spp1)[0], hs_x_y(
            -x, -y, -z, hpp0, spp0, hpp1, spp1)[1]


def hs_y_y(x, y, z, hpp0, spp0, hpp1, spp1):
    return y**2*hpp0+(1-y**2)*hpp1, y**2*spp0+(1-y**2)*spp1


def hs_y_z(x, y, z, hpp0, spp0, hpp1, spp1):
    return y*z*hpp0-y*z*hpp1, y*z*spp0-y*z*spp1


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

