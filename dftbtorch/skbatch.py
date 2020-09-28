"""
Todo
----
- Extra validation needed for final dot product evaluation when working
  in multi-system mode. [Priority: High]
- Implement f orbital transformations. [Priority: Low]
- Enable use of ase objects. [Priority: QOL]
- Replace conditional reshape operation with single reshape operation
  by using .../None as the first value. [Priority: QOL]
- Should add information to the transformation functions to describe
  the anticipated shape of the returned tensor. [Priority: Low]
- Abstract shape conversion operations for r & integrals parameters to
  a wrapper function. [Priority: High]
- Make the shape of the tensors returned form _sk functions consistent


Notes the tensors returned from sk operations MUST be batch major, i.e.
the first dimension must iterate over the sk groups.

Check that the transpose/permute before scattering does not spaghetti
ss and sp interactions

ss: Nx1x1  #<-- not strictly needed, but maintains consistency
sp: Nx1x3
sd: Nx1x5
pp: Nx3x3
pd: Nx3x5
dd: Nx5x5

Multi system mode will require an additional check during the final grouping
phase to make sure blocks from different Hamiltonians don't get grouped
together just because they have the same row number.

use __ALL__ method to hide things from the import * statment

"""
import torch
import numpy as np
from dftbmalt.utils.utilities import split_by_size
from IO.basis import Basis, Bases

# Static module-level constants
_SQR3 = np.sqrt(3.)
_HSQR3 = 0.5 * np.sqrt(3.)


def skt(systems, orbital_id, integral_feed, **kwargs):
    """Constructs a Hamiltonian or overlap matrix from integral values
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
    # or orbitals found on any system in the batch. Note that in the comments
    # below "B" will be referred to as the system index dimension.
    HS = torch.zeros(orbital_id.shape)

    # Detect if this is a batch mode operation (mostly done to clarify the code)
    batch = HS.dim() == 3

    # Notes
    # -----
    # The "*_mat_*" variables hold data that is; used to build masks, gathered
    # by other masks, or both. The _f, _b & _a suffixes indicate whether the
    # tensor is full, block-wise or atom-wise resolved. See the basis.Basis
    # class for more information.

    # Masking Data Initialisation
    # ---------------------------
    # Matrices used in building getter & setter masks are initialised here.
    # These are the full (f) & block-wise (b) azimuthal identity matrices.
    l_mat_f = orbital_id.azimuthal_matrix(mask=True, mask_lower=True)
    l_mat_b = orbital_id.azimuthal_matrix(block=True, mask=True, mask_lower=False)

    # The masks will then gather data from the matrices initialised below to
    # 1) pass on to various functions & 2) create new "secondary" masks. These
    # matrices are similar in structure to the azimuthal identity matrices.
    i_mat_b = orbital_id.index_matrix(block=True)  # <- atom indices
    an_mat_a = orbital_id.atomic_number_matrix(atomic=True)  # <- atomic numbers
    # Atomic distances:
    dist_mat_a = torch.cdist(systems.positions, systems.positions, p=2)
    # directional cosigns:
    vec_mat_a = systems.positions.unsqueeze(-3) - systems.positions.unsqueeze(-2)
    vec_mat_a = torch.nn.functional.normalize(vec_mat_a, p=2, dim=-1)

    # Loop over each type of azimuthal-pair interaction & its associated
    # SK-function
    for l_pair, f in zip(_SK_interactions, _SK_functions):
        # Build a bool mask that is True wherever l_mat_b = l_pair & convert
        # it to a set of index positions (aka an index mask).
        index_mask_b = torch.where((l_mat_b == l_pair).all(dim=-1))
        print("_SK_interactions, _SK_functions", l_mat_b, l_pair)

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
        groupings = torch.stack(index_mask_b[:-1]).unique_consecutive(dim=1, return_counts=True)[1]
        groups = split_by_size(sk_data, groupings)

        # Concatenate each group of SK blocks & flatten the result. Then
        # concatenate each of the now flatted block groups.
        sk_data_shaped = torch.cat([group.transpose(1, 0).flatten() for group in groups])

        # Create the full size index mask which will assign the results
        index_mask_f = torch.where((l_mat_f == l_pair).all(dim=-1))

        # Assign the data, flip the mask's indices & assign a 2'nd time to
        # ensure symmetry (this sets the azimuthal major SK blocks too).
        HS[index_mask_f] = sk_data_shaped
        if batch:  # Catch to prevent transposition of the system index number
            HS[index_mask_f[0], index_mask_f[2], index_mask_f[1]] = sk_data_shaped
        else:
            HS[index_mask_f[1], index_mask_f[0]] = sk_data_shaped
        print("HS", HS)

    return HS


def _batch_integral_retrieve(distances, atom_pairs, integral_feed, l_pair, **kwargs):
    """Mediates integral retrieval operation by splitting requests
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
    integrals = torch.zeros((len(atom_pairs), l_pair.min() + 1),
                            dtype=distances.dtype)

    # Identify all unique element-element pairs
    unique_atom_pairs = atom_pairs.unique(dim=0)

    # Loop over each of the unique atom_pairs
    for atom_pair in unique_atom_pairs:
        # Construct an index mask for gather & scatter operations
        index_mask = torch.where((atom_pairs == atom_pair).all(1))

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

    [torch.cat(list(map(on_site_blocks.__getitem__, i))) for i in on_site_element_blocks]


def _skt_ss(r, integrals):
    """Performs Slater-Koster transformations for ss interactions.

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
    """Performs Slater-Koster transformations for sp interactions.

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
    """Performs Slater-Koster transformations for sd interactions.

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
    rot = torch.stack([
        _SQR3 * x * y,
        _SQR3 * y * z,
        z2 - 0.5 * (x2 + y2),
        _SQR3 * x * z,
        0.5 * _SQR3 * (x2 - y2)
    ])
    # Apply the transformation and return the result
    return (rot.T * integrals).unsqueeze(1)


def _skt_sf(r, integrals):
    """Performs Slater-Koster transformations for sf interactions.

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
    """Performs Slater-Koster transformations for pp interactions.

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
    outer = torch.bmm(r.unsqueeze(2), r.unsqueeze(1)).view(-1, 9)  # Outer product
    tmp = torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=r.dtype)  # Signs
    rot = torch.stack((outer, tmp-outer), 2) # Reshape

    # Calculate & reshape the final sk blocks
    return torch.bmm(rot, integrals.unsqueeze(2)).view(r.shape[0], 3, 3)


def _skt_pd(r, integrals):
    """Performs Slater-Koster transformations for pd interactions.

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
    column_1 = torch.stack((
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

    column_2 = torch.stack((
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
    rot = torch.stack((column_1, column_2), -1)

    # Calculate, reshape and return the SK blocks
    return (rot @ integrals.unsqueeze(2)).view(-1, 3, 5)


def _skt_pf(r, integrals):
    """Performs Slater-Koster transformations for pf interactions.

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
    """Performs Slater-Koster transformations for dd interactions.

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
    column_1 = torch.stack([
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

    column_2 = torch.stack([
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

    column_3 = torch.stack([
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

    rot = torch.stack((column_1, column_2, column_3), -1)
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
_SK_interactions = torch.tensor([[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]])
_SK_functions = [_skt_ss, _skt_sp, _skt_sd, _skt_pp, _skt_pd, _skt_dd]
