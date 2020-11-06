import torch
import glob
from dftbmalt.io.skf import SKF_File
from dftbmalt.structures.basis import Basis, Bases
from dftbmalt.dftb.slaterkoster import skt
from dftbmalt.structures.system import System
from dftbmalt.utils.batch import pack
from scipy.interpolate import CubicSpline
from os import path
from ase.atoms import Atoms
from ase.build import molecule as molecule_database


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

        # The interactions: ddσ, ddπ, ddδ, ...
        interactions = [(2, 2, 0), (2, 2, 1), (2, 2, 2), (1, 2, 0), (1, 2, 1),
                        (1, 1, 0), (1, 1, 1), (0, 2, 0), (0, 1, 0), (0, 0, 0)]

        # Find all the skf files
        skf_files = glob.glob(path.join(directory, '*.skf'))

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
        if type(distances) == torch.Tensor:
            distances = distances.numpy()

        # Retrieve the appropriate splines
        splines = [self.spline_dict[(*atom_pair.tolist(), *l_pair.tolist(), b, mat_type)]
                   for b in range(min(l_pair)+1)]

        # Evaluate the splines at the requested distances, convert the result
        # into a torch tensor and return it.
        return torch.tensor([spline(distances) for spline in splines]).T

class FakeAtoms:
    def __init__(self, positions):
        self.positions = positions


def ase2pytorch(atoms):
    for key, value in atoms.arrays.items():
        atoms.arrays[key] = torch.tensor(value)


def single_run_example():

    # Section 0: Defining variables
    # Path to where the directory in which the skf files are store
    skf_path = '/home/gz_fan/Documents/ML/dftb/DFTBMaLT/auorg-1-1'

    # Ase atoms object for the molecule we want to look at
    molecule = molecule_database('CH4')

    # What operator do we wish to construct (H or S)
    operator = 'H'

    # Section 1: Initial setup and static variables
    # The ase atoms object uses numpy array which are not compatible with
    # many pytorch applications. Thus we must convert it to an atoms like
    # object that is based on pytorch.
    molecule = System.from_ase_atoms(molecule)

    # Static property section
    # Max angular momentum associated with each element type
    max_l_key = {1: 0, 6: 1, 7: 1, 8: 1, 79: 2}

    # Section 2: Creating a Slater Koster interpolation object
    # Build the Slater-Koster integral generator object
    sk_integral_generator = SKIntegralGenerator.from_dir(skf_path)

    # Section 3: Creating a basis object
    basis_info = Basis(molecule.get_atomic_numbers(), max_l_key)

    # Section 4:
    matrix = skt(molecule,
                 basis_info, sk_integral_generator, mat_type=operator)


def batch_run_example():
    # Section 0: Defining variables
    # Path to where the directory in which the skf files are store
    skf_path = '/home/ajmhpc/Documents/Work/DFTB/parameters/auorg-1-1'

    # Ase atoms objects for the molecules we want to look at
    molecules = [molecule_database('CH4'),
                 molecule_database('H2O'),
                 molecule_database('C2H5'),
                 molecule_database('CH3O'),
                 molecule_database('H2COH')]

    # What operator do we wish to construct (H or S)
    operator = 'H'

    # Section 1: Initial setup and static variables
    # The ase atoms object uses numpy array which are not compatible with
    # many pytorch applications. Thus we must convert it to an atoms like
    # object that is based on pytorch.
    molecules = [System.from_ase_atoms(molecule) for molecule in molecules]

    # Molecules must be packed into a single batch object. We just use a fake
    # atoms object as the skf function only needs the positions.
    molecules_batched = FakeAtoms(pack([m.positions for m in molecules]))

    # Static property section
    # Max angular momentum associated with each element type
    max_l_key = {1: 0, 6: 1, 7: 1, 8: 1, 79: 2}

    # Section 2: Creating a Slater Koster interpolation object
    # Build the Slater-Koster integral generator object
    sk_integral_generator = SKIntegralGenerator.from_dir(skf_path)

    # Section 3: Creating a basis object
    basis_info_list = [Basis(molecule.get_atomic_numbers(), max_l_key)
                       for molecule in molecules]

    # We need to batch all the basis info together into a single object
    bases_info = Bases(basis_info_list, max_l_key)

    # Section 4:
    matrix = skt(molecules_batched,
                 bases_info, sk_integral_generator, mat_type=operator)

if __name__ == '__main__':
    # Warnings: Results will be incorrect here as the ase molecules use angstroms
    # rather than Bohrs. If you need the correct answer you must make this
    # conversion. It was not done in this example to save time.

    # Note that currently there is not support for the diagonal matrix elements

    single_run_example()
    batch_run_example()
    print('Done')
