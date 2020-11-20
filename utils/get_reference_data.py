"""Build hdf type data for machine learning reference.

The data will include:
    geometry information,
    physical properties

define para['dataType'] as original dataset, e.g 'ani'

"""
import dftbtorch.initparams as initpara
from utils.aset import DFTB, AseAims
from IO.dataloader import LoadData
import dftbtorch.parameters as constpara
ATOMNUM = {'H': 1, 'C': 6, 'N': 7, 'O': 8}


class RefDFTBPlus:
    """Physical properties are from DFTB-ASE calculations."""

    def __init__(self, para):
        """Load original dataset and then write into hdf type."""
        # load constant parameters
        self.para = constpara.constant_parameter(para)

        self.ml, self.dataset = {}, {}

        # how to run FHI-aims
        self.ml['reference'] = 'dftbase'

        # runani: read dataset and run FHI-aims calculations
        # writeinput: read dataset and write FHI-aims input without calculation
        self.dataset['datasetType'] = 'rundftbplus'

        # SKF path
        self.para['directorySK'] = '../slko/mio/'

        # read and run different molecule species dataset size
        self.dataset['sizeDataset'] = [200, 200, 200]

        # do not mix different molecule specie in dataset
        self.dataset['LdatasetMixture'] = False

        # define dataset as input
        self.dataset['dataset'] = '../data/dataset/an1/ani_gdb_s01.h5'

        # get parameters for generating reference data
        self.para = initpara.dftb_parameter(self.para)
        self.skf = initpara.skf_parameter(self.para)
        self.dataset = initpara.init_dataset(self.dataset)
        self.para, self.dataset, self.skf, self.ml = \
            initpara.init_ml(self.para, self.dataset, self.skf, self.ml)

        # LoadData will load ani dataset (call function load_ani)
        # then run load_data_hdf
        LoadData(self.para, self.dataset, self.ml)

    def run_dftbplus(self, end, coorall):
        """Run DFTB-ase and save as hdf data."""
        # DFTB+ as reference
        DFTB(self.para, setenv=True).run_dftb(end, coorall, hdf=self.f,
                                              group=self.g)

    def run_dftbase():
        pass

    def get_property():
        pass

    def save_data():
        pass


class RefAims:
    """Run FHI-aims calculations and save results as reference data for ML.

    Note:
    set para['reference'] = 'aimsase'
    """

    def __init__(self, para):
        """Load geometry data, run FHI-aims and analyze the results."""
        # load constant parameters
        self.para = constpara.constant_parameter(para)

        self.ml, self.dataset = {}, {}

        # how to run FHI-aims
        self.ml['reference'] = 'aimsase'

        # runaims: read dataset and run FHI-aims calculations
        # writeinput: read dataset and only generate geometry.in
        self.dataset['datasetType'] = 'runaims'

        # read and run different molecule species dataset size
        # self.dataset['dataset'] = '../data/dataset/an1/ani_gdb_s01.h5'
        # self.dataset['sizeDataset'] = [200, 200, 200]

        # do not mix different molecule specie in dataset
        self.dataset['LdatasetMixture'] = False

        # define dataset as input
        self.dataset['dataset'] = '../data/dataset/an1/ani_gdb_s03.h5'
        self.dataset['sizeDataset'] = [1] * 25

        # get parameters for generating reference data
        self.para = initpara.dftb_parameter(self.para)
        self.skf = initpara.skf_parameter(self.para)
        self.dataset = initpara.init_dataset(self.dataset)
        self.para, self.dataset, self.skf, self.ml = \
            initpara.init_ml(self.para, self.dataset, self.skf, self.ml)

        # LoadData will load ani dataset (call function load_ani)
        # then run load_data_hdf
        LoadData(self.para, self.dataset, self.ml)

    def load_data_hdf(self):
        """Load original data for FHI-aims calculations."""
        pass

    def run_aims(self):
        """Run FHI-aims calculations."""
        pass


if __name__ == '__main__':
    """Main function."""
    para = {'task': 'get_aims_hdf'}

    # get_dftb_hdf means get reference data from DFTB+ calculations
    if para['task'] == 'get_dftbplus_hdf':
        RefDFTBPlus(para)

    # get_dftb_hdf means get reference data from FHI-aims calculations
    elif para['task'] == 'get_aims_hdf':
        RefAims(para)
