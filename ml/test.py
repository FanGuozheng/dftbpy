import time
import torch as t
import numpy as np
import dftbtorch.initparams as initpara
import utils.plot as plot
import dftbtorch.dftbcalculator as dftbcalculator
import ml.interface as interface
from dftbtorch.sk import SKTran, GetSKTable, GetSK_
from ml.feature import Dscribe as dscribe
from IO.dataloader import LoadData, LoadReferenceData
from utils.runcalculation import RunReference
from ml.padding import pad1d, pad2d


class DFTBMLTest:
    """Test DFTB machine learning."""

    def __init__(self, parameter=None, dataset=None, skf=None, ml=None):
        """Initialize parameters."""
        time_begin = time.time()

        # initialize DFTB, dataset, sk parameters
        self.init = dftbcalculator.Initialization(parameter, dataset, skf, ml)

        # read input file if defined in command line
        # then get default DFTB, dataset, sk parameters
        self.init.initialize_parameter()
        self.parameter = self.init.parameter
        self.dataset = self.init.dataset
        self.skf = self.init.skf
        self.ml = ml

        # detect automatically
        t.autograd.set_detect_anomaly(True)

        # set the print precision from very beginning
        t.set_printoptions(precision=14)

        # set the precision control
        if self.parameter['precision'] in (t.float64, t.float32):
            t.set_default_dtype(d=self.parameter['precision'])
        else:
            raise ValueError('please select either t.float64 or t.float32')

        # initialize machine learning parameters
        self.parameter, self.dataset, self.skf, self.ml = \
            initpara.init_ml(self.parameter, self.dataset, self.skf, self.ml)

        # load data
        self.load_data()

        # run machine learning optimization
        self.run_dftb()

        # plot ML results
        plot.plot_ml(self.parameter, self.ml)
        time_end = time.time()
        print('Total time:', time_end - time_begin)

    def load_data(self):
        """Load reference dataset for machine learning."""
        # run DFT(B) to get reference data
        if self.ml['runReference']:
            LoadData(self.parameter, self.dataset, self.ml)
            RunReference(self.parameter, self.dataset, self.skf, self.ml)

        # directly get reference data from dataset
        else:
            LoadReferenceData(self.parameter, self.dataset, self.skf, self.ml)


    def run_dftb(self):
        """Run machine learning optimization."""
        # optimize integrals directly
        if self.parameter['task'] == 'testIntegral':
            Integral(self.parameter, self.dataset, self.skf, self.ml)

        # optimize compression radius and then generate integrals
        elif self.parameter['task'] == 'testCompressionR':
            CompressionR(self.parameter, self.dataset, self.skf, self.ml)


class Integral:

    def __init__(self):
        pass


class CompressionR:

    def __init__(self, para, dataset, skf, ml):
        self.para, self.dataset, self.skf, self.ml = para, dataset, skf, ml

        self.nbatch = self.dataset['nbatch'] = self.dataset['nfile']

        # load data, get optimized compression radii
        self.load_data()

        # generate compression r ML parameters, and return compression r
        self.fit_compression_r()

        # run DFRB calculations
        self.run_dftb()

        # process ML prediction DFTB and mio DFTB results
        self.process_test_data()

    def load_data(self):
        """Load optimized compression radii."""
        self.steps = self.ml['mlSteps']

        # get the total system size of compression radii
        nsys = int(self.steps * self.nbatch)
        compr_dat = np.fromfile(self.para['CompressionRData'], sep=' ')
        self.max_molecule_size = int(len(compr_dat) / nsys)

        # get optimized compression radii
        self.ml['optCompressionR'] = t.from_numpy(
            compr_dat[-int(self.max_molecule_size * self.nbatch):].reshape(
                self.nbatch, self.max_molecule_size))

    def fit_compression_r(self):
        """Fit compression radii and predict for new geometry."""
        interface.MLPara(self.para, self.dataset, self.ml)
        dscribe(self.para, self.dataset, self.ml)

        # the predicted compression radii
        self.para['compr_ml'] = self.para['compr_pred']

    def run_dftb(self):
        """Run DFTB calculations."""
        # get DFTB system information
        dftbcalculator.Initialization(
            self.para, self.dataset, self.skf, self.ml).initialize_dftb()

        # get SK parameters (integral tables)
        self.slako = GetSK_(self.para, self.dataset, self.skf, self.ml)

        # get nbatch * natom * natom * [ncompr, ncompr, 20] integrals
        self.skf['hs_compr_all_'] = []
        self.ml['CompressionRInit'] = []
        maxorb = max(self.dataset['norbital'])
        ham = t.zeros(self.nbatch, maxorb, maxorb)
        over = t.zeros(self.nbatch, maxorb, maxorb)

        # get all the integral
        for ibatch in range(self.nbatch):
            if self.ml['reference'] == 'hdf':
                natom = self.dataset['natomAll'][ibatch]
                print("self.nbatch", self.nbatch)
                # Get integral at certain distance, read integrals from hdf5
                self.slako.genskf_interp_dist_hdf(ibatch, natom)
                self.skf['hs_compr_all_'].append(self.skf['hs_compr_all'])

            # skf interpolation
            self.skf['hs_compr_all'] = self.skf['hs_compr_all_'][ibatch]
            self.slako.genskf_interp_compr(ibatch)

            # SK transformations
            SKTran(self.para, self.dataset, self.skf, self.ml, ibatch)
            iorb = self.dataset['norbital'][ibatch]
            ham[ibatch, :iorb, :iorb] = self.skf['hammat']
            over[ibatch, :iorb, :iorb] = self.skf['overmat']
        self.skf['hammat_'] = ham
        self.skf['overmat_'] = over
        dftbcalculator.Rundftbpy(self.para, self.dataset, self.skf, self.nbatch)

    def process_test_data(self):
        import matplotlib.pyplot as plt
        """Compare predicted results."""
        ref = pad1d(self.dataset['refDipole'])
        self.ml['referenceDataset'] = '../data/dataset/ani01_all_dftbplus.hdf5'
        LoadReferenceData(self.para, self.dataset, self.skf, self.ml)
        dftb = self.para['dipole']
        print('ref', ref.flatten())
        dftbplus = pad1d(self.dataset['refDipole'])
        plt.plot(ref, ref, 'k')
        plt.plot(ref, dftb, 'rx')
        plt.plot(ref, dftbplus, 'bv')
        plt.show()

