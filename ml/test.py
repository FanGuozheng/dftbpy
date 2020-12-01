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
from IO.save import Save1D, Save2D


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

        self.nbatch = self.dataset['nbatch']
        self.ntest = self.dataset['ntest']

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
        nsys = int(self.steps * self.dataset['nbatch'])
        compr_dat = np.fromfile(self.para['CompressionRData'], sep=' ')
        self.max_molecule_size = int(len(compr_dat) / nsys)

        # get optimized compression radii
        self.ml['optCompressionR'] = t.from_numpy(
            compr_dat[-int(self.max_molecule_size * self.nbatch):].reshape(
                self.nbatch, self.max_molecule_size))

    def fit_compression_r(self):
        """Fit compression radii and predict for new geometry."""
        interface.MLPara(self.para, self.dataset, self.ml)
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
        ham = t.zeros(self.ntest, maxorb, maxorb)
        over = t.zeros(self.ntest, maxorb, maxorb)

        # get all the integral
        for ibatch in range(self.ntest):
            if self.ml['reference'] == 'hdf':
                natom = self.dataset['natomAll'][ibatch]

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
        dftbcalculator.Rundftbpy(self.para, self.dataset, self.skf, self.ntest)

    def process_test_data(self):
        """Compare predicted results."""
        import matplotlib.pyplot as plt
        if 'dipole' in self.ml['target']:
            ref = pad1d(self.dataset['refDipole'])
            self.ml['referenceDataset'] = self.ml['referenceMioDataset']
            pred = self.para['dipole']
            LoadReferenceData(self.para, self.dataset, self.skf, self.ml)
            mio = pad1d(self.dataset['refDipole'])
            Save2D(ref.detach().numpy(), name='refdip.dat', dire='.', ty='w')
            Save2D(pred.detach().numpy(), name='preddip.dat', dire='.', ty='w')
            Save2D(mio.detach().numpy(), name='miodip.dat', dire='.', ty='w')
            print('difference ratio', t.abs(ref - pred).sum() / t.abs(ref - mio).sum())
        elif 'charge' in self.ml['target']:
            ref = pad1d(self.dataset['refCharge'])
            self.ml['referenceDataset'] = self.ml['referenceMioDataset']
            pred = self.para['fullCharge']
            LoadReferenceData(self.para, self.dataset, self.skf, self.ml)
            mio = pad1d(self.dataset['refCharge']) + self.para['fullCharge'] - self.para['charge']
            Save2D(ref.detach().numpy(), name='refcha.dat', dire='.', ty='w')
            Save2D(pred.detach().numpy(), name='predcha.dat', dire='.', ty='w')
            Save2D(mio.detach().numpy(), name='miocha.dat', dire='.', ty='w')
            print('difference ratio', t.abs(ref - pred).sum() / t.abs(ref - mio).sum())
        elif 'gap' in self.ml['target']:
            refhl = pad1d(self.dataset['refHOMOLUMO'])
            hl = self.para['homo_lumo']
            ref = refhl[:, 1] - refhl[:, 0]
            pred = hl[:, 1] - hl[:, 0]
            self.ml['referenceDataset'] = self.ml['referenceMioDataset']
            LoadReferenceData(self.para, self.dataset, self.skf, self.ml)
            miohl = pad1d(self.dataset['refHOMOLUMO'])
            print('miohl', miohl)
            mio = miohl[:, 1] - miohl[:, 0]
            Save1D(ref.detach().numpy(), name='refgap.dat', dire='.', ty='w')
            Save1D(pred.detach().numpy(), name='predgap.dat', dire='.', ty='w')
            Save1D(mio.detach().numpy(), name='miogap.dat', dire='.', ty='w')
            print('difference ratio', t.abs(ref - pred).sum() / t.abs(ref - mio).sum())
        elif 'HOMOLUMO' in self.ml['target']:
            ref = pad1d(self.dataset['refHOMOLUMO'])
            pred = self.para['homo_lumo']
            self.ml['referenceDataset'] = self.ml['referenceMioDataset']
            LoadReferenceData(self.para, self.dataset, self.skf, self.ml)
            mio = pad1d(self.dataset['refHOMOLUMO'])
            Save2D(ref.detach().numpy(), name='refhl.dat', dire='.', ty='w')
            Save2D(pred.detach().numpy(), name='predhl.dat', dire='.', ty='w')
            Save2D(mio.detach().numpy(), name='miohl.dat', dire='.', ty='w')
            print('difference ratio', t.abs(ref - pred).sum() / t.abs(ref - mio).sum())
        elif 'cpa' in self.ml['target']:
            ref = pad1d(self.dataset['refHirshfeldVolume'])
            pred = self.para['cpa']
            self.ml['referenceDataset'] = self.ml['referenceMioDataset']
            LoadReferenceData(self.para, self.dataset, self.skf, self.ml)
            mio = pad1d(self.dataset['refHOMOLUMO'])
            Save2D(ref.detach().numpy(), name='refcpa.dat', dire='.', ty='w')
            Save2D(pred.detach().numpy(), name='predcpa.dat', dire='.', ty='w')
            Save2D(mio.detach().numpy(), name='miocpa.dat', dire='.', ty='w')
            print('difference ratio', t.abs(ref - pred).sum() / t.abs(ref - mio).sum())
        plt.plot(ref, ref, 'k')
        plt.plot(ref, pred, 'rx')
        plt.plot(ref, mio, 'bv')
        plt.show()
