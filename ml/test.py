import time
import os
import torch as t
import numpy as np
import h5py
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

    def __init__(self, para, dataset, skf, ml):
        self.para, self.dataset, self.skf, self.ml = para, dataset, skf, ml

        self.nbatch = self.dataset['nbatch']
        self.ntest = self.dataset['ntest']

        # load data, get optimized compression radii
        self.load_data()

        # run DFRB calculations
        self.run_dftb()

        # process ML prediction DFTB and mio DFTB results
        self.process_test_data()

    def load_data(self):
        """Load optimized compression radii."""
        self.steps = self.ml['mlSteps']

        # get the total system size of compression radii
        nsys = int(self.steps * self.dataset['nbatch'])

        # check if skf dataset exists
        if not os.path.isfile(self.para['datasetSK']):
            raise FileNotFoundError('%s not found' % self.para['datasetSK'])

        self.skf['hs_compr_all'] = []
        with h5py.File(self.para['datasetSK'], 'r') as f:
            for ispecie in self.dataset['specieGlobal']:
                for jspecie in self.dataset['specieGlobal']:
                    nameij = ispecie + jspecie
                    grid_distance = f[nameij + '/grid_dist'][()]
                    ngrid = f[nameij + '/ngridpoint'][()]
                    yy = t.from_numpy(f[nameij + '/hs_all'][()])
                    xx = t.arange(0., ngrid * grid_distance, grid_distance, dtype=yy.dtype)
                    self.skf['polySplinex' + nameij] = xx
                    spla = np.fromfile(os.path.join(self.dataset['datasetSpline'],
                                                    nameij + 'spl_a.dat'), sep=' ')
                    splb = np.fromfile(os.path.join(self.dataset['datasetSpline'],
                                                    nameij + 'spl_b.dat'), sep=' ')
                    splc = np.fromfile(os.path.join(self.dataset['datasetSpline'],
                                                    nameij + 'spl_c.dat'), sep=' ')
                    spld = np.fromfile(os.path.join(self.dataset['datasetSpline'],
                                                    nameij + 'spl_d.dat'), sep=' ')
                    lena = int(spla.shape[0] / self.steps)
                    lenb = int(splb.shape[0] / self.steps)
                    lenc = int(splc.shape[0] / self.steps)
                    lend = int(spld.shape[0] / self.steps)
                    self.skf['polySplinea' + nameij] = t.from_numpy(
                        spla[-lena:].reshape(int(lena / 20), 20))
                    self.skf['polySplineb' + nameij] = t.from_numpy(
                        splb[-lenb:].reshape(int(lenb / 20), 20))
                    self.skf['polySplinec' + nameij] = t.from_numpy(
                        splc[-lenc:].reshape(int(lenc / 20), 20))
                    self.skf['polySplined' + nameij] = t.from_numpy(
                        spld[-lend:].reshape(int(lend / 20), 20))

    def run_dftb(self):
        '''DFTB optimization for given dataset'''
        # get DFTB system information
        dftbcalculator.Initialization(
            self.para, self.dataset, self.skf, self.ml).initialize_dftb()

        maxorb = max(self.dataset['norbital'])
        # calculate one by one to optimize para
        self.sktran = SKTran(self.para, self.dataset, self.skf, self.ml)
        ham = t.zeros(self.ntest, maxorb, maxorb)
        over = t.zeros(self.ntest, maxorb, maxorb)
        for ibatch in range(self.ntest):
            # get integral at certain distance, read raw integral from binary hdf
            # SK transformations
            # SKTran(self.para, self.dataset, self.skf, self.ml, ibatch)
            iham, iover = self.sktran(ibatch)
            iorb = self.dataset['norbital'][ibatch]
            ham[ibatch, :iorb, :iorb] = iham  # self.skf['hammat']
            over[ibatch, :iorb, :iorb] = iover  # self.skf['overmat']
        # self.sktran.save_spl_param()
        self.skf['hammat_'] = ham
        self.skf['overmat_'] = over

        # run each DFTB calculation separatedly
        dftbcalculator.Rundftbpy(self.para, self.dataset, self.skf, self.ntest)

    def process_test_data(self):
        """Compare predicted results."""
        refdip = pad1d(self.dataset['refDipole'])
        refhl = pad1d(self.dataset['refHOMOLUMO'])
        refcha = pad1d(self.dataset['refCharge'])
        refcpa = pad1d(self.dataset['refHirshfeldVolume'])
        if 'dipole' in self.ml['target']:
            self.ml['referenceDataset'] = self.ml['referenceMioDataset']
            pred = self.para['dipole']
            LoadReferenceData(self.para, self.dataset, self.skf, self.ml)
            mio = pad1d(self.dataset['refDipole'])
            Save2D(refdip.detach().numpy(), name='refdip.dat', dire='.', ty='w')
            Save2D(pred.detach().numpy(), name='preddip.dat', dire='.', ty='w')
            Save2D(mio.detach().numpy(), name='miodip.dat', dire='.', ty='w')
            print('difference ratio', t.abs(refdip - pred).sum() / t.abs(refdip - mio).sum())
        if 'charge' in self.ml['target']:
            self.ml['referenceDataset'] = self.ml['referenceMioDataset']
            pred = self.para['fullCharge']
            LoadReferenceData(self.para, self.dataset, self.skf, self.ml)
            mio = pad1d(self.dataset['refCharge']) + self.para['fullCharge'] - self.para['charge']
            Save2D(refcha.detach().numpy(), name='refcha.dat', dire='.', ty='w')
            Save2D(pred.detach().numpy(), name='predcha.dat', dire='.', ty='w')
            Save2D(mio.detach().numpy(), name='miocha.dat', dire='.', ty='w')
            print('difference ratio', t.abs(refcha - pred).sum() / t.abs(refcha - mio).sum())
        if 'gap' in self.ml['target']:
            hl = self.para['homo_lumo']
            ref = refhl[:, 1] - refhl[:, 0]
            pred = hl[:, 1] - hl[:, 0]
            self.ml['referenceDataset'] = self.ml['referenceMioDataset']
            LoadReferenceData(self.para, self.dataset, self.skf, self.ml)
            print("self.dataset['refHOMOLUMO']", self.dataset['refHOMOLUMO'])
            miohl = pad1d(self.dataset['refHOMOLUMO'])
            mio = miohl[:, 1] - miohl[:, 0]
            Save1D(ref.detach().numpy(), name='refgap.dat', dire='.', ty='w')
            Save1D(pred.detach().numpy(), name='predgap.dat', dire='.', ty='w')
            Save1D(mio.detach().numpy(), name='miogap.dat', dire='.', ty='w')
            print('difference ratio', t.abs(ref - pred).sum() / t.abs(ref - mio).sum())
        if 'HOMOLUMO' in self.ml['target']:
            pred = self.para['homo_lumo']
            self.ml['referenceDataset'] = self.ml['referenceMioDataset']
            LoadReferenceData(self.para, self.dataset, self.skf, self.ml)
            mio = pad1d(self.dataset['refHOMOLUMO'])
            Save2D(refhl.detach().numpy(), name='refhl.dat', dire='.', ty='w')
            Save2D(pred.detach().numpy(), name='predhl.dat', dire='.', ty='w')
            Save2D(mio.detach().numpy(), name='miohl.dat', dire='.', ty='w')
            print('difference ratio', t.abs(refhl - pred).sum() / t.abs(refhl - mio).sum())
        if 'cpa' in self.ml['target']:
            pred = self.para['cpa']
            self.ml['referenceDataset'] = self.ml['referenceMioDataset']
            LoadReferenceData(self.para, self.dataset, self.skf, self.ml)
            mio = pad1d(self.dataset['refCPA'])
            Save2D(refcpa.detach().numpy(), name='refcpa.dat', dire='.', ty='w')
            Save2D(pred.detach().numpy(), name='predcpa.dat', dire='.', ty='w')
            Save2D(mio.detach().numpy(), name='miocpa.dat', dire='.', ty='w')
            print('difference ratio', t.abs(refcpa - pred).sum() / t.abs(refcpa - mio).sum())


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

        # get optimized compression radii
        if not self.ml['globalCompR']:
            self.max_molecule_size = int(len(compr_dat) / nsys)
            self.ml['optCompressionR'] = t.from_numpy(
                compr_dat[-int(self.max_molecule_size * self.nbatch):].reshape(
                    self.nbatch, self.max_molecule_size))
        elif self.ml['globalCompR']:
            self.max_molecule_size = int(len(compr_dat) / self.steps)
            optr = t.from_numpy(compr_dat[-self.max_molecule_size:])
            sglo =  list(self.dataset['specieGlobal'])
            self.ml['optCompressionR'] = pad1d([t.tensor([optr[sglo.index(ii)] for ii in isym])
                                                for isym in self.dataset['symbols'][:self.nbatch]])

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
        self.sktran = SKTran(self.para, self.dataset, self.skf, self.ml)
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
            # SKTran(self.para, self.dataset, self.skf, self.ml, ibatch)
            iham, iover = self.sktran(ibatch)
            iorb = self.dataset['norbital'][ibatch]
            ham[ibatch, :iorb, :iorb] = iham  # self.skf['hammat']
            over[ibatch, :iorb, :iorb] = iover # self.skf['overmat']
        self.skf['hammat_'] = ham
        self.skf['overmat_'] = over
        dftbcalculator.Rundftbpy(self.para, self.dataset, self.skf, self.ntest)

    def process_test_data(self):
        """Compare predicted results."""
        refdip = pad1d(self.dataset['refDipole'])
        refhl = pad1d(self.dataset['refHOMOLUMO'])
        refcha = pad1d(self.dataset['refCharge'])
        refcpa = pad1d(self.dataset['refHirshfeldVolume'])
        if 'dipole' in self.ml['target']:
            self.ml['referenceDataset'] = self.ml['referenceMioDataset']
            pred = self.para['dipole']
            LoadReferenceData(self.para, self.dataset, self.skf, self.ml)
            mio = pad1d(self.dataset['refDipole'])
            Save2D(refdip.detach().numpy(), name='refdip.dat', dire='.', ty='w')
            Save2D(pred.detach().numpy(), name='preddip.dat', dire='.', ty='w')
            Save2D(mio.detach().numpy(), name='miodip.dat', dire='.', ty='w')
            print('difference ratio', t.abs(refdip - pred).sum() / t.abs(refdip - mio).sum())
        if 'charge' in self.ml['target']:
            self.ml['referenceDataset'] = self.ml['referenceMioDataset']
            pred = self.para['fullCharge']
            LoadReferenceData(self.para, self.dataset, self.skf, self.ml)
            mio = pad1d(self.dataset['refCharge']) + self.para['fullCharge'] - self.para['charge']
            Save2D(refcha.detach().numpy(), name='refcha.dat', dire='.', ty='w')
            Save2D(pred.detach().numpy(), name='predcha.dat', dire='.', ty='w')
            Save2D(mio.detach().numpy(), name='miocha.dat', dire='.', ty='w')
            print('difference ratio', t.abs(refcha - pred).sum() / t.abs(refcha - mio).sum())
        if 'gap' in self.ml['target']:
            hl = self.para['homo_lumo']
            ref = refhl[:, 1] - refhl[:, 0]
            pred = hl[:, 1] - hl[:, 0]
            self.ml['referenceDataset'] = self.ml['referenceMioDataset']
            LoadReferenceData(self.para, self.dataset, self.skf, self.ml)
            print("self.dataset['refHOMOLUMO']", self.dataset['refHOMOLUMO'])
            miohl = pad1d(self.dataset['refHOMOLUMO'])
            mio = miohl[:, 1] - miohl[:, 0]
            Save1D(ref.detach().numpy(), name='refgap.dat', dire='.', ty='w')
            Save1D(pred.detach().numpy(), name='predgap.dat', dire='.', ty='w')
            Save1D(mio.detach().numpy(), name='miogap.dat', dire='.', ty='w')
            print('difference ratio', t.abs(ref - pred).sum() / t.abs(ref - mio).sum())
        if 'HOMOLUMO' in self.ml['target']:
            pred = self.para['homo_lumo']
            self.ml['referenceDataset'] = self.ml['referenceMioDataset']
            LoadReferenceData(self.para, self.dataset, self.skf, self.ml)
            mio = pad1d(self.dataset['refHOMOLUMO'])
            Save2D(refhl.detach().numpy(), name='refhl.dat', dire='.', ty='w')
            Save2D(pred.detach().numpy(), name='predhl.dat', dire='.', ty='w')
            Save2D(mio.detach().numpy(), name='miohl.dat', dire='.', ty='w')
            print('difference ratio', t.abs(refhl - pred).sum() / t.abs(refhl - mio).sum())
        if 'cpa' in self.ml['target']:
            pred = self.para['cpa']
            self.ml['referenceDataset'] = self.ml['referenceMioDataset']
            LoadReferenceData(self.para, self.dataset, self.skf, self.ml)
            mio = pad1d(self.dataset['refCPA'])
            Save2D(refcpa.detach().numpy(), name='refcpa.dat', dire='.', ty='w')
            Save2D(pred.detach().numpy(), name='predcpa.dat', dire='.', ty='w')
            Save2D(mio.detach().numpy(), name='miocpa.dat', dire='.', ty='w')
            print('difference ratio', t.abs(refcpa - pred).sum() / t.abs(refcpa - mio).sum())
