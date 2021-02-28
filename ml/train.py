import os
import numpy as np
from torch.autograd import Variable
import torch as t
import time
import tb.dftb.scc as dftbcalculator
from common.structures.system import System
from dftbtorch.sk import SKTran, GetSK_
import dftbtorch.initparams as initpara
from IO.loadhdf import LoadHdf
from IO.save import Save1D, Save2D
from common.batch import pack
import utils.plot as plot
ATOMNUM = {'H': 1, 'C': 6, 'N': 7, 'O': 8}
VAL_ORB = {"H": 1, "C": 2, "N": 2, "O": 2, "Ti": 3}
AIMS_ENERGY = {"H": -0.45891649, "C": -37.77330663, "N": -54.46973501,
               "O": -75.03140052}
DFTB_ENERGY = {"H": -0.238600544, "C": -1.398493891, "N": -2.0621839400,
               "O": -3.0861916005}


class DFTBMLTrain:
    """DFTB machine learning."""

    def __init__(self, parameter=None, skf=None, ml=None):
        """Initialize parameters."""
        time_begin = time.time()

        # initialize DFTB, dataset, sk parameters
        self.init = dftbcalculator.Initialization(parameter, skf, ml)
        print("self.parameter['task']", parameter['task'])

        # read input file if defined in command line
        self.init.initialize_parameter()
        self.parameter = self.init.parameter
        self.skf = self.init.skf
        self.ml = ml

        # detect automatically
        t.autograd.set_detect_anomaly(True)

        # set the print precision
        t.set_printoptions(precision=14)
        t.set_default_dtype(t.float64)

        # initialize machine learning parameters
        self.initialization_ml()

        numbers, positions, self.data = LoadHdf.load_reference(
            self.ml['referenceDataset'], 3, self.ml['target'])
        self.sys = System(numbers, positions)

        # run machine learning optimization
        self.run_ml()

        time_end = time.time()
        print('Total time:', time_end - time_begin)

    def initialization_ml(self):
        """Initialize machine learning parameters."""
        # remove some documents
        os.system('rm *.dat')

        self.parameter, self.skf, self.ml = \
            initpara.init_ml(self.parameter, self.skf, self.ml)

    def run_ml(self):
        """Run machine learning optimization."""
        # optimize integrals directly
        if self.parameter['task'] == 'mlIntegral':
            MLIntegral(self.parameter, self.skf, self.ml, self.sys, self.data)

        # optimize compression radius and then generate integrals
        elif self.parameter['task'] == 'mlCompressionR':
            MLCompressionR(self.parameter, self.skf, self.ml, self.sys, self.data)


class MLIntegral:
    """Optimize integrals."""

    def __init__(self, parameter, skf, ml, sys, ref):
        """Initialize parameters."""
        self.para = parameter
        self.skf = skf
        self.ml = ml
        self.sys = sys
        self.ref = ref

        # get natom * natom * [ncompr, ncompr, 20] for interpolation DFTB
        self.skf['hs_compr_all_'] = []
        self.para['compr_init_'] = []

        # get spline integral
        self.slako = GetSK_(self.para, self.skf, self.ml, self.sys)
        self.ml_variable = self.slako.integral_spline_parameter()  # 0.2 s, too long!!!

        # get loss function type
        if self.ml['lossFunction'] == 'MSELoss':
            self.criterion = t.nn.MSELoss(reduction='sum')
        elif self.ml['lossFunction'] == 'L1Loss':
            self.criterion = t.nn.L1Loss(reduction='sum')

        # get optimizer
        if self.ml['optimizer'] == 'SCG':
            self.optimizer = t.optim.SGD(self.ml_variable, lr=self.ml['lr'])
        elif self.ml['optimizer'] == 'Adam':
            self.optimizer = t.optim.Adam(self.ml_variable, lr=self.ml['lr'])

        # total batch size
        self.nbatch = self.sys.size_batch
        self.ml_integral_batch()

    def ml_integral_batch(self):
        '''DFTB optimization for given dataset'''
        # calculate one by one to optimize para
        self.sktran = SKTran(self.para, self.skf, self.ml, self.sys)
        self.para['loss'] = []
        for istep in range(self.ml['mlSteps']):
            ham = t.zeros(self.sys.hs_shape)
            over = t.zeros(self.sys.hs_shape)
            for ibatch in range(self.nbatch):
                iham, iover = self.sktran(ibatch)
                iorb = iham.shape[-1]
                ham[ibatch, :iorb, :iorb] = iham
                over[ibatch, :iorb, :iorb] = iover
            self.sktran.save_spl_param()
            self.skf['hammat_'] = ham
            self.skf['overmat_'] = over

            # run each DFTB calculation separatedly
            dftbcalculator.Rundftbpy(self.para, self.skf, self.sys, self.nbatch)

            loss = 0.
            if 'dipole' in self.ml['target']:
                loss += self.criterion(self.para['dipole'],
                                       pack(self.ref['dipole']))
                self.para['loss'].append(loss.detach())

            if 'charge' in self.ml['target']:
                loss += self.criterion(self.para['charge'],
                                       pack(self.ref['charge']))
            if 'HOMOLUMO' in self.ml['target']:
                loss += self.criterion(self.para['homo_lumo'],
                                       pack(self.ref['refHOMOLUMO']))
            if 'gap' in self.ml['target']:
                homolumo = self.para['homo_lumo']
                refhl = pack(self.ref['refHOMOLUMO'])
                gap = homolumo[:, 1] - homolumo[:, 0]
                refgap = refhl[:, 1] - refhl[:, 0]
                loss += self.criterion(gap, refgap)
            if 'polarizability' in self.ml['target']:
                loss += self.criterion(self.para['alpha_mbd'],
                                       pack(self.ref['refMBDAlpha']))
            print("step:", istep + 1, "loss:", loss, loss.device.type)

            # clear gradients and define back propagation
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()

            # check and save machine learning parameters
        import matplotlib.pyplot as plt
        steps = len(self.para['loss'])
        plt.plot(np.linspace(1, steps, steps), self.para['loss'])
        plt.show()


class MLCompressionR:
    """Optimize compression radii."""

    def __init__(self, para, skf, ml, sys, ref):
        """Initialize parameters."""
        self.para = para
        self.skf = skf
        self.ml = ml
        self.sys = sys
        self.ref = ref

        # batch size
        self.nbatch = self.sys.size_batch

        self.process_dataset()

        # get optimizer
        if self.ml['optimizer'] == 'SCG':
            self.optimizer = t.optim.SGD([self.para['compr_ml']], lr=self.ml['lr'])
        elif self.ml['optimizer'] == 'Adam':
            self.optimizer = t.optim.Adam([self.para['compr_ml']], lr=self.ml['lr'])

        # get loss function type
        if self.ml['lossFunction'] == 'MSELoss':
            self.criterion = t.nn.MSELoss(reduction='sum')
        elif self.ml['lossFunction'] == 'L1Loss':
            self.criterion = t.nn.L1Loss(reduction='sum')

        # batch calculations, all systems in batch together
        self.ml_compr_batch()

    def process_dataset(self):
        """Get all parameters for compression radii machine learning."""
        # get SK parameters (integral tables)
        self.slako = GetSK_(self.para, self.skf, self.ml, self.sys)
        # get nbatch * natom * natom * [ncompr, ncompr, 20] integrals
        self.skf['hs_compr_all_'] = []

        # loop in batch
        for ibatch in range(self.nbatch):
            natom = self.sys.size_system[ibatch]
            # Get integral at certain distance, read integrals from hdf5
            self.slako.genskf_interp_dist_hdf(ibatch, natom)
            self.skf['hs_compr_all_'].append(self.skf['hs_compr_all'])

        self.para['compr_ml'] = Variable(t.ones(
            self.nbatch, max(self.sys.size_system)) * 3.5, requires_grad=True)

    def ml_compr_batch(self):
        """DFTB optimization of compression radius for given dataset."""
        self.para['loss'] = []

        for istep in range(self.ml['mlSteps']):
            ham = t.zeros(self.sys.hs_shape)  # t.zeros(self.nbatch, maxorb, maxorb)
            over = t.zeros(self.sys.hs_shape)
            self.sktran = SKTran(self.para, self.skf, self.ml, self.sys)

            for ibatch in range(self.nbatch):
                self.skf['hs_compr_all'] = self.skf['hs_compr_all_'][ibatch]
                self.slako.genskf_interp_compr(ibatch)

                # SK transformations
                iham, iover = self.sktran(ibatch)
                iorb = iham.shape[-1]
                ham[ibatch, :iorb, :iorb] = iham
                over[ibatch, :iorb, :iorb] = iover
            self.skf['hammat_'] = ham
            self.skf['overmat_'] = over
            dftbcalculator.Rundftbpy(self.para, self.skf, self.sys, self.nbatch)

            # get loss function
            loss = 0.
            if 'dipole' in self.ml['target']:
                lossdip = self.criterion(self.para['dipole'],
                                         pack(self.ref['dipole']))
                loss = loss + lossdip
            if 'HOMOLUMO' in self.ml['target']:
                losshl = self.criterion(self.para['homo_lumo'],
                                       pack(self.ref['refHOMOLUMO']))
                loss = loss + losshl
            if 'gap' in self.ml['target']:
                homolumo = self.para['homo_lumo']
                refhl = pack(self.ref['refHOMOLUMO'])
                gap = homolumo[:, 1] - homolumo[:, 0]
                refgap = refhl[:, 1] - refhl[:, 0]
                lossgap = 0.001 * self.criterion(gap, refgap)
                loss = loss + lossgap
            if 'polarizability' in self.ml['target']:
                losspol = self.criterion(self.para['alpha_mbd'],
                                       pack(self.ref['refMBDAlpha']))
                loss = loss + losspol
            if 'charge' in self.ml['target']:
                lossq = self.criterion(self.para['charge'],
                                       pack(self.ref['charge']))
                loss = loss + lossq
            if 'cpa' in self.ml['target']:
                losscpa = 0.5 * self.criterion(
                    self.para['cpa'], pack(self.ref['refHirshfeldVolume']))
                loss = loss + losscpa
            if 'pdos' in self.ml['target']:
                loss += self.criterion(
                    self.para['cpa'], pack(self.ref['refHirshfeldVolume']))
            self.para['loss'].append(loss.detach())
            print("istep:", istep, '\n loss', loss, 'loss device', loss.device.type)
            print('gradient', self.para['compr_ml'].grad)
            print('compression radii', self.para['compr_ml'])

            # save data
            Save1D(np.array([self.para['loss']]), name='loss.dat', dire='.', ty='a')
            if not self.ml['globalCompR']:
                Save2D(self.para['compr_ml'].detach().numpy(),
                       name='compr.dat', dire='.', ty='a')
            else:
                Save1D(self.para['compr_ml'].detach().cpu().squeeze().numpy(),
                       name='compr.dat', dire='.', ty='a')

            # clear gradients and define back propagation
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()

            # check and save machine learning variables
            self._check()

        import matplotlib.pyplot as plt
        steps = len(self.para['loss'])
        plt.plot(np.linspace(1, steps, steps), self.para['loss'])
        plt.show()

    def _check(self):
        """Check the machine learning variables each step.

        When training compression radii, sometimes the compression radii will
        be out of range of given grid points and go randomly, therefore here
        the code makes sure the compression radii is in the defined range.
        """
        # detach remove initial graph and make sure compr_ml is leaf tensor
        compr_ml = self.para['compr_ml'].detach().clone()
        min_mask = compr_ml[compr_ml != 0].lt(self.ml['compressionRMin'])
        max_mask = compr_ml[compr_ml != 0].gt(self.ml['compressionRMax'])
        if True in min_mask or True in max_mask:
            with t.no_grad():
                self.para['compr_ml'].clamp_(self.ml['compressionRMin'],
                                             self.ml['compressionRMax'])
