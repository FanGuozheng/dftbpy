import os
import numpy as np
from torch.autograd import Variable
import torch as t
import time
import dftbtorch.dftbcalculator as dftbcalculator
from dftbtorch.sk import SKTran, GetSKTable, GetSK_
import dftbtorch.initparams as initpara
from ml.padding import pad1d
from IO.dataloader import LoadData, LoadReferenceData
from IO.save import Save1D, Save2D
from utils.runcalculation import RunReference
from ml.padding import pad1d, pad2d
import utils.plot as plot
ATOMNUM = {'H': 1, 'C': 6, 'N': 7, 'O': 8}
VAL_ORB = {"H": 1, "C": 2, "N": 2, "O": 2, "Ti": 3}
AIMS_ENERGY = {"H": -0.45891649, "C": -37.77330663, "N": -54.46973501,
               "O": -75.03140052}
DFTB_ENERGY = {"H": -0.238600544, "C": -1.398493891, "N": -2.0621839400,
               "O": -3.0861916005}


class DFTBMLTrain:
    """DFTB machine learning."""

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

        # set the print precision
        t.set_printoptions(precision=14)

        # set the precision control
        if self.parameter['device'] == 'cpu':
            if self.parameter['precision'] in (t.float64, t.float32):
                t.set_default_dtype(self.parameter['precision'])  # cpu device
            else:
                raise ValueError('device is cpu, please select float64 or float32')
        elif self.parameter['device'] == 'cuda':
            if self.parameter['precision'] in (t.cuda.DoubleTensor, t.cuda.FloatTensor):
                t.set_default_tensor_type(self.parameter['precision'])  # gpu device
            else:
                raise ValueError('device is cuda, please select cuda.FloatTensor or cuda.DoubleTensor')
        else:
            raise ValueError('device only support cpu and cuda')

        # initialize machine learning parameters
        self.initialization_ml()

        # 1. read dataset then run DFT(B) to get reference data
        # 2. directly get reference data from reading input dataset
        self.get_reference()

        # load SKF dataset
        self.load_skf()

        # run machine learning optimization
        self.run_ml()

        # plot ML results
        # plot.plot_ml(self.parameter, self.ml)
        time_end = time.time()
        print('Total time:', time_end - time_begin)

    def initialization_ml(self):
        """Initialize machine learning parameters."""
        # remove some documents
        os.system('rm *.dat')

        self.parameter, self.dataset, self.skf, self.ml = \
            initpara.init_ml(self.parameter, self.dataset, self.skf, self.ml)

    def get_reference(self):
        """Load reference dataset for machine learning."""
        # run DFT(B) to get reference data
        if self.ml['runReference']:
            LoadData(self.parameter, self.dataset, self.ml)
            RunReference(self.parameter, self.dataset, self.skf, self.ml)

        # directly get reference data from dataset
        else:
            LoadReferenceData(self.parameter, self.dataset, self.skf, self.ml)

    def load_skf(self):
        pass

    def run_ml(self):
        """Run machine learning optimization."""
        # optimize integrals directly
        if self.parameter['task'] == 'mlIntegral':
            MLIntegral(self.parameter, self.dataset, self.skf, self.ml)

        # optimize compression radius and then generate integrals
        elif self.parameter['task'] == 'mlCompressionR':
            MLCompressionR(self.parameter, self.dataset, self.skf, self.ml)


class MLIntegral:
    """Optimize integrals."""

    def __init__(self, parameter, dataset, skf, ml):
        """Initialize parameters."""
        self.para = parameter
        self.dataset = dataset
        self.skf = skf
        self.ml = ml

        # get the ith coordinates
        get_coor(self.dataset, self.para['precision'])

        # initialize DFTB calculations with datasetmetry and input parameters
        # read skf according to global atom species
        dftbcalculator.Initialization(
            self.para, self.dataset, self.skf).initialize_dftb()

        # get natom * natom * [ncompr, ncompr, 20] for interpolation DFTB
        self.skf['hs_compr_all_'] = []
        self.para['compr_init_'] = []

        # get spline integral
        self.slako = GetSK_(self.para, self.dataset, self.skf, self.ml)
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
        self.nbatch = self.dataset['nfile']
        if self.para['Lbatch']:
            self.ml_integral_batch()
        else:
            self.ml_integral()

    def ml_integral(self):
        '''DFTB optimization for given dataset'''
        # calculate one by one to optimize para
        for istep in range(self.ml['mlSteps']):
            loss = 0.
            for ibatch in range(self.nbatch):
                print("step:", istep + 1, "ibatch:", ibatch + 1)

                # do not perform batch calculation
                self.para['Lbatch'] = False

                # get integral at certain distance, read raw integral from binary hdf
                # SK transformations
                SKTran(self.para, self.dataset, self.skf, self.ml, ibatch)

                # run each DFTB calculation separatedly
                dftbcalculator.Rundftbpy(self.para, self.dataset, self.skf, ibatch)

                # define loss function
                # get loss function
                if 'dipole' in self.ml['target']:
                    loss += self.criterion(self.para['dipole'].squeeze(), self.dataset['refDipole'][ibatch])

                # clear gradients and define back propagation
                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                self.optimizer.step()
            Save1D(np.array([loss]), name='loss.dat', dire='.', ty='a')

    def ml_integral_batch(self):
        '''DFTB optimization for given dataset'''
        maxorb = max(self.dataset['norbital'])
        # calculate one by one to optimize para
        self.sktran = SKTran(self.para, self.dataset, self.skf, self.ml)
        for istep in range(self.ml['mlSteps']):
            ham = t.zeros(self.nbatch, maxorb, maxorb)
            over = t.zeros(self.nbatch, maxorb, maxorb)
            for ibatch in range(self.nbatch):
                # get integral at certain distance, read raw integral from binary hdf
                # SK transformations
                # SKTran(self.para, self.dataset, self.skf, self.ml, ibatch)
                iham, iover = self.sktran(ibatch)
                iorb = self.dataset['norbital'][ibatch]
                ham[ibatch, :iorb, :iorb] = iham  # self.skf['hammat']
                over[ibatch, :iorb, :iorb] = iover  # self.skf['overmat']
            self.sktran.save_spl_param()
            self.skf['hammat_'] = ham
            self.skf['overmat_'] = over

            # run each DFTB calculation separatedly
            dftbcalculator.Rundftbpy(self.para, self.dataset, self.skf, self.nbatch)

            loss = 0.
            if 'dipole' in self.ml['target']:
                loss += self.criterion(self.para['dipole'],
                                       pad1d(self.dataset['refDipole']))
                if self.para['dipole'].device.type == 'cuda':
                    Save2D(self.para['dipole'].detach().cpu().numpy(),
                           name='dipole.dat', dire='.', ty='a')
                elif self.para['dipole'].device.type == 'cpu':
                    Save2D(self.para['dipole'].detach().numpy(),
                           name='dipole.dat', dire='.', ty='a')
            elif 'HOMOLUMO' in self.ml['target']:
                loss += self.criterion(self.para['homo_lumo'],
                                       pad1d(self.dataset['refHOMOLUMO']))
            elif 'gap' in self.ml['target']:
                homolumo = self.para['homo_lumo']
                refhl = pad1d(self.dataset['refHOMOLUMO'])
                gap = homolumo[:, 1] - homolumo[:, 0]
                refgap = refhl[:, 1] - refhl[:, 0]
                loss += self.criterion(gap, refgap)
            elif 'polarizability' in self.ml['target']:
                loss += self.criterion(self.para['alpha_mbd'],
                                       pad1d(self.dataset['refMBDAlpha']))
            print("step:", istep + 1, "loss:", loss, loss.device.type)
            # for ii, ivar in enumerate(self.ml_variable):
            #     Save2D(ivar.detach().numpy(), name=str(ii)+'var.dat', dire='.', ty='a')

            # clear gradients and define back propagation
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()

            # check and save machine learning parameters
            self._check()
            Save1D(np.array([loss]), name='loss.dat', dire='.', ty='a')

    def _check(self):
        """Check the machine learning variables each step."""
        pass

class MLCompressionR:
    """Optimize compression radii."""

    def __init__(self, para, dataset, skf, ml):
        """Initialize parameters."""
        self.para = para
        self.dataset = dataset
        self.skf = skf
        self.ml = ml

        # batch size
        self.nbatch = self.dataset['nfile']

        # process dataset for machine learning
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
        if self.para['Lbatch']:
            self.ml_compr_batch()

        # single calculations, run optimization system by system in batch
        elif not self.para['Lbatch']:
            self.ml_compr_single()

    def process_dataset(self):
        """Get all parameters for compression radii machine learning."""
        # deal with coordinates type
        get_coor(self.dataset, self.para['precision'])

        # get DFTB system information
        dftbcalculator.Initialization(
            self.para, self.dataset, self.skf, self.ml).initialize_dftb()

        # get SK parameters (integral tables)
        self.slako = GetSK_(self.para, self.dataset, self.skf, self.ml)

        # get nbatch * natom * natom * [ncompr, ncompr, 20] integrals
        self.skf['hs_compr_all_'] = []
        self.ml['CompressionRInit'] = []

        # loop in batch
        for ibatch in range(self.nbatch):
        # if self.ml['reference'] == 'hdf':
            natom = self.dataset['natomAll'][ibatch]
            atomname = self.dataset['symbols'][ibatch]

            # Get integral at certain distance, read integrals from hdf5
            self.slako.genskf_interp_dist_hdf(ibatch, natom)
            self.skf['hs_compr_all_'].append(self.skf['hs_compr_all'])

            if not self.ml['globalCompR']:
                # get initial compression radius
                self.ml['CompressionRInit'].append(
                    genml_init_compr(atomname, self.ml))

        if self.ml['globalCompR']:
            self.ml['CompressionRInit'] = [genml_init_compr(
                ispe, self.ml, self.ml['globalCompR']) for ispe in self.dataset['specieGlobal']]

        self.para['compr_ml'] = \
            Variable(pad1d(self.ml['CompressionRInit']), requires_grad=True)

    def ml_compr_batch(self):
        """DFTB optimization of compression radius for given dataset."""
        maxorb = max(self.dataset['norbital'])

        for istep in range(self.ml['mlSteps']):
            ham = t.zeros(self.nbatch, maxorb, maxorb)
            over = t.zeros(self.nbatch, maxorb, maxorb)
            self.sktran = SKTran(self.para, self.dataset, self.skf, self.ml)

            for ibatch in range(self.nbatch):
                self.skf['hs_compr_all'] = self.skf['hs_compr_all_'][ibatch]
                self.slako.genskf_interp_compr(ibatch)

                # SK transformations
                # SKTran(self.para, self.dataset, self.skf, self.ml, ibatch)
                iham, iover = self.sktran(ibatch)
                iorb = self.dataset['norbital'][ibatch]
                ham[ibatch, :iorb, :iorb] = iham  # self.skf['hammat']
                over[ibatch, :iorb, :iorb] = iover  # self.skf['overmat']
            self.skf['hammat_'] = ham
            self.skf['overmat_'] = over
            dftbcalculator.Rundftbpy(self.para, self.dataset, self.skf, self.nbatch)

            # dftb formation energy calculations
            self.para['formation_energy'] = get_formation_energy(
                self.para['electronic_energy'],
                self.dataset['symbols'], ibatch)

            # get loss function
            loss = 0.
            if 'dipole' in self.ml['target']:
                lossdip = self.criterion(self.para['dipole'],
                                         pad1d(self.dataset['refDipole']))
                loss = loss + lossdip
                if self.para['dipole'].device.type == 'cuda':
                    Save2D(self.para['dipole'].detach().cpu().numpy(),
                           name='dipole.dat', dire='.', ty='a')
                    Save1D(np.array([lossdip.cpu()]), name='lossdip.dat', dire='.', ty='a')
                elif self.para['dipole'].device.type == 'cpu':
                    Save2D(self.para['dipole'].detach().numpy(),
                           name='dipole.dat', dire='.', ty='a')
                    Save1D(np.array([lossdip]), name='lossdip.dat', dire='.', ty='a')
            if 'HOMOLUMO' in self.ml['target']:
                losshl = self.criterion(self.para['homo_lumo'],
                                       pad1d(self.dataset['refHOMOLUMO']))
                loss = loss + losshl
                if self.para['homo_lumo'].device.type == 'cuda':
                    Save2D(self.para['homo_lumo'].detach().cpu().numpy(),
                           name='homolumo.dat', dire='.', ty='a')
                    Save1D(np.array([losshl.cpu()]), name='losshl.dat', dire='.', ty='a')
                elif self.para['homo_lumo'].device.type == 'cpu':
                    Save2D(self.para['homo_lumo'].detach().numpy(),
                           name='homolumo.dat', dire='.', ty='a')
                    Save1D(np.array([losshl]), name='losshl.dat', dire='.', ty='a')
            if 'gap' in self.ml['target']:
                homolumo = self.para['homo_lumo']
                refhl = pad1d(self.dataset['refHOMOLUMO'])
                gap = homolumo[:, 1] - homolumo[:, 0]
                refgap = refhl[:, 1] - refhl[:, 0]
                lossgap = 0.001 * self.criterion(gap, refgap)
                loss = loss + lossgap
                if gap.device.type == 'cuda':
                    Save1D(gap.detach().cpu().numpy(), name='gap.dat', dire='.', ty='a')
                    Save1D(np.array([lossgap.cpu()]), name='lossgap.dat', dire='.', ty='a')
                elif gap.device.type == 'cpu':
                    Save1D(gap.detach().numpy(), name='gap.dat', dire='.', ty='a')
                    Save1D(np.array([lossgap]), name='lossgap.dat', dire='.', ty='a')
            if 'polarizability' in self.ml['target']:
                losspol = self.criterion(self.para['alpha_mbd'],
                                       pad2d(self.dataset['refMBDAlpha']))
                loss = loss + losspol
                if self.para['alpha_mbd'].device.type == 'cuda':
                    Save2D(self.para['alpha_mbd'].detach().cpu().numpy(),
                           name='pol.dat', dire='.', ty='a')
                    Save1D(np.array([losspol.cpu()]), name='losspol.dat', dire='.', ty='a')
                elif self.para['alpha_mbd'].device.type == 'cpu':
                    Save2D(self.para['alpha_mbd'].detach().numpy(),
                           name='pol.dat', dire='.', ty='a')
                    Save1D(np.array([losspol]), name='losspol.dat', dire='.', ty='a')
            if 'charge' in self.ml['target']:
                lossq = self.criterion(self.para['fullCharge'],
                                       pad1d(self.dataset['refCharge']))
                loss = loss + lossq
                if self.para['fullCharge'].device.type == 'cuda':
                    Save2D(self.para['fullCharge'].detach().cpu().numpy(),
                        name='charge.dat', dire='.', ty='a')
                    Save1D(np.array([lossq.cpu()]), name='lossq.dat', dire='.', ty='a')
                elif self.para['fullCharge'].device.type == 'cpu':
                    Save2D(self.para['fullCharge'].detach().numpy(),
                        name='charge.dat', dire='.', ty='a')
                    Save1D(np.array([lossq]), name='lossq.dat', dire='.', ty='a')
            if 'cpa' in self.ml['target']:
                losscpa = 0.5 * self.criterion(
                    self.para['cpa'], pad1d(self.dataset['refHirshfeldVolume']))
                loss = loss + losscpa
                if self.para['cpa'].device.type == 'cuda':
                    Save2D(self.para['cpa'].detach().cpu().numpy(),
                           name='cpa.dat', dire='.', ty='a')
                    Save1D(np.array([losscpa.cpu()]), name='losscpa.dat', dire='.', ty='a')
                elif self.para['cpa'].device.type == 'cpu':
                    Save2D(self.para['cpa'].detach().numpy(),
                           name='cpa.dat', dire='.', ty='a')
                    Save1D(np.array([losscpa]), name='losscpa.dat', dire='.', ty='a')
            if 'pdos' in self.ml['target']:
                loss += self.criterion(
                    self.para['cpa'], pad1d(self.dataset['refHirshfeldVolume']))
            print("istep:", istep, '\n loss', loss, 'loss device', loss.device.type)
            # print("compression radius:", self.para['compr_ml'])
            print('gradient', self.para['compr_ml'].grad)

            # save data
            if loss.device.type == 'cuda':
                Save1D(np.array([loss.cpu()]), name='loss.dat', dire='.', ty='a')
                if not self.ml['globalCompR']:
                    Save2D(self.para['compr_ml'].detach().cpu().numpy(),
                           name='compr.dat', dire='.', ty='a')
                else:
                    Save1D(self.para['compr_ml'].detach().cpu().squeeze().numpy(),
                           name='compr.dat', dire='.', ty='a')
            elif loss.device.type == 'cpu':
                Save1D(np.array([loss]), name='loss.dat', dire='.', ty='a')
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

    def ml_compr_single(self):
        """DFTB optimization of compression radius for given dataset."""
        # do not perform batch calculation
        self.para['Lbatch'] = False
        self.sktran = SKTran(self.para, self.dataset, self.skf, self.ml)

        for istep in range(self.ml['mlSteps']):

            # set compr_ml every several steps, to avoid memory problem
            # self.para['compr_ml'] = self.para['compr_ml'].clone().requires_grad_(True)
            loss = 0
            if 'offsetenergy' in self.ml['target']:
                initenergy = []

            for ibatch in range(self.nbatch):
                self.skf['hs_compr_all'] = self.skf['hs_compr_all_'][ibatch]
                if self.ml['reference'] == 'hdf':
                    self.slako.genskf_interp_compr(ibatch)

                # SK transformations
                self.skf['hammat'], self.skf['overmat'] = self.sktran(ibatch)
                
                # run each DFTB calculation separatedly
                dftbcalculator.Rundftbpy(self.para, self.dataset, self.skf, ibatch)

                # get loss function
                if 'dipole' in self.ml['target']:
                    loss += self.criterion(self.para['dipole'].squeeze(), self.dataset['refDipole'][ibatch])
                elif 'charge' in self.ml['target']:
                    loss += self.criterion(self.para['charge'].squeeze(), self.dataset['refCharge'][ibatch])
                elif 'homo_lumo' in self.ml['target']:
                    loss += self.criterion(self.para['homo_lumo'].squeeze(), self.dataset['refHomoLumo'][ibatch])
                elif 'formationenergy' in self.ml['target']:
                    self.para['formation_energy'] = self.cal_optfor_energy(
                        self.para['electronic_energy'], ibatch)
                    loss += self.criterion(self.para['formation_energy'], self.dataset['refFormEnergy'][ibatch])
                elif 'offsetenergy' in self.ml['target']:
                    initenergy.append(self.para['electronic_energy'])
                elif 'cpa' in self.ml['target']:
                    loss += self.criterion(self.para['cpa'].squeeze(), self.dataset['refHirshfeldVolume'][ibatch])
                    print('cpa', self.criterion(self.para['cpa'].squeeze(), self.dataset['refHirshfeldVolume'][ibatch]))

                print("*" * 50, "\n istep:", istep + 1, "\n ibatch", ibatch + 1)
                print("loss:", loss, self.para['compr_ml'].grad)

            # get loss function
            if 'offsetenergy' in self.ml['target']:
                offset = self.cal_offset_energy(initenergy, self.dataset['refFormEnergy'])
                self.para['offsetEnergy'] = pad1d(initenergy) + offset
                print(self.para['offsetEnergy'], self.dataset['refFormEnergy'])
                loss = self.criterion(self.para['offsetEnergy'], self.dataset['refFormEnergy'])

            # save data
            Save1D(np.array([loss]), name='loss.dat', dire='.', ty='a')
            Save2D(self.para['compr_ml'].detach().numpy(), name='compr.dat', dire='.', ty='a')

            # clear gradients and define back propagation
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()

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


def get_coor(dataset, dtype, ibatch=None):
    """get the ith coor according to data type"""
    # for batch system
    if ibatch is None:
        if type(dataset['positions']) is t.Tensor:
            coordinate = dataset['positions']
        elif type(dataset['positions']) is np.ndarray:
            coordinate = t.from_numpy(dataset['positions']).type(dtype)
        elif type(dataset['positions']) is list:
            coordinate = dataset['positions']
        dataset['positions'] = pad2d(coordinate)
    # for single system
    else:
        if type(dataset['positions'][ibatch]) is t.Tensor:
            dataset['positions'] = dataset['positions'][ibatch][:, :]
        elif type(dataset['positions'][ibatch]) is np.ndarray:
            dataset['positions'] = \
                t.from_numpy(dataset['positions'][ibatch][:, :]).type(dtype)


def get_formation_energy(energy, atomname, ibatch):
    """Calculate formation energy"""
    return energy - sum([DFTB_ENERGY[ina] for ina in atomname[ibatch]])


def genml_init_compr(atomname, ml=None, global_R=False):
    """Get initial compression radius for each atom in system."""
    if not global_R:
        return t.tensor([ml[ia + '_init_compr'] for ia in atomname])
    elif global_R:
        return t.tensor([ml[ia + '_init_compr'] for ia in atomname])


def cal_offset_energy(self, energy, refenergy):
    A = pad2d(self.dataset['numberatom'])
    B = t.tensor(refenergy) - pad1d(energy)
    offset, _ = t.lstsq(B, A)
    return A @ offset
