import os
import numpy as np
from torch.autograd import Variable
import torch as t
import time
import dftbtorch.dftbcalculator as dftbcalculator
from dftbtorch.sk import SKTran, GetSKTable, GetSK_
import dftbtorch.initparams as initpara
import ml.interface as interface
from ml.feature import ACSF as acsfml
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
        if self.parameter['precision'] in (t.float64, t.float32):
            t.set_default_dtype(d=self.parameter['precision'])
        else:
            raise ValueError('please select either t.float64 or t.float32')

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
        plot.plot_ml(self.parameter, self.ml)
        time_end = time.time()
        print('Total time:', time_end - time_begin)

    def initialization_ml(self):
        """Initialize machine learning parameters."""
        # remove some documents
        os.system('rm loss.dat')

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
        get_coor(self.dataset)

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
        for istep in range(self.ml['mlSteps']):
            ham = t.zeros((self.nbatch, maxorb, maxorb), dtype=t.float64)
            over = t.zeros((self.nbatch, maxorb, maxorb), dtype=t.float64)
            for ibatch in range(self.nbatch):
                print("step:", istep + 1, "ibatch:", ibatch + 1)
                # get integral at certain distance, read raw integral from binary hdf
                # SK transformations
                SKTran(self.para, self.dataset, self.skf, self.ml, ibatch)
                iorb = self.dataset['norbital'][ibatch]
                ham[ibatch, :iorb, :iorb] = self.skf['hammat']
                over[ibatch, :iorb, :iorb] = self.skf['overmat']
            self.skf['hammat_'] = ham
            self.skf['overmat_'] = over

            # run each DFTB calculation separatedly
            dftbcalculator.Rundftbpy(self.para, self.dataset, self.skf, self.nbatch)

            # define loss function
            # get loss function
            if 'dipole' in self.ml['target']:
                loss = self.criterion(self.para['dipole'], pad1d(self.dataset['refDipole']))

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
        os.system('rm compr.dat')
        # deal with coordinates type
        get_coor(self.dataset)

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
            if self.ml['reference'] == 'hdf':
                natom = self.dataset['natomAll'][ibatch]
                atomname = self.dataset['symbols'][ibatch]

                # Get integral at certain distance, read integrals from hdf5
                self.slako.genskf_interp_dist_hdf(ibatch, natom)
                self.skf['hs_compr_all_'].append(self.skf['hs_compr_all'])

                # get initial compression radius
                self.ml['CompressionRInit'].append(
                    genml_init_compr(self.ml, atomname))

        self.para['compr_ml'] = \
            Variable(pad1d(self.ml['CompressionRInit']), requires_grad=True)
        # pad1d(self.para['compr_init_']).requires_grad_(True)

    def ml_compr_batch(self):
        """DFTB optimization of compression radius for given dataset."""
        maxorb = max(self.dataset['norbital'])

        for istep in range(self.ml['mlSteps']):
            ham = t.zeros((self.nbatch, maxorb, maxorb), dtype=t.float64)
            over = t.zeros((self.nbatch, maxorb, maxorb), dtype=t.float64)

            for ibatch in range(self.nbatch):

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

            # dftb formation energy calculations
            self.para['formation_energy'] = get_formation_energy(
                self.para['electronic_energy'],
                self.dataset['symbols'], ibatch)

            # get loss function
            if 'dipole' in self.ml['target']:
                loss = self.criterion(self.para['dipole'], pad1d(self.dataset['refDipole']))
            print("istep:", istep, '\n loss', loss)
            print("compression radius:", self.para['compr_ml'])
            print('gradient', self.para['compr_ml'].grad)

            # save data
            Save1D(np.array([loss]), name='loss.dat', dire='.', ty='a')
            Save2D(self.para['compr_ml'].detach().numpy(),
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
                SKTran(self.para, self.dataset, self.skf, self.ml, ibatch)

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
                    loss += self.criterion(self.para['homo_lumo'].squeeze(), self.dataset['refHirshfeldVolume'][ibatch])

                print("*" * 50, "\n istep:", istep + 1, "\n ibatch", ibatch + 1)
                print("loss:", loss, self.para['compr_ml'].grad)

            # get loss function
            if 'offsetenergy' in self.ml['target']:
                offset = self.cal_offset_energy(initenergy, self.dataset['refFormEnergy'])
                self.para['offsetEnergy'] = pad1d(initenergy) + offset
                print(self.para['offsetEnergy'], self.dataset['refFormEnergy'])
                loss = self.criterion(self.para['offsetEnergy'], self.dataset['refFormEnergy'])

            # save data
            self.save.save1D(np.array([loss]), name='loss.dat', dire='.', ty='a')
            self.save.save2D(self.para['compr_ml'].detach().numpy(),
                             name='compr.dat', dire='.', ty='a')

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
            '''if self.ml['optimizer'] == 'SCG':
                self.optimizer = t.optim.SGD([self.para['compr_ml']], lr=self.ml['lr'])
            elif self.ml['optimizer'] == 'Adam':
                self.optimizer = t.optim.Adam([self.para['compr_ml']], lr=self.ml['lr'])'''


def get_coor(dataset, ibatch=None):
    """get the ith coor according to data type"""
    # for batch system
    if ibatch is None:
        if type(dataset['positions']) is t.Tensor:
            coordinate = dataset['positions']
        elif type(dataset['positions']) is np.ndarray:
            coordinate = t.from_numpy(dataset['positions'])
        elif type(dataset['positions']) is list:
            coordinate = dataset['positions']
        dataset['positions'] = pad2d(coordinate)
    # for single system
    else:
        if type(dataset['positions'][ibatch]) is t.Tensor:
            dataset['positions'] = dataset['positions'][ibatch][:, :]
        elif type(dataset['positions'][ibatch]) is np.ndarray:
            dataset['positions'] = \
                t.from_numpy(dataset['positions'][ibatch][:, :])


def get_formation_energy(energy, atomname, ibatch):
    """Calculate formation energy"""
    return energy - sum([DFTB_ENERGY[ina] for ina in atomname[ibatch]])


def genml_init_compr(ml, atomname):
    """Get initial compression radius for each atom in system."""
    return t.tensor([ml[ia + '_init_compr']
                     for ia in atomname], dtype=t.float64)


def cal_offset_energy(self, energy, refenergy):
    A = pad2d(self.dataset['numberatom'])
    B = t.tensor(refenergy) - pad1d(energy)
    offset, _ = t.lstsq(B, A)
    return A @ offset


class MLACSF:
    def __init__(self):
        pass

    def ml_acsf(self, para):
        """DFTB optimization of ACSF parameters radius for given dataset."""
        # calculate one by one to optimize para
        ml = interface.ML(para)
        acsfml(self.para)  # get acsf_dim
        acsf_dim = self.para['acsf_dim']
        # define linear ML weight parameters
        acsf_w = t.ones((acsf_dim), dtype=t.float64) / 5
        bias_ = t.Tensor([2])
        bias = bias_.cpu().requires_grad_(True)
        acsf_weight = acsf_w.cpu().requires_grad_(True)
        optimizer = t.optim.SGD([acsf_weight, bias], lr=para['lr'])

        for it in range(para['mlsteps']):
            for ibatch in range(self.nbatch):
                para['ibatch'] = ibatch
                self.get_coor(ibatch)
                nat = self.para['coor'].shape[0]
                dftbcalculator.Initialization(self.para)
                # get natom * natom * [ncompr, ncompr, 20] by interpolation
                self.slako.genskf_interp_dist()
                self.para['LreadSKFinterp'] = False  # read SKF list only once

                # get compr by weight parameter and atomic structure parameter
                ml.dataprocess_atom()  # get atomic structure parameter
                para['compr_ml'] = (para['acsf_mlpara'] @ acsf_weight) + bias

                # dftb calculations (choose scc or nonscc)
                if para['Lml_compr_global']:
                    self.genml.genml_init_compr()

                # 2D interpolation with compression radius of atom pairs
                if self.para['dataType'] == 'hdf':
                    self.slako.genskf_interp_compr_vec()
                else:
                    self.slako.genskf_interp_compr()

                self.runcal.idftb_torchspline()
                self.para['formation_energy'] = self.cal_optfor_energy(
                        self.para['electronic_energy'], self.para['coor'])

                # get optimizer
                if self.para['optimizer'] == 'SCG':
                    optimizer = t.optim.SGD([para['compr_ml']], lr=para['lr'])
                elif self.para['optimizer'] == 'Adam':
                    optimizer = t.optim.Adam([para['compr_ml']], lr=para['lr'])

                # for each molecule we will run mlsteps
                savestep_ = 0
                # get loss function type
                if self.para['loss_function'] == 'MSELoss':
                    self.criterion = t.nn.MSELoss(reduction='sum')
                elif self.para['loss_function'] == 'L1Loss':
                    self.criterion = t.nn.L1Loss(reduction='sum')

                # get loss function
                loss = self.get_loss(ibatch)

                # clear gradients and define back propagation
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

                # save and print information
                if (it + 1) % para['save_steps'] == 0 or it == 0:
                    savestep_ += 1
                    if self.para['loss_function'] == 'MSELoss':
                        self.loss_ = t.sqrt(loss.detach()) / nat
                    elif self.para['loss_function'] == 'L1Loss':
                        self.loss_ = loss.detach() / nat
                    self.para["nsteps"][ibatch] = savestep_
                    print('-' * 100)
                    print('ibatch: {} steps: {} target: {}'.format(
                              ibatch + 1, it + 1, para['target']))
                    print('average loss: {}'.format(self.loss_))
                    print('compr_ml.grad', para['compr_ml'].grad.detach())
                    print("para['compr_ml']", para['compr_ml'])
                    # print('homo_lumo: {}, homo_lumo_ref: {}'.format(
                    #      self.para['homo_lumo'], self.homo_lumo_ref))
                    print('-' * 100)
                    self.save_idftbml()
                    self.save.save1D(bias.detach().numpy(),
                                     name='bias.dat', dire='.', ty='a')
                    self.save.save1D(acsf_weight.detach().numpy(),
                                     name='weight.dat', dire='.', ty='a')

                # convergence or break condition
                if t.lt(para['compr_ml'], para['compr_min']).all():
                    print("there is compression R smaller than {}".format(
                        para['compr_min']))
                    break
                if t.gt(para['compr_ml'], para['compr_max']).all():
                    print("there is compression larger than {}".format(
                        para['compr_max']))
                    break
        self.save.save1D(para['nsteps'].detach().numpy(),
                         name='nsave.dat', dire='.', ty='a')

    def test_pred_compr(self, para):
        '''DFTB optimization for given dataset'''
        dire = self.para['dire_data']
        os.system('rm ' + dire + '/*pred.dat')
        self.nbatch = self.para['npred']

        for ibatch in range(self.nbatch):
            para['ibatch'] = ibatch
            self.get_coor(ibatch)
            dftbcalculator.Initialization(self.para)

            self.para['LreadSKFinterp'] = False
            dftbcalculator.Initialization(self.para)
            self.slako.genskf_interp_dist()
            self.genml.genml_init_compr()

            para['compr_ml'] = self.para['compr_pred'][ibatch]

            # dftb calculations (choose scc or nonscc)
            # slako.genskf_interp_r(para)
            self.slako.genskf_interp_compr()
            self.runcal.idftb_torchspline()
            homo_lumo = para['homo_lumo']
            dipole = para['dipole']
            gap = t.abs(homo_lumo[1] - homo_lumo[0])
            self.save.save1D(
                    homo_lumo.numpy(), name='eigpred.dat', dire=dire, ty='a')
            self.save.save1D(
                    dipole.numpy(), name='dippred.dat', dire=dire, ty='a')
            self.save.save1D(para['charge'].numpy(),
                             name='qatompred.dat', dire=dire, ty='a')
            self.save.save1D(self.para['alpha_mbd'].numpy(),
                             name='polpred.dat', dire=dire, ty='a')
            self.save.save1D(homo_lumo.numpy(),
                             name='HLpred.dat', dire=dire, ty='a')
            self.save.save1D(gap.numpy(),
                             name='gappred.dat', dire=dire, ty='a')
            self.save.save1D(para['compr_ml'].numpy(),
                             name='comprpred.dat', dire=dire, ty='a')

    def ml_compr_interp(self, para):
        """Test the interpolation gradients."""

        # select the first molecule
        if type(para['coordinate'][0]) is t.Tensor:
            para['coor'] = para['coordinate'][0][:, :]
        elif type(para['coordinate'][0]) is np.ndarray:
            para['coor'] = t.from_numpy(para['coordinate'][0][:, :])
        dftbcalculator.Initialization(para)
        # interpskf(para)
        self.genml.genml_init_compr()
        para['compr_ml'] = para['compr_init'] + 1
        self.slako.genskf_interp_dist(para)
        self.slako.genskf_interp_r(para)
        hs_ref = para['h_s_all']

        para['compr_ml'] = Variable(para['compr_init'], requires_grad=True)
        optimizer = t.optim.SGD([para['compr_ml']], lr=1e-1)

        # for each molecule we will run mlsteps
        for it in range(0, para['mlsteps']):
            self.slako.genskf_interp_r(para)

            # define loss function
            criterion = t.nn.MSELoss(reduction='sum')
            # criterion = t.nn.L1Loss()
            loss = criterion(para['h_s_all'], hs_ref)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            print('compr_ml.grad', para['compr_ml'].grad)
            optimizer.step()

            if it % para['save_steps'] == 0:
                print('-' * 50)
                print('steps: {} loss: {} target: {}'.format(
                      it, loss.item(), para['target']))
                print("para['compr_ml']", para['compr_ml'])
                self.save.save1D(para['compr_ml'].detach().numpy(),
                                 name='comprbp.dat', dire='.', ty='a')


def readhs_ij_line(iat, jat, para):
    '''
    deal with the integrals each line in .skf
    '''
    nameij, nameji = iat + jat, jat + iat
    ncompr = int(t.sqrt(t.Tensor([para['nfile_rall' + nameij]])))
    lmax = max(VAL_ORB[iat], VAL_ORB[jat])
    lmin = min(VAL_ORB[iat], VAL_ORB[jat])
    if lmax == 2 and lmin == 1:
        if VAL_ORB[iat] == 2:
            for ii in range(0, ncompr):
                for jj in range(0, ncompr):
                    para['hs_all_rall' + nameij][ii, jj, :, 8] = \
                        -para['hs_all_rall' + nameji][jj, ii, :, 8]
                    para['hs_all_rall' + nameij][ii, jj, :, 18] = \
                        -para['hs_all_rall' + nameji][jj, ii, :, 18]
        elif VAL_ORB[jat] == 2:
            for ii in range(0, ncompr):
                for jj in range(0, ncompr):
                    para['hs_all_rall' + nameji][ii, jj, :, 8] = \
                        -para['hs_all_rall' + nameij][jj, ii, :, 8]
                    para['hs_all_rall' + nameji][ii, jj, :, 18] = \
                        -para['hs_all_rall' + nameij][jj, ii, :, 18]
    elif lmax == 2 and lmin == 2:
        pass
