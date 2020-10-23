import os
import numpy as np
from torch.autograd import Variable
import torch as t
import time
import dftbtorch.dftbcalculator as dftbcalculator
from dftbtorch.sk import SKTran, SKinterp
import utils.plot as plot
import dftbtorch.initparams as initpara
import ml.interface as interface
from ml.feature import ACSF as acsfml
from ml.padding import pad1d
from IO.dataloader import LoadData, LoadReferenceData
from IO.save import Save1D, Save2D
from utils.runcalculation import RunReference
import dftbmalt.utils.maths as maths
from ml.padding import pad1d, pad2d
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

        # general, dataset, skf, ML parameters (dictionaries)
        self.parameter = [parameter, {}][parameter is None]
        self.dataset = [dataset, {}][dataset is None]
        self.skf = [skf, {}][skf is None]
        self.ml = [ml, {}][ml is None]

        # initialize general, DFTB, dataset parameters
        dftbcalculator.Initialization(
            self.parameter, self.dataset, self.skf, self.ml)

        # detect automatically
        t.autograd.set_detect_anomaly(True)

        # set the print precision
        t.set_printoptions(precision=14)

        # set the precision control
        if self.parameter['precision'] in (t.float64, t.float32):
            t.set_default_dtype(d=self.parameter['precision'])
        else:
            raise ValueError('please select either t.float64 or t.float32')

        # initialize machine learning parameters based on
        # self.ml['reference'] = 'dftbplus'
        # self.ml['runReference'] = True
        # self.parameter['directorySK'] = '../slko/test'
        self.initialization_ml()

        # 1. read dataset then run DFT(B) to get reference data
        # 2. directly get reference data from reading input dataset
        self.load_dataset()

        # load SKF dataset
        self.load_skf()

        # run machine learning optimization
        self.run_ml()

        time_end = time.time()

    def initialization_ml(self):
        """Initialize machine learning parameters."""
        self.parameter, self.ml, self.dataset = \
            initpara.init_ml(self.parameter, self.dataset, self.ml)

    def load_dataset(self):
        """Load dataset for machine learning."""
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

        # total batch size
        self.nbatch = self.dataset['nfile']
        self.ml_integral()

    def ml_integral(self):
        '''DFTB optimization for given dataset'''
        # get the ith coordinates
        get_coor(self.dataset)

        # initialize DFTB calculations with datasetmetry and input parameters
        # read skf according to global atom species
        dftbcalculator.Initialization(self.para, self.dataset, self.skf).initialization_dftb()
        print("self.para['task']1", self.para['task'])

        # get natom * natom * [ncompr, ncompr, 20] for interpolation DFTB
        self.skf['hs_compr_all_'] = []
        self.para['compr_init_'] = []

        # get spline integral
        slako = SKinterp(self.para, self.dataset, self.skf, self.ml)
        ml_variable = slako.integral_spline_parameter()  # 0.2 s, too long!!!

        # get loss function type
        if self.ml['lossFunction'] == 'MSELoss':
            self.criterion = t.nn.MSELoss(reduction='sum')
        elif self.ml['lossFunction'] == 'L1Loss':
            self.criterion = t.nn.L1Loss(reduction='sum')

        # get optimizer
        if self.ml['optimizer'] == 'SCG':
            optimizer = t.optim.SGD(ml_variable, lr=self.ml['lr'])
        elif self.ml['optimizer'] == 'Adam':
            optimizer = t.optim.Adam(ml_variable, lr=self.ml['lr'])

        # calculate one by one to optimize para
        for istep in range(self.ml['mlSteps']):
            loss = 0.
            for ibatch in range(self.nbatch):
                print("step:", istep + 1, "ibatch:", ibatch + 1)

                # do not perform batch calculation
                self.para['Lbatch'] = False

                # get integral at certain distance, read raw integral from binary hdf
                # SK transformations
                print("self.para['task']", self.para['task'])
                SKTran(self.para, self.dataset, self.skf, self.ml, ibatch)
                print("self.skf['hammat']", self.skf['hammat'])

                # run each DFTB calculation separatedly
                dftbcalculator.Rundftbpy(self.para, self.dataset, self.skf, ibatch)

                # define loss function
                # get loss function
                if 'dipole' in self.ml['target']:
                    loss += self.criterion(self.para['dipole'].squeeze(), self.dataset['refDipole'][ibatch])

                # clear gradients and define back propagation
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
            Save1D(np.array([loss]), name='loss.dat', dire='.data', ty='a')


class MLCompressionR:
    """Optimize compression radii."""

    def __init__(self, para, dataset, skf, ml):
        """Initialize parameters."""
        self.para = para
        self.dataset = dataset
        self.skf = skf
        self.ml = ml

        # batch size
        self.nbatch = self.para['nfile']

        # process dataset for machine learning
        self.process_dataset()

        # batch calculations, all systems in batch together
        if self.para['Lbatch']:
            self.ml_compr_batch()

        # single calculations, run optimization system by system in batch
        elif not self.para['Lbatch']:
            self.ml_compr_single()

    def process_dataset(self):
        """Get all parameters for compression radii machine learning."""
        # remove some documents
        os.system('rm loss.dat compr.dat')

        # set coordinates type
        get_coor(self.dataset)

        dftbcalculator.Initialization(self.para, self.dataset, self.skf).initialization_dftb()
        self.slako = SKinterp(self.para, self.dataset, self.skf, self.ml)
        self.genmlpara = GenMLPara(self.ml)

        # get nbatch * natom * natom * [ncompr, ncompr, 20] integrals
        self.skf['hs_compr_all_'] = []
        self.ml['CompressionRInit'] = []

        # loop in batch
        for ibatch in range(self.nbatch):
            if self.ml['reference'] == 'hdf':
                natom = self.dataset['natomAll'][ibatch]
                atomname = self.dataset['atomNameAll'][ibatch]

                # get integral at certain distance, read integrals from hdf5
                self.slako.genskf_interp_dist_hdf(ibatch, natom)
                self.skf['hs_compr_all_'].append(self.skf['hs_compr_all'])

                # get initial compression radius
                self.ml['CompressionRInit'].append(
                    self.genmlpara.genml_init_compr(atomname))

        self.para['compr_ml'] = \
            Variable(pad1d(self.ml['CompressionRInit']), requires_grad=True)
        # pad1d(self.para['compr_init_']).requires_grad_(True)

    def ml_compr_batch(self):
        """DFTB optimization of compression radius for given dataset."""
        maxorb = max(self.dataset['norbital'])
        # get optimizer
        if self.ml['optimizer'] == 'SCG':
            optimizer = t.optim.SGD([self.para['compr_ml']], lr=self.ml['lr'])
        elif self.ml['optimizer'] == 'Adam':
            optimizer = t.optim.Adam([self.para['compr_ml']], lr=self.ml['lr'])

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
                self.dataset['atomNameAll'], ibatch)

            # get loss function type
            if self.ml['lossFunction'] == 'MSELoss':
                self.criterion = t.nn.MSELoss(reduction='sum')
            elif self.ml['lossFunction'] == 'L1Loss':
                self.criterion = t.nn.L1Loss(reduction='sum')

            # get loss function
            if 'dipole' in self.ml['target']:
                loss = self.criterion(self.para['dipole'], pad1d(self.dataset['refDipole']))
            print("istep:", istep, '\n loss', loss)
            print("compression radius:", self.para['compr_ml'])

            # save data
            Save1D(np.array([loss]), name='loss.dat', dire='.data', ty='a')

            # clear gradients and define back propagation
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
        print(self.para['compr_ml'].grad)

    def ml_compr_single(self):
        """DFTB optimization of compression radius for given dataset."""
        # get optimizer
        if self.ml['optimizer'] == 'SCG':
            optimizer = t.optim.SGD([self.para['compr_ml']], lr=self.ml['lr'])
        elif self.ml['optimizer'] == 'Adam':
            optimizer = t.optim.Adam([self.para['compr_ml']], lr=self.ml['lr'])

        # get loss function type
        if self.ml['lossFunction'] == 'MSELoss':
            self.criterion = t.nn.MSELoss(reduction='sum')
        elif self.ml['lossFunction'] == 'L1Loss':
            self.criterion = t.nn.L1Loss(reduction='sum')

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
            self.save.save1D(np.array([loss]), name='loss.dat', dire='.data', ty='a')
            self.save.save2D(self.para['compr_ml'].detach().numpy(),
                             name='compr.dat', dire='.data', ty='a')

            # clear gradients and define back propagation
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()


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



    def cal_offset_energy(self, energy, refenergy):
        A = pad2d(self.dataset['numberatom'])
        B = t.tensor(refenergy) - pad1d(energy)
        offset, _ = t.lstsq(B, A)
        return A @ offset

    def save_ref_idata(self, ref, ibatch, LWHL=False, LWeigenval=False,
                       LWenergy=False, LWdipole=False, LWpol=False):
        """Save data for single molecule calculation result."""
        if LWHL:
            self.save.save1D(self.dataset['refhomo_lumo'][ibatch].numpy(),
                             name='HL'+ref+'.dat', dire=self.dire_res, ty='a')
        if LWeigenval:
            self.para['refeigval'] = self.para['eigenvalue']
            self.save.save1D(self.para['eigenvalue'].detach().numpy(),
                             name='eigval'+ref+'.dat',
                             dire=self.dire_res, ty='a')
        if LWenergy:
            self.save.save1D(self.dataset['refTotEenergy'][ibatch],
                             name='totalenergy'+ref+'.dat',
                             dire=self.dire_res, ty='a')
        if LWdipole:
            self.save.save1D(self.dataset['refdipole'][ibatch].detach().numpy(),
                             name='dip'+ref+'.dat', dire=self.dire_res, ty='a')
        if LWpol:
            self.save.save1D(self.dataset['refalpha_mbd'][ibatch],
                             name='pol'+ref+'.dat', dire=self.dire_res, ty='a')

    def save_ref_data(self, ref, LWHL=False, LWeigenval=False,
                      LWenergy=False, LWdipole=False, LWpol=False):
        """Save data for all molecule calculation results."""
        if LWHL:
            self.para['refhomo_lumo'] = self.para['homo_lumo']
            self.save.save2D(self.para['homo_lumo'].detach().numpy(),
                             name='HL'+ref+'.dat', dire=self.dire_res, ty='a')
        if LWeigenval:
            self.para['refeigval'] = self.para['eigenvalue']
            self.save.save2D(self.para['eigenvalue'].detach().numpy(),
                             name='eigval'+ref+'.dat',
                             dire=self.dire_res, ty='a')
        if LWenergy:
            self.save.save1D(self.para['refenergy'],
                             name='refenergy'+ref+'.dat',
                             dire=self.dire_res, ty='a')
            self.save.save1D(self.para['totalenergy'],
                             name='totalenergy'+ref+'.dat',
                             dire=self.dire_res, ty='a')
        if LWdipole:
            self.save.save2D(self.para['refdipole'].detach().numpy(),
                             name='dip'+ref+'.dat', dire=self.dire_res, ty='a')
        if LWpol:
            self.save.save2D(self.para['alpha_mbd'].detach().numpy(),
                             name='pol'+ref+'.dat', dire=self.dire_res, ty='a')

    def update_compr_para(self, ibatch):
        nbatchml_ = int(para['opt_ml_step'] * self.nbatch)
        if ibatch == 0:
            if para['opt_ml_all']:  # use training compR from beginning
                npred_ = min(ibatch + int(nbatchml_), self.nbatch)
                self.para['ntrain'] = ibatch
                self.para['npred'] = npred_
                interface.ML(para)
            else:
                self.genml.genml_init_compr()
        elif ibatch % nbatchml_ == 0:
            self.para['dire_data'] = '.data'
            npred_ = min(ibatch + int(nbatchml_), self.nbatch)
            self.para['ntrain'] = ibatch
            self.para['npred'] = npred_
            interface.ML(para)
        elif ibatch > nbatchml_:
            para['compr_init'] = self.para['compr_pred'][ibatch]

    def get_iref(self, ibatch, nat):
        """Get reference data for each single molecule."""
        # self.homo_lumo_ref = self.para['refhomo_lumo'][ibatch]
        self.dipref = self.dataset['refdipole'][ibatch]
        if para['LMBD_DFTB']:
            self.pol_ref = self.para['refalpha_mbd'][ibatch][:nat]
            self.volref = self.para['refvol'][ibatch][:nat]
        if 'hstable' in self.ml['target']:
            self.hatableref = self.para['refhammat'][ibatch]
        if 'eigval' in self.ml['target']:
            self.eigvalref = self.para['refeigval'][ibatch]
        if 'qatomall' in self.ml['target']:
            self.qatomall_ref = para['refqatom'][ibatch]
        if self.para['Lenergy']:
            self.energy_ref = self.para['refenergy'][ibatch]

    def get_loss(self, ibatch):
        if len(self.ml['target']) == 1:
            homo_lumo = para['homo_lumo']
            dipole = para['dipole']
            gap = t.abs(homo_lumo[ibatch][1] - homo_lumo[ibatch][0])
            eigval = para['eigenvalue']
            qatomall = para['charge']
            energy = para['energy']
            if 'homo_lumo' in self.ml['target']:
                loss = self.criterion(self.homo_lumo, self.homo_lumo_ref)
            elif 'gap' in self.ml['target']:
                loss = self.criterion(gap, self.gapref)
            elif 'dipole' in self.ml['target']:
                loss = self.criterion(dipole, self.dipref)
            elif 'hstable' in self.ml['target']:
                hstable = self.skf['hammat']
                loss = self.criterion(hstable, self.hatableref)
            elif 'eigval' in self.ml['target']:
                loss = self.criterion(eigval, self.eigvalref)
            elif 'qatomall' in self.ml['target']:
                loss = self.criterion(qatomall, self.qatomall_ref)
            elif 'energy' in self.ml['target']:
                loss = self.criterion(energy, self.energy_ref)
            elif 'polarizability' in self.ml['target']:
                pol = self.para['alpha_mbd']
                loss = self.criterion(pol, self.pol_ref)
            elif 'cpa' in self.ml['target']:
                cpa = self.para['cpa']
                vol_ratio_ref = self.get_hirsh_vol_ratio(self.volref)
                loss = self.criterion(cpa, vol_ratio_ref)
            elif 'pdos' in self.para['target']:
                pdosref = self.para['pdosdftbplus'][ibatch]
                loss = maths.hellinger(para['pdos'], pdosref).sum()
                # loss = self.criterion(para['pdos'], pdosref)
                self.para['shape_pdos'][ibatch][0] = self.para['pdos'].shape[0]
                self.para['shape_pdos'][ibatch][1] = self.para['pdos'].shape[1]
            elif len(self.ml['target']) == 2:
                if 'dipole' and 'polarizability' in self.ml['target']:
                    dipole = self.para['dipole']
                    pol = self.para['alpha_mbd']
                    loss = 2 * self.criterion(pol, self.pol_ref) + \
                        self.criterion(dipole, self.dipref)
            return loss

    def save_idftbml(self):
        #self.save.save1D(para['homo_lumo'].detach().numpy(),
        #                 name='HLbp.dat', dire='.data', ty='a')
        if self.para['Ldipole']:
            print('dipole: {}, dipref: {}'.format(
                self.para['dipole'], self.dipref))
            self.save.save1D(self.para['dipole'].detach().numpy(),
                             name='dipbp.dat', dire='.data', ty='a')
        if self.para['Lenergy']:
            print('energy: {}, energyref: {}'.format(
                self.para['formation_energy'], self.energy_ref))
            self.save.save1D(self.para['formation_energy'].detach().numpy().squeeze(),
                             name='form_energybp.dat', dire='.data', ty='a')
            if self.para['Lrepulsive']:
                self.save.save1D(self.para['rep_energy'].detach().numpy(),
                                 name='repenergybp.dat', dire='.data', ty='a')
        if self.para['LMBD_DFTB']:
            self.save.save1D(para['alpha_mbd'].detach().numpy(),
                             name='polbp.dat', dire='.data', ty='a')
            self.save.save1D(self.para['cpa'].detach().numpy(),
                             name='cpa.dat', dire='.data', ty='a')
        if self.para['Lpdos']:
            self.save.save2D(self.para['pdos'].detach().numpy(),
                             name='pdosbp.dat', dire='.data', ty='a')
            self.save.save1D(self.para['hammat'].detach().numpy(),
                             name='hambp.dat', dire='.data', ty='a')
            self.save.save1D(self.para['compr_ml'].detach().numpy(),
                             name='comprbp.dat', dire='.data', ty='a')
            self.save.save1D(self.para['charge'].detach().numpy(),
                             name='qatombp.dat', dire='.data', ty='a')
        self.save.save1D(self.loss_, name='lossbp.dat', dire='.data', ty='a')

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
                                     name='bias.dat', dire='.data', ty='a')
                    self.save.save1D(acsf_weight.detach().numpy(),
                                     name='weight.dat', dire='.data', ty='a')

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
                         name='nsave.dat', dire='.data', ty='a')


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
                                 name='comprbp.dat', dire='.data', ty='a')


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


class GenMLPara:
    """Aims to get parameters for ML.

    genenvir: atomic environment parameters
    get_spllabel: get how many row lines of c parameters in bspline

    """

    def __init__(self, ml):
        self.ml = ml

    def genml_init_compr(self, atomname):
        """Get initial compression radius for each atom in system."""
        return t.tensor([self.ml[ia + '_init_compr']
                         for ia in atomname], dtype=t.float64)
