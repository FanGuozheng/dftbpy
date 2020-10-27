import os
import torch as t
import numpy as np
from utils.aset import DFTB, AseAims
import IO.write_output as write
from ml.padding import pad1d, pad2d
from IO.save import Save1D, Save2D
from dftbtorch.dftbcalculator import DFTBCalculator, Initialization, Rundftbpy
AIMS_ENERGY = {"H": -0.45891649, "C": -37.77330663, "N": -54.46973501,
               "O": -75.03140052}
DFTB_ENERGY = {"H": -0.238600544, "C": -1.398493891, "N": -2.0621839400,
               "O": -3.0861916005}


class RunReference:
    """Run DFT(B) with different interface."""

    def __init__(self, para, dataset, skf, ml):
        """Initialize reference parameters."""
        self.para = para
        self.ml = ml
        self.dataset = dataset
        self.skf = skf

        # total batch size
        self.nbatch = self.dataset['nfile']

        # check data and path, rm *.dat, get path for saving data
        self.dire_res = check_data(self.para, rmdata=True)

        # run reference calculations (e.g., DFTB+ ...) before ML
        if self.ml['runReference']:

            # run DFTB python code as reference
            if self.ml['reference'] == 'dftb':
                self.dftb_ref()

            # run DFTB+ as reference
            elif self.ml['reference'] == 'dftbplus':
                self.dftbplus_ref(self.para)

            # run DFTB+ with ASE interface as reference
            elif self.ml['reference'] == 'dftbase':
                DFTB(self.para, self.dataset, self.ml, setenv=True).run_dftb(
                    self.nbatch, self.dataset['coordinateAll'])

            # FHI-aims as reference
            elif self.ml['reference'] == 'aims':
                self.aims_ref()

            # FHI-aims as reference
            elif self.ml['reference'] == 'aimsase':
                AseAims(self.para, self.dataset, self.ml, True).run_aims(self.nbatch)

    def dftb_ref(self):
        """Calculate reference with DFTB(torch)"""
        get_coor(self.dataset)
        DFTBCalculator(self.para, self.dataset, self.skf, self.dataset['nfile'])
        # Rundftbpy(self.para, self.dataset, self.skf, self.para['nfile'])

    def dftbplus_ref(self, para):
        """Calculate reference (DFTB+)"""
        # get the binary aims
        dftb = write.Dftbplus(self.para)
        bdftb = self.ml['dftbplus']

        # copy executable dftb+ as ./dftbplus/dftb+
        self.dir_ref = os.getcwd() + '/dftbplus'
        os.system('cp ' + bdftb + ' ./dftbplus/dftb+')

        # check binary FHI-aims
        if os.path.isfile(bdftb) is False:
            raise FileNotFoundError("Could not find binary, executable DFTB+")

        # get the coordinates
        get_coor(self.dataset)
        for ibatch in range(self.nbatch):
            # check if atom specie is the same to the former
            self.run_dftbplus(ibatch, self.dir_ref)

        # calculate formation energy
        self.dataset['refTotEenergy'] = write.Dftbplus(self.para).read_energy(
            self.nbatch, self.dir_ref)
        self.dataset['refFormEnergy'] = cal_for_energy(
            self.dataset, self.ml['reference'], self.dataset['refTotEenergy'], self.nbatch)
        self.dataset['refHomoLumo'] = dftb.read_bandenergy(
            self.dataset['nfile'], self.dir_ref)
        self.dataset['refDipole'] = dftb.read_dipole(
            self.dataset['nfile'], self.dir_ref, 'debye', 'eang')
        if self.para['LMBD_DFTB'] is True:
            self.dataset['refMBDAlpha'] = dftb.read_alpha(
                self.dataset['natomAll'], self.para['nfile'], self.dir_ref)

        # save results for each single molecule
        save_ref_data(self.dataset, ref='dftbplus', LWHL=True,
                      LWeigenval=False, LWenergy=True, LWdipole=True,
                      LWpol=self.para['LMBD_DFTB'])

    def run_dftbplus(self, ibatch, dire):
        """Perform DFTB+ to calculate."""
        dftb = write.Dftbplus(self.para)
        coor = self.dataset['coordinateAll'][ibatch]
        atomname = self.dataset['symbols'][ibatch]
        specie = self.dataset['specie'][ibatch]
        self.para['natom'] = int(self.dataset['natomAll'][ibatch])
        scc = self.para['scc']
        dftb.geo_nonpe(dire, coor, atomname, specie)
        dftb.write_dftbin(dire, scc, atomname, specie)
        os.system('bash ' + dire + '/run.sh ' + dire + ' ' + str(ibatch))

    def aims_ref(self):
        """Calculate reference (FHI-aims)"""
        # get the binary aims
        baims = self.ml['aims']

        # check binary FHI-aims
        if os.path.isfile(baims) is False:
            raise FileNotFoundError("Could not find binary FHI-aims")

        if os.path.isdir('aims'):

            # if exist aims folder, remove files
            os.system('rm ./aims/*.dat')

        elif os.path.isdir('aims'):
            # if exist aims folder
            os.system('mkdir aims')

        # copy executable aims as ./aims/aims
        self.dir_ref = os.getcwd() + '/aims'
        os.system('cp ' + baims + ' ./aims/aim')

        for ibatch in range(self.nbatch):
            # get the nth coordinates
            get_coor(self.dataset)

            # check if atom specie is the same to the former
            self.run_aims(ibatch, self.dir_ref)

        # read results, including energy, dipole ...
        aimsio = write.FHIaims(self.dataset)
        self.dataset['refTotEenergy'] = aimsio.read_energy(
            self.nbatch, self.dir_ref)
        self.dataset['refFormEnergy'] = cal_for_energy(
            self.dataset, self.ml['reference'],
            self.dataset['refTotEenergy'], self.nbatch)
        self.dataset['refDipole'] = aimsio.read_dipole(
            self.nbatch, self.dir_ref, 'eang', 'eang')
        self.dataset['refHomoLumo'] = aimsio.read_bandenergy(self.nbatch, self.dir_ref)
        self.dataset['refMBDAlpha'] = aimsio.read_alpha(self.nbatch, self.dir_ref)
        self.dataset['refHirshfeldVolume'] = aimsio.read_hirshfeld_vol(self.nbatch, self.dir_ref)

    def run_aims(self, ibatch, dire):
        """DFT means FHI-aims here."""
        self.natom = int(self.dataset['natomAll'][ibatch])
        coor = self.dataset['coordinateAll'][ibatch][:self.natom]
        write.FHIaims(self.dataset).geo_nonpe_hdf(ibatch, coor)
        os.rename('geometry.in.{}'.format(ibatch), 'aims/geometry.in')
        os.system('bash ' + dire + '/run.sh ' + dire + ' ' + str(ibatch) +
                  ' ' + str(self.natom))


def check_data(para, rmdata=False):
    """Check and build folder/data before run or read reference part."""
    if 'dire_result' in para.keys():
        dire_res = para['dire_result']

        # rm .dat file
        if rmdata:
            os.system('rm ' + dire_res + '/*.dat')

    # build new directory for saving results
    else:
        # do not have .data folder
        if os.path.isdir('.data'):

            # rm all the .dat files
            if rmdata:
                os.system('rm .data/*.dat')

        # .data folder exist
        else:
            os.system('mkdir .data')

        # new path for saving result
        dire_res = os.path.join(os.getcwd(), '.data')

    return dire_res


def get_coor(dataset, ibatch=None):
    """Get the coordinates according to data type."""
    # for batch system
    if ibatch is None:
        if type(dataset['coordinateAll']) is t.Tensor:
            coordinate = dataset['coordinateAll']
        elif type(dataset['coordinateAll']) is np.ndarray:
            coordinate = t.from_numpy(dataset['coordinateAll'])
        elif type(dataset['coordinateAll']) is list:
            coordinate = dataset['coordinateAll']
        dataset['coordinate'] = pad2d(coordinate)
    # for single system
    else:
        if type(dataset['coordinateAll'][ibatch]) is t.Tensor:
            dataset['coordinate'] = dataset['coordinateAll'][ibatch][:, :]
        elif type(dataset['coordinateAll'][ibatch]) is np.ndarray:
            dataset['coordinate'] = \
                t.from_numpy(dataset['coordinateAll'][ibatch][:, :])


def cal_for_energy(dataset, ref, energy, nbatch):
    """Calculate formation energy for molecule."""
    if ref == 'aims':
        for ibatch in range(nbatch):
            natom = len(dataset['coordinateAll'][ibatch])
            iname = dataset['symbols'][ibatch]
            for iat in range(natom):
                energy[ibatch] -= AIMS_ENERGY[iname[iat]]
    elif ref in ('dftb', 'dftbplus'):
        for ibatch in range(nbatch):
            natom = len(dataset['coordinateAll'][ibatch])
            iname = dataset['symbols'][ibatch]
            for iat in range(0, natom):
                energy[ibatch] -= DFTB_ENERGY[iname[iat]]
    return energy

def save_ref_data(dataset, ref, LWHL=False, LWeigenval=False,
                  LWenergy=False, LWdipole=False, LWpol=False):
    """Save data for all molecule calculation results."""
    if LWHL:
        Save2D(dataset['refHomoLumo'].detach().numpy(),
               name='HL'+ref+'.dat', dire='.', ty='a')
    if LWeigenval:
        Save2D(dataset['refEigval'].detach().numpy(),
               name='eigval'+ref+'.dat', dire='.', ty='a')
    if LWenergy:
        Save1D(dataset['refFormEnergy'],
               name='refenergy'+ref+'.dat', dire='.', ty='a')
        Save1D(dataset['refTotEenergy'],
               name='totalenergy'+ref+'.dat', dire='.', ty='a')
    if LWdipole:
        Save2D(dataset['refDipole'].detach().numpy(),
               name='dip'+ref+'.dat', dire='.', ty='a')
    if LWpol:
        Save2D(dataset['refMBDAlpha'].detach().numpy(),
               name='pol'+ref+'.dat', dire='.', ty='a')
