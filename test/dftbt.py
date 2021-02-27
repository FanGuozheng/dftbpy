""""""
import torch as t
import torchvision.models as models
import torch.autograd.profiler as profiler
from dftbtorch.dftbcalculator import DFTBCalculator, Initialization
import dftbtorch.parser as parsert
from ml.train import DFTBMLTrain
from ml.test import DFTBMLTest
import IO.readt as readt


def main(parameter=None, dataset=None):
    """Start main function of DFTB-ML framework."""
    # define general DFTB parameters dictionary
    parameter = [parameter, {}][parameter is None]
    dataset = [dataset, {}][dataset is None]
    ml = {}

    # example 1: if use this code directly to run DFTB
    # parameter['task'] = 'dftb'
    # parameter['LReadInput'] = True  # default is False
    # # parameter['LCPA'], parameter['LMBD'] = True, True
    # parameter['inputName'] = 'dftb_in.dftb'
    # # parameter['mixMethod'] = 'simple'
    # dataset['nfile'] = 1

    # example 2.1: if use this code directly to optimize compression radii
    parameter['task'] = 'mlCompressionR'
    parameter['device'] = 'cpu'
    parameter['Lbatch'], parameter['dynamicSCC'] = True, True
    # parameter['dynamicSCC'] = False
    parameter['profiler'] = False
    ml['globalCompR'] = False
    # dipole, charge, HOMOLUMO, gap, cpa, polarizability
    ml['target'] = 'dipole'
    ml['referenceDataset'] = '../data/dataset/ani01_2000.hdf5'
    dataset['sizeDataset'] = [1, 1, 1]
    ml['lr'] = 5E-2
    ml['mlSteps'] = 15
    parameter['datasetSK'] = '../slko/hdf/skf.hdf5'

    # example 2.2: test compression radii
    # parameter['task'] = 'testCompressionR'
    # parameter['CompressionRData'] = '../data/results/ani_result/ani3/dipole_20mol_100step_lr_005_comprmin15_base_ani1.dat'
    # # parameter['CompressionRData'] = '../data/results/ani_result/ani_mul/dip_cpa/compr_20mol_ani01_100step_total_scale1_1.dat'
    # ml['globalCompR'] = False
    # ml['referenceDataset'] = '../data/dataset/ani03_1000.hdf5'
    # ml['testDataset'] = '../data/dataset/ani01_2000.hdf5'
    # ml['referenceMioDataset'] = '../data/dataset/ani01_2000_mio.hdf5'
    # dataset['sizeDataset'] = [20] * 18  # this should be consistent with compr.dat
    # dataset['sizeTest'] = [20] * 3
    # # ml['referenceMioDataset'] = '../data/dataset/ani01_2000_dftb_mio.hdf5'
    # ml['target'] = 'dipole'
    # ml['mlSteps'] = 100  # this should be consistent with compr.dat
    # ml['MLmodel'] = 'svm'
    # # ml['featureType'] = 'cm'
    # parameter['datasetSK'] = '../slko/hdf/skf.hdf5'

    #  example 3.1: if use this code directly to optimize compression radii
    # parameter['task'] = 'mlIntegral'
    # parameter['device'] = 'cpu'
    # dataset['sizeDataset'] = [1, 1, 1]
    # parameter['datasetSK'] = '../slko/hdf/skfmio.hdf5'
    # # dipole, charge, HOMOLUMO, gap, cpa, polarizability
    # parameter['dynamicSCC'] = True
    # ml['target'] = 'dipole'
    # ml['referenceDataset'] = '../data/dataset/ani01_2000.hdf5'
    # ml['mlSteps'] = 10
    # ml['lr'] = 1E-3

    #  example 3.2: if use this code directly to optimize compression radii
    # parameter['task'] = 'testIntegral'
    # dataset['datasetSpline'] = '../data/results/ani_result/spline/dipole/'
    # parameter['device'] = 'cpu'
    # dataset['sizeDataset'] = [20] * 3
    # dataset['sizeTest'] = [100] * 3
    # ml['referenceDataset'] = '../data/dataset/ani01_2000.hdf5'
    # parameter['datasetSK'] = '../slko/hdf/skfsingle.hdf5'
    # # dipole, charge, HOMOLUMO, gap, cpa, polarizability
    # ml['target'] = ['dipole', 'charge']
    # ml['mlSteps'] = 2
    # ml['lr'] = 1E-3
    # ml['referenceMioDataset'] = '../data/dataset/ani01_2000_mio.hdf5'

    # get command line parameters, add t in parsert to avoid naming conflicts
    parameter = parsert.parser_cmd_args(parameter)

    # return/update DFTB, geometric, skf parameters from input file
    if 'LReadInput' in parameter.keys():
        if parameter['LReadInput']:
            _para_read = readt.ReadInput(parameter, dataset, ml=ml)
            parameter, dataset = _para_read.parameter, _para_read.dataset
            ml = _para_read.ml

    # run optional task
    if parameter['task'] == 'dftb':
        DFTBCalculator(parameter, dataset)

    elif parameter['task'] in ('mlCompressionR', 'mlIntegral'):
        DFTBMLTrain(parameter, dataset, ml=ml)

    elif parameter['task'] in ('testCompressionR', 'testIntegral'):
        DFTBMLTest(parameter, dataset, ml=ml)


if __name__ == "__main__":
    main()
