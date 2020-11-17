""""""
import torch as t
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
    # parameter['LCPA'], parameter['LMBD'] = True, True
    # parameter['inputName'] = 'dftb_in.dftb'

    # example 2.1: if use this code directly to optimize compression radii
    parameter['task'] = 'mlCompressionR'
    # # dipole, charge, HOMOLUMO, gap, cpa, polarizability
    ml['target'] = 'polarizability'
    ml['referenceDataset'] = '../data/dataset/ani01_200.hdf5'
    dataset['sizeDataset'] = [1, 1, 1]
    ml['mlSteps'] = 5
    parameter['datasetSK'] = '../slko/hdf/skf.hdf5'

    # example 2.2: test compression radii
    # parameter['CompressionRData'] = '../data/results/ani_result/ani1/compr_50mol_50step_dipole.dat'
    # dataset['sizeDataset'] = [50, 50, 50]  # this should be consistent with compr.dat
    # dataset['sizeTest'] = [200, 200, 200]
    # ml['target'] = 'dipole'
    # ml['mlSteps'] = 50  # this should be consistent with compr.dat
    # parameter['task'] = 'testCompressionR'
    # ml['referenceDataset'] = '../data/dataset/ani01_200.hdf5'
    # parameter['datasetSK'] = '../slko/hdf/skf.hdf5'

    #  example 3: if use this code directly to optimize compression radii
    # parameter['task'] = 'mlIntegral'
    # dataset['sizeDataset'] = [2, 2, 2]
    # parameter['datasetSK'] = '../slko/hdf/skfmio.hdf5'

    # get command line parameters, add t in parsert to avoid naming conflicts
    parameter = parsert.parser_cmd_args(parameter)

    # return/update DFTB, geometric, skf parameters from input file
    if 'LReadInput' in parameter.keys():
        if parameter['LReadInput']:
            _para_read = readt.ReadInput(parameter, dataset)
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
