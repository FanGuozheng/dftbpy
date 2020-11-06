""""""
import torch as t
from dftbtorch.dftbcalculator import DFTBCalculator, Initialization
import dftbtorch.parser as parsert
from ml.train import DFTBMLTrain
from ml.test import DFTBMLTest


def main(parameter=None, dataset=None):
    """Start main function of DFTB-ML framework."""
    # define general DFTB parameters dictionary
    parameter = [parameter, {}][parameter is None]
    dataset = [dataset, {}][dataset is None]

    # example 1: if use this code directly to run DFTB
    # parameter['task'] = 'dftb'
    # parameter['LReadInput'] = True  # default is False
    # parameter['inputName'] = 'dftb_in.dftb'

    #  example 2: if use this code directly to optimize compression radii
    # parameter['task'] = 'mlCompressionR'
    # parameter['datasetSK'] = '../slko/hdf/skf.hdf5'

    #  example 3: if use this code directly to optimize compression radii
    parameter['task'] = 'mlIntegral'
    parameter['datasetSK'] = '../slko/hdf/skfmio.hdf5'

    # get command line parameters, add t in parsert to avoid naming conflicts
    parameter = parsert.parser_cmd_args(parameter)

    # run optional task
    if parameter['task'] == 'dftb':
        DFTBCalculator(parameter)

    elif parameter['task'] in ('mlCompressionR', 'mlIntegral'):
        DFTBMLTrain(parameter)

    elif parameter['task'] == 'test':
        DFTBMLTest(parameter)


if __name__ == "__main__":
    main()
