""""""
import torch as t
from dftbtorch.dftbcalculator import DFTBCalculator
import dftbtorch.parser as parsert
from ml.train import DFTBMLTrain
from ml.test import DFTBMLTest


def main(parameter=None):
    """Start main function of DFTB-ML framework."""
    # define general DFTB parameters dictionary
    parameter = [parameter, {}][parameter is None]

    # get command line parameters, add t in parsert to avoid naming conflicts
    parameter = parsert.parser_cmd_args(parameter)

    # run optional task
    parameter['task'] = 'mlIntegral'
    if parameter['task'] == 'dftb':
        DFTBCalculator(parameter)

    elif parameter['task'] in ('mlCompressionR', 'mlIntegral'):
        DFTBMLTrain(parameter)

    elif parameter['task'] == 'test':
        DFTBMLTest(parameter)


if __name__ == "__main__":
    main()
