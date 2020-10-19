import torch as t
from dftbtorch.dftbcalculator import DFTBCalculator
import dftbtorch.parser as parsert
from ml.train import DFTBMLTrain
from ml.test import DFTBMLTest
import logging


def main(parameter=None):
    """Start main function of DFTB-ML framework."""
    # precision control
    # t.set_default_dtype(d=t.float64)

    # define general DFTB parameters dictionary
    parameter = [parameter, {}][parameter is None]

    # get command line parameters, add t in parsert to avoid naming conflicts
    parameter = parsert.parser_cmd_args(parameter)

    # run optional task
    parameter['task'] = 'mlCompressionR'
    if parameter['task'] == 'dftb':
        logging.info("started")
        DFTBCalculator(parameter)

    elif parameter['task'] in ('mlCompressionR', 'mlIntegral'):
        t.autograd.set_detect_anomaly(True)

        # set the print precision
        t.set_printoptions(precision=15)

        # set the data type precision
        t.set_default_dtype(d=t.float64)
        DFTBMLTrain(parameter)

    elif parameter['task'] == 'test':
        DFTBMLTest(parameter)


if __name__ == "__main__":
    main()
