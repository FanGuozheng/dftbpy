"""Created on Thu Jul 23 21:12:39 2020."""
import os
import argparse


def parser_cmd_args(para=None):
    """Interface to command line.

    define input information, including path, names of files, etc.

    """
    para = [para, {}][para is None]

    _description = 'Test script demonstrating argparse'
    parser = argparse.ArgumentParser(description=_description)

    # default task: opt, test
    if 'task' not in para.keys():
        msg = 'task'
        parser.add_argument('-t', '--task', type=str, default='dftb', help=msg)

    # default running path
    if 'directory' not in para.keys():
        msg = 'Running directory (default: .)'
        parser.add_argument('-d', '--directory', default='.', help=msg)

    # default SK path
    if 'directorySK' not in para.keys():
        msg = 'Slater-Koster directory (default: .)'
        parser.add_argument('-dsk', '--directorySK', default='.', help=msg)

    # default input data path
    if 'directoryData' not in para.keys():
        msg = 'Directory saving data (default: .)'
        parser.add_argument('-dsave', '--directoryData', default='.', help=msg)

    # default input file name
    if 'inputName' not in para.keys():
        msg = 'input file name'
        parser.add_argument('-in', '--inputname', type=str, default='dftb_in',
                            metavar='NAME', help=msg)

    # L means logical, LReadInput defines if read parameters from dftb_in
    if 'LReadInput' not in para.keys():
        msg = 'Read input from dftb_in. default: True'
        parser.add_argument('-Lread', '--LReadInput', default=True, help=msg)

    args = parser.parse_args()

    # get current path
    path = os.getcwd()

    # add task
    if 'task' not in para:
        para['task'] = args.task

    # add calculation path
    if 'directory' not in para:
        para['directory'] = os.path.join(path, args.directory)

    # add sk path
    if 'directorySK' not in para:
        para['directorySK'] = os.path.join(path, args.directorySK)

    # add input data path
    if 'directoryData' not in para:
        para['directoryData'] = os.path.join(path, args.directoryData)

    # add input file name
    if 'inputName' not in para:
        para['inputName'] = args.inputname

    # add if read parameters from dftb_in
    if 'LReadInput' not in para:
        para['LReadInput'] = args.LReadInput

    return para
