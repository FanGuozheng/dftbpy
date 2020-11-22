"""Created on Thu Jul 23 21:12:39 2020."""
import os
import argparse


def parser_cmd_args(para=None):
    """Interface to command line.

    define input information, including path, names of files, etc.

    """
    para = [para, {}][para is None]

    _description = 'get DFTB-ML parameters'
    parser = argparse.ArgumentParser(description=_description)

    # get task from command line else return None
    msg = 'task of DFTB-ML framework'
    parser.add_argument('-t', '--task', help=msg)

    # default running path is current path
    msg = 'Running directory (default: .)'
    parser.add_argument('-d', '--directory', default='.', help=msg)

    # get directory of SKF from command line else return None
    msg = 'Slater-Koster directory'
    parser.add_argument('-dsk', '--directorySK', help=msg)

    # get directory of input dataset from command line else return None
    msg = 'Directory saving data'
    parser.add_argument('-dsave', '--directoryData', help=msg)

    # get directory of input file name from command line else return None
    msg = 'input file name'
    parser.add_argument('-in', '--inputname', metavar='NAME', help=msg)

    # LReadInput defines if read parameters from dftb_in
    # it will automatically switch to True if you define input file name
    msg = 'Read input from dftb_in'
    parser.add_argument('-Lread', '--LReadInput', help=msg)

    # if use pytorch profiler
    msg = 'PyTirch profiler'
    parser.add_argument('-pro', '--profiler', default=False, help=msg)

    args, unkown = parser.parse_known_args()

    # get current path
    path = os.getcwd()

    # add task
    if args.task is not None:
        para['task'] = args.task

    # add calculation path
    if args.directory is not None:
        para['directory'] = os.path.join(path, args.directory)

    # add sk path
    if args.directorySK is not None:
        para['directorySK'] = os.path.join(path, args.directorySK)

    # add input data path
    if args.directoryData is not None:
        para['directoryData'] = os.path.join(path, args.directoryData)

    # add input file name, in this case read the input is True
    if args.inputname is not None:
        para['inputName'] = args.inputname

        # if you define input file, it suggests the code will read the input
        para['LReadInput'] = True

    if args.profiler is not None:
        para['profiler'] = args.profiler


    return para
