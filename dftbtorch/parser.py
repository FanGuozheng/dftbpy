"""Created on Thu Jul 23 21:12:39 2020."""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse


def parser_cmd_args(para):
    """Interface to command line.

    define input information, including path, names of files, etc.

    """
    _description = 'Test script demonstrating argparse'

    # default running path
    parser = argparse.ArgumentParser(description=_description)
    msg = 'Directory (default: .)'
    parser.add_argument('-d', '--directory', default='.', help=msg)

    # default SK path
    msg = 'Directory_SK (default: .)'
    parser.add_argument('-dsk', '--directorySK', default='../slko/test', help=msg)

    # default input data path
    msg = 'Directory saving data (default: .data)'
    parser.add_argument('-ds', '--directoryData', default='.data', help=msg)
    msg = 'input filename'
    parser.add_argument('-fn', '--filename', type=str, default='dftb_in',
                        metavar='NAME', help=msg)

    # default task: opt, test
    msg = 'task'
    parser.add_argument('-t', '--task', type=str, default='opt', help=msg)

    args = parser.parse_args()

    # get current path
    path = os.getcwd()

    # add input file name
    if 'filename' not in para:
        para['filename'] = args.filename

    # add calculation path
    if 'direInput' not in para:
        para['direInput'] = os.path.join(path, args.directory)

    # add sk path
    if 'direSK' not in para:
        para['direSK'] = os.path.join(path, args.directorySK)

    # add input data path
    if 'dire_data' not in para:
        para['dire_data'] = os.path.join(path, args.directoryData)

    # add task
    if 'task' not in para:
        para['task'] = args.task
