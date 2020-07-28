"""Created on Thu Jul 23 21:12:39 2020."""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse


def parser_cmd_args(para):
    """Interface to command line.

    raed some input information, including path, names of files, etc.
    default path of input: current path
    default path of .skf file: ./slko
    default inout name: dftb_in (json type).
    """
    _description = 'Test script demonstrating argparse'
    parser = argparse.ArgumentParser(description=_description)
    msg = 'Directory (default: .)'
    parser.add_argument('-d', '--directory', default='.', help=msg)
    msg = 'Directory_SK (default: .)'
    parser.add_argument('-dk', '--directorySK', default='slko', help=msg)
    msg = 'Directory saving data (default: .data)'
    parser.add_argument('-ds', '--directoryData', default='.data', help=msg)
    msg = 'input filename'
    parser.add_argument('-fn', '--filename', type=str, default='dftb_in',
                        metavar='NAME', help=msg)
    msg = 'task'
    parser.add_argument('-t', '--task', type=str, default='opt', help=msg)
    args = parser.parse_args()
    path = os.getcwd()
    if 'filename' not in para:
        para['filename'] = args.filename
    if 'direInput' not in para:
        para['direInput'] = os.path.join(path, args.directory)
    if 'direSK' not in para:
        para['direSK'] = os.path.join(path, args.directorySK)
    if 'dire_data' not in para:
        para['dire_data'] = os.path.join(path, args.directoryData)
    if 'task' not in para:
        para['task'] = args.task
