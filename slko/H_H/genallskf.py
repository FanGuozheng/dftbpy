#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is an example that how to use the code to generate the whole skf file
-------------------------------------------------------------------------------
usage: genallskf.py [-h] [-d DIRECTORY] [-g Grid0] [-m Gridmesh]
                    R1 R2 ATOM1 ATOM2
Example input1: python genallskf.py 3 7 Si Sii
Example input2: python genallskf.py -d . 3 7 Si Sii -g 0.4 -m 0.02
DIRECTORY is the path of your input files, the default is .
Grid0, Gridmesh are GridStart and GridSeparation in skdef.hsd, the default is
0.4 bohr, 0.02 bohr
atom1, atom2 are the name of atoms
R1, R2 are the interpolation radius of atom1, atom2 (bohr)
-------------------------------------------------------------------------------
Input format
For the database (.skf file), you should name like:
    atomname1-atomename2.skf.compression_r1.compression_r2
Attention: when R < 10.00, we should add prefix 0 to make sure the sequence is
correct
Example name of input files (ATOM1 is Si, ATOM2 is H, R1=5, R2=11):
Si-Si.05.00.05.00 Si-H.05.00.11.00 H-Si.11.00.05.00 H-H.11.00.11.00
-------------------------------------------------------------------------------
Other options
if you only want to generate interpolation of a single integral, you can use
getintegral in SkInterpolator class, you should offer the following parameters:
    R1, R2, integraltype, distance, gridpointarr, superskf, generalpara
    integraltype is like [l1, l2, m, type_hs], the l for s orbital is 0, for p
is 1, for d is 2, the m for σ is 0, π is 1 and δ is 2, for Hamiltonian type_hs
is 0 and overlap is 1, e.g., for dd0 Hamiltonian, integraltype is [2, 2, 0, 0];
    distance unit is bohr;
    gridpointarr = np.array([x0, y0]);
    generalpara = np.array([Grid0, Gridmesh]);
    superskf = np.array((len(x0), len(y0), line_skf, numintegral)), line_skf =
int((distance - generalpara[0])/generalpara[1]), numintegral is the
corresponding numberof integral type in integraltyperef.
"""
import numpy as np
import argparse
from geninterp import SkInterpolator


def main():
    x0 = np.array([2.00, 2.34, 2.77, 3.34, 4.07, 5.03, 6.28, 7.90, 10.00])
    y0 = np.array([2.00, 2.34, 2.77, 3.34, 4.07, 5.03, 6.28, 7.90, 10.00])
    dire, name1, name2, r1, r2, grid0, gridmesh = parser_cmd_args()
    skinter = SkInterpolator(grid0, gridmesh)
    nameall = [[name1, name1], [name1, name2], [name2, name1], [name2, name2]]
    # Example1: generate whole skf file
    # we will generate 4 skf files, therefore num will be 0, 1, 2 and 3
    for num in range(1, 3):
        skffile = skinter.readskffile(num, nameall[num], dire)
        hs_skf, ninterpline = skinter.getallgenintegral(num, skffile, r1, r2,
                                                        x0, y0)
        hs_skf = skinter.polytozero(hs_skf, ninterpline)
        # polytozero function adds 5 lines to make the tail more smooth
        skinter.saveskffile(num, nameall[num], skffile, hs_skf, ninterpline+5)
        num += 1
    """
    # Example 2
    # generate only one integral, ss0 at certain distance
    atom_atom_type = 0
    distance = 3
    skffile = skinter.readskffile(atom_atom_type, nameall[atom_atom_type],
                                  dire)
    integral = skinter.getintegral(r1, r2, '[0, 0, 0, 0]', distance, x0, y0,
                                   skffile["intergrals"])
    print('integral is: ', integral)
    """


def parser_cmd_args():
    """read the parameters from shell:
    directory is the current directory
    atom are the name of atom, e.g., Si, C, H....
    rr1, rr2 are the corresponding compression radius"""
    _DESCRIPTION = 'Test script demonstrating argparse'
    parser = argparse.ArgumentParser(description=_DESCRIPTION)
    msg = 'Directory (default: .)'
    parser.add_argument('-d', '--directory', default='.', help=msg)
    msg = 'compression radius 1: angstrom'
    parser.add_argument('compressr1', type=float, metavar='R1', help=msg)
    msg = 'compression radius 2: angstrom'
    parser.add_argument('compressr2', type=float, metavar='R2', help=msg)
    msg = 'name of atom1'
    parser.add_argument('atomname1', type=str, metavar='ATOM1', help=msg)
    msg = 'name of atom2'
    parser.add_argument('atomname2', type=str, metavar='ATOM2', help=msg)
    # msg = 'list of skf files'
    # parser.add_argument('list_skf', type=str, metavar='LIST', help=msg)
    msg = 'GridStart: bohr'
    parser.add_argument('-g', '--grid0', type=float, metavar='Grid0',
                        default=0.4, help=msg)
    msg = 'GridSeparation: bohr'
    parser.add_argument('-m', '--gridmesh', type=float, metavar='Gridmesh',
                        default=0.02, help=msg)
    args = parser.parse_args()
    print("directory is: {:s}".format(args.directory))
    print("compression radius of atom1: {:f}".format(args.compressr1))
    print("compression radius of atom2: {:f}".format(args.compressr2))
    print("Atom1: {:s} Atom2: {:s}".format(args.atomname1, args.atomname2))
    directory = args.directory
    atomname1 = args.atomname1
    atomname2 = args.atomname2
    r1 = args.compressr1
    r2 = args.compressr2
    grid0 = args.grid0
    mesh = args.gridmesh
    return directory, atomname1, atomname2, r1, r2, grid0, mesh


if __name__ == "__main__":
    main()
