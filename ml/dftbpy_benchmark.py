#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import argparse
from geninterp import SkInterpolator
intergraltyperef = {'[2, 2, 0, 0]': 0, '[2, 2, 1, 0]': 1, '[2, 2, 2, 0]': 2,
                    '[1, 2, 0, 0]': 3, '[1, 2, 1, 0]': 4, '[1, 1, 0, 0]': 5,
                    '[1, 1, 1, 0]': 6, '[0, 2, 0, 0]': 7, '[0, 1, 0, 0]': 8,
                    '[0, 0, 0, 0]': 9, '[2, 2, 0, 1]': 10, '[2, 2, 1, 1]': 11,
                    '[2, 2, 2, 1]': 12, '[1, 2, 0, 1]': 13, '[1, 2, 1, 1]': 14,
                    '[1, 1, 0, 1]': 15, '[1, 1, 1, 1]': 16, '[0, 2, 0, 1]': 17,
                    '[0, 1, 0, 1]': 18, '[0, 0, 0, 1]': 19}


compress_r = np.array([2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9,
                       3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9,
                       4.0, 4.2, 4.4, 4.6, 5.0])


def gensk(distance, r1, r2, skffile, nameall, grid0, gridmesh):
    skinter = SkInterpolator(grid0, gridmesh)
    x0 = np.array([2.00, 2.34, 2.77, 3.34, 4.07, 5.03, 6.28, 7.90, 10.00])
    y0 = np.array([2.00, 2.34, 2.77, 3.34, 4.07, 5.03, 6.28, 7.90, 10.00])
    # Example1: generate whole skf file
    # we will generate 4 skf files, therefore num will be 0, 1, 2 and 3
    '''for num in range(0, 4):
        hs_skf, ninterpline = skinter.getallgenintegral(num, skffileall[num],
                                                        r1, r2, x0, y0)
        hs_skf = skinter.polytozero(hs_skf, ninterpline)
        # polytozero function adds 5 lines to make the tail more smooth
        skinter.saveskffile(num, nameall[num], skffileall[num], hs_skf,
                            ninterpline+5)
        num += 1'''
    # Example 2
    # generate only one integral, ss0 at certain distance
    # atom_atom_type = ty
    '''skffile = skinter.readskffile(atom_atom_type, nameall[atom_atom_type],
                                  dire)
    integral = skinter.getintegral(r1, r2, '[0, 0, 0, 0]', distance, x0, y0,
                                   skffileall["intergrals"])'''
    interg = skffile["intergrals"]
    nfile = skffile["nfilenamelist"]
    row = int(np.sqrt(nfile))
    xneigh = (np.abs(x0 - r1)).argmin()
    yneigh = (np.abs(y0 - r2)).argmin()
    ninterp = round(xneigh*row + yneigh)
    ninterpline = int(skffile["gridmeshpoint"][ninterp, 1])
    hs_skf = np.empty(20)
    counti = 0
    for intergrali in intergraltyperef:
        znew3 = skinter.getintegral(r1, r2, intergrali, distance, x0, y0, interg)
        hs_skf[counti] = znew3
        counti += 1
    return hs_skf


def readsk(dire, name1, name2, grid0, gridmesh):
    # dire, name1, name2, r1, r2, grid0, gridmesh = parser_cmd_args()
    skinter = SkInterpolator(grid0, gridmesh)
    nameall = [[name1, name1], [name1, name2], [name2, name1], [name2, name2]]
    skffileall = []
    # Example1: generate whole skf file
    # we will generate 4 skf files, therefore num will be 0, 1, 2 and 3
    for num in range(0, 4):
        skffile = skinter.readskffile(num, nameall[num], dire)
        skffileall.append(skffile)
    return skffileall, nameall


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


'''
if __name__ == "__main__":
    main()'''
