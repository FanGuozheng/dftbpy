"""Read DFTB parameters, SKF files."""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import os
import scipy
import numpy as np
import torch as t
err = 1E-4
BOHR = 0.529177249
ATOMNAME = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg",
            "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr",
            "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br",
            "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd",
            "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La",
            "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er",
            "Tm", "Yb", "Lu", "Hf", "Ta", "W ", "Re", "Os", "Ir", "Pt", "Au",
            "Hg", "Tl", "Pb", "Bi", "Po", "At"]
VAL_ORB = {"H": 1, "C": 2, "N": 2, "O": 2, "Ti": 3}
intergraltyperef = {'[2, 2, 0, 0]': 0, '[2, 2, 1, 0]': 1, '[2, 2, 2, 0]': 2,
                    '[1, 2, 0, 0]': 3, '[1, 2, 1, 0]': 4, '[1, 1, 0, 0]': 5,
                    '[1, 1, 1, 0]': 6, '[0, 2, 0, 0]': 7, '[0, 1, 0, 0]': 8,
                    '[0, 0, 0, 0]': 9, '[2, 2, 0, 1]': 10, '[2, 2, 1, 1]': 11,
                    '[2, 2, 2, 1]': 12, '[1, 2, 0, 1]': 13, '[1, 2, 1, 1]': 14,
                    '[1, 1, 0, 1]': 15, '[1, 1, 1, 1]': 16, '[0, 2, 0, 1]': 17,
                    '[0, 1, 0, 1]': 18, '[0, 0, 0, 1]': 19}
ATOM_NUM = {"H": 1, "C": 6, "N": 7, "O": 8}


class ReadInt:
    """Read from .hsd input file.

    Returns:
        DFTB parameters for calculations

    """

    def __init__(self, para):
        """Initialize parameters for DFTB."""
        self.para = para
        self.get_constant()

    def get_constant(self):
        """TO DO list!!! move this to a specified code."""
        self.para['boltzmann_constant_H'] = 3.166811429e-6  # Hartree / K
        self.para['t_zero_max'] = 5.0

    def get_task(self, para):
        """Read the general information from .json file."""
        filename = para['filename']
        direct = para['direInput']
        with open(os.path.join(direct, filename), 'r') as fp:
            fpinput = json.load(fp)

            # parameter: scf
            if 'scf' in fpinput['general']:
                scf = fpinput['general']['scf']
                if scf in ('True', 'T', 'true'):
                    para['scf'] = True
                elif scf in ('False', 'F', 'false'):
                    para['scf'] = False
                else:
                    raise ImportError('Error: scf not defined correctly')
            else:
                para['scf'] = True

            # parameter: task
            if 'task' in fpinput['general']:
                para['task'] = fpinput['general']['task']
            else:
                para['task'] = 'ground'

            # parameter: scc
            if 'scc' in fpinput['general']:
                scc = fpinput['general']['scc']
                if scc in ('scc', 'nonscc', 'xlbomd'):
                    para['scc'] = scc
                else:
                    raise ImportError('Error: scc not defined correctly')
            else:
                para['scc'] = False

            if 'Lml' in fpinput['general']:
                Lml = fpinput['general']['Lml']
                if Lml in ('T', 'True', 'true'):
                    para['Lml'] = True
                elif Lml in ('F', 'False', 'false'):
                    para['Lml'] = False
                else:
                    raise ImportError('Error: scc not defined correctly')
            else:
                para['Lml'] = False

            # parameter: mixFactor
            if 'mixFactor' in fpinput['general']:
                para['mixFactor'] = fpinput['general']['mixFactor']
            else:
                para['mixFactor'] = 0.2

            # parameter: tElec
            if 'tElec' in fpinput['general']:
                para['tElec'] = fpinput['general']['tElec']
            else:
                para['tElec'] = 0

            # parameter: mixing parameters
            if 'mixMethod' in fpinput['general']:
                para['mixMethod'] = fpinput['general']['mixMethod']
            else:
                para['mixMethod'] = 'anderson'
            if 'maxIter' in fpinput['general']:
                para['maxIter'] = fpinput['general']['maxIter']
            else:
                para['maxIter'] = 60

            # convergence for energy, charge ...
            if 'convergenceType' in fpinput['general']:
                para['convergenceType'] = fpinput['general']['convergenceType']
            else:
                para['convergenceType'] = 'charge'
            if 'energy_tol' in fpinput['general']:
                para['energy_tol'] = fpinput['general']['energy_tol']
            else:
                para['energy_tol'] = 1e-7
            if 'charge_tol' in fpinput['general']:
                para['charge_tol'] = fpinput['general']['charge_tol']
            else:
                para['charge_tol'] = 1e-7

            if 'general_tol' in fpinput['general']:
                para['general_tol'] = fpinput['general']['general_tol']
            else:
                para['general_tol'] = 1e-4

            # --------------------------skf----------------------------
            # ninterp: the number of points for interp when read .skf
            if 'ninterp' in fpinput['skf']:
                para['ninterp'] = fpinput['skf']['ninterp']
            else:
                para['ninterp'] = 8

            if 'grid0' in fpinput['skf']:
                para['grid0'] = fpinput['skf']['grid0']
            else:
                para['grid0'] = 0.4

            if 'dist_tailskf' in fpinput['skf']:
                para['dist_tailskf'] = fpinput['skf']['dist_tailskf']
            else:
                para['dist_tailskf'] = 1.0
            if 'delta_r_skf' in fpinput['skf']:
                para['delta_r_skf'] = fpinput['skf']['delta_r_skf']
            else:
                para['delta_r_skf'] = 1E-5

            # --------------------------analysis----------------------------
            if 'dipole' in fpinput['analysis']:
                dipole = fpinput['analysis']['dipole']
                if dipole in ('True', 'T', 'true'):
                    para['dipole'] = True
                elif dipole in ('False', 'F', 'false'):
                    para['dipole'] = False
                else:
                    ImportError('Error: dipole not defined correctly')
            else:
                para['dipole'] = False

            # --------------------------geometry----------------------------
            # parameter: periodic
            if 'periodic' in fpinput['general']:
                period = fpinput['general']['periodic']
                if period in ('True', 'T', 'true'):
                    para['periodic'] = True
                elif period in ('False', 'F', 'false'):
                    para['periodic'] = False
                else:
                    para['periodic'] = False
                    print('Warning: periodic not defined correctly')
            else:
                para['periodic'] = False

            if 'type' in fpinput['geometry']:
                para['coorType'] = fpinput['geometry']['periodic']
            else:
                para['coorType'] = 'C'

            # interphs: use interpolation to generate SK instead of read .skf
            if 'Lml_HS' in fpinput['general']:
                scc = fpinput['general']['Lml_HS']
                if scc in ('True', 'T', 'true'):
                    para['Lml_HS'] = True
                elif scc in ('False', 'F', 'false'):
                    para['Lml_HS'] = False
                else:
                    raise ImportError('Error: Lml_HS not defined correctly')
            else:
                para['Lml_HS'] = False
        return para

    def get_coor(self):
        """Read / process coor information."""
        filename = self.para['filename']
        direct = self.para['direInput']
        with open(os.path.join(direct, filename), 'r') as f:
            fpinput = json.load(f)
            try:
                coor = fpinput['geometry']['coor']
            except IOError:
                print('please define the coordination')
        self.para['coor'] = t.from_numpy(np.asarray(coor))
        self.para['atomNumber'] = self.para['coor'][:, 0]

    def cal_coor(self):
        """Generate vector, ditance according to input geometry.

        Args:
            coor: [natom, 4], the first column is atom number
        Returns:
            natomtype: the type of atom, the 1st is 0, the 2nd different is 1
            atomind: how many orbitals of each atom in DFTB calculations

        """
        self.para['coorbohr'] = self.para['coor'][:, 1:] / BOHR
        coor = self.para['coor']
        natom = np.shape(coor)[0]
        self.para['atomNumber'] = self.para['coor'][:, 0]
        distance = t.zeros((natom, natom), dtype=t.float64)
        dnorm = t.zeros((natom, natom, 3), dtype=t.float64)
        dvec = t.zeros((natom, natom, 3), dtype=t.float64)
        atomnamelist, atom_lmax, atomind = [], [], []
        self.para['natomtype'] = []

        atomind.append(0)
        [atomnamelist.append(ATOMNAME[int(num) - 1]) for num in coor[:, 0]]

        for iat in range(0, natom):
            atom_lmax.append(VAL_ORB[ATOMNAME[int(coor[iat, 0] - 1)]])
            atomind.append(int(atomind[iat] + atom_lmax[iat]**2))
            for jat in range(0, natom):
                [xx, yy, zz] = (coor[jat, 1:] - coor[iat, 1:]) / BOHR
                dd = np.sqrt(xx * xx + yy * yy + zz * zz)
                distance[iat, jat] = dd
                if dd > err:
                    dnorm[iat, jat, :] = t.Tensor([xx, yy, zz]) / dd
                    dvec[iat, jat, :] = t.Tensor([xx, yy, zz])

        dictat = dict(zip(dict(enumerate(set(atomnamelist))).values(),
                          dict(enumerate(set(atomnamelist))).keys()))
        [self.para['natomtype'].append(dictat[ati]) for ati in atomnamelist]
        self.para['atomind2'] = int(atomind[natom] * (atomind[natom] + 1) / 2)
        self.para['atomspecie'] = list(set(atomnamelist))
        self.para['distance'], self.para['dnorm'] = distance, dnorm
        self.para['dvec'], self.para['natom'] = dvec, natom
        self.para['lmaxall'], self.para['atomind'] = atom_lmax, atomind
        self.para['atomnameall'] = atomnamelist
        self.cal_neighbour()

    def cal_neighbour(self):
        """Get number of neighbours."""
        natom = self.para['natom']
        # cutoff =
        self.para['Nneighbour'] = t.zeros(natom)


class ReadSlaKo:
    """Read .skf files.

    1. read directly from normal .skf files
    2. read from a list of files with given geometry, compression radius,
    onsite, etc.
    """

    def __init__(self, para):
        """Read all data from .skf files among different requests.

        Args:
            general DFTB parameters
            geometry
        Returns:
            gridDist, nGridPoints
            onsite, SPE, Hubbard U
            mass,  polynomial coefficients, cutoff radius
            integral tables

        """
        self.para = para

    def read_sk_specie(self):
        """Read the SKF table raw data according to atom specie."""
        atomspecie = self.para['atomspecie']
        diresk = self.para['direSK']
        nspecie = len(atomspecie)

        for iat in range(nspecie):
            for jat in range(nspecie):
                nameij = atomspecie[iat] + atomspecie[jat]
                skname = atomspecie[iat] + '-' + atomspecie[jat] + '.skf'
                fp = open(os.path.join(diresk, skname), "r")
                words = fp.readline().split()
                self.para['grid_dist' + nameij] = float(words[0])
                self.para['ngridpoint' + nameij] = int(words[1])
                nitem = int(words[1]) * 20

                if atomspecie[iat] == atomspecie[jat]:  # Homo-nuclear
                    fp_line = [float(ii) for ii in fp.readline().split()]
                    fp_line_ = t.from_numpy(np.asarray(fp_line))
                    self.para['onsite' + nameij] = fp_line_[0:3]
                    self.para['spe' + nameij] = fp_line_[3]
                    self.para['uhubb' + nameij] = fp_line_[4:7]
                    self.para['occ_skf' + nameij] = fp_line_[7:10]
                    data = np.fromfile(fp, dtype=float, count=20, sep=' ')
                    self.para['mass_cd' + nameij] = t.from_numpy(data)
                    hs_all = np.fromfile(fp, dtype=float, count=nitem, sep=' ')
                    hs_all.shape = (int(words[1]), 20)
                    self.para['hs_all' + nameij] = hs_all

                else:  # Hetero-nuclear
                    data = np.fromfile(fp, dtype=float, count=20, sep=' ')
                    self.para['mass_cd' + nameij] = t.from_numpy(data)
                    hs_all = np.fromfile(fp, dtype=float, count=nitem, sep=' ')
                    hs_all.shape = (int(words[1]), 20)
                    self.para['hs_all' + nameij] = hs_all

                spline = fp.readline().split()
                if 'Spline' in spline:
                    nint_cutoff = fp.readline().split()
                    nint_ = int(nint_cutoff[0])
                    self.para['nint_rep' + nameij] = nint_
                    self.para['cutoff_rep' + nameij] = float(nint_cutoff[1])
                    a123 = fp.readline().split()
                    self.para['a1_rep' + nameij] = float(a123[0])
                    self.para['a2_rep' + nameij] = float(a123[1])
                    self.para['a3_rep' + nameij] = float(a123[2])
                    datarep = np.fromfile(fp, dtype=float,
                                          count=(nint_ - 1) * 6, sep=' ')
                    datarep.shape = (nint_ - 1, 6)
                    self.para['rep' + nameij] = t.from_numpy(datarep)
                    datarepend = np.fromfile(fp, dtype=float,
                                             count=8, sep=' ')
                    self.para['repend' + nameij] = t.from_numpy(datarepend)
        self.get_cutoff_all()

    def read_sk_specie_old(self):
        """Old code for reading .skf. ATTENTION!!! not maintained."""
        atomspecie, diresk = self.para['atomspecie'], self.para['direSK']
        nspecie = len(atomspecie)

        for iat in range(0, nspecie):
            for jat in range(0, iat + 1):
                nameij = atomspecie[iat] + atomspecie[jat]
                if atomspecie[iat] == atomspecie[jat]:   # Homo-nuclear
                    skname = atomspecie[iat] + '-' + atomspecie[jat] + '.skf'
                    fp = open(os.path.join(diresk, skname))
                    skdata = []
                    try:
                        for line in fp:
                            each_line = line.strip().split()
                            skdata.append(each_line)
                    except IOError:
                        print('failed to open Slater-Koster file')
                    self.para['grid_dist' + nameij] = float(skdata[0][0])
                    self.para['ngridpoint' + nameij] = int(skdata[0][1])
                    line1_temp, hs_all = [], []
                    self.para['mass_cd' + nameij] = []
                    [line1_temp.append(float(ix)) for ix in skdata[1]]
                    self.para['onsite' + nameij] = line1_temp[0:3]
                    self.para['spe' + nameij] = line1_temp[3]
                    self.para['uhubb' + nameij] = line1_temp[4:7]
                    self.para['occ_skf' + nameij] = line1_temp[7:10]
                    for imass_cd in skdata[2]:
                        self.para['mass_cd' + nameij].append(float(imass_cd))
                    for iline in range(0, int(skdata[0][1])):
                        hs_all.append([float(ij) for ij in skdata[iline + 3]])
                    self.para['hs_all' + nameij] = hs_all

                else:  # Hetero-nuclear
                    nameji = atomspecie[jat] + atomspecie[iat]
                    self.para['mass_cd' + nameij] = []
                    self.para['mass_cd' + nameji] = []
                    sknameij = atomspecie[iat] + '-' + atomspecie[jat] + '.skf'
                    sknameji = atomspecie[jat] + '-' + atomspecie[iat] + '.skf'
                    fp_ij = open(os.path.join(diresk, sknameij))
                    fp_ji = open(os.path.join(diresk, sknameji))
                    skdataij, skdataji, hs_ij, hs_ji = [], [], [], []
                    try:
                        for line in fp_ij:
                            each_line = line.strip().split()
                            skdataij.append(each_line)
                        for line in fp_ji:
                            each_line = line.strip().split()
                            skdataji.append(each_line)
                    except IOError:
                        print('failed to open Slater-Koster file')
                    self.para['grid_dist' + nameij] = float(skdataij[0][0])
                    self.para['ngridpoint' + nameij] = int(skdataij[0][1])
                    self.para['grid_dist' + nameji] = float(skdataji[0][0])
                    self.para['ngridpoint' + nameji] = int(skdataji[0][1])
                    for imass_cdij in skdataij[1]:
                        self.para['mass_cd' + nameij].append(float(imass_cdij))
                    for imass_cdji in skdataji[1]:
                        self.para['mass_cd' + nameji].append(float(imass_cdji))
                    assert int(skdataij[0][1]) == int(skdataji[0][1])
                    for iline in range(0, int(skdataij[0][1])):
                        self.readhs_ij_line(
                            iline, skdataij, skdataji, atomspecie[iat],
                            atomspecie[jat], hs_ij, hs_ji)
                    self.para['hs_all' + nameij] = hs_ij
                    self.para['hs_all' + nameji] = hs_ji

    def readhs_ij_line(self, iline, skdataij, skdataji, iat, jat, hsij, hsji):
        """Deal with the integrals each line in .skf."""
        lmax = max(VAL_ORB[iat], VAL_ORB[jat])
        lmin = min(VAL_ORB[iat], VAL_ORB[jat])
        if lmax == 1:
            [hsij.append(float(ij)) for ij in skdataij[iline + 2]]
            [hsji.append(float(ji)) for ji in skdataji[iline + 2]]
        elif lmax == 2 and lmin == 1:
            if VAL_ORB[iat] == 2:
                hsji.append([float(ji) for ji in skdataji[iline + 2]])
                hsij.append(hsji[-1])
                hsij[-1][8], hsij[-1][18] = -hsij[-1][8], -hsij[-1][18]
            elif VAL_ORB[jat] == 2:
                hsij.append([float(ij) for ij in skdataij[iline + 2]])
                hsji.append(hsij[-1])
                hsji[-1][8], hsji[-1][18] = -hsji[-1][8], -hsji[-1][18]
        elif lmax == 2 and lmax == 1:
            pass

    def read_sk_atom(self):
        """Read SKF table atom by atom, ATTENTION!!! not maintained."""
        atomname, natom = self.para['atomnameall'], self.para['natom']
        self.para['onsite'] = t.zeros((natom, 3), dtype=t.float64)
        self.para['spe'] = t.zeros((natom), dtype=t.float64)
        self.para['uhubb'] = t.zeros((natom, 3), dtype=t.float64)
        self.para['occ_atom'] = t.zeros((natom, 3), dtype=t.float64)

        icount = 0
        for namei in atomname:
            for namej in atomname:
                self.read_sk(namei, namej)
                self.get_cutoff(namei, namej)
            nameii = namei + namei
            self.para['onsite'][icount, :] = \
                t.FloatTensor(self.para['onsite' + nameii])
            self.para['spe'][icount] = self.para['spe' + nameii]
            self.para['uhubb'][icount, :] = \
                t.FloatTensor(self.para['uhubb' + nameii])
            self.para['occ_atom'][icount, :] = t.FloatTensor(
                self.para['occ_skf' + nameii])
            icount += 1

    def read_sk(self, namei, namej):
        """Read homo- type .skf file."""
        nameij = namei + namej
        skfname = namei + '-' + namej + '.skf'
        direc = self.para['direSK']
        fp_sk = open(os.path.join(direc, skfname))
        allskfdata = []
        try:
            for line in fp_sk:
                each_line = line.strip().split()
                allskfdata.append(each_line)
        except IOError:
            print('open Slater-Koster file ERROR')
        grid_dist = float(allskfdata[0][0])
        ngridpoint = int(allskfdata[0][1])
        mass_cd = []
        hs_all = []
        if namei == namej:
            line1_temp = []
            [line1_temp.append(float(ix)) for ix in allskfdata[1]]
            self.para['onsite' + nameij] = line1_temp[0:3]
            self.para['spe' + nameij] = line1_temp[3]
            self.para['uhubb' + nameij] = line1_temp[4:7]
            self.para['occ_skf' + nameij] = line1_temp[7:10]
            for imass_cd in allskfdata[2]:
                mass_cd.append(float(imass_cd))
            for iline in range(0, ngridpoint):
                hs_all.append([float(ii) for ii in allskfdata[iline + 3]])
        else:
            for imass_cd in allskfdata[1]:
                mass_cd.append(float(imass_cd))
            for iline in range(0, int(ngridpoint)):
                hs_all.append([float(ii) for ii in allskfdata[iline + 2]])
        self.para['grid_dist' + nameij] = grid_dist
        self.para['ngridpoint' + nameij] = ngridpoint
        self.para['mass_cd' + nameij] = mass_cd
        self.para['hs_all' + nameij] = hs_all

    def get_cutoff(self, namei, namej):
        """Get the cutoff of atomi-atomj in .skf file."""
        nameij = namei + namej
        grid = self.para['grid_dist' + nameij]
        ngridpoint = self.para['ngridpoint' + nameij]
        disttailsk = self.para['dist_tailskf']
        cutoff = grid * ngridpoint + disttailsk
        lensk = grid * ngridpoint
        self.para['cutoffsk' + nameij] = cutoff
        self.para['lensk' + nameij] = lensk

    def get_cutoff_all(self):
        """Get the cutoff of atomi-atomj in .skf file."""
        atomspecie = self.para['atomspecie']
        disttailsk = self.para['dist_tailskf']
        for iat in range(0, len(atomspecie)):
            for jat in range(0, len(atomspecie)):
                nameij = atomspecie[iat] + atomspecie[jat]
                grid = self.para['grid_dist' + nameij]
                ngridpoint = self.para['ngridpoint' + nameij]
                cutoff = grid * ngridpoint + disttailsk - grid
                lensk = grid * ngridpoint
                self.para['cutoffsk' + nameij] = cutoff
                self.para['lensk' + nameij] = lensk


class SkInterpolator:
    """Interpolate SKF from a list of .skf files.

    These files with various compression radius
    """

    def __init__(self, para, gridmesh):
        """Generate integrals by interpolation method.

        For the moment, onsite will be included in machine learning, therefore
        onsite will be offered as input!!!
        """
        self.para = para
        self.gridmesh = gridmesh

    def readskffile(self, namei, namej, directory):
        """Read a list of .skf files.

        Args:
            all .skf file, atom names, directory
        Returns:
            gridmesh_points, onsite_spe_u, mass_rcut, integrals

        """
        nameij = namei + namej
        filenamelist = self.getfilenamelist(namei, namej, directory)
        nfile = len(filenamelist)
        ncompr = int(np.sqrt(len(filenamelist)))
        ngridpoint = t.empty((nfile), dtype=t.float64)
        grid_dist = t.empty((nfile), dtype=t.float64)
        onsite = t.empty((nfile, 3), dtype=t.float64)
        spe = t.empty((nfile), dtype=t.float64)
        uhubb = t.empty((nfile, 3), dtype=t.float64)
        occ_skf = t.empty((nfile, 3), dtype=t.float64)
        mass_rcut = t.empty((nfile, 20), dtype=t.float64)
        integrals, atomname_filename, self.para['rest'] = [], [], []

        icount = 0
        for filename in filenamelist:
            fp = open(os.path.join(directory, filename), 'r')
            words = fp.readline().split()
            grid_dist[icount] = float(words[0])
            ngridpoint[icount] = int(words[1])
            nitem = int(ngridpoint[icount] * 20)
            atomname_filename.append((filename.split('.')[0]).split("-"))
            data = np.fromfile(fp, dtype=float, count=20, sep=' ')
            mass_rcut[icount, :] = t.from_numpy(data)
            data = np.fromfile(fp, dtype=float, count=nitem, sep=' ')
            data.shape = (int(ngridpoint[icount]), 20)
            integrals.append(data)
            self.para['rest'].append(fp.read())
            icount += 1

        if self.para['Lrepulsive']:
            fp = open(os.path.join(
                directory, namei + '-' + namej + '.rep'), "r")
            first_line = fp.readline().split()
            assert 'Spline' in first_line
            nInt_cutoff = fp.readline().split()
            nint_ = int(nInt_cutoff[0])
            self.para['nint_rep' + nameij] = nint_
            self.para['cutoff_rep' + nameij] = float(nInt_cutoff[1])
            a123 = fp.readline().split()
            self.para['a1_rep' + nameij] = float(a123[0])
            self.para['a2_rep' + nameij] = float(a123[1])
            self.para['a3_rep' + nameij] = float(a123[2])
            datarep = np.fromfile(fp, dtype=float,
                                  count=(nint_-1)*6, sep=' ')
            datarep.shape = (nint_ - 1, 6)
            self.para['rep' + nameij] = t.from_numpy(datarep)
            datarepend = np.fromfile(fp, dtype=float, count=8, sep=' ')
            self.para['repend' + nameij] = t.from_numpy(datarepend)

        self.para['skf_line_tail' + nameij] = int(max(ngridpoint) + 5)
        superskf = t.zeros(ncompr, ncompr,
                           self.para['skf_line_tail' + nameij], 20)
        if self.para['Lonsite']:
            mass_rcut_ = t.zeros((ncompr, ncompr, 20), dtype=t.float64)
            onsite_ = t.zeros((ncompr, ncompr, 3), dtype=t.float64)
            spe_ = t.zeros((ncompr, ncompr), dtype=t.float64)
            uhubb_ = t.zeros((ncompr, ncompr, 3), dtype=t.float64)
            occ_skf_ = t.zeros((ncompr, ncompr, 3), dtype=t.float64)
        ngridpoint_ = t.zeros((ncompr, ncompr), dtype=t.float64)
        grid_dist_ = t.zeros((ncompr, ncompr), dtype=t.float64)

        # transfer 1D [nfile, n] to 2D [ncompr, ncompr, n]
        for skfi in range(0, nfile):
            rowi = int(skfi // ncompr)
            colj = int(skfi % ncompr)
            ingridpoint = int(ngridpoint[skfi])
            superskf[rowi, colj, :ingridpoint, :] = \
                t.from_numpy(integrals[skfi])
            grid_dist_[rowi, colj] = grid_dist[skfi]
            ngridpoint_[rowi, colj] = ngridpoint[skfi]
            if self.para['Lonsite']:
                mass_rcut_[rowi, colj, :] = mass_rcut[skfi, :]
                onsite_[rowi, colj, :] = onsite[skfi, :]
                spe_[rowi, colj] = spe[skfi]
                uhubb_[rowi, colj, :] = uhubb[skfi, :]
                occ_skf_[rowi, colj, :] = occ_skf[skfi, :]
        if self.para['Lonsite']:
            self.para['massrcut_rall' + nameij] = mass_rcut_
            self.para['onsite_rall' + nameij] = onsite_
            self.para['spe_rall' + nameij] = spe_
            self.para['uhubb_rall' + nameij] = uhubb_
            self.para['occ_skf_rall' + nameij] = occ_skf_
        self.para['nfile_rall' + nameij] = nfile
        self.para['grid_dist_rall' + nameij] = grid_dist_
        self.para['ngridpoint_rall' + nameij] = ngridpoint_
        self.para['hs_all_rall' + nameij] = superskf
        self.para['atomnameInSkf' + nameij] = atomname_filename
        self.para['interpcutoff'] = int(ngridpoint_.max())

    def getfilenamelist(self, namei, namej, directory):
        """Read all the skf files name.

        Returns:
            lists of skf file names according to sort types, therefore you
            should follow the name style:
        Namestyle:
            e.g: C-H.skf.02.77.03.34, compression radius of C, H are 2.77 and
            3.34, respectively. You can choose only read the integrals, since
            if only tune the compression radius, the onsite remains unchanged.

        """
        filename = namei + '-' + namej + '.skf.'
        filenamelist = []
        filenames = os.listdir(directory)
        filenames.sort()
        for name in filenames:
            if name.startswith(filename):
                filenamelist.append(name)
        return filenamelist

    def getallgenintegral(self, ninterpfile, skffile, r1, r2, gridarr1,
                          gridarr2):
        """Generate the whole integrals, an example code."""
        superskf = skffile["intergrals"]
        nfile = skffile["nfilenamelist"]
        row = int(np.sqrt(nfile))
        xneigh = (np.abs(gridarr1 - r1)).argmin()
        yneigh = (np.abs(gridarr2 - r2)).argmin()
        ninterp = round(xneigh * row + yneigh)
        ninterpline = int(skffile["gridmeshpoint"][ninterp, 1])
        hs_skf = np.empty((ninterpline + 5, 20))
        for lineskf in range(0, ninterpline):
            distance = lineskf * self.gridmesh + self.grid0
            counti = 0
            for intergrali in intergraltyperef:
                znew3 = SkInterpolator.getintegral(self, r1, r2, intergrali,
                                                   distance, gridarr1,
                                                   gridarr2, superskf)
                hs_skf[lineskf, counti] = znew3
                counti += 1
        return hs_skf, ninterpline

    def getintegral(self, interpr1, interpr2, integraltype, distance,
                    gridarr1, gridarr2, superskf):
        """Generate interpolation at given distance and compression radius."""
        numgridpoints = len(gridarr1)
        numgridpoints2 = len(gridarr2)
        if numgridpoints != numgridpoints2:
            print('Error: the dimension is not equal')
        skftable = np.empty((numgridpoints, numgridpoints))
        numline = int((distance - self.grid0)/self.gridmesh)
        numtypeline = intergraltyperef[integraltype]
        skftable = superskf[:, :, numline, numtypeline]
        # print('skftable', skftable)
        funcubic = scipy.interpolate.interp2d(gridarr2, gridarr1, skftable,
                                              kind='cubic')
        interporbital = funcubic(interpr2, interpr1)
        return interporbital

    def polytozero(self, hs_skf, ninterpline):
        """Here, we fit the tail of skf file (5lines, 5th order)."""
        ni = ninterpline
        dx = self.gridmesh * 5
        ytail = hs_skf[ni - 1, :]
        ytailp = (hs_skf[ni - 1, :] - hs_skf[ni - 2, :]) / self.gridmesh
        ytailp2 = (hs_skf[ni - 2, :]-hs_skf[ni - 3, :]) / self.gridmesh
        ytailpp = (ytailp - ytailp2) / self.gridmesh
        xx = np.array([self.gridmesh * 4, self.gridmesh * 3, self.gridmesh * 2,
                       self.gridmesh, 0.0])
        nline = ninterpline
        for xxi in xx:
            dx1 = ytailp * dx
            dx2 = ytailpp * dx * dx
            dd = 10.0 * ytail - 4.0 * dx1 + 0.5 * dx2
            ee = -15.0 * ytail + 7.0 * dx1 - 1.0 * dx2
            ff = 6.0 * ytail - 3.0 * dx1 + 0.5 * dx2
            xr = xxi / dx
            yy = ((ff * xr + ee) * xr + dd) * xr * xr * xr
            hs_skf[nline, :] = yy
            nline += 1
        return hs_skf

    def saveskffile(self, ninterpfile, atomnameall, skffile, hs_skf,
                    ninterpline):
        """Save all parts in skf file."""
        atomname1 = atomnameall[0]
        atomname2 = atomnameall[1]
        nfile = skffile["nfilenamelist"]
        if ninterpfile in (0, 3):
            print('generate {}-{}.skf'.format(atomname1, atomname2))
            with open('{}-{}.skf'.format(atomname1, atomname1), 'w') as fopen:
                fopen.write(str(skffile["gridmeshpoint"][nfile-1][0])+" ")
                fopen.write(str(int(ninterpline)))
                fopen.write('\n')
                np.savetxt(fopen, skffile["onsitespeu"], fmt="%s", newline=" ")
                fopen.write('\n')
                np.savetxt(fopen, skffile["massrcut"][nfile-1], newline=" ")
                fopen.write('\n')
                np.savetxt(fopen, hs_skf)
                fopen.write('\n')
                fopen.write(skffile["rest"])
        elif ninterpfile in (1, 2):
            print('generate {}-{}.skf'.format(atomname1, atomname2))
            with open('{}-{}.skf'.format(atomname1, atomname2), 'w') as fopen:
                fopen.write(str(skffile["gridmeshpoint"][nfile-1][0])+" ")
                fopen.write(str(int(ninterpline)))
                fopen.write('\n')
                np.savetxt(fopen, skffile["massrcut"][nfile-1], newline=" ")
                fopen.write('\n')
                np.savetxt(fopen, hs_skf)
                fopen.write('\n')
                fopen.write(skffile["rest"])
