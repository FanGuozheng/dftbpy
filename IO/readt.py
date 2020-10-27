"""Read DFTB parameters, SKF files."""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import os
import scipy
import time
import numpy as np
import torch as t
VAL_ORB = {"H": 1, "C": 2, "N": 2, "O": 2, "Ti": 3}
intergraltyperef = {'[2, 2, 0, 0]': 0, '[2, 2, 1, 0]': 1, '[2, 2, 2, 0]': 2,
                    '[1, 2, 0, 0]': 3, '[1, 2, 1, 0]': 4, '[1, 1, 0, 0]': 5,
                    '[1, 1, 1, 0]': 6, '[0, 2, 0, 0]': 7, '[0, 1, 0, 0]': 8,
                    '[0, 0, 0, 0]': 9, '[2, 2, 0, 1]': 10, '[2, 2, 1, 1]': 11,
                    '[2, 2, 2, 1]': 12, '[1, 2, 0, 1]': 13, '[1, 2, 1, 1]': 14,
                    '[1, 1, 0, 1]': 15, '[1, 1, 1, 1]': 16, '[0, 2, 0, 1]': 17,
                    '[0, 1, 0, 1]': 18, '[0, 0, 0, 1]': 19}


def interpskf(para, skftype, atomspecie, dire_interpSK):
    """Read .skf data from skgen with various compression radius.

    Args:
        para: dictionary which store parameters
        skftype: the type (parameters in skgen) of the skf file, such as
            superposition type
        atomspecie: the species of atom in dataset
        dire_interpSK: path of skf files

    Returns:
        interpolation integrals with dimension [nspecie, nspecie, ncompress,
                                                ncompress, distance, 20]
    """
    time0 = time.time()

    # optimize wavefunction compression radius
    if skftype == 'wavefunction':
        nametail = '_wav'

    # optimize density compression radius
    elif skftype == 'density':
        nametail = '_den'

    # optimize all the compression radius and keep all the same
    elif skftype == 'all':
        nametail = '_all'

    # read skf according to atom specie
    for namei in atomspecie:
        for namej in atomspecie:

            # get atom number and read corresponding directory
            if para['atomno_' + namei] < para['atomno_' + namej]:

                # generate folder name
                dire = dire_interpSK + '/' + namei + \
                    '_' + namej + nametail

                # get integral by interpolation
                SkInterpolator(para, gridmesh=0.2).readskffile(
                    namei, namej, dire)
            else:

                # generate folder name
                dire = dire_interpSK + '/' + namej + \
                    '_' + namei + nametail

                # get integral by interpolation
                SkInterpolator(para, gridmesh=0.2).readskffile(
                    namei, namej, dire)

    # get total time
    time1 = time.time()
    time_ = time1 - time0
    dire_ = para['dire_data']

    # save to log
    with open(os.path.join(dire_, 'log.dat'), 'a') as fp:
        fp.write('Reading all the SKF files time is: ' + str(time_) + '\n')
        fp.close


class ReadInput:
    """Read from .hsd input file.

    Returns:
        DFTB parameters for calculations
    Contains:
        read_input: read json type input file
        get_coor: read coor from json type input file
        cal_coor: calculate input coordinate, return distance, atom specie...
    """

    def __init__(self, parameter=None, dataset=None, skf=None):
        """Read parameters for DFTB from hsd, json files.

        Parameters
        ----------
        parameter: `dictionary`
            parameters for DFTB calculations
        dataset: `dictionary`
            parameters for all geometric information

        Returns
        -------
        parameter: `dictionary`
            update parameter for DFTB calculations
        dataset: `dictionary`
            update dataset for all geometric information
        """
        self.para = parameter

        # geometry information
        self.dataset = [dataset, {}][dataset is None]

        # skf information
        self.skf = [skf, {}][skf is None]

        # input file name
        filename = self.para['inputName']

        # multi options for parameters: defined json file,
        if 'LReadInput' in self.para.keys():
            if self.para['LReadInput']:

                # directory of input file
                direct = self.para['directory']
                self.inputfile = os.path.join(direct, filename)

                # if file exists, read inputfile, else use default parameters
                if os.path.isfile(self.inputfile):
                    # read parameters from input json files if file exists
                    with open(self.inputfile, 'r') as fp:
                        # load json type file
                        fpinput = json.load(fp)
                        if 'general' in fpinput.keys():
                            self.dftb_parameter = self.read_general_param()
                        if 'analysis' in fpinput.keys():
                            self.dftb_parameter = self.read_analysis_param()
                        if 'geometry' in fpinput.keys():
                            self.dftb_parameter = self.read_geometry_param()

    def read_general_param(self):
        """Read the general information from .json file."""
        print("Read parameters from: ", self.inputfile)
        with open(self.inputfile, 'r') as fp:
            # load json type file
            fpinput = json.load(fp)

            # parameter of task
            if 'task' in fpinput['general']:
                task = fpinput['general']['task']
                if task in ('dftb', 'mlCompressionR', 'mlIntegral'):
                    self.para['task'] = task
                else:
                    raise ValueError('task value not defined correctly')

            # parameter of task
            if 'directorySK' in fpinput['general']:
                self.para['directorySK'] = fpinput['general']['directorySK']

            # parameter of scc
            if 'scc' in fpinput['general']:
                scc = fpinput['general']['scc']
                if scc in ('scc', 'nonscc', 'xlbomd'):
                    self.para['scc'] = scc
                else:
                    raise ValueError('scc not defined correctly')

            # perform ML or not
            if 'Lml' in fpinput['general']:
                Lml = fpinput['general']['Lml']
                if Lml in ('T', 'True', 'true'):
                    self.para['Lml'] = True
                elif Lml in ('F', 'False', 'false'):
                    self.para['Lml'] = False
                else:
                    raise ValueError('Lml not defined correctly')

            # parameter: mixing parameters
            if 'mixMethod' in fpinput['general']:
                mixmethod = fpinput['general']['mixMethod']
                if mixmethod in ('simple', 'anderson', 'broyden'):
                    self.para['mixMethod'] = mixmethod
                else:
                    raise NotImplementedError('not implement the method', mixmethod)

            # parameter: max iteration
            if 'maxIteration' in fpinput['general']:
                maxiter = fpinput['general']['maxIteration']
                if type(maxiter) is int and 6 <= maxiter <= 100:
                    self.para['maxIteration'] = maxiter
                else:
                    raise ValueError('maxIteration not defined correctly')

            # parameter: mixFactor
            if 'mixFactor' in fpinput['general']:
                mixfactor = fpinput['general']['mixFactor']
                if 0 < mixfactor < 1:
                    self.para['mixFactor'] = mixfactor
                else:
                    raise ValueError('mixFactor not defined correctly')

            # parameter of temperature: tElectron
            if 'tElectron' in fpinput['general']:
                telect = fpinput['general']['tElectron']
                if 0. <= telect <= 1E3:
                    self.para['tElec'] = telect
                else:
                    raise ValueError('tElectron not defined correctly')

            # convergence type: energy, charge ...
            if 'convergenceType' in fpinput['general']:
                convergence = fpinput['general']['convergenceType']
                if convergence in ('charge', 'energy'):
                    self.para['convergenceType'] = convergence
                else:
                    raise ValueError('convergenceType not defined correctly')

            # convergence tolerance
            if 'convergenceTolerance' in fpinput['general']:
                tol = fpinput['general']['convergenceTolerance']
                if 0 < tol < 1E-2:
                    self.para['convergenceTolerance'] = tol
                else:
                    raise ValueError('convergenceTolerance not defined correctly')

            # periodic or molecule
            if 'Lperiodic' in fpinput['general']:
                period = fpinput['general']['periodic']
                if period in ('True', 'T', 'true'):
                    self.para['Lperiodic'] = True
                elif period in ('False', 'F', 'false'):
                    self.para['Lperiodic'] = False
                else:
                    raise ValueError('Lperiodic not defined correctly')

    def read_analysis_param(self):
        with open(self.inputfile, 'r') as fp:
            # load json type file
            fpinput = json.load(fp)
            # if write dipole or not
            if 'Ldipole' in fpinput['analysis']:
                dipole = fpinput['analysis']['Ldipole']
                if dipole in ('True', 'T', 'true'):
                    self.para['Ldipole'] = True
                elif dipole in ('False', 'F', 'false'):
                    self.para['Ldipole'] = False
                else:
                    raise ValueError('dipole not defined correctly')

            # if perform MBD-DFTB with CPA
            if 'LMBD_DFTB' in fpinput['analysis']:
                dipole = fpinput['analysis']['LMBD_DFTB']
                if dipole in ('True', 'T', 'true'):
                    self.para['LMBD_DFTB'] = True
                elif dipole in ('False', 'F', 'false'):
                    self.para['LMBD_DFTB'] = False
                else:
                    raise ValueError('LMBD_DFTB not defined correctly')

            # if calculate repulsive part
            if 'Lrepulsive' in fpinput['analysis']:
                dipole = fpinput['analysis']['Lrepulsive']
                if dipole in ('True', 'T', 'true'):
                    self.para['Lrepulsive'] = True
                elif dipole in ('False', 'F', 'false'):
                    self.para['Lrepulsive'] = False
                else:
                    raise ValueError('Lrepulsive not defined correctly')

    def read_geometry_param(self):
        with open(self.inputfile, 'r') as fp:
            # load json type file
            fpinput = json.load(fp)
            # type of coordinate C (Cartesian)
            if 'type' in fpinput['geometry']:
                coortype = fpinput['geometry']['periodic']
                if coortype in ('C', 'Cartesian', 'cartesian'):
                    self.para['positionType'] = 'C'
                else:
                    raise ValueError('positionType not defined correctly')

            # read coordinate and atom number
            if 'positions' in fpinput['geometry']:
                self.dataset['positions'] = t.tensor(np.asarray(
                    fpinput['geometry']['positions']))
            if 'numbers' in fpinput['geometry']:
                self.dataset['numbers'] = t.from_numpy(np.asarray(
                    fpinput['geometry']['numbers']))

    def cal_neighbour(self):
        """Get number of neighbours, this is for solid."""
        natom = self.para['natom']
        self.para['Nneighbour'] = t.zeros(natom)


class ReadSlaKo:
    """Read .skf files.

    1. read directly from normal .skf files
    2. read from a list of files with given geometry, compression radius,
    onsite, etc.
    """

    def __init__(self, parameter, dataset, skf):
        """Read integral with different ways."""
        self.para = parameter
        self.dataset = dataset
        self.skf = skf

    def read_sk_specie(self):
        """Read the SKF table raw data according to atom specie."""
        # the atom specie in the system
        atomspecie = self.dataset['specieGlobal']

        # number of specie
        nspecie = len(atomspecie)

        for iat in range(nspecie):
            for jat in range(nspecie):

                # atom name
                nameij = atomspecie[iat] + atomspecie[jat]

                # name of skf file
                skname = atomspecie[iat] + '-' + atomspecie[jat] + '.skf'
                fp = open(os.path.join(self.para['directorySK'], skname), "r")

                # get the first line information
                words = fp.readline().split()

                # distance of grid points and number of grid points
                self.skf['grid_dist' + nameij] = float(words[0])
                self.skf['ngridpoint' + nameij] = int(words[1])

                # total integral number
                nitem = int(words[1]) * 20

                # if the atom specie is the same
                if atomspecie[iat] == atomspecie[jat]:

                    # read the second line: onsite, U...
                    fp_line = [float(ii) for ii in fp.readline().split()]
                    fp_line_ = t.from_numpy(np.asarray(fp_line))
                    self.skf['onsite' + nameij] = fp_line_[0:3]
                    self.skf['spe' + nameij] = fp_line_[3]
                    self.skf['uhubb' + nameij] = fp_line_[4:7]
                    self.skf['occ_skf' + nameij] = fp_line_[7:10]

                    # if orbital resolved
                    if not self.skf['LOrbitalResolve']:
                        self.skf['uhubb' + nameij][:] = fp_line_[6]

                    # read third line: mass...
                    data = np.fromfile(fp, dtype=float, count=20, sep=' ')
                    self.skf['mass_cd' + nameij] = t.from_numpy(data)

                    # read all the integral and reshape
                    hs_all = np.fromfile(fp, dtype=float, count=nitem, sep=' ')
                    hs_all.shape = (int(words[1]), 20)
                    self.skf['hs_all' + nameij] = hs_all

                # atom specie is different
                else:

                    # read the second line: mass...
                    data = np.fromfile(fp, dtype=float, count=20, sep=' ')
                    self.skf['mass_cd' + nameij] = t.from_numpy(data)

                    # read all the integral and reshape
                    hs_all = np.fromfile(fp, dtype=float, count=nitem, sep=' ')
                    hs_all.shape = (int(words[1]), 20)
                    self.skf['hs_all' + nameij] = hs_all

                # read spline part
                spline = fp.readline().split()
                if 'Spline' in spline:

                    # read first line of spline
                    nint_cutoff = fp.readline().split()
                    nint_ = int(nint_cutoff[0])
                    self.skf['nint_rep' + nameij] = nint_
                    self.skf['cutoff_rep' + nameij] = float(nint_cutoff[1])

                    # read second line of spline
                    a123 = fp.readline().split()
                    self.skf['a1_rep' + nameij] = float(a123[0])
                    self.skf['a2_rep' + nameij] = float(a123[1])
                    self.skf['a3_rep' + nameij] = float(a123[2])

                    # read the rest of spline but not the last
                    datarep = np.fromfile(fp, dtype=float,
                                          count=(nint_ - 1) * 6, sep=' ')
                    datarep.shape = (nint_ - 1, 6)
                    self.para['rep' + nameij] = t.from_numpy(datarep)

                    # raed the end line: start end c0 c1 c2 c3 c4 c5
                    datarepend = np.fromfile(fp, dtype=float,
                                             count=8, sep=' ')
                    self.para['repend' + nameij] = t.from_numpy(datarepend)
        self.get_cutoff_all()

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
        self.skf['onsite'] = t.zeros((natom, 3), dtype=t.float64)
        self.skf['spe'] = t.zeros((natom), dtype=t.float64)
        self.skf['uhubb'] = t.zeros((natom, 3), dtype=t.float64)
        self.skf['occ_atom'] = t.zeros((natom, 3), dtype=t.float64)

        icount = 0
        for namei in atomname:
            for namej in atomname:
                self.read_sk(namei, namej)
                self.get_cutoff(namei, namej)
            nameii = namei + namei
            self.skf['onsite'][icount, :] = \
                t.FloatTensor(self.skf['onsite' + nameii])
            self.skf['spe'][icount] = self.para['spe' + nameii]

            # if orbital resolved
            if not self.para['Lorbres']:
                self.skf['uhubb'][icount, :] = \
                    t.FloatTensor(self.skf['uhubb' + nameii])[-1]
            else:
                self.skf['uhubb'][icount, :] = \
                    t.FloatTensor(self.skf['uhubb' + nameii])

            self.skf['occ_atom'][icount, :] = t.FloatTensor(
                self.skf['occ_skf' + nameii])
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
            self.skf['onsite' + nameij] = line1_temp[0:3]
            self.skf['spe' + nameij] = line1_temp[3]
            self.skf['uhubb' + nameij] = line1_temp[4:7]
            if not self.para['Lorbres']:
                self.para['uhubb' + nameij][:] = line1_temp[6]
            self.skf['occ_skf' + nameij] = line1_temp[7:10]
            for imass_cd in allskfdata[2]:
                mass_cd.append(float(imass_cd))
            for iline in range(0, ngridpoint):
                hs_all.append([float(ii) for ii in allskfdata[iline + 3]])
        else:
            for imass_cd in allskfdata[1]:
                mass_cd.append(float(imass_cd))
            for iline in range(0, int(ngridpoint)):
                hs_all.append([float(ii) for ii in allskfdata[iline + 2]])
        self.skf['grid_dist' + nameij] = grid_dist
        self.skf['ngridpoint' + nameij] = ngridpoint
        self.skf['mass_cd' + nameij] = mass_cd
        self.skf['hs_all' + nameij] = hs_all

    def get_cutoff(self, namei, namej):
        """Get the cutoff of atomi-atomj in .skf file."""
        nameij = namei + namej
        grid = self.para['grid_dist' + nameij]
        ngridpoint = self.para['ngridpoint' + nameij]
        disttailsk = self.skf['dist_tailskf']
        cutoff = grid * ngridpoint + disttailsk
        lensk = grid * ngridpoint
        self.skf['cutoffsk' + nameij] = cutoff
        self.skf['lensk' + nameij] = lensk

    def get_cutoff_all(self):
        """Get the cutoff of atomi-atomj in .skf file."""
        atomspecie = self.dataset['symbols'][self.ibatch]
        disttailsk = self.skf['tailSKDistance']
        for iat in range(0, len(atomspecie)):
            for jat in range(0, len(atomspecie)):
                nameij = atomspecie[iat] + atomspecie[jat]
                grid = self.skf['grid_dist' + nameij]
                ngridpoint = self.skf['ngridpoint' + nameij]
                cutoff = grid * ngridpoint + disttailsk - grid
                lensk = grid * ngridpoint
                self.skf['cutoffsk' + nameij] = cutoff
                self.skf['lensk' + nameij] = lensk


class SkInterpolator:
    """Interpolate SKF from a list of .skf files.

    Returns:
        Integrals with various compression radius

    """

    def __init__(self, para, dataset, skf, gridmesh):
        """Generate integrals by interpolation method.

        For the moment, onsite will be included in machine learning, therefore
        onsite will be offered as input!!!
        """
        self.para = para
        self.dataset = dataset
        self.skf = skf
        self.gridmesh = gridmesh

    def readskffile(self, namei, namej, directory):
        """Read a list of .skf files.

        Args:
            filename: getfilenamelist
            atom species
            path of .skf files

        Returns:
            gridmesh_points, onsite_spe_u, mass_rcut, H0 and S integrals

        """
        # atom name pair
        nameij = namei + namej

        # get skf file list
        filenamelist = self.getfilenamelist(namei, namej, directory)
        nfile = len(filenamelist)

        # get number of compression radius grid point
        ncompr = int(np.sqrt(len(filenamelist)))

        # build grid distance, integral point number matrices for all skf files
        ngridpoint = t.empty((nfile), dtype=t.float64)
        grid_dist = t.empty((nfile), dtype=t.float64)

        # build onsite, spe... matrices for all skf files, second line in skf
        onsite = t.empty((nfile, 3), dtype=t.float64)
        spe = t.empty((nfile), dtype=t.float64)
        uhubb = t.empty((nfile, 3), dtype=t.float64)
        occ_skf = t.empty((nfile, 3), dtype=t.float64)

        # build matrices for third line in skf file
        mass_rcut = t.empty((nfile, 20), dtype=t.float64)
        integrals, atomname_filename, self.skf['rest'] = [], [], []

        # number for skf files
        icount = 0
        for filename in filenamelist:

            # open all the files one by one
            fp = open(os.path.join(directory, filename), 'r')

            # read first line
            words = fp.readline().split()
            grid_dist[icount] = float(words[0])
            ngridpoint[icount] = int(words[1])

            # read second line
            nitem = int(ngridpoint[icount] * 20)
            atomname_filename.append((filename.split('.')[0]).split("-"))

            # read third line
            data = np.fromfile(fp, dtype=float, count=20, sep=' ')
            mass_rcut[icount, :] = t.from_numpy(data)

            # read all the integrals, float64 precision
            data = np.fromfile(fp, dtype=float, count=nitem, sep=' ')
            data.shape = (int(ngridpoint[icount]), 20)
            integrals.append(data)

            # the rest part in skf file
            self.skf['rest'].append(fp.read())
            icount += 1

        # read repulsive parameters
        if self.para['Lrepulsive']:
            fp = open(os.path.join(
                directory, namei + '-' + namej + '.rep'), "r")
            first_line = fp.readline().split()
            assert 'Spline' in first_line
            nInt_cutoff = fp.readline().split()
            nint_ = int(nInt_cutoff[0])
            self.skf['nint_rep' + nameij] = nint_
            self.skf['cutoff_rep' + nameij] = float(nInt_cutoff[1])
            a123 = fp.readline().split()
            self.skf['a1_rep' + nameij] = float(a123[0])
            self.skf['a2_rep' + nameij] = float(a123[1])
            self.skf['a3_rep' + nameij] = float(a123[2])
            datarep = np.fromfile(fp, dtype=float,
                                  count=(nint_-1)*6, sep=' ')
            datarep.shape = (nint_ - 1, 6)
            self.skf['rep' + nameij] = t.from_numpy(datarep)
            datarepend = np.fromfile(fp, dtype=float, count=8, sep=' ')
            self.skf['repend' + nameij] = t.from_numpy(datarepend)

        # 5 more lines to smooth the tail, 4 more lines for interpolation
        self.skf['skf_line_tail' + nameij] = int(max(ngridpoint) + 9)

        # read onsite parameters
        if self.skf['Lonsite']:
            mass_rcut_ = t.zeros((ncompr, ncompr, 20), dtype=t.float64)
            onsite_ = t.zeros((ncompr, ncompr, 3), dtype=t.float64)
            spe_ = t.zeros((ncompr, ncompr), dtype=t.float64)
            uhubb_ = t.zeros((ncompr, ncompr, 3), dtype=t.float64)
            occ_skf_ = t.zeros((ncompr, ncompr, 3), dtype=t.float64)
        ngridpoint_ = t.zeros((ncompr, ncompr), dtype=t.float64)
        grid_dist_ = t.zeros((ncompr, ncompr), dtype=t.float64)

        # build integrals with various compression radius
        superskf = t.zeros((ncompr, ncompr,
                            self.skf['skf_line_tail' + nameij], 20),
                           dtype=t.float64)

        # transfer 1D [nfile, n] to 2D [ncompr, ncompr, n]
        for skfi in range(nfile):
            rowi = int(skfi // ncompr)
            colj = int(skfi % ncompr)

            # number of grid points
            ingridpoint = int(ngridpoint[skfi])
            superskf[rowi, colj, :ingridpoint, :] = \
                t.from_numpy(integrals[skfi])

            # grid distance and number grid points
            grid_dist_[rowi, colj] = grid_dist[skfi]
            ngridpoint_[rowi, colj] = ngridpoint[skfi]

            # smooth the tail
            if self.skf['LSmoothTail']:
                self.polytozero(grid_dist_, superskf, ngridpoint_, rowi, colj)

            # transfer from 1D [nfile, n] to 2D [ncompr, ncompr, n]
            if self.skf['Lonsite']:
                mass_rcut_[rowi, colj, :] = mass_rcut[skfi, :]
                onsite_[rowi, colj, :] = onsite[skfi, :]
                spe_[rowi, colj] = spe[skfi]
                uhubb_[rowi, colj, :] = uhubb[skfi, :]
                occ_skf_[rowi, colj, :] = occ_skf[skfi, :]

                # if orbital resolved
                if not self.skf['Lorbres']:
                    uhubb_[rowi, colj, :] = uhubb[skfi, -1]

        # transfer onsite, Hubbert ... to dictionary
        if self.skf['Lonsite']:
            self.skf['massrcut_rall' + nameij] = mass_rcut_
            self.skf['onsite_rall' + nameij] = onsite_
            self.skf['spe_rall' + nameij] = spe_
            self.skf['uhubb_rall' + nameij] = uhubb_
            self.skf['occ_skf_rall' + nameij] = occ_skf_

        # save integrals, gridmesh values to dictionary
        self.skf['nfile_rall' + nameij] = nfile
        self.skf['grid_dist_rall' + nameij] = grid_dist_
        self.skf['ngridpoint_rall' + nameij] = ngridpoint_
        self.skf['hs_all_rall' + nameij] = superskf
        self.skf['atomnameInSkf' + nameij] = atomname_filename
        self.skf['interpcutoff'] = int(ngridpoint_.max())

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

    def polytozero(self, griddist, hs_skf, ninterpline, rowi, colj):
        """Fit the tail of skf file (5lines, 5th order).

        griddist: gridmesh distance, 2D
        ninterpline: number of grid points, 2D
        hs_skf: integrals, 4D
        """
        ni = int(ninterpline[rowi, colj])
        dx = float(griddist[rowi, colj]) * 5
        ytail = hs_skf[rowi, colj, ni - 1, :]
        ytailp = (hs_skf[rowi, colj, ni - 1, :] - hs_skf[rowi, colj, ni - 2, :]) / griddist[rowi, colj]
        ytailp2 = (hs_skf[rowi, colj, ni - 2, :]-hs_skf[rowi, colj, ni - 3, :]) / griddist[rowi, colj]
        ytailpp = (ytailp - ytailp2) / griddist[rowi, colj]
        xx = t.tensor([4, 3, 2, 1, 0.0], dtype=t.float64) * griddist[rowi, colj]
        for xxi in xx:
            dx1 = ytailp * dx
            dx2 = ytailpp * dx * dx
            dd = 10.0 * ytail - 4.0 * dx1 + 0.5 * dx2
            ee = -15.0 * ytail + 7.0 * dx1 - 1.0 * dx2
            ff = 6.0 * ytail - 3.0 * dx1 + 0.5 * dx2
            xr = xxi / dx
            yy = ((ff * xr + ee) * xr + dd) * xr * xr * xr
            hs_skf[rowi, colj, ni, :] = yy
            ni += 1
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
