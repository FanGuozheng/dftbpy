"""Interface to DFTB+, FHI-aims."""
import os
import numpy as np
import torch as t
default_r = {"H": 3, "C": 3.5, "N": 2.2, "O": 2.3, "S": 3.8}
ATOMNUM = {"H": 1, "C": 6, "N": 7, "O": 8}


class Dftbplus:
    """Interface to DFTB+."""

    def __init__(self, para):
        """Read output from DFTB+, write DFTB+ input files."""
        self.para = para

    def geo_nonpe_ml(self, file, coor, specie, speciedict, symbols):
        """Write geo.gen in a dataset for non-periodic condition."""
        natom = len(coor)
        compressr0 = np.zeros(natom)
        with open('geo.gen.{}'.format(file), 'w') as fp:
            fp.write(str(natom)+" "+"C")
            fp.write('\n')
            for iname in specie:
                atom_i = 0
                for iatom in range(0, speciedict[iname]):
                    atom_i += 1
                    fp.write(iname+str(atom_i)+' ')
            fp.write('\n')
            iatom = 0
            for atom in specie:
                for natom in range(0, speciedict[atom]):
                    compressr0[iatom] = default_r[atom]
                    iatom += 1
                    fp.write(str(iatom)+' '+str(iatom)+' ')
                    np.savetxt(fp, coor[iatom-1], fmt="%s", newline=" ")
                    fp.write('\n')
        return compressr0

    def geo_nonpe(self, dire, coor, atomname, specie):
        """Write geo.gen in a dataset for non-periodic condition.

        Args:
            coor: first coulumn should be the atom number

        """
        natom = len(coor)
        with open(os.path.join(dire, 'geo.gen'), 'w') as fp:
            # lst line, number of atoms
            fp.write(str(natom) + " " + "C")
            fp.write('\n')
            # 2nd line: atom specie name
            for ispe in specie:
                fp.write(ispe + ' ')
            fp.write('\n')
            # coordination
            iatom = 0
            for jatom in range(natom):
                nspe = specie.index(atomname[jatom]) + 1
                iatom += 1
                fp.write(str(iatom) + " " + str(nspe) + " ")
                np.savetxt(fp, coor[jatom], fmt="%s", newline=" ")
                fp.write('\n')

    def geo_pe():
        """Write geo.gen in a dataset for periodic condition."""
        pass

    def write_dftbin(self, dire, scc, atomname, specie):
        """Write dftb_in.hsd."""
        with open(os.path.join(dire, 'dftb_in.hsd'), 'w') as fp:
            fp.write('Geometry = GenFormat { \n')
            fp.write('  <<< "geo.gen" \n } \n')
            fp.write('Driver {} \n')
            fp.write('Hamiltonian = DFTB { \n')
            if scc == 'scc':
                fp.write('Scc = Yes \n')
                fp.write('SccTolerance = 1e-8 \n MaxSccIterations = 100 \n')
                fp.write('Mixer = Broyden { \n MixingParameter = 0.2 } \n')
            elif scc == 'nonscc':
                fp.write('Scc = No \n')
            fp.write('MaxAngularMomentum { \n')
            for ispe in specie:
                if ispe == 'H':
                    fp.write('H="s" \n')
                elif ispe == 'C':
                    fp.write('C="p" \n')
                elif ispe == 'N':
                    fp.write('N="p" \n')
                elif ispe == 'O':
                    fp.write('O="p" \n')
            fp.write('} \n Charge = 0.0 \n SpinPolarisation {} \n')
            fp.write('Eigensolver = DivideAndConquer {} \n')
            fp.write('Filling = Fermi { \n')
            fp.write('Temperature [Kelvin] = 0.0 \n } \n')
            fp.write('SlaterKosterFiles = Type2FileNames { \n')
            fp.write('Separator = "-" \n Suffix = ".skf" \n } \n')
            if self.para['LMBD_DFTB']:
                fp.write('Dispersion = Mbd { \n')
                fp.write('ReferenceSet = "ts" \n')
                fp.write('NOmegaGrid = 15 \n')
                fp.write('Beta = 1.05 \n KGrid = 3 3 3 \n')
                fp.write('VacuumAxis = Yes Yes Yes \n } \n')
            fp.write('} \n')
            fp.write('Analysis { \n CalculateForces = Yes \n } \n')
            fp.write('Options { \n WriteAutotestTag = Yes \n } \n')
            fp.write('ParserOptions { \n ParserVersion = 5 \n } \n')
            fp.write('Parallel { \n UseOmpThreads = Yes \n } \n')

    def read_bandenergy(self, nfile, dire, inunit='H', outunit='H'):
        """Read file bandenergy.dat, which is HOMO and LUMO data."""
        fp = open(os.path.join(dire, 'bandenergy.dat'))
        bandenergy = np.zeros((nfile, 2), dtype=float)
        for ifile in range(0, nfile):
            ibandenergy = np.fromfile(fp, dtype=float, count=2, sep=' ')
            if inunit == outunit:
                bandenergy[ifile, :] = ibandenergy[:]
        return t.from_numpy(bandenergy)

    def read_dipole(self, nfile, dire, unit='eang', outunit='debye'):
        """Read file dip.dat, which is dipole data."""
        fp = open(os.path.join(dire, 'dip.dat'))
        dipole = np.zeros((nfile, 3), dtype=float)
        for ifile in range(0, nfile):
            idipole = np.fromfile(fp, dtype=float, count=3, sep=' ')
            if unit == 'eang' and outunit == 'debye':
                dipole[ifile, :] = idipole[:] / 0.2081943
            elif unit == outunit:
                dipole[ifile, :] = idipole[:]
            elif unit == 'debye' and outunit == 'eang':
                dipole[ifile, :] = idipole[:] * 0.2081943
        return t.from_numpy(dipole)

    def read_energy(self, nfile, dire, inunit='H', outunit='H'):
        """Read file dip.dat, which is dipole data."""
        fp = open(os.path.join(dire, 'energy.dat'))
        energy = np.zeros((nfile), dtype=float)

        for ifile in range(nfile):
            iener = np.fromfile(fp, dtype=float, count=1, sep=' ')
            if inunit == outunit:
                energy[ifile] = iener
        return t.from_numpy(energy)

    def read_hstable(self, para, nfile, dire):
        """Read file bandenergy.dat, which is HOMO and LUMO data."""
        fp = open(os.path.join(dire, 'hstable_ref'))
        hstable_ref = np.zeros((36), dtype=float)

        for ifile in range(0, nfile):
            ibandenergy = np.fromfile(fp, dtype=float, count=36, sep=' ')
            hstable_ref[:] = ibandenergy[:]
        return t.from_numpy(hstable_ref)

    def read_alpha(self, natomall, nfile, dire):
        """Read alpha data.

        Returns:
            polarizability_MBD.

        """
        nmaxatom = int(max(natomall))
        alpha = np.zeros((nfile, nmaxatom), dtype=float)

        fp = open(os.path.join(dire, 'poldftbplus.dat'))
        for ifile in range(nfile):
            natom = int(natomall[ifile])
            ial = np.fromfile(fp, dtype=float, count=natom, sep=' ')
            print(alpha[ifile, :natom], ial[:])
            alpha[ifile, :natom] = ial[:]
        return t.from_numpy(alpha)


class FHIaims:
    """Interface to FHI-aims."""

    def __init__(self, dataset):
        """Initialize parameters."""
        self.dataset = dataset

    def geo_nonpe_hdf(self, ibatch, coor):
        """Input is from hdf data, output is FHI-aims input: geo.in."""
        specie = self.dataset['specie'][ibatch]
        speciedict = self.dataset['speciedict'][ibatch]
        symbols = self.dataset['symbols'][ibatch]
        with open('geometry.in.{}'.format(ibatch), 'w') as fp:
            ispecie = 0
            iatom = 0
            for atom in specie:
                ispecie += 1
                for natom in range(speciedict[atom]):
                    fp.write('atom ')
                    iatom += 1
                    np.savetxt(fp, coor[iatom - 1], fmt='%s', newline=' ')
                    fp.write(symbols[iatom - 1])
                    fp.write('\n')

    def geo_pe():
        """Generate periodic coordinate input."""
        pass

    def read_bandenergy(self, nfile, dire, inunit='H', outunit='H'):
        """Read file bandenergy.dat, which is HOMO and LUMO data."""
        fp = open(os.path.join(dire, 'bandenergy.dat'))
        bandenergy = np.zeros((nfile, 2))
        for ifile in range(0, nfile):
            ibandenergy = np.fromfile(fp, dtype=float, count=2, sep=' ')
            if inunit == outunit:
                bandenergy[ifile, :] = ibandenergy[:]
        return t.from_numpy(bandenergy)

    def read_dipole(self, nfile, dire, unit='eang', outunit='debye'):
        """Read file dip.dat, which is dipole data."""
        fp = open(os.path.join(dire, 'dip.dat'))
        dipole = np.zeros((nfile, 3), dtype=float)

        for ifile in range(nfile):
            idipole = np.fromfile(fp, dtype=float, count=3, sep=' ')
            if unit == 'eang' and outunit == 'debye':
                dipole[ifile, :] = idipole[:] / 0.2081943
            elif unit == outunit:
                dipole[ifile, :] = idipole[:]
            elif unit == 'debye' and outunit == 'eang':
                dipole[ifile, :] = idipole[:] * 0.2081943
        return t.from_numpy(dipole)

    def read_energy(self, nfile, dire, inunit='H', outunit='H'):
        """Read file dip.dat, which is dipole data."""
        fp = open(os.path.join(dire, 'energy.dat'))
        energy = np.zeros((nfile), dtype=float)

        for ifile in range(0, nfile):
            iener = np.fromfile(fp, dtype=float, count=1, sep=' ')
            if inunit == outunit:
                energy[ifile] = iener[:]
        return t.from_numpy(energy)

    def read_qatom(self, nfile, dire, inunit='e', outunit='e'):
        """Read file dip.dat, which is dipole data."""
        nmaxatom = int(max(self.dataset['natomAll']))
        qatom = np.zeros((nfile, nmaxatom), dtype=float)

        fp = open(os.path.join(dire, 'qatomref.dat'))
        for ifile in range(0, nfile):
            natom = int(self.dataset['natomAll'][ifile])
            iqatom = np.fromfile(fp, dtype=float, count=natom, sep=' ')
            if inunit == outunit:
                qatom[ifile, :natom] = iqatom[:]
        return t.from_numpy(qatom)

    def read_alpha(self, nfile, dire):
        """Read alpha data.

        Returns:
            polarizability_MBD.

        """
        nmaxatom = int(max(self.dataset['natomAll']))
        alpha = np.zeros((nfile, nmaxatom), dtype=float)

        fp = open(os.path.join(dire, 'pol.dat'))
        for ifile in range(nfile):
            natom = int(self.dataset['natomAll'][ifile])
            ial = np.fromfile(fp, dtype=float, count=natom, sep=' ')
            alpha[ifile, :natom] = ial[:]
        return t.from_numpy(alpha)

    def read_hirshfeld_vol(self, nfile, dire):
        """Read Hirshfeld volume."""
        nmaxatom = int(max(self.dataset['natomAll']))
        vol = np.zeros((nfile, nmaxatom), dtype=float)

        fp = open(os.path.join(dire, 'vol.dat'))
        for ifile in range(nfile):
            natom = int(self.dataset['natomAll'][ifile])
            ivol = np.fromfile(fp, dtype=float, count=natom, sep=' ')
            vol[ifile, :natom] = ivol[:]
        return t.from_numpy(vol)


class NWchem:
    pass
