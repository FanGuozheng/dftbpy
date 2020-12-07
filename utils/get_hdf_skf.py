"""Write skf to hdf5 binary file.

The skf include normal skf files or skf with a list of compression radii.
"""
import os
import h5py
import numpy as np
import dftbtorch.initparams as initpara
import IO.readt as readt
import dftbtorch.parameters as constpara
VAL_ORB = {"H": 1, "C": 2, "N": 2, "O": 2, "Ti": 3}


class WriteSKNormal:
    """Transfer SKF files from skf files to hdf binary type.

    Args:
        para: general parameter dictionary.

    """

    def __init__(self, para):
        """Read skf, smooth the tail, write to hdf."""
        self.para = para
        self.ml = {}
        self.para['directorySK'] = '../slko/test/compr35/'
        self.dataset = {}
        self.dataset['Dataset'] = '../data/dataset/testfile.hdf5'

        # load constant parameters
        constpara.constant_parameter(self.para)

        # get default general parameters
        self.para = initpara.dftb_parameter(self.para)
        self.skf = initpara.skf_parameter(self.para)
        self.dataset = initpara.init_dataset(self.dataset)
        self.para, self.dataset, self.skf, self.ml = \
            initpara.init_ml(self.para, self.dataset, self.skf, self.ml)
        self.ibatch = 0

        # read the dataset and then read the corresponding skf files
        self.read_dataset()

        # interpolate integrals
        self.interpskf()

    def read_dataset(self):
        """Read the corresponding hdf data of geometries.

        Notes:
            define para['dataType'] == 'hdf'
        """
        # join the path and hdf data
        hdf_ = '../data/dataset/ani01_2000.hdf5'

        # read the hdf data, get the global atom specie
        with h5py.File(hdf_, 'r') as f:
            self.specie_global = f['globalGroup'].attrs['specieGlobal']
            self.dataset['specieGlobal'] = [self.specie_global]

    def interpskf(self):
        """Read .skf data from skgen with various compR."""
        os.system('rm skfsingle.hdf5')
        # optimize wavefunction compression radius

        # read skf according to atom specie
        for namei in self.specie_global:
            for namej in self.specie_global:

                # generate folder name
                dire = self.para['directorySK'] + '/' + namei + '-' + namej + '.skf'

                # get integral by interpolation
                # read normal skf file by atom specie
                readt.ReadSlaKo(self.para,
                                self.dataset, self.skf).read_sk_specie()

                # write intepolation data as hdf
                self.write_hdf5(namei, namej)

    def write_hdf5(self, namei, namej):
        """Write each atom pair skf to hdf."""
        nameij = namei + namej

        # if onsite is fixed (False) or tunable (True)
        if self.skf['Lonsite']:
            name_onsite = 'onsite'
            onsite_ = self.skf['onsite' + nameij]

            # mass and r_cut
            name_massr = 'massrcut'
            mass_rcut_ = self.skf['massrcut' + nameij]

            # spe
            namespe = 'spe'
            spe_ = self.skf['spe' + nameij]

            # Hubbert
            namehubbert = 'uhubb'
            hubbert = self.skf['uhubb' + nameij]

        # gridmesh distance
        namedist = 'grid_dist'
        distance = self.skf['grid_dist' + nameij]

        # number of gridmesh
        namengridpoint = 'ngridpoint'
        ngridpoint = self.skf['ngridpoint' + nameij]

        # integrals
        namehs = 'hs_all'
        hs = self.skf['hs_all' + nameij]

        if self.para['Lrepulsive']:

            namenint = 'nint_rep'
            nint_ = self.skf['nint_rep' + nameij]

            namenInt_cutoff = 'cutoff_rep'
            nInt_cutoff = self.skf['cutoff_rep' + nameij]

            a1 = self.skf['a1_rep' + nameij]
            a2 = self.skf['a2_rep' + nameij]
            a3 = self.skf['a3_rep' + nameij]
            rep = self.skf['rep' + nameij]
            repend = self.skf['repend' + nameij]

        # write the above to hdf
        with h5py.File('skfsingle.hdf5', 'a') as f:
            print('specie pair', nameij)
            g = f.create_group(nameij)

            # if onsite is fixed (False) or tunable (True)
            if self.skf['Lonsite']:
                g.create_dataset(name_onsite, data=onsite_)
                g.create_dataset(name_massr, data=mass_rcut_)
                g.create_dataset(namespe, data=spe_)
                g.create_dataset(namehubbert, data=hubbert)
            g.create_dataset(namedist, data=distance)
            g.create_dataset(namengridpoint, data=ngridpoint)
            g.create_dataset(namehs, data=hs)

            # create global group
            try:
                glob = f.create_group('globalgroup')
                glob.attrs['grid_dist'] = distance[0, 0]
            except:
                pass


class WriteSKComprR:
    """Transfer SKF files from skf files to hdf binary type.

    Args:
        para (dict): general parameter
        dataset (dict): dataset parameter
    """

    def __init__(self, para):
        """Read skf, smooth the tail, write to hdf."""
        self.para = para
        self.ml = {}

        # define parameters, reference dataset (get global specie), skf path
        self.para['referenceDataset'] = './testfile.hdf5'
        self.para['directorySK'] = '/home/gz_fan/Documents/ML/dftb/slko/test/grid0.1'  # '../slko/uniform'

        # load initial parameters
        self.para = initpara.dftb_parameter(self.para)
        self.para = initpara.dftb_parameter(self.para)
        self.skf = initpara.skf_parameter(self.para)
        self.dataset = initpara.init_dataset()
        self.para, self.dataset, self.skf, self.ml = \
            initpara.init_ml(self.para, self.dataset, self.skf, self.ml)

        # load constant parameters
        self.para = constpara.constant_parameter(self.para)

        # read the dataset and then read the corresponding skf files
        # self.readdataset()
        self.specie_global = ['C']

        # interpolate integrals
        self.interpskf()

    def readdataset(self):
        """Read the corresponding hdf data of geometries.

        Notes:
            define para['dataType'] == 'hdf'
        """
        # read the hdf data, get the global atom specie
        # with h5py.File(self.para['referenceDataset'], 'r') as f:
        #     self.specie_global = f['globalGroup'].attrs['specieGlobal']

    def interpskf(self):
        """Read .skf data from skgen with various compR."""
        os.system('rm skf.hdf5')
        # optimize wavefunction compression radius
        if self.ml['typeSKinterpR'] == 'wavefunction':
            nametail = '_wav'

        # optimize density compression radius
        elif self.ml['typeSKinterpR'] == 'density':
            nametail = '_den'

        # optimize all the compression radius and keep all the same
        elif self.ml['typeSKinterpR'] == 'all':
            nametail = '_all'

        # read skf according to atom specie
        for namei in self.specie_global:
            for namej in self.specie_global:

                # get atom number and read corresponding directory
                if self.para['atomNumber_' + namei] < self.para['atomNumber_' + namej]:

                    # generate folder name
                    dire = self.para['directorySK'] + '/' + namei + \
                        '_' + namej + nametail

                    # get integral by interpolation
                    readt.SkInterpolator(
                        self.para, self.dataset,
                        self.skf, gridmesh=0.2).readskffile(
                            namei, namej, dire)

                    # write intepolation data as hdf
                    self.write_hdf5(namei, namej)
                else:

                    # generate folder name
                    dire = self.para['directorySK'] + '/' + namej + \
                        '_' + namei + nametail

                    # get integral by interpolation
                    readt.SkInterpolator(
                        self.para, self.dataset,
                        self.skf, gridmesh=0.2).readskffile(
                            namei, namej, dire)

                    # write intepolation data as hdf
                    self.write_hdf5(namei, namej)

    def write_hdf5(self, namei, namej):
        """Write each atom pair skf to hdf."""
        nameij = namei + namej

        # if onsite is fixed (False) or tunable (True)
        if self.skf['Lonsite']:
            name_onsite = 'onsite_rall'
            onsite_ = self.skf['onsite_rall' + nameij]

            # mass and r_cut
            name_massr = 'massrcut_rall'
            mass_rcut_ = self.skf['massrcut_rall' + nameij]

            # spe
            namespe = 'spe_rall'
            spe_ = self.skf['spe_rall' + nameij]

            # Hubbert
            namehubbert = 'uhubb_rall'
            hubbert = self.skf['uhubb_rall' + nameij]

        # gridmesh distance
        namedist = 'grid_dist_rall'
        distance = self.skf['grid_dist_rall' + nameij]

        # number of gridmesh
        namengridpoint = 'ngridpoint_rall'
        ngridpoint = self.skf['ngridpoint_rall' + nameij]

        # integrals
        namehs = 'hs_all_rall'
        hs = self.skf['hs_all_rall' + nameij]

        # number of compression radius
        namecompr = 'ncompr'
        compr = int(np.sqrt(self.skf['nfile_rall' + nameij]))

        # the repulsive parameters are dictionary parameters now
        if self.para['Lrepulsive']:
            pass

        # write the above to hdf
        with h5py.File('skf.hdf5', 'a') as f:
            print(nameij)
            g = f.create_group(nameij)

            # if onsite is fixed (False) or tunable (True)
            if self.skf['Lonsite']:
                g.create_dataset(name_onsite, data=onsite_)
                g.create_dataset(name_massr, data=mass_rcut_)
                g.create_dataset(namespe, data=spe_)
                g.create_dataset(namehubbert, data=hubbert)
            g.create_dataset(namedist, data=distance)
            g.create_dataset(namengridpoint, data=ngridpoint)
            g.create_dataset(namehs, data=hs)
            g.attrs[namecompr] = compr

            # create global group
            try:
                glob = f.create_group('globalgroup')
                glob.attrs['grid_dist'] = distance[0, 0]
            except:
                pass

    def read_skf(self):
        pass


if __name__ == '__main__':
    """Main function."""
    para = {'task': 'get_hdf_normal'}

    # generate hdf5 binary dataset of skf with various compression radii
    if para['task'] == 'get_hdf_compr':
        WriteSKComprR(para)

    # generate hdf5 binary dataset of normal skf
    elif para['task'] == 'get_hdf_normal':
        WriteSKNormal(para)
