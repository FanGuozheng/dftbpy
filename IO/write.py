"""This code is to print calculation information."""
import torch as t


class Print:
    """Print DFTB results."""

    def __init__(self, para, geo, skf):
        """Initialize parameters."""
        self.para = para
        self.geo = geo
        self.skf = skf

        # set print precision
        t.set_printoptions(precision=15)

    def print_scf_title(self):
        """Print DFTB title before running DFTB."""
        # non SCC molecule
        if not self.para['Lperiodic'] and self.para['scc'] == 'nonscc':
            print('*' * 80)
            print(' ' * 25, 'non-periodic non-SCC-DFTB')
            print('*' * 80)

        # SCC molecule
        elif not self.para['Lperiodic'] and self.para['scc'] == 'scc':
            print('*' * 80)
            print(' ' * 30, 'non-periodic SCC-DFTB')
            print('*' * 80)

        # XLBOMD molecule
        elif not self.para['Lperiodic'] and self.para['scc'] == 'xlbomd':
            print('*' * 80)
            print(' ' * 25, 'non-periodic xlbomd-DFTB')
            print('*' * 80)

        # SCC solid
        elif self.para['Lperiodic'] and self.para['scc'] == 'scc':
            print('*' * 80)
            print(' ' * 30, 'periodic SCC-DFTB')
            print('*' * 80)

    def print_energy(self, iiter, energy, batch, nbatch=None, Lprint=False):
        """Print energy in each SCC loops."""
        # for the 0th loop, dE == energy
        if iiter == 0:
            dE = energy[iiter].detach()

            # single system
            if not batch:

                # print the first title line
                print('iteration', ' ' * 8, 'energy', ' ' * 20, 'dE')
                energy_ = energy[iiter].squeeze(0)
                dE_ = dE.squeeze(0)

                # print 0th loop energy
                print(f'{iiter:5} {energy_.detach():25}', f'{dE_:25}')
                return dE

            # multi system
            elif batch:

                # if print the energy
                if Lprint:
                    assert nbatch is not None
                    # print the first title line
                    print('iteration', ' ' * 8,
                          'energy list', ' ' * 20, 'dE list')

                    # print 0th loop energy
                    for i in range(nbatch):
                        print(f'{iiter:5} {energy[iiter].detach():25}',
                              f'{dE:25}')
                return dE

        # for loops >= 1
        elif iiter >= 1:

            # single system
            if not batch:

                # get energy
                dE = energy[iiter].detach() - energy[iiter - 1].detach()
                energy_ = energy[iiter].squeeze(0)
                dE_ = dE.squeeze(0)

                # print nth loop energy
                print(f'{iiter:5} {energy_.detach():25}', f'{dE_:25}')
                return dE

            # multi system
            elif batch:

                # get energy
                dE = energy[iiter].detach() - energy[iiter - 1].detach()

                # if print the energy
                if Lprint:
                    assert nbatch is not None

                    # print nth loop energy
                    for i in range(nbatch):
                        print(f'{iiter:5} {energy[iiter].detach():25}',
                              f'{dE:25}')
                return dE

    def print_charge(self, iiter, charge, nat):
        """Print charge in each SCC loops."""
        # for the 0th loop, dQ == energy
        if iiter == 0:
            dQ = charge[0].detach().sum() / nat

            # print title
            print('iteration', ' ' * 8, 'total charge', ' ' * 20, 'dQ')

            # print 0th loop charge
            print(f'{iiter:5} {charge[-1].detach().sum():25}', f'{dQ:25}')
            return dQ

        # print dE and charge
        elif iiter >= 1:

            # get charge
            dQ = abs((charge[-2].detach() - charge[-1].detach())).sum() / nat

            # print nth loop charge
            print(f'{iiter:5} {charge[-1].detach().sum():25}', f'{dQ:25}')
            return dQ

    def print_dftb_tail(self):
        """Print DFTB calculation physical results."""
        # print atoms
        print('list of atoms in batch:', self.geo['atomnameall'])

        # print charge
        print('\n charge (e): ', self.para['charge'])

        # print dipole
        print('\n dipole (eAng): ', self.para['dipole'])

        # print energy
        print('\n TS energy (Hartree): ', self.para['H0_energy'])

        # print SCC energy
        if self.para['scc'] == 'scc':
            print('\n Coulomb energy (Hartree): ', -self.para['coul_energy'])

        # print repulsive energy
        if self.para['Lrepulsive']:
            print('\n repulsive energy (Hartree): ', self.para['rep_energy'])
            print('\n total energy (Hartree): ', self.para['energy'])

        # do not calculate repulsive energy
        else:
            print('\n repulsive energy (Hartree): ', 0.)
            print('\n total energy (Hartree): ', self.para['energy'])

        # print MBD-DFTB information
        if self.para['LMBD_DFTB']:

            # print charge population analysis
            print('\n CPA: ', self.para['cpa'])

            # print polarizability
            print('\n polarizability: ',
                  self.para['alpha_mbd'].detach().float())
