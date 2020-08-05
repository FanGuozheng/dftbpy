import torch as t
from abc import ABC, abstractmethod


"""
This module contains the abstract base class on which all mixers are based.


Notes
-----
Updating parameters using gradients that have been backpropagated through a
large number of mixing iterations may result in poor gradients.

Todo
----
- Enable toggling of vector override behaviour.
- Amend terminology; dQ is not necessarily whats being passed in. More commonly
  it is the current atomic charges q.
  [Priority: Low]
- Replace pytorch math operations with their inplace equivalents wherever
  applicable. [Priority: Low]
"""

# <Helper_Functions>
# def __increment(self):
#     """Wrapper func
#     tion to increment the step ``mixing_step`` parameter.
#     """

# </Helper_Functions>


class Mixer(ABC):
    """This is the abstract base class up on which all, charge, mixers are based.

    Parameters
    ----------
    mix_param : `float`
        Mixing parameter, ∈(0, 1), controls the extent of charge mixing. Larger
        values result in more aggressive mixing.

    **kwargs
        Additional keyword arguments:


    Properties
    ----------
    converged : `bool`
        Boolean indicating convergence status according to charges & and other
        mixer specific properties.
    _step_number : `int`:
        Integer which tracks the number of mixing iterations performed thus far.
        Updated via wrapper function prior to mixing operation.
    _delta : `torch.tensor` [`float`]
        Zero dimensional tensor holding the difference between the current and
        previous charges. Use for checking convergence.

    Notes
    -----
    `dQ_init` is handed elsewhere.

    Todo
    ----
    - Fully vectorise mixing operations to enable multiple systems to be mixed
      simultaneously. Will need code to purge selective parts of the history
      associated with converged systems no longer requiring SCC operations,
      and some code to deal with saving iteration counts for each system.
      [Moderate]
    - Abstract _mixing_step incrementation to an external wrapper function.
        [Low]
    - Add verbosity setting. [Low]
    - Add logging options (i.e. save fock history) [Low]
    """

    # Maximum permitted difference in charge between successive iterations.
    charge_threashold = 1E-5

    def __init__(self, mix_param, **kwargs):

        self.mix_param = mix_param

        self._step_number = 0

        self._delta = None

    @abstractmethod
    def __call__(self, dQ_new, **kwargs):
        """Performs the mixing operation & returns the newly mixed dQ.
        """
        # Mixing operation would best be performed in a different function
        # this cleans up
        pass

    @property
    def converged(self):
        """Getter for ``converged`` property.

        Returns
        -------
        converged : `bool`
            Boolien indicating convergence status
        """
        # By default, only the charge difference is considered however other
        # mixer specific terms could also be added by overriding this locally.
        return t.max(t.abs(self._delta), -1).values < self.charge_threashold

    def reset(self):
        """Resets the mixer to its initial state. At the very least this must
        reset the iteration counter.
        """

        raise NotImplementedError(
            'Function not implemented or is invalid for the '
            f'"{type(self).__name__}" mixer')

    def _initialisation(self):
        """Conducts any required initialisation operations. This helps to clean
        up the __init__ function and should only ever be called there.

        Notes
        -----
        This may not actually be necessary.
        """
        raise NotImplementedError('Function is not yet implemented')

    def __repr__(self):
        """Returns representative string (mostly used for logging).

        Returns
        -------
        r: `str`
            A string representation of the mixer.
        """
        pass


class Simple(Mixer):
    """Implementation of simple mixer.

    Parameters
    ----------
    mix_param : `float`, optional
        Mixing parameter, ∈(0, 1), controls the extent of charge mixing. Larger
        values result in more aggressive mixing. [DEFAULT=0.05]

    **kwargs
        Additional keyword arguments:
            ``dQ_init``:
                Pre-initialises the mixer so ``dQ_old`` does not have to be
                passed in the first mixing call. (`torch.tensor`). [DEFAULT=None]

    Properties
    ----------
    converged : `bool`
        Boolean indicating convergence status according to charges & and other
        mixer specific properties.
    _step_number : `int`:
        Integer which tracks the number of mixing iterations performed thus far.
        Updated via wrapper function prior to mixing operation.
    _delta : `torch.tensor` [`float`]
        Zero dimensional tensor holding the difference between the current and
        previous charges. Use for checking convergence.

    Todo
    ----
    - Document dQ_old. [Low]

    """
    def __init__(self, mix_param=0.05, **kwargs):
        super().__init__(mix_param, **kwargs)

        #|!>
        # `dQ_init`'s behavior is mixer specific, thus it must
        # be a child rather than parent property.
        self.dQ_old = kwargs.get('dQ_init', None)

    def __call__(self, dQ_new, dQ_old=None, **kwargs):
        """Performs the simple mixing operation & returns the mixed dQ vector.

        Parameters
        ----------
        dQ_new : `torch.tensor` [`float`]
            The new dQ vector that is to be mixed.
        dQ_old : `torch.tensor` [`float`], optional
            Previous dQ with which ``dQ_new`` is to be mixed. Only required for
            the first mix operation, and only if ``dQ_init`` was not specified
            at instantiation. [see notes]

        **kwargs
            Additional keyword arguments:
                ``mix_param``:
                    Updates the mixing parameter to ``x`` for the current and all
                    subsequent mixing operations (`float`).
                ``inplace``: `bool`
                    By default a new tensor instance is returned, however if
                    ``inplace`` is set to True then dQ_old will be overridden.
                    (`bool`) [DEFAULT=False]

        Returns
        -------
        dQ_mixed : `torch.tensor` [`float`]
            Mixed dQ vector.

        Notes
        -----
        The ``dQ_old`` parameter becomes optional after the first mixing call.
        This is because ``dQ_old`` is commonly just the``dQ_mixed`` value from
        the previous cycle; which is stored locally. If ``dQ_init`` is specified
        during class instantiation; ``dQ_old`` can be omitted completely. If
        specified, ``dQ_old`` will be used rather than the stored value.

        This will automatically assign the newly mixed dQ vector to the class'
        ``self.dQ_old`` parameter.

        Warnings
        --------
        This currently overwrites the dQ_old vector.

        Todo
        ----
        - For the ``inplace`` keyword to result in true inplace behaviour no
          tensor should be returned. [Priority: Low]
        - Update comments regarding self assignment
        """

        # Increment _step_number variable (should be abstracted to a wrapper)
        self._step_number += 1

        # Assign dQ_old to the stored value if it was not specified
        dQ_old = self.dQ_old if dQ_old is None else dQ_old

        # Check if a new mixing parameter "x" has been provided.
        if 'mix_param' in kwargs:
            # If so update the class' mixing parameter to the new value.
            self.mix_param = kwargs['mix_param']

        # Calculate the difference between the new and old dQ vectors & add
        # x times that difference to the old dQ vector to yield the mixed dQ.
        delta = (dQ_new - dQ_old)
        dQ_mixed = dQ_old + delta * self.mix_param

        # Update the dQ_old attribute
        self.dQ_old = dQ_mixed

        # Update the self._delta property
        self._delta = (dQ_new - dQ_old)

        # Return the mixed dQ vector.
        return dQ_mixed

    def reset(self, dQ_init=None):
        """Resets the mixer instance read to be used again.

        Parameters
        ----------
        dQ_init : `torch.tensor` [`float`], optional
            Pre-initialises the mixer so ``dQ_old`` does not have to be passed
            in the first mixing call. (`torch.tensor`). [DEFAULT=None]
        """
        # Reset the step iteration counter
        self._step_number = 0

        # Check if dQ_init was specified
        if dQ_init is not None:
            # If so assign it
            self.dQ_old = dQ_init

        # Purge the _delta attribute
        self._delta = None

    def __repr__(self):
        """Returns representative string (mostly used for logging).

        Returns
        -------
        r: `str`
            A string representation of the mixer.
        """
        # Return an informational string.
        return f'{self.__class__.__qualname__}: step={self._step_number}'

    def cull(self, cull_list):
        """Purge specific systems from the mixer when in multi-system mode. This
        is useful when a subset of systems have converged during the mixing of a
        large number of systems.

        Parameters
        ----------
        cull_list : `torch.tensor` [`bool`]
            Tensor with a boolean per-system: True indicates a system should be
            removed from history and that it will no longer be specified during
            mixing, False indicates a system should be kept.
        """
        # Invert the cull_list, gather & reassign self.dQ_old and self._delta
        # so only those marked False remain.
        self.dQ_old = self.dQ_old[~cull_list]
        self._delta = self._delta[~cull_list]


class Anderson(Mixer):
    """Performs Anderson mixing of input vectors through an iterative process.
    Upon instantiation, a callable class instance will be returned. Calls to
    this instance will take, as its arguments, two input vectors and return
    a single mixed vector.

    Parameters
    ----------
    mix_param : `float`, optional
        Mixing parameter, ∈(0, 1), controls the extent of charge mixing. Larger
        values result in more aggressive mixing. [DEFAULT=0.05]
    generations : `int`, optional
        Number of generations to consider during mixing. [DEFAULT=4]
    init_mix_param : `float`, optional
        Simple mixing parameter to use until `_step_number` >= `generations`.
        [DEFAULT=0.01]
    diagonal_offset : `float`, optioinal
        Value added to the equation system's diagonal elements to help prevent
        the emergence of linear dependencies during the mixing processes. If
        set to None then rescaling will be disabled. [DEFAULT=0.01]

    **kwargs
        Additional keyword arguments:
            ``dQ_init``:
                Pre-initialises the mixer so ``dQ_old`` does not have to be
                passed in the first mixing call. (`torch.tensor`). [DEFAULT=None]

    Properties
    ----------
    converged : `bool`
        Boolean indicating convergence status according to charges & and other
        mixer specific properties.
    _step_number : `int`:
        Integer which tracks the number of mixing iterations performed thus far.
        Updated via wrapper function prior to mixing operation.
    _delta : `torch.tensor` [`float`]
        Zero dimensional tensor holding the difference between the current and
        previous charges. Use for checking convergence.

    Attributes
    ----------
    _dQs `torch.tensor` [`float`]
        History of dQ vectors (dQ_old).
    _F : `torch.tensor` [`float`]
        History of difference between dQ_new and dQ_old vectors.

    Notes
    -----
    The Anderson mixing functions primarily follow the equations set out by
    Eyert [1]. However, the borrows heavily from the DFTB+ implementation [2].
    A more in depth discussion on the topic is given by Anderson [3].

    This will perform simple mixing until the number of interactions surpasses
    the number of generations requested.


    Warnings
    --------
    Setting `generations` too high can lead to a linearly dependent set of
    equations. However, this effect can be mitigated through the use of the
    ``diagonal_offset`` parameter.

    This deviates from the DFTB+ implementation in that it does not compute or
    use the theta zero values. Which are currently causing stability issues,
    see the "to do" list for more information.

    .. [1] Eyert, V. (1996). A Comparative Study on Methods for Convergence
       Acceleration of Iterative Vector Sequences. Journal of Computational
       Physics, 124(2), 271–285.
    .. [2] Hourahine, B., Aradi, B., Blum, V., Frauenheim, T. et al., (2020).
       DFTB+, a software package for efficient approximate density functional
       theory based atomistic simulations. The Journal of Chemical Physics,
       152(12), 124101.
    .. [3] Anderson, D. G. M. (2018). Comments on “Anderson Acceleration,
       Mixing and Extrapolation.” Numerical Algorithms, 80(1), 135–234.

    Todo
    ----
    - Identify why using theta 0 in a manner similar to DFTB+ causes such
      highly divergent behaviour. Lines concerned:
            theta_0 = 1 - torch.sum(thetas)
            dQ_bar += theta_0 * self._dQs[0]
            F_bar += theta_0 * self._F[0]
      [Priority: Critical]
    - Document dQ_list [Priority: Low]
    """
    def __init__(self, mix_param=0.05, generations=4, init_mix_param=0.01,
                 diagonal_offset=0.01, **kwargs):
        super().__init__(mix_param, **kwargs)

        self.generations = generations

        self.init_mix_param = init_mix_param

        self.diagonal_offset = diagonal_offset

        # If `dQ_init` is specified; _dQs & _F can be generated here
        if 'dQ_init' in kwargs:
            self.__build_F_and_dQs(kwargs['dQ_init'])
            # Assign dQ_init to the _dQs list
            self._dQs[0] = kwargs['dQ_init']
        # Otherwise create placeholders to be overridden
        else:
            self._dQs = None
            self._F = None

    def __build_F_and_dQs(self, dQ_first):
        """Builds the F and dQs matrices. This is mainly used during the
        initialisation prices, and has been abstracted to avoid code repetition
        and help tidy things up.

        Parameters
        ----------
        dQ_first : `torch.tensor`
            The first dQ tensor on which the new are to be based.

        Notes
        -----
        This function auto-assigns to the class variables so it returns nothing.

        Todo
        ----
        - Rename this function to "_initialisation" to bring it into line with
          the base class definition.
        """
        # Clone size and type settings.
        size = (self.generations + 1, *tuple(dQ_first.shape))
        dtype = dQ_first.dtype
        self._F = t.zeros(size, dtype=dtype)
        self._dQs = t.zeros(size, dtype=dtype)

    def __call__(self, dQ_new, dQ_old=None, **kwargs):
        """This performs the actual Anderson mixing operation.

        Parameters
        ----------
        dQ_new : `torch.tensor` [`float`]
            The newly calculated, pure, dQ vector that is to be mixed.
        dQ_old : `torch.tensor` [`float`], optional
            The previous, mixed, dQ vector which represents the current dQ value
            that is to be updated. As this


        Notes
        -----
        This implementation is based off the Eyert paper, DFTB+ implementation
        and "Comments on Anderson Acceleration, Mixing and Extrapolation".

        This solver is currently equipped process the mixing of vectors only.
        Passing in Fock matrices for example is likely to result in a crash.
        The method used to set up the linear equation must be updated to enable
        matrix mixing.

        Todo
        ----
        - Find a more elegant approach to construct equation 4.3 (Eyert) that
          does not require two implementations. [Priority: Low]
        - Merge single & batch operations into one shape agnostic operation.
          This primarily converns equations 4.1-4.3. [Priority: Moderate]

        Warnings
        --------
        When vectoring the code for multi-system mixing, consideration must be
        given to the torch.solve operation along with the squeeze & unsqueeze
        operations associate with it.

        """
        # In this function "F" is used to refer to the delta:
        #   F = dQ_new - dQ_previous
        # this is in line with the notation used by Eyert. However, "dQs"
        # takes the place of what would be "x" in Eyert's equations. These
        # are the previous, mixed, charge values.

        # Batch-wise operations must be handled slightly different so identity
        # if this is a batch operation (i.e. multi-system)
        batch = dQ_new.dim() != 1

        # Increment _step_number variable (should be abstracted to a wrapper)
        self._step_number += 1

        # If this is the 1st cycle; then the F & dQs arrays must be built.
        # This will have been done during the init if dQ_init was given.
        if self._step_number == 1 and self._F is None:
            self.__build_F_and_dQs(dQ_new)

        # If dQ_old specified; overwrite last entry in self._dQs.
        if dQ_old is not None:
            self._dQs[0] = dQ_old

        # Calculate dQ_new - dQ_old delta & assign to the delta history _F
        self._F[0] = dQ_new - dQ_old

        # If a sufficient history has been built up then use Anderson mixing
        if self._step_number > self.generations:
            # Setup and solve the linear equation system, as described in equation
            # 4.3 (Eyert), to get the coefficients "thetas":
            #   a(i,j) =  <F(l) - F(l-i)|F(l) - F(l-j)>
            #   b(i)   =  <F(l) - F(l-i)|F(l)>
            # here dF = <F(l) - F(l-i)|
            dF = self._F[0] - self._F[1:]
            # Approach taken depends on if it is a single or a batch of vectors
            # that are to be mixed. (At least if we want to avoid for loops)
            if not batch:
                a, b = dF @ dF.T, dF @ self._F[0]
            else:
                dF = dF.transpose(1, 0)
                a = dF @ dF.transpose(1, 2)
                b = t.squeeze(t.bmm(dF, t.unsqueeze(self._F[0], -1)))

            # Rescale diagonals to prevent linear dependence in a manner akin
            # to equation 8.2 (Eyert)
            if self.diagonal_offset is not None:
                # Add (1 + diagonal_offset**2) to the diagonals of "a"
                a += t.eye(a.shape[-1]) * (self.diagonal_offset ** 2)

            # Solve for the coefficients. As torch.solve cannot solve for 1D
            # tensors a blank dimension must be added
            thetas = t.squeeze(t.solve(t.unsqueeze(b, -1), a)[0])

            # Construct the 2nd terms of equations 4.1 and 4.2 (Eyert). These are
            # the "averaged" histories of x and F respectively:
            #   x_bar = sum(j=1 -> m) ϑ_j(l) * (|x(l-j)> - |x(l)>)
            #   F_bar = sum(j=1 -> m) ϑ_j(l) * (|F(l-j)> - |F(l)>)
            # These are not the x_bar & F_var values of equations 4.1 & 4.2 (Eyert)
            # yet as they are still missing the 1st terms.
            if not batch:
                dQ_bar = t.sum(thetas * (self._dQs[1:] - self._dQs[0]).T, 1)
                F_bar = t.sum(thetas * (-dF).T, 1)
            else:
                dQ_bar = t.sum(thetas.T * (self._dQs[1:] - self._dQs[0]).transpose(1, 0).T, 1).T
                F_bar = t.sum(thetas.T * (-dF).T, 1).T

            # The first terms of equations 4.1 & 4.2 (Eyert):
            #   4.1: |x(l)> and & 4.2: |F(l)>
            # Have been replaced by:
            #   ϑ_0(l) * |x(j)> and ϑ_0(l) * |x(j)>
            # respectively, where "ϑ_0(l)" is the coefficient for the current step
            # and is defined as (Anderson):
            #   ϑ_0(l) = 1 - sum(j=1 -> m) ϑ_j(l)

            # Currently using the same method as DFTB+ causes stability issues.
            theta_0 = 1 - t.sum(thetas)
            # dQ_bar += theta_0 * self._dQs[0]
            # F_bar += theta_0 * self._F[0]
            dQ_bar += self._dQs[0]
            F_bar += self._F[0]

            # Calculate the new mixed dQ following equation 4.4 (Eyert):
            #   |x(l+1)> = |x_bar(l)> + beta(l)|F_bar(l)>
            # where "beta" is the mixing parameter
            dQ_mixed = dQ_bar + (self.mix_param * F_bar)

        # If there is insufficient history for Anderson; use simple mixing
        else:
            dQ_mixed = self._dQs[0] + (self._F[0] * self.init_mix_param)

        # Shift F & dQ histories over; a roll follow by a reassignment is
        # necessary to avoid a pytorch inplace error. (gradients remain intact)
        self._F = t.roll(self._F, 1, 0)
        self._dQs = t.roll(self._dQs, 1, 0)

        # Assign the mixed dQ to the dQs history array. The last dQ_mixed value
        # is saved on the assumption that it will be used in the next step.
        self._dQs[0] = dQ_mixed

        # Save the last difference to _delta
        self._delta = self._F[1]

        # Return the mixed parameter
        return dQ_mixed

    def reset(self, dQ_init=None):
        """Resets the mixer instance read to be used again.

        Parameters
        ----------
        dQ_init : `torch.tensor` [`float`], optional
            Pre-initialises the mixer so ``dQ_old`` does not have to be passed
            in the first mixing call. (`torch.tensor`). [DEFAULT=None]
        """
        # Reset the step iteration counter
        self._step_number = 0

        # Check if dQ_init was specified
        if dQ_init is not None:
            # Build the F and dQs lists
            self.__build_F_and_dQs(dQ_init)
            # Assign dQ_init to the _dQs list
            self._dQs[0] = dQ_init
        # Otherwise
        else:
            # Purge self._F & _dQs attributes
            self._dQs = None
            self._F = None

        # Reset the _delta attribute
        self._delta = None

    def __repr__(self):
        """Returns representative string (mostly used for logging).

        Returns
        -------
        r: `str`
            A string representation of the mixer.
        """
        # Return an informational string.
        return f'{self.__class__.__qualname__}: step={self._step_number}'

    def cull(self, cull_list):
        """Purge specific systems from the mixer when in multi-system mode. This
        is useful when a subset of systems have converged during the mixing of a
        large number of systems.

        Parameters
        ----------
        cull_list : `torch.tensor` [`bool`]
            Tensor with a boolean per-system: True indicates a system should be
            removed from history and that it will no longer be specified during
            mixing, False indicates a system should be kept.
        """
        # Invert the cull_list, gather & reassign self._delta self._dQs &
        # self._F so only those marked False remain.
        self._delta = self._delta[~cull_list]
        self._dQs = self._dQs[:, ~cull_list]
        self._F = self._F[:, ~cull_list]
        # If only a single system remains; strip the extra axis
        if t.sum(~cull_list) == 1:
            self._delta = t.squeeze(self._delta)
            self._dQs = t.squeeze(self._dQs)
            self._F = t.squeeze(self._F)


class Broyden(Mixer):
    """Broyden mixing method.

    Reference:
        D. D. Johnson, PRB, 38 (18), 1988.

    """

    def __init__(self, mix_param=0.05, generations=60, init_mix_param=0.01,
                 diagonal_offset=0.01, **kwargs):
        super().__init__(mix_param, **kwargs)

        self.init_mix_param = init_mix_param

        self.generations = generations

        self.diagonal_offset = diagonal_offset

        # If `dQ_init` is specified; _dQs & _F can be generated here
        if 'dQ_init' in kwargs:
            self.__build_F_and_dQs(kwargs['dQ_init'])
            # Assign dQ_init to the _dQs list
            self._dQs[0] = kwargs['dQ_init']
        # Otherwise create placeholders to be overridden
        else:
            self._dQs = None
            self._F = None

    def __build_F_and_dQs(self, dQ_first):
        """Builds the F and dQs matrices. This is mainly used during the
        initialisation prices, and has been abstracted to avoid code repetition
        and help tidy things up.

        Parameters
        ----------
        dQ_first : `torch.tensor`
            The first dQ tensor on which the new are to be based.

        Notes
        -----
        This function auto-assigns to the class variables so it returns nothing.

        Todo
        ----
        - Rename this function to "_initialisation" to bring it into line with
          the base class definition.
        """
        # Clone size and type settings.
        size = (self.generations + 1, *tuple(dQ_first.shape))
        dtype = dQ_first.dtype
        self._F = t.zeros(size, dtype=dtype)
        self._dQs = t.zeros(size, dtype=dtype)
        self.ww = t.zeros((self.generations + 1), dtype=dtype)

    def __call__(self, dQ_new, dQ_old=None, **kwargs):
        # Batch-wise operations must be handled slightly different so identity
        # if this is a batch operation (i.e. multi-system)
        batch = dQ_new.dim() != 1

        # Increment _step_number variable (should be abstracted to a wrapper)
        self._step_number += 1

        # If this is the 1st cycle; then the F & dQs arrays must be built.
        # This will have been done during the init if dQ_init was given.
        if self._step_number == 1 and self._F is None:
            self.__build_F_and_dQs(dQ_new)

        # If dQ_old specified; overwrite last entry in self._dQs.
        if dQ_old is not None:
            self._dQs[0] = dQ_old

        # Calculate dQ_new - dQ_old delta & assign to the delta history _F
        self._F[0] = dQ_new - dQ_old

        cc = t.zeros((self._step_number, self._step_number), dtype=t.float64)

        # temporal a parameter for current interation
        aa_ = []

        # temporal c parameter for current interation
        cc_ = []

        weightfrac = 1e-2

        omega0 = 1e-2

        # max weight
        maxweight = 1E5

        # min weight
        minweight = 1.0

        alpha = self.mix_param

        # get temporal parameter of last interation: <dF|dF>
        ww_ = t.sqrt(self._F[0] @ self._F[0])

        # build omega (ww) according to charge difference
        # if weight from last loop is larger than: weightfrac / maxweight
        if ww_ > weightfrac / maxweight:
            self.ww[self._step_number - 1] = weightfrac / ww_

        # here the gradient may break
        else:
            self.ww[self._step_number - 1] = maxweight

        # if weight is smaller than: minweight
        if self.ww[self._step_number - 1] < minweight:

            # here the gradient may break
            self.ww[self._step_number - 1] = minweight

        # get updated charge difference
        qdiff.append(qatom_ - qmix[-1])

        # temporal (latest) difference of charge difference
        df_ = qdiff[-1] - qdiff[-2]

        # get normalized difference of charge difference
        ndf = 1 / t.sqrt(df_ @ df_) * df_

        if self._step_number >= 2:
            # build loop from first loop to last loop
            [aa_.append(t.tensor(idf, dtype=t.float64) @ ndf) for idf in
             self.df[:-1]]

            # build loop from first loop to last loop
            [cc_.append(t.tensor(idf, dtype=t.float64) @ t.tensor(qdiff[-1],
             dtype=t.float64)) for idf in self.df[:-1]]

            # update last a parameter
            self.aa[: self._step_number - 1, self._step_number] = t.tensor(aa_, dtype=t.float64)
            self.aa[self._step_number, : self._step_number - 1] = t.tensor(aa_, dtype=t.float64)

            # update last c parameter
            self.cc[: self._step_number - 1, self._step_number] = t.tensor(cc_, dtype=t.float64)
            self.cc[self._step_number, : self._step_number - 1] = t.tensor(cc_, dtype=t.float64)

        self.aa[self._step_number - 1, self._step_number - 1] = 1.0

        # update last c parameter
        self.cc[self._step_number - 1, self._step_number - 1] = self.ww[self._step_number - 1] * (ndf @ qdiff[-1])

        for ii in range(self._step_number):
            self.beta[:self._step_number, ii] = self.ww[:self._step_number] * self.ww[ii] * \
                self.aa[:self._step_number, ii]

            self.beta[ii, ii] = self.beta[ii, ii] + omega0 ** 2

        self.beta[: self._step_number, : self._step_number] = t.inverse(self.beta[: self._step_number, : self._step_number])

        gamma = t.mm(cc[: self._step_number, : self._step_number], self.beta[: self._step_number, : self._step_number])

        # add difference of charge difference
        self.df.append(ndf)

        df = alpha * ndf + 1 / t.sqrt(df_ @ df_) * (qmix[-1] - qmix[-2])

        qmix_ = qmix[-1] + alpha * qdiff[-1]

        for ii in range(self._step_number):
            qmix_ = qmix_ - self.ww[ii] * gamma[0, ii] * self.uu[ii]

        qmix_ = qmix_ - self.ww[self._step_number - 1] * gamma[0, self._step_number - 1] * df

        self.uu.append(df)
        # Shift F & dQ histories over; a roll follow by a reassignment is
        # necessary to avoid a pytorch inplace error. (gradients remain intact)
        self._F = t.roll(self._F, 1, 0)
        self._dQs = t.roll(self._dQs, 1, 0)

        # Assign the mixed dQ to the dQs history array. The last dQ_mixed value
        # is saved on the assumption that it will be used in the next step.
        self._dQs[0] = dQ_mixed

        # Save the last difference to _delta
        self._delta = self._F[1]

        # Return the mixed parameter
        return dQ_mixed

    def __build_F_and_dQs(self, dQ_first):
        """Builds the F and dQs matrices. This is mainly used during the
        initialisation prices, and has been abstracted to avoid code repetition
        and help tidy things up.

        Parameters
        ----------
        dQ_first : `torch.tensor`
            The first dQ tensor on which the new are to be based.

        Notes
        -----
        This function auto-assigns to the class variables so it returns nothing.

        Todo
        ----
        - Rename this function to "_initialisation" to bring it into line with
          the base class definition.
        """
        # Clone size and type settings.
        size = (self.generations + 1, *tuple(dQ_first.shape))
        dtype = dQ_first.dtype
        self._F = t.zeros(size, dtype=dtype)
        self._dQs = t.zeros(size, dtype=dtype)

    def reset(self, dQ_init=None):
        """Resets the mixer instance read to be used again.

        Parameters
        ----------
        dQ_init : `torch.tensor` [`float`], optional
            Pre-initialises the mixer so ``dQ_old`` does not have to be passed
            in the first mixing call. (`torch.tensor`). [DEFAULT=None]
        """
        # Reset the step iteration counter
        self._step_number = 0

        # Check if dQ_init was specified
        if dQ_init is not None:
            # Build the F and dQs lists
            self.__build_F_and_dQs(dQ_init)
            # Assign dQ_init to the _dQs list
            self._dQs[0] = dQ_init
        # Otherwise
        else:
            # Purge self._F & _dQs attributes
            self._dQs = None
            self._F = None

        # Reset the _delta attribute
        self._delta = None

    def __repr__(self):
        pass