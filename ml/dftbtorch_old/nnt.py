#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch


class DFTBtorch(torch.nn.module):

    def __init__(self, modules):
        super(DFTBtorch, self).__init__()
        self.module_list = torch.nn.ModuleList(modules)

    def __getitem__(self, i):
        return self.module_list[i]

    def forward(self, species_aev):
        species, aev = species_aev
        species_ = species.flatten()
        aev = aev.flatten(0, 1)

        output = torch.full(species_.shape, self.padding_fill,
                            dtype=aev.dtype)
        i = 0
        for m in self.module_list:
            mask = (species_ == i)
            i += 1
            midx = mask.nonzero().flatten()
            if midx.shape[0] > 0:
                input_ = aev.index_select(0, midx)
                output.masked_scatter_(mask, m(input_).flatten())
        output = output.view_as(species)
        return species, torch.sum(output, dim=1)


class Ensemble(torch.nn.Module):
    """Compute the average output of an ensemble of modules."""

    def __init__(self, modules):
        super(Ensemble, self).__init__()
        assert len(modules) == 8
        self.model0 = modules[0]
        self.model1 = modules[1]
        self.model2 = modules[2]
        self.model3 = modules[3]
        self.model4 = modules[4]
        self.model5 = modules[5]
        self.model6 = modules[6]
        self.model7 = modules[7]

    def __getitem__(self, i):
        return [self.model0, self.model1, self.model2, self.model3,
                self.model4, self.model5, self.model6, self.model7][i]

    def forward(self, species_input):
        species, _ = species_input
        sum_ = self.model0(species_input)[1] + self.model1(species_input)[1] \
            + self.model2(species_input)[1] + self.model3(species_input)[1] \
            + self.model4(species_input)[1] + self.model5(species_input)[1] \
            + self.model6(species_input)[1] + self.model7(species_input)[1]
        return species, sum_ / 8.0


class Sequential(torch.nn.Module):
    """Modified Sequential module that accept Tuple type as input"""

    def __init__(self, *modules):
        super(Sequential, self).__init__()
        self.modules_list = torch.nn.ModuleList(modules)

    def forward(self, input_):
        for module in self.modules_list:
            input_ = module(input_)
        return input_


class Gaussian(torch.nn.Module):
    """Gaussian activation"""
    def forward(self, x):
        return torch.exp(- x * x)
