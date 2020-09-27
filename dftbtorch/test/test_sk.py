#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test slater-koster transformation for batch calculation.

@author: gz_fan
"""
import numpy as np
import torch as t
import dftbtorch.slaterkoster as slaterkoster
from IO.basis import Basis, Bases


def main():
    batch = False

    systems = GetPosition(batch)

    if batch:
        atomic_numbers = t.tensor([[1, 1], [1, 1], [1, 1], [1, 1]])
        max_ls = t.tensor([1, 1, 1, 1])
    else:
        atomic_numbers = t.tensor([1, 1])
        max_ls = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

    orbital_id = Basis(atomic_numbers, max_ls)
    slaterkoster.skt(systems, orbital_id, integral_feed=12)


class GetPosition:

    def __init__(self, batch):
        if batch:
            self.position()
        else:
            self.position_batch()

    def position(self):
        return t.randn(2, 2)

    def position_batch(self):
        return t.randn(4, 2, 2)

if __name__ == "__main__":
    main()