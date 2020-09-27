# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module Description
------------------

The code here is for batch calculations.
"""
import torch as t


def block_diag(in_tensor):
    """Replace Pytorch block_diag because AttributeError problem.

    Parameters
    ----------
    in_tensor : `torch.tensor`
        A list of 2D tensor, each tensor shape[0] == shape[1].

    Returns
    -------
    out_tensor : `torch.tensor`
        Return 3D tensor where each in_tensor will be in the diagonal matrix.

    """
    # get the size of each tensor and define the output tensor
    size = [isize.shape[0] for isize in in_tensor]
    out_tensor = t.zeros((len(in_tensor), sum(size), sum(size)),
                         dtype=in_tensor[0].dtype)

    # copy each tensor to the diagonal
    for itensor, ivalue in enumerate(in_tensor):
        out_tensor[itensor, :ivalue.shape[0], :ivalue.shape[1]] = ivalue

    # return output
    return out_tensor


