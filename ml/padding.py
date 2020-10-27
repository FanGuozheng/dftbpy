#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 18:02:56 2020

@author: gz_fan
"""
import torch as t
import time
from torch.nn.utils.rnn import pad_sequence


def pad1d(in_tensor, transpose=True):
    """Pad a list of 1D tensor when size is not the same.

    Parameters
    ----------
    in_tensor : `torch.tensor`
        A list of 1D tensor.
    transpose : `bool`
        If transpose is True, the 1st dimension will be len(in_tensor).

    Returns
    -------
    out_tensor : `torch.tensor`
        Return 2D tensor, one dimension is len(in_tensor), one dimension
        is the max length of input tensor.

    """
    # input should be a list of tensor
    assert type(in_tensor) == list

    # return output
    if transpose:
        return pad_sequence(in_tensor).T
    elif not transpose:
        return pad_sequence(in_tensor)


def pad1d_(in_tensor):
    """Pad a list of 1D tensor when size is not the same.

    Parameters
    ----------
    in_tensor : `torch.tensor`
        A list of 1D tensor.
    transpose : `bool`
        If transpose is True, the 1st dimension will be len(in_tensor).

    Returns
    -------
    out_tensor : `torch.tensor`
        Return 2D tensor, one dimension is len(in_tensor), one dimension
        is the max length of input tensor.

    """
    # number in a batch
    nbatch = len(in_tensor)

    # max size in batch
    size = max(ibatch.shape for ibatch in in_tensor)

    # define output tensor
    out_batch = t.empty(nbatch, *size, dtype=in_tensor[0].dtype)

    # get the output tensor
    for ibtch, ivalue in enumerate(in_tensor):
        out_batch[ibtch, :ivalue.shape[0]] = ivalue

    return out_batch


def pad2d(in_tensor):
    """Pad a list of 2D tensor when size is not the same.

    Parameters
    ----------
    in_tensor : `torch.tensor`
        A list of 2D tensor.

    Returns
    -------
    out_tensor : `torch.tensor`
        Return 3D tensor, one dimension is len(in_tensor), one dimension
        is the max length of input tensor.

    """
    # number in a batch
    nbatch = len(in_tensor)

    # max size in batch
    size = max(ibatch.shape for ibatch in in_tensor)

    # define output tensor
    out_batch = t.zeros(nbatch, *size, dtype=in_tensor[0].dtype)

    # get the output tensor
    for ibtch, ivalue in enumerate(in_tensor):
        out_batch[ibtch, :ivalue.shape[0], :ivalue.shape[1]] = ivalue

    return out_batch
