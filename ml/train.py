#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 11:09:30 2020

"""
import os
import time
import dftbtorch.init_parameter as initpara
import dftbtorch.dftb_torch as dftbt
from IO.load import LoadData

class Train:

    # plot data from ML
    #if para['Lml_acsf']:
    #   plot.plot_ml_feature(para)
    #else:
    #    plot.plot_ml_compr(para)

    def __init__(self, task='compression_r', dataset=None):
        """"""
        time_begin = time.time()
        # main task of machine learning,
        # e.g., compression_r: training compression radius
        self.task = task

        # the training dataset
        if 'dataType' not in dataset.keys():
            self.datasettype = 'hdf'
        elif 'dataType' in dataset.keys():
            self.datasettype = dataset['dataType']

        # dataset name and path
        if 'path_dataset' and 'name_dataset' in dataset.keys():

            self.path_name_dataset = os.path.join(dataset['path_dataset'],
                                                  dataset['name_dataset'])
        else:
            self.path_name_dataset = os.path.join(
                os.getcwd(), '../data/dataset/testfile.hdf5')

        # get the initial parameter dictionary
        self.init_para()

        # load dataset
        self.load_dataset()

        # get initial parameters
        dftbt.Initialization()
        geometry = {}

        if task == 'compression_r':
            self.train_compression_r()
        time_end = time.time()
        print("total training time:", time_end - time_begin)


    def init_para(self):
        self.para = {}
        self.dataset = {}
        self.ml = {}
        self.skf = {}
        self.geometry = {}

    def load_dataset(self, dataset_type='hdf'):
        if not self.datasettype == 'hdf':

            if self.dataset['dataType'] == 'ani':
                LoadData().load_ani()   # test!
                LoadData().get_specie_all()  # test!

            # dataset is qm7
            elif self.dataset['dataType'] == 'qm7':
                LoadData().loadqm7()  # test!

            # dataset is json
            elif self.dataset['dataType'] == 'json':
                LoadData().load_json_data()  # test!
                LoadData().get_specie_all()  # test!
        if self.datasettype == 'hdf'

    def train_compression_r(self):
        RunML(para, ml, dataset, geometry, skf)

        # run reference calculations, either dft or dftb
        runml.ref()

        # run dftb in ML process
        runml.mldftb()

    def train_hs(self):
        pass

Train('compression_r')