#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from setuptools import setup

setup (
  name='dftbtorch',
  version='0.0.1',
  packages=['dftbtorch'],
  package_data={
    'dftbtorch': ['test/*'],
  },
)