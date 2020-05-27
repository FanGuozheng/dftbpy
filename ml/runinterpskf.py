#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is an example that how to use the code to generate the whole skf file
You should type one of the following commands:
python gen_interp_skf.py -d . 3.2 3.4 Si Sii -g 0.4 -m 0.02 2 8 7
python gen_interp_skf.py 3.2 3.4 Si Sii 2 8 7
"""
import numpy as np
from geninterp import Getinterp

x0 = np.array([2.00, 2.34, 2.77, 3.34, 4.07, 5.03, 6.28, 7.90, 10.00])
y0 = np.array([2.00, 2.34, 2.77, 3.34, 4.07, 5.03, 6.28, 7.90, 10.00])
nbegin = 0
nend = 4
dire, name1, name2, genpara = Getinterp.readshell()
r1 = genpara[0]
r2 = genpara[1]
grid0 = genpara[2]
gridmesh = genpara[3]
num = nbegin
nameall = [[name1, name1], [name1, name2], [name2, name1], [name2, name2]]
while True:
    if num < nend:
        skffile = Getinterp.readskffile(num, nameall[num], grid0, dire)
        hs_skf, ninterpline = Getinterp.genallgenintegral(num, skffile,
                                                          genpara, x0, y0)
        Getinterp.saveskffile(num, nameall[num], skffile, hs_skf, ninterpline)
        num += 1
    else:
        break
