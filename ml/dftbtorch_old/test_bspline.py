#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from scipy.interpolate import BSpline
import matplotlib.pyplot as plt
import numpy as np


def B(x, k, i, t):
    if k == 0:
        return 1.0 if t[i] <= x < t[i+1] else 0.0
    if t[i+k] == t[i]:
        c1 = 0.0
    else:
        c1 = (x - t[i])/(t[i+k] - t[i]) * B(x, k-1, i, t)
        if t[i+k+1] == t[i+1]:
            c2 = 0.0
        else:
            c2 = (t[i+k+1] - x)/(t[i+k+1] - t[i+1]) * B(x, k-1, i+1, t)
        return c1 + c2


def bspline(x, t, c, k):
    n = len(t) - k - 1
    assert (n >= k+1) and (len(c) >= n)
    return sum(c[i] * B(x, k, i, t) for i in range(n))


k = 2
t = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
c = [6, 2, 4, 2, 0, 1, 0, 1, 2, 3, 5]
spl = BSpline(t, c, k)
bspline(2.5, t, c, k)
fig, ax = plt.subplots()
xx = np.linspace(0, 10, 50)
ax.plot(xx, [bspline(x, t, c, k) for x in xx], 'r-', lw=3, label='naive k=2')
ax.plot(xx, [bspline(x, t, c, 3) for x in xx], 'y-', lw=3, label='naive k=3')
ax.plot(xx, spl(xx), 'b-', lw=4, alpha=0.7, label='BSpline')
ax.grid(True)
ax.legend(loc='best')
plt.show()
