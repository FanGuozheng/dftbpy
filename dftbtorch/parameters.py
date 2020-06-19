"""Definition of parameters for DFTB and Machine Learning."""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch as t


def dftb_parameter(para):
    """Parameters for ML and DFTB."""
    para["AUEV"] = 27.2113845


def mbd_parameter(para):
    """Parameters are from MBD-DFTB+."""
    if para['n_omega_grid'] == 15:
        gauss_legendre_grid15(para)


def gauss_legendre_grid15(para):
    """Call this function if N_omega_grid is 15."""
    omega, omega_weight = t.zeros(16), t.zeros(16)
    omega[0] = 0.0000000000000000
    omega_weight[0] = 0.0000000000000000
    omega[1] = 0.0036240021641946
    omega_weight[1] = 0.0093377589930291
    omega[2] = 0.0194272861575501
    omega_weight[2] = 0.0224989668490614
    omega[3] = 0.0492780681398096
    omega_weight[3] = 0.0376452136210375
    omega[4] = 0.0958870685315184
    omega_weight[4] = 0.0563236089195038
    omega[5] = 0.1638582152651668
    omega_weight[5] = 0.0808455433033145
    omega[6] = 0.2607386870266663
    omega_weight[6] = 0.1149344187522071
    omega[7] = 0.3990059108653618
    omega_weight[7] = 0.1650309702105489
    omega[8] = 0.6000000000000000
    omega_weight[8] = 0.2430938903106735
    omega[9] = 0.9022422731012530
    omega_weight[9] = 0.3731722103362876
    omega[10] = 1.3806926931529038
    omega_weight[10] = 0.6086136045730677
    omega[11] = 2.1970213664137801
    omega_weight[11] = 1.0839821838000396
    omega[12] = 3.7544165810186048
    omega_weight[12] = 2.2053264790411675
    omega[13] = 7.3054811925383074
    omega_weight[13] = 5.5809087182012016
    omega[14] = 18.5306376341242682
    omega_weight[14] = 21.4605477286440589
    omega[15] = 99.3376890214983490
    omega_weight[15] = 255.9577387044332113
    para['omega'], para['omega_weight'] = omega, omega_weight


def mbd_vdw_para(para, iat):
    """MBD-DFTB parameters."""
    atom_num = para['atomNumber'][iat]
    if atom_num == 1:
        alpha = 4.500000
        C6 = 6.500000
        R0 = 3.100000
    elif atom_num == 2:
        alpha = 1.380000
        C6 = 1.460000
        R0 = 2.650000
    elif atom_num == 3:
        alpha = 164.200000
        C6 = 1387.000000
        R0 = 4.160000
    elif atom_num == 4:
        alpha = 38.000000
        C6 = 214.000000
        R0 = 4.170000
    elif atom_num == 5:
        alpha = 21.000000
        C6 = 99.500000
        R0 = 3.890000
    elif atom_num == 6:
        alpha = 12.000000
        C6 = 46.600000
        R0 = 3.590000
    elif atom_num == 7:
        alpha = 7.400000
        C6 = 24.200000
        R0 = 3.340000
    elif atom_num == 8:
        alpha = 5.400000
        C6 = 15.600000
        R0 = 3.190000
    elif atom_num == 9:
        alpha = 3.800000
        C6 = 9.520000
        R0 = 3.040000
    elif atom_num == 10:
        alpha = 2.670000
        C6 = 6.380000
        R0 = 2.910000
    elif atom_num == 11:
        alpha = 162.700000
        C6 = 1556.000000
        R0 = 3.730000
    elif atom_num == 12:
        alpha = 71.000000
        C6 = 627.000000
        R0 = 4.270000
    elif atom_num == 13:
        alpha = 60.000000
        C6 = 528.000000
        R0 = 4.330000
    elif atom_num == 14:
        alpha = 37.000000
        C6 = 305.000000
        R0 = 4.200000
    elif atom_num == 15:
        alpha = 25.000000
        C6 = 185.000000
        R0 = 4.010000
    elif atom_num == 16:
        alpha = 19.600000
        C6 = 134.000000
        R0 = 3.860000
    elif atom_num == 17:
        alpha = 15.000000
        C6 = 94.600000
        R0 = 3.710000
    elif atom_num == 18:
        alpha = 11.100000
        C6 = 64.300000
        R0 = 3.550000
    elif atom_num == 19:
        alpha = 292.900000
        C6 = 3897.000000
        R0 = 3.710000
    elif atom_num == 20:
        alpha = 160.000000
        C6 = 2221.000000
        R0 = 4.650000
    elif atom_num == 21:
        alpha = 120.000000
        C6 = 1383.000000
        R0 = 4.590000
    elif atom_num == 22:
        alpha = 98.000000
        C6 = 1044.000000
        R0 = 4.510000
    elif atom_num == 23:
        alpha = 84.000000
        C6 = 832.000000
        R0 = 4.440000
    elif atom_num == 24:
        alpha = 78.000000
        C6 = 602.000000
        R0 = 3.990000
    elif atom_num == 25:
        alpha = 63.000000
        C6 = 552.000000
        R0 = 3.970000
    elif atom_num == 26:
        alpha = 56.000000
        C6 = 482.000000
        R0 = 4.230000
    elif atom_num == 27:
        alpha = 50.000000
        C6 = 408.000000
        R0 = 4.180000
    elif atom_num == 28:
        alpha = 48.000000
        C6 = 373.000000
        R0 = 3.820000
    elif atom_num == 29:
        alpha = 42.000000
        C6 = 253.000000
        R0 = 3.760000
    elif atom_num == 30:
        alpha = 40.000000
        C6 = 284.000000
        R0 = 4.020000
    elif atom_num == 31:
        alpha = 60.000000
        C6 = 498.000000
        R0 = 4.190000
    elif atom_num == 32:
        alpha = 41.000000
        C6 = 354.000000
        R0 = 4.200000
    elif atom_num == 33:
        alpha = 29.000000
        C6 = 246.000000
        R0 = 4.110000
    elif atom_num == 34:
        alpha = 25.000000
        C6 = 210.000000
        R0 = 4.040000
    elif atom_num == 35:
        alpha = 20.000000
        C6 = 162.000000
        R0 = 3.930000
    para['alpha_free'][iat] = alpha
    para['C6_free'][iat] = C6
    para['R_vdw_free'][iat] = R0
