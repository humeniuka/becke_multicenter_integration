#!/usr/bin/env python
import numpy as np
from becke.lebedev_quad_points import LebedevGridPoints, Lebedev_Npts, Lebedev_Lmax, Lebedev_L2max
from becke.SphericalHarmonics import spherical_harmonics_it

def outerN(a, b):
    """
    compute the outer product of two arrays a and b, which do not 
    have to be vectors but can have any shape.
    """
    ar = a.ravel()
    br = b.ravel()
    axbr = np.outer(ar,br)
    axb = axbr.reshape(a.shape + b.shape)
    return axb

def get_lebedev_grid(order):
    """find grid closest to requested order"""
    n = abs(np.array(Lebedev_L2max) - order).argmin()
    if order != Lebedev_L2max[n]:
        print( "No grid for order %s, using grid which integrates up to L2max = %s exactly instead." \
            % (order, Lebedev_L2max[n]) )
    th,phi,w = np.array(LebedevGridPoints[Lebedev_Npts[n]]).transpose()
    return th,phi,w

def spherical_wave_expansion_it(f, r, order):
    th,phi,w = get_lebedev_grid(order)
    x = outerN(r, np.sin(th)*np.cos(phi))
    y = outerN(r, np.sin(th)*np.sin(phi))
    z = outerN(r, np.cos(th))
    fxyz = f(x,y,z)
    sph_it = spherical_harmonics_it(th,phi)
    for Ylm,l,m in sph_it:
        flm = 4.0*pi*np.sum(w*fxyz*Ylm.conjugate(), axis=-1)
        yield flm, l, m
        if m == -Lebedev_L2max[n]:
            return

def sph_synthesis(flm_it, r, th, phi, lmax=17):
    ylm_it = spherical_harmonics_it(th,phi)
    f = 0.0
    for (flm,l1,m1),(ylm,l,m) in zip(flm_it, ylm_it):
        assert l==l1 and m==m1
        f += flm*ylm
    return f
