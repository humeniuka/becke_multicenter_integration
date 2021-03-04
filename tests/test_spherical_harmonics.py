#!/usr/bin/env python
# coding: utf-8
import unittest
import numpy as np
import numpy.linalg as la
from scipy.special import sph_harm

from becke.SphericalHarmonics import spherical_harmonics_it

class TestSphericalHarmonics(unittest.TestCase):
    def test_spherical_harmonics(self):
        """
        compare spherical harmonics up to l=50 with scipy
        """
        th = np.linspace(0.000001, np.pi, 10)
        phi = np.linspace(0, 2.0*np.pi, 10, endpoint=False)
        Omega = np.outer(th, phi)
        TH = np.outer(th, np.ones_like(phi))
        PHI = np.outer(np.ones_like(th), phi)
        Ylm_it = spherical_harmonics_it(TH, PHI)
        for l in range(0, 50):
            for m in range(0, l+1):
                Ylm, l, m = next(Ylm_it)
                Ylm_scipy = sph_harm(m,l, PHI, TH)
                if (m > 0):
                    Yl_minus_m, l, mminus = next(Ylm_it)
                    Ylm_scipy_minus_m = sph_harm(-m,l, PHI, TH)
                difplus = Ylm - Ylm_scipy
                if (m > 0):
                    difminus = Yl_minus_m - Ylm_scipy_minus_m
                    deltaY = (la.norm(difplus) + la.norm(difminus))/(la.norm(Yl_minus_m) + la.norm(Ylm))
                else:
                    deltaY = la.norm(difplus)/la.norm(Ylm)
                #print("|deltaY(%s,%s)| = %s" % (l,m,deltaY))
                self.assertLess(deltaY, 1.0e-6)
        
if __name__ == "__main__":
    unittest.main()

    
