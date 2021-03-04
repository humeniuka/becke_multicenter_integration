#!/usr/bin/env python
# coding: utf-8
import unittest
import numpy as np

from becke.LebedevQuadrature import LebedevGridPoints, Lebedev_Npts, Lebedev_L2max
from becke.SphericalHarmonics import spherical_harmonics_it

class TestLebedevQuadarture(unittest.TestCase):
    def test_lebedev_quadrature(self):
        for order in [0,1,2,3,4,5]: #range(0, len(Lebedev_Npts)):
            with self.subTest(order=order):
                self._check_lebedev_quadrature(order)

    def _check_lebedev_quadrature(self, order, tolerance=1e-9):
        """
        check that spherical harmonics are orthonormal

          <l1,m1|l2,m2> = delta       delta
                               l1,l2       m1,m2

        The integral over the solid angle is computed on Lebedev grids of all orders.
        """
        th,phi,w = np.array(LebedevGridPoints[Lebedev_Npts[order]]).transpose()
        sph_it1 = spherical_harmonics_it(th,phi)
        for Ylm1,l1,m1 in sph_it1:
            if l1 > Lebedev_L2max[order]:
                break
            sph_it2 = spherical_harmonics_it(th,phi)
            for Ylm2,l2,m2 in sph_it2:
                if l2 > Lebedev_L2max[order]:
                    break
                I = 4.0*np.pi*np.sum(w*Ylm1*Ylm2.conjugate())
                #print( "<%s %s|%s %s> = %s" % (l1,m1,l2,m2,I) )
                if l1 == l2 and m1 == m2:
                    #print( "|I-1.0| = %s" % abs(I-1.0) )
                    self.assertLess( abs(I-1.0),  tolerance )
                else:
                    self.assertLess( abs(I), tolerance )

if __name__ == "__main__":
    unittest.main()
    
