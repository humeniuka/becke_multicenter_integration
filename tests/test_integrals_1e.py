#!/usr/bin/env python
# coding: utf-8
import unittest
import numpy as np
import numpy.linalg as la

from becke.Ints1e import overlap, kinetic, nuclear, electronic_dipole
from becke import GaussianIntegrals, CheckpointFile
from becke import settings

class TestOneElectronIntegrals(unittest.TestCase):
    def test_integrals_h2(self):
        # Gaussian-type 1s function with exponent alpha
        # centered at position Rc=(xc,yc,zc)
        def gf_1s(alpha, xc,yc,zc, x,y,z):
            return (2*alpha/np.pi)**(0.75) * np.exp(-alpha*( (x-xc)**2 + (y-yc)**2 + (z-zc)**2 ))

        # contracted STO-3G basis function for hydrogen, see eqn. (3.225) in Szabo & Ostlund
        def cgf_1s(xc,yc,zc, x,y,z):
            wfn =   0.444635 * gf_1s(0.168856, xc,yc,zc, x,y,z) \
                  + 0.535328 * gf_1s(0.623913, xc,yc,zc, x,y,z) \
                  + 0.154329 * gf_1s(3.42525 , xc,yc,zc, x,y,z)
            return wfn

        # molecular integrals for STO-3G hydrogen molecule
        # with an internuclear distance of R=1.4 bohr
        R = 1.4
        atomlist = [(1, (0.0, 0.0, 0.0)),
                    (1, (0.0, 0.0, R))]

        # The basis set consists of two functions phi_1 and phi_2
        # centered on each proton
        def phi_1(x,y,z):
            return cgf_1s(0.0, 0.0, 0.0, x,y,z)
        def phi_2(x,y,z):
            return cgf_1s(0.0, 0.0, R  , x,y,z)
        basis = [phi_1, phi_2]
        nbfs = len(basis)

        # overlap, kinetic and nuclear matrix elements
        S = np.zeros((nbfs,nbfs))
        T = np.zeros((nbfs,nbfs))
        N = np.zeros((nbfs,nbfs))

        for i in range(0, nbfs):
            for j in range(0, nbfs):
                S[i,j] = overlap(atomlist, basis[i], basis[j])
                T[i,j] = kinetic(atomlist, basis[i], basis[j])
                N[i,j] = nuclear(atomlist, basis[i], basis[j])

        print( "" )
        print( "  The matrix elements for STO-3G H2 should be compared" )
        print( "  with those in chapter 3.5.2 of Szabo & Ostlund." )
        print( "" )
        print( "overlap matrix S, compare with eqn. (3.229)" )
        print( S )
        print( "kinetic energy T, compare with eqn. (3.230)" )
        print( T )
        print( "nuclear attraction V, compare with sums of eqns. (3.231) and (3.232)" )
        print( N )

        # compare with Szabo & Ostlund
        S_exact = np.array([[1.0000, 0.6593],
                            [0.6593, 1.0000]])
        T_exact = np.array([[0.7600, 0.2365],
                            [0.2365, 0.7600]])
        N_exact = ( np.array([[-1.2266, -0.5974],
                              [-0.5974, -0.6538]])
                   +np.array([[-0.6538, -0.5974],
                              [-0.5974, -1.2266]]) )
        self.assertLess( la.norm(S_exact - S), 1.0e-4 )
        self.assertLess( la.norm(T_exact - T), 1.0e-4 )
        self.assertLess( la.norm(N_exact - N), 1.0e-4 )
        
    def test_dipole_integrals(self):
        """
        compare numerical integrals for dipole matrix elements with
        exact values for Gaussian orbitals
        """
        # load test data with Gaussian AO basis
        res = CheckpointFile.G09ResultsDFT("data/h2o_hf_sv.fchk")

        # geometry
        atomlist = []
        for i in range(0, res.nat):
            Z = res.atomic_numbers[i]
            posi = res.coordinates[i,:]
            atomlist.append( (Z, posi) )

        # Dipole matrix elements are computed from analytic expressions
        # for Gaussian orbitals
        dip_exact = GaussianIntegrals.basis_dipoles(res.basis)

        # Dipole matrix elements are computed by numerical integration

        # increase resolution of multicenter grid
        settings.radial_grid_factor = 20
        settings.lebedev_order = 41

        nbfs = res.basis.nbfs
        dip_numeric = np.zeros((3,nbfs,nbfs))

        # Computing all integrals numerically takes a lot of time. We only take
        # the last 2 AOs to speed up the test. 
        for i in range(nbfs-2, nbfs): #range(0, nbfs):
            # 
            orb_i = np.zeros(nbfs)
            orb_i[i] = 1.0
            # define wavefunctions for AO i
            def ao_i(x,y,z):
                return res.basis.wavefunction(orb_i, x,y,z)

            for j in range(nbfs-2, nbfs): #range(0, nbfs):
                print( "computing dipole matrix elements of AO pair %d-%d" % (i,j) )
                # 
                orb_j = np.zeros(nbfs)
                orb_j[j] = 1.0

                # define wavefunctions for AO j
                def ao_j(x,y,z):
                    return res.basis.wavefunction(orb_j, x,y,z)

                dip_numeric[:,i,j] = electronic_dipole(atomlist, ao_i, ao_j)

                print( "  exact   = %s" % dip_exact[:,i,j] )
                print( "  numeric = %s" % dip_numeric[:,i,j] )

                self.assertLess( la.norm(dip_numeric[:,i,j] - dip_exact[:,i,j]), 1.0e-3 )

    
if __name__ == "__main__":
    unittest.main()

    
