#!/usr/bin/env python
# coding: utf-8
import unittest
import numpy as np
import numpy.linalg as la

from becke.BeckeMulticenterGrid import BeckeMulticenterGrid

class TestBeckeMulticenterGrid(unittest.TestCase):
    def test_integral_single_center(self):
        # Define the integrand, a normalized Gaussian centered at the origin.
        def integrand(x,y,z):
            r = np.sqrt(x*x+y*y+z*z)

            # Gaussian exponent
            alpha = 0.5
            norm = np.sqrt(np.pi / alpha)**3
            gr = np.exp(-alpha * r*r) / norm

            return gr
        
        # Create an integration grid with a single center.
        atomic_coordinates = np.zeros((3,1))
        atomic_numbers = np.array([1])

        # Choose a fine grid.
        rfac=4
        Lmax=23

        grid = BeckeMulticenterGrid(atomic_coordinates, atomic_numbers,
                                    radial_grid_factor=rfac, lebedev_order=Lmax)

        # Evaluate the integrand on the grid.
        integrand_values = grid.evaluate(integrand)

        # Compute the integral
        integral = grid.integrate(integrand_values)

        # The integrand is normalized to 1.
        self.assertAlmostEqual(integral, 1.0)

    def test_integral_two_centers(self):
        # Define the integrand, a normalized Gaussian centered at the origin.
        def integrand(x,y,z):
            r = np.sqrt(x*x+y*y+z*z)

            # Gaussian exponent
            alpha = 0.5
            norm = np.sqrt(np.pi / alpha)**3
            gr = np.exp(-alpha * r*r) / norm

            return gr
        
        # Create an integration grid with two centers at -x and +x.
        atomic_coordinates = np.array([
            [-0.5, 0.0, 0.0],
            [ 0.5, 0.0, 0.0]]).transpose()
        # Radial grids for hydrogen.
        atomic_numbers = np.array([1,1])

        # Choose a fine grid.
        rfac=4
        Lmax=23

        grid = BeckeMulticenterGrid(atomic_coordinates, atomic_numbers,
                                    radial_grid_factor=rfac, lebedev_order=Lmax)

        # Evaluate the integrand on the grid.
        integrand_values = grid.evaluate(integrand)

        # Compute the integral
        integral = grid.integrate(integrand_values)

        # The integrand is normalized to 1.
        self.assertAlmostEqual(integral, 1.0)

    def test_laplacian_hydrogen_1s(self):
        """
        compute the Laplacian for the 1s hydrogen wavefunction
        
          psi(r) = 2/sqrt(4 pi) exp(-r)

        The exact result is

          __2                             2
          \/  psi(r) = 2/sqrt(4 pi) (1 - --- ) exp(-r) 
                                          r
        """
        def psi_1s(x,y,z):
            """wavefunction of 1s hydrogen electron"""
            r = np.sqrt(x*x+y*y+z*z)
            psi = 1.0/np.sqrt(np.pi) * np.exp(-r)
            return psi

        # Create an integration grid with single center
        atomic_coordinates = np.zeros((3,1))
        atomic_numbers = np.array([1])

        rfac=4
        Lmax=23

        grid = BeckeMulticenterGrid(atomic_coordinates, atomic_numbers,
                                    radial_grid_factor=rfac, lebedev_order=Lmax)
        
        # Evaluate the orbital on the grid.
        psi_1s_values = grid.evaluate(psi_1s)

        # Compute the Laplacian on the grid
        lap_1s_numer = grid.laplacian(psi_1s_values)

        # Evaluate the exact Laplacian on the grid.
        coords = grid.coordinates()
        x, y, z = coords[:,0], coords[:,1], coords[:,2]
        r = np.sqrt(x*x+y*y+z*z)
        
        lap_1s_exact = 1/np.sqrt(np.pi) * (1.0-2.0/r)*np.exp(-r)

        self.assertLess(la.norm(lap_1s_numer - lap_1s_exact)/la.norm(lap_1s_exact), 1.0e-3)

if __name__ == "__main__":
    unittest.main()
