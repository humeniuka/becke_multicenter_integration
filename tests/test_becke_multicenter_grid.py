#!/usr/bin/env python
# coding: utf-8
import unittest
import numpy as np
import numpy.linalg as la

from becke.BeckeMulticenterGrid import BeckeMulticenterGrid

class TestBeckeMulticenterGrid(unittest.TestCase):
    def single_center_grid(self):
        """
        Create an integration grid with a single hydrogen center at the origin.
        """
        atomlist = [
            (1, [0.0, 0.0, 0.0])]

        # Choose a grid.
        grid = BeckeMulticenterGrid(
            atomlist,
            radial_grid_factor=8,
            lebedev_order=31)

        return grid

    def two_center_grid(self):
        """
        Create an integration grid with two hydrogen centers at -x and +x.
        """
        atomlist = [
            (1, [-0.5, 0.0, 0.0]),
            (1, [ 0.5, 0.0, 0.0])]

        # Choose a grid.
        grid = BeckeMulticenterGrid(
            atomlist,
            radial_grid_factor=8,
            lebedev_order=31)

        return grid

    def three_center_grid(self):
        """
        Create an integration grid with three centers.
        """
        atomlist = [
            (1, [-0.5, 0.0, 0.0]),
            (1, [ 0.5, 0.0, 0.0]),
            (1, [ 0.0, 0.0, 1.0])]

        # Choose a grid.
        grid = BeckeMulticenterGrid(
            atomlist,
            radial_grid_factor=8,
            lebedev_order=23)

        return grid

    def compare_original_vs_interpolated(
            self,
            grid,
            function,
            relative_tolerance=1.0e-3):
        """ 
        We 
         (1) evaluate a function on a grid, then
         (2) generate the interpolated function and check that
         (3) evaluating the interpolation function on the original grid
             gives the same values back.
        """
        # (1) function -> grid values
        function_values = grid.evaluate(function)
        # (2) grid values -> interpolation function
        interpolation_function = grid.interpolate(function_values)
        # (3) inpoterlation function -> grid values
        interpolated_values = grid.evaluate(interpolation_function)

        ### DEBUG
        import matplotlib.pyplot as plt
        Npts = 1000
        # plot slice along x-axis
        x = np.linspace(-10.0, 10.0, Npts)
        y = np.zeros(Npts)+0.1
        z = np.zeros(Npts)+0.25

        r = x
        
        plt.plot(r, function(x,y,z).real, lw=2, alpha=0.5, label="Re[f]")
        plt.plot(r, function(x,y,z).imag, lw=2, alpha=0.5, label="Im[f]")
        plt.plot(r, interpolation_function(x,y,z).real, ls="--", label="Re[f] (interpolated)")
        plt.plot(r, interpolation_function(x,y,z).imag, ls="-.", label="Im[f] (interpolated)")
        plt.legend()
        plt.show()  
        ###
        
        # Compare grid values before and after interpolation.
        error = np.sqrt(grid.integrate(abs(function_values - interpolated_values)**2))
        norm = np.sqrt(grid.integrate(abs(function_values)**2))
        relative_error = error / norm
        self.assertLess(error, relative_tolerance)

    def compare_numerical_vs_exact_gradients(
            self,
            grid,
            function,
            dfdx_function,
            dfdy_function,
            dfdz_function,
            relative_tolerance=1.0e-3):
        # Evaluate the function on the grid
        function_values = grid.evaluate(function)

        # Compute gradient numerically from the function values
        dfdx_values, dfdy_values, dfdz_values = grid.gradient(function_values)

        """
        ### DEBUG        
        dfdx_interpolation_function = grid.interpolate(dfdx_values)
        dfdy_interpolation_function = grid.interpolate(dfdy_values)
        dfdz_interpolation_function = grid.interpolate(dfdz_values)

        import matplotlib.pyplot as plt
        Npts = 1000
        """
        """
        # plot slice along x-axis
        x = np.linspace(-10.0, 10.0, Npts)
        y = np.zeros(Npts)+0.0
        z = np.zeros(Npts)

        r = x
        """
        """
        # plot slice along y-axis
        y = np.linspace(-10.0, 10.0, Npts)
        x = np.zeros(Npts)+0.0
        z = np.zeros(Npts)

        r = y
        """
        """
        # plot slice along z-axis
        z = np.linspace(-10.0, 10.0, Npts)
        x = np.zeros(Npts)+0.0
        y = np.zeros(Npts)

        r = z
        
        plt.plot(r, dfdx_function(x,y,z).real, lw=2, alpha=0.5, label="Re[df/dx]")
        plt.plot(r, dfdx_function(x,y,z).imag, lw=2, alpha=0.5, label="Im[df/dx]")
        plt.plot(r, dfdx_interpolation_function(x,y,z).real, ls="--", label="Re[df/dx] (numerical)")
        plt.plot(r, dfdx_interpolation_function(x,y,z).imag, ls="-.", label="Im[df/dx] (numerical)")
        plt.legend()
        plt.show()  
        ###
        """
        
        dfdx_values_exact = grid.evaluate(dfdx_function)
        dfdy_values_exact = grid.evaluate(dfdy_function)
        dfdz_values_exact = grid.evaluate(dfdz_function)

        # Compare numerical and exact gradients
        #  |df/dx(numerical) - df/dx(exact)|/|df/dx(exact)|
        error_x = np.sqrt(grid.integrate(abs(dfdx_values - dfdx_values_exact)**2))
        norm_x = np.sqrt(grid.integrate(abs(dfdx_values_exact)**2))
        relative_error_x = error_x / norm_x

        #  |df/dy(numerical) - df/dy(exact)|/|df/dy(exact)|
        error_y = np.sqrt(grid.integrate(abs(dfdy_values - dfdy_values_exact)**2))
        norm_y = np.sqrt(grid.integrate(abs(dfdy_values_exact)**2))
        relative_error_y = error_y / norm_y

        #  |df/dz(numerical) - df/dz(exact)|/|df/dz(exact)|
        error_z = np.sqrt(grid.integrate(abs(dfdz_values - dfdz_values_exact)**2))
        norm_z = np.sqrt(grid.integrate(abs(dfdz_values_exact)**2))
        relative_error_z = error_z / norm_z

        """
        ### DEBUG
        print(error_x)
        print(norm_x)
        print(error_y)
        print(norm_y)
        print(error_z)
        print(norm_z)

        print("relative error df/dx= ", relative_error_x)
        print("relative error df/dy= ", relative_error_y)
        print("relative error df/dz= ", relative_error_z)
        ###
        """
        
        self.assertLess(relative_error_x, relative_tolerance)
        self.assertLess(relative_error_y, relative_tolerance)
        self.assertLess(relative_error_z, relative_tolerance)
        
    def test_interpolation_spherically_symmetric_single_center(self):
        # Spherically symmetric Gaussian.
        alpha = 0.5
        def function(x,y,z):
            r2 = x*x+y*y+z*z
            f = np.exp(-alpha*r2)
            return f

        grid = self.single_center_grid()
        self.compare_original_vs_interpolated(grid, function)
        
    def test_interpolation_spherically_symmetric_two_centers(self):
        # Spherically symmetric Gaussian.
        alpha = 0.5
        def function(x,y,z):
            r2 = x*x+y*y+z*z
            f = np.exp(-alpha*r2)
            return f

        grid = self.two_center_grid()
        self.compare_original_vs_interpolated(grid, function)
        
    def test_interpolation_nonsymmetric_three_centers(self):
        x0 = 0.6456
        alpha = 0.2352
        def function(x,y,z):
            r2 = x*x+y*y+z*z
            f = (np.sin(x-x0)/(x-x0) * y + z) * np.exp(-alpha*r2)
            return f

        grid = self.three_center_grid()
        self.compare_original_vs_interpolated(grid, function)        
                    
    def test_integral_single_center(self):
        # Define the integrand, a normalized Gaussian centered at the origin.
        def integrand(x,y,z):
            r = np.sqrt(x*x+y*y+z*z)

            # Gaussian exponent
            alpha = 0.5
            norm = np.sqrt(np.pi / alpha)**3
            gr = np.exp(-alpha * r*r) / norm

            return gr
        
        grid = self.single_center_grid()

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

        # Create a multicenter grid.
        grid = self.two_center_grid()

        # Evaluate the integrand on the grid.
        integrand_values = grid.evaluate(integrand)

        # Compute the integral
        integral = grid.integrate(integrand_values)

        # The integrand is normalized to 1.
        self.assertAlmostEqual(integral, 1.0)

    def test_complex_integral_two_centers(self):
        # Define the integrand, a normalized Gaussian centered at the origin.
        def integrand(x,y,z):
            r = np.sqrt(x*x+y*y+z*z)

            # Complex Gaussian exponent
            alpha = 0.5+0.1j
            norm = np.sqrt(np.pi / alpha)**3
            gr = np.exp(-alpha * r*r) / norm

            return gr

        # Create a multicenter grid.
        grid = self.two_center_grid()

        # Evaluate the integrand on the grid.
        integrand_values = grid.evaluate(integrand)

        # Compute the integral
        integral = grid.integrate(integrand_values)

        # The integrand is normalized to 1.
        self.assertAlmostEqual(integral, 1.0)
                
    def test_gradient_single_center_complex(self):
        """
        compare the numerical gradient with the analytical ones
        """
        # Define the function whose gradient should be calculated.
        alpha = 0.5+0.1j
        norm = pow(np.pi/alpha, 3.0/2.0)
        def function(x,y,z):
            # spherically symmetric Gaussian
            r2 = x*x+y*y+z*z
            f = np.exp(-alpha*r2) / norm
            return f

        # analytical gradients
        def dfdx_function(x,y,z):
            r2 = x*x+y*y+z*z
            dfdx = -2*alpha*x*np.exp(-alpha*r2) / norm
            return dfdx
            
        def dfdy_function(x,y,z):
            r2 = x*x+y*y+z*z
            dfdy = -2*alpha*y*np.exp(-alpha*r2) / norm
            return dfdy

        def dfdz_function(x,y,z):
            r2 = x*x+y*y+z*z
            dfdz = -2*alpha*z*np.exp(-alpha*r2) / norm
            return dfdz

        grid = self.single_center_grid()
        
        self.compare_numerical_vs_exact_gradients(
            grid,
            function,
            dfdx_function,
            dfdy_function,
            dfdz_function,
            relative_tolerance=1.0e-3)

    @unittest.skip("gradient operator is not numerically stable at boundaries of Voronoi cells")
    def test_gradient_two_centers_complex(self):
        """
        Test gradients of complex function, f: Real^3 --> Complex.
        """
        # Define the function whose gradient should be calculated.
        # Making the exponent complex allows us to check the gradients for complex
        # functions.
        alpha = 0.5+0.1j
        norm = pow(np.pi/alpha, 3.0/2.0)
        # The maximum of the function should coincide with a center of the
        # integration grid.
        x0 = 0.5
        def function(x,y,z):
            # spherically symmetric Gaussian
            r2 = (x-x0)*(x-x0)+y*y+z*z
            f = np.exp(-alpha*r2) / norm
            return f

        # analytical gradients
        def dfdx_function(x,y,z):
            r2 = (x-x0)*(x-x0)+y*y+z*z
            dfdx = -2*alpha*(x-x0)*np.exp(-alpha*r2) / norm
            return dfdx
            
        def dfdy_function(x,y,z):
            r2 = (x-x0)*(x-x0)+y*y+z*z
            dfdy = -2*alpha*y*np.exp(-alpha*r2) / norm
            return dfdy

        def dfdz_function(x,y,z):
            r2 = (x-x0)*(x-x0)+y*y+z*z
            dfdz = -2*alpha*z*np.exp(-alpha*r2) / norm
            return dfdz

        grid = self.two_center_grid()

        self.compare_numerical_vs_exact_gradients(
            grid,
            function,
            dfdx_function,
            dfdy_function,
            dfdz_function,
            relative_tolerance=1.0e-3)

    @unittest.skip("gradient operator is not numerically stable at boundaries of Voronoi cells")
    def test_gradient_three_centers(self):
        # Define the function whose gradient should be calculated
        x0 = 0.1
        y0 = 0.2
        z0 = 0.3
        n = 3
        alpha = 0.2342
        def function(x,y,z):
            r2 = x*x+y*y+z*z
            f = np.sin(x-x0)*np.cos(y-y0)*pow(z-z0,n) * np.exp(-alpha*r2)
            return f

        # analytical gradients
        def dfdx_function(x,y,z):
            r2 = x*x+y*y+z*z
            dfdx = np.cos(y-y0)*pow(z-z0,n)*(np.cos(x-x0) - 2*alpha*x*np.sin(x-x0))*np.exp(-alpha*r2)
            return dfdx
            
        def dfdy_function(x,y,z):
            r2 = x*x+y*y+z*z
            dfdy = np.sin(x-x0)*pow(z-z0,n)*(-np.sin(y-y0) - 2*alpha*y*np.cos(y-y0))*np.exp(-alpha*r2)
            return dfdy

        def dfdz_function(x,y,z):
            r2 = x*x+y*y+z*z
            dfdz = np.sin(x-x0)*np.cos(y-y0)*(n*pow(z-z0,n-1) - 2*alpha*z*pow(z-z0,n))*np.exp(-alpha*r2)
            return dfdz

        grid = self.three_center_grid()

        self.compare_numerical_vs_exact_gradients(
            grid,
            function,
            dfdx_function,
            dfdy_function,
            dfdz_function,
            relative_tolerance=1.0e-3)
        
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

        # Create an integration grid with single center at the origin.
        grid = self.single_center_grid()
        
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
