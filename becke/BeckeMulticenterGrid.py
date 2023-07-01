#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Instead of operating on functions, the BeckeMulticenterGrid class implements operations
such as integration, gradient, laplacian directly on the grid values.
"""
import numpy as np
import numpy.linalg as la
from scipy import interpolate

from becke.AtomicData import atom_names, slater_radii
from becke.LebedevQuadrature import outerN
from becke.SphericalCoords import cartesian2spherical
from becke.SphericalHarmonics import (
    spherical_harmonics_it, spherical_harmonics_block_it)
from becke.MulticenterIntegration import (
    number_of_radial_points, select_angular_grid, radial_difop)


class BeckeMulticenterGrid(object):
    def __init__(self,
                 atomic_coordinates,
                 atomic_numbers,
                 lebedev_order=23,
                 radial_grid_factor=1):
        """
        Parameters
        ----------
        atomic_coordinates : numpy array with shape (3,Nat), atomic_coordinates[:,i]
                             is the cartesian position of atom i
        atomic_numbers     : numpy array with shape (Nat)
    
        Optional
        --------
        lebedev_order      : order Lmax of the Lebedev grid
        radial_grid_factor : the number of radial grid points is increased by this factor
        """
        # Consistency checks
        assert atomic_coordinates.shape == (3, len(atomic_numbers))

        # Make copies of molecular configuration.
        self._atomic_coordinates = np.copy(atomic_coordinates)
        self._atomic_numbers = np.copy(atomic_numbers)
        # Number of centers.
        self.Nat = atomic_coordinates.shape[1]
        # (Nr,Nang) for each center
        self._grid_sizes = []
        # Cartesian coordinates relative to the origin.
        self._x_coordinates_global = []
        self._y_coordinates_global = []
        self._z_coordinates_global = []
        # Cartesian coordinates relative to each center.
        self._x_coordinates_local = []
        self._y_coordinates_local = []
        self._z_coordinates_local = []
        # radial grids
        #  coverging [-1,1]
        self._transformed_radial_grids = []
        #  coverging [0,+oo]
        self._radial_grids = []
        # 1/2 Slater radii for each center
        self._slater_radii_half = []
        # weights
        self._volume_weights = []
        self._voronoi_weights = []
        # _start_index[I] is the index into the stacked array
        # at which the grid points belonging to center I start.
        # The grid points in the range _start_index[I]:_start_index[I+1]
        # belong to center I.
        self._start_index = [0]
        
        # angular grid
        self.Lmax, self._angular_grid = select_angular_grid(lebedev_order)
        th,ph,angular_weights = self._angular_grid
        Nang = len(th)
        sc = np.sin(th)*np.cos(ph)
        ss = np.sin(th)*np.sin(ph)
        c  = np.cos(th)
        
        # for nuclear weight functions
        def s(mu, k=3):
            f = mu
            for ik in range(0, k):
                f = 1.5 * f -0.5 * f**3
            return 0.5*(1-f)
    
        atomic_names = [atom_names[Z-1] for Z in atomic_numbers]
        
        Nat = self.Nat
        R = np.zeros((Nat,Nat))     # distances between atoms i and j
        a = np.zeros((Nat,Nat))     # scaling factor used in eqn. A2
        for i in range(0, Nat):
            for j in range(i+1, Nat):
                R[i,j] = la.norm(atomic_coordinates[:,i] - atomic_coordinates[:,j])
                R[j,i] = R[i,j]

                # ratio of Slater radii
                chi = slater_radii[atomic_names[i]] / slater_radii[atomic_names[j]]
                uij = (chi-1)/(chi+1)
                a[i,j] = uij/(uij**2 - 1)
                a[j,i] = -a[i,j]

        # atom-centered subintegral
        for I in  range(0, Nat):
            # radial grid
            Nr = number_of_radial_points(atomic_numbers[I])
            # increase number of grid points is requested
            Nr *= radial_grid_factor
            rm = 0.5*slater_radii[atomic_names[I]]

            k = np.array(range(1,Nr+1))
            zr = k/(Nr+1.0)
            # grid points on interval [-1,1]
            xr = np.cos(zr * np.pi)
            # weights
            radial_weightsI = np.pi/(Nr+1.0) * np.sin(k/(Nr+1.0) * np.pi)**2
            # from variable transformation
            g = 2 * rm**3 * np.sqrt(((1+xr)/(1-xr)**3)**3)
            radial_weightsI *= g
            # radial grid points on interval [0,infinity]
            r = rm * (1+xr)/(1-xr)

            volume_weightsI = outerN(radial_weightsI, 4.0*np.pi * angular_weights).flatten()

            # Cartesian coordinates of grid relative to center I.
            x = outerN(r, sc)
            y = outerN(r, ss)
            z = outerN(r, c )
            
            # Cartesian coordinates of grid relative to origin.
            xI = (x + atomic_coordinates[0,I])
            yI = (y + atomic_coordinates[1,I])
            zI = (z + atomic_coordinates[2,I])

            Npts = Nr*Nang
            # distance between grid points and atom i
            dist = np.zeros((Nr,Nang, Nat))
            for i in range(0, Nat):
                dist[:,:,i] = np.sqrt( (xI - atomic_coordinates[0,i])**2   \
                                      +(yI - atomic_coordinates[1,i])**2   \
                                      +(zI - atomic_coordinates[2,i])**2 )

            # P_i(r) as defined in eqn. (13)
            P = np.ones((Nr,Nang, Nat))
            for i in range(0, Nat):
                for j in range(0, Nat):
                    if i==j:
                        continue
                    # mu_ij as defined in eqn. (11)
                    mu = (dist[:,:,i]-dist[:,:,j])/R[i,j]
                    nu = mu + a[i,j]*(1-mu**2)
                    P[:,:,i] *= s(nu)
            Ptot = np.sum(P, axis=-1)
    
            # weight function
            voronoi_weightsI = P[:,:,I]/Ptot

            self._grid_sizes.append((Nr, Nang))
            # append coordinates relative to origin
            self._x_coordinates_global.append(xI)
            self._y_coordinates_global.append(yI)
            self._z_coordinates_global.append(zI)
            # append coordinates relative to center I
            self._x_coordinates_local.append(x)
            self._y_coordinates_local.append(y)
            self._z_coordinates_local.append(z)
            # radial grids
            self._transformed_radial_grids.append(xr)
            self._radial_grids.append(r)
            self._slater_radii_half.append(rm)
            # append weights from center I
            self._volume_weights.append(volume_weightsI)
            self._voronoi_weights.append(voronoi_weightsI)
            # Index where the grid for the next center starts.
            self._start_index.append(self._start_index[-1] + Npts)

        # Coordinates of all grid points combined into single vectors.
        self._x_all = np.hstack([xI.flatten() for xI in self._x_coordinates_global])
        self._y_all = np.hstack([yI.flatten() for yI in self._y_coordinates_global])
        self._z_all = np.hstack([zI.flatten() for zI in self._z_coordinates_global])
            
    def coordinates(self):
        """
        Cartesian coordinates of the grid points. The grid points
        of different atomic centers are stacked in the order of the
        atoms in the configuration.

        Returns
        -------
        coords: np.ndarray of shape (npts, 3)
          arrays with Cartesian coordinates of grid points
        """
        # The coordinates around each center a combined into a single large array.
        coords = np.vstack([self._x_all, self._y_all, self._z_all]).transpose()

        return coords

    def integration_weights(self):
        """
        Integration weights of the grid points. The grid points
        of different atomic centers are stacked in the order of the
        atoms in the configuration.

        Returns
        -------
        ws: np.ndarray of shape (npts,)
          integration weights of grid points
        """
        # The weights from different centers are stack into a single large vector.
        weights = np.hstack(
            [dV.flatten() * wr.flatten()
             for (dV, wr) in zip(self._volume_weights, self._voronoi_weights)])
        return weights

    def evaluate(self, function):
        """
        Evaluate the function f(x,y,z) on the multicenter grid
        and return its function values on the grid points.

        Parameters
        ----------
        function   : callable, f(x,y,z) should evaluate the function at the 
                     grid points specified by x = [x0,x1,...,xn], y = [y0,y1,...yn]
                     and z = [z0,z1,...,zn]

        Returns
        -------
        values     : numpy array of shape (npts,)
                     with the function values on the multicenter grid.
        """
        function_values = function(self._x_all, self._y_all, self._z_all)

        return function_values

    def _spherical_wave_expansion(
            self,
            function_values,
            calculate_derivative=False,
            calculate_laplacian=False):
        """
        compute radial splines f^I_{l,m}(r)
        """
        # Spherical grid corresponding to the function values.
        th,ph,angular_weights = self._angular_grid
        
        # Lists of radial splines for each center. For each center the splines
        # are stored as a dictionary with the angular moment (l,m) as keys.
        radial_functions = []
        if (calculate_derivative):
            radial_derivatives = []
        if (calculate_laplacian):
            radial_laplacians = []

        for I in  range(0, self.Nat):
            # Grid indices in the flattened array belonging to center I
            indicesI = range(self._start_index[I],self._start_index[I+1])

            # Weights of fuzzy Voronoi partitioning.
            voronoi_weightsI = self._voronoi_weights[I]

            # Number of radial and angular points in the grid I.
            Nr, Nang = self._grid_sizes[I]

            # Transform radial grid.
            k = np.array(range(1,Nr+1))
            zr = k/(Nr+1.0)
            xr = self._transformed_radial_grids[I]
            r = self._radial_grids[I]
            rm = self._slater_radii_half[I]
            
            # The function values are stored in a one-dimensional vectors.
            # The values belonging to the grid on center I are extracted
            # and put into a matrix of shape (Nr,Nang).
            fI = voronoi_weightsI * np.reshape(function_values[indicesI], (Nr,Nang))
            
            # expand function f_I(r) into spherical harmonics
            #
            #  f_I(x,y,z) = sum_l sum_{m=-l}^{l}  f_lm(r) Y_{l,m}(th,ph)
            #
            # and create an interpolation spline for the radial function f_lm(r).

            # Create a new dictionary for the radial splines for each (l,m)
            radial_functions.append( {} )
            if (calculate_derivative):
                radial_derivatives.append( {} )
            if (calculate_laplacian):
                radial_laplacians.append( {} )

            # Loop over spherical harmonics.
            sph_it = spherical_harmonics_it(th,ph)
            for Ylm,l,m in sph_it:
                wYlm = outerN(np.ones(Nr), angular_weights*Ylm.conjugate())
                fI_lm = 4.0*np.pi * np.sum(fI*wYlm, axis=-1)
                
                # z-array is sorted in assending order,
                #   0 < z[0] < z[1] < ... < z[-1] < 1
                # while associated r-array is sorted in descending order
                #   infinity > r[0] > r[1] > ... > r[-1] > 0

                # Spline real and imaginary parts of fI_lm(r) separately
                # and store them.
                spline_lm_real = interpolate.splrep(zr, fI_lm.real, s=0)
                spline_lm_imag = interpolate.splrep(zr, fI_lm.imag, s=0)
                radial_functions[-1][(l,m)] = spline_lm_real, spline_lm_imag

                if (calculate_derivative):
                    # coefficients from chain rule
                    #  df/dr = df/dzr * dzr/dr
                    dfdzr = -1.0/np.pi * np.sqrt(rm/r) * 1.0/(rm+r)
                    zero = 0*r

                    # Compute df/dr using finite differences
                    dfIdr_lm = radial_difop(zero,dfdzr,zero, fI_lm)
                    
                    # Spline real and imaginary parts of d(fI_lm)/dr(r) separately
                    # and store them.
                    spline_deriv_lm_real = interpolate.splrep(zr, dfIdr_lm.real, s=0)
                    spline_deriv_lm_imag = interpolate.splrep(zr, dfIdr_lm.imag, s=0)
                    radial_derivatives[-1][(l,m)] = spline_deriv_lm_real, spline_deriv_lm_imag

                if (calculate_laplacian):
                    omx = 1-xr
                    opx = 1+xr
                    # coefficients for Laplacian operator after coordinate transformation
                    c0 = -l*(l+1)/rm**2 * (omx/opx)**2
                    c1 = 1.0/(4.0*np.pi*rm**2) * omx**(2.5) * (1+opx) / opx**(1.5)
                    c2 = 1.0/(4.0*np.pi**2*rm**2) * omx**3 / opx

                    #  
                    # f_lm(r) = 1/r u_lm
                    #
                    uI_lm = r*fI_lm

                    # apply Laplacian operator
                    #            d^2     l(l+1)
                    # L u_lm = ( ----  - ------ ) u_lm
                    #            dr^2     r^2
                    LuI_lm = radial_difop(c0,c1,c2, uI_lm)

                    # __2  (I)              1    d^2     l(l+1)
                    # \/  f    = sum_{l,m} --- ( ----  - ------ ) u_lm(r) Y_lm(th,ph)
                    #                       r    dr^2     r^2
                    #                         
                    #          = sum_{l,m} lapI_lm(r) Y_lm(th,ph)
                    #
                    #lapI_lm = LuI_lm/r
            
                    # z-array is sorted in assending order,
                    #   0 < z[0] < z[1] < ... < z[-1] < 1
                    # while associated r-array is sorted in descending order
                    #   infinity > r[0] > r[1] > ... > r[-1] > 0
                    
                    # It seems better to spline LuI_lm instead of LuI_lm/r,
                    # because the first function is smoother and has no
                    # singularity at r=0.

                    spline_lapl_lm_real = interpolate.splrep(zr, LuI_lm.real, s=0)
                    spline_lapl_lm_imag = interpolate.splrep(zr, LuI_lm.imag, s=0)
                    radial_laplacians[-1][(l,m)] = spline_lapl_lm_real, spline_lapl_lm_imag
                if m == -(self.Lmax-1)/2:
                    break
                
        splines = {}
        splines['radial_functions'] = radial_functions
        if (calculate_derivative):
            splines['radial_derivatives'] = radial_derivatives
        if (calculate_laplacian):
            splines['radial_laplacians'] = radial_laplacians
            
        return splines

    def interpolate(self, function_values):
        """
        Create an interpolation function from the grid values.

        Parameters
        ----------
        values     : numpy array of shape (npts,)
                     with the function values on the multicenter grid.

        Returns
        -------
        function   : callable, f(x,y,z) that evaluates the interpolated
                     function at any other grid points. The Cartesian coordinates
                     x,y,z should be specified as one-dimensional arrays.
        """
        splines = self._spherical_wave_expansion(function_values)
        radial_functions = splines['radial_functions']
        
        # The interpolation function assembles the function values at the new grid
        # points by interpolating the radial splines for each center.
        def interpolation_function(x,y,z):
            """
            interpolation function for evaluating f(x,y,z)
            """
            values = 0j*x
            # build the whole function from the parts
            #  f = sum_n  f^(n)
            for I in range(0, self.Nat):
                xI = x - self._atomic_coordinates[0,I]
                yI = y - self._atomic_coordinates[1,I]
                zI = z - self._atomic_coordinates[2,I]
                # spherical coordinates of the new grid relative to center I.
                rI,thI,phI = cartesian2spherical((xI,yI,zI))

                # Transform to the radial coordinates z(r) on which the interpolation
                # splines are defined.
                rm = self._slater_radii_half[I]
                xr = (rI-rm)/(rI+rm)
                zr = np.arccos(xr) / np.pi

                # Loop over spherical harmonics
                sph_it = spherical_harmonics_it(thI,phI)
                for Ylm,l,m in sph_it:
                    # Fetch interpolation spline for the real and imaginary parts of
                    # the radial function fI_lm(r)
                    spline_lm_real, spline_lm_imag = radial_functions[I][(l,m)]
                    # interpolate on the new grid.
                    fI_lm =        interpolate.splev(zr, spline_lm_real, der=0, ext=0) \
                            + 1.0j*interpolate.splev(zr, spline_lm_imag, der=0, ext=0)

                    # Add the contribution from center I with angular momentum (l,m).
                    values += fI_lm*Ylm

                    # Up to (including) l = (Lmax-1)/2 the integration on
                    # the Lebedev grid is exact. If higher spherical harmonics
                    # were to be included, interpolated function can blow
                    # up after several iterations.
                    if m == -(self.Lmax-1)/2:
                        break

            return values

        return interpolation_function
    
    def integrate(self, function_values):
        """
        Compute the integral over the entire space.

        Parameters
        ----------
        function_values : np.ndarray of shape (npts,)
          The values of the function f(x,y,z) on the multicenter grid,
          which can be obtained with .evaluate(f).
        """
        integral = np.sum(self.integration_weights() * function_values)
        return integral
    
    def gradient(self, function_values):
        """
        compute the gradient of a function f in Cartesian coordinates
        __
        \/  f(r) = [ df/dx(r), df/dy(r), df/dz(r) ]^T

        numerically on a multicenter spherical grid using the grid values
        of the function.

        Parameters
        ----------
        function_values : np.ndarray of shape (npts,)
          The values of the function f(x,y,z) on the multicenter grid,
          which can be obtained with .evaluate(f).

        Returns
        -------
        dfdx_values : np.ndarray of shape (npts,)
          The values of the df/dx(r) at the grid points.
        dfdy_values : np.ndarray of shape (npts,)
          The values of the df/dy(r) at the grid points.
        dfdz_values : np.ndarray of shape (npts,)
          The values of the df/dz(r) at the grid points.
        """
        # The function f_I(r) is expanded into spherical harmonics
        #
        #  f_I(x,y,z) = sum_l sum_{m=-l}^{l}  f_lm(r) Y_{l,m}(th,ph)
        #
        # and the gradient in spherical coordinates is applied to each (l,m) component.
        #  __   (I)          d f^I_lm(r)                             f^I_lm(r)
        #  \/  f    = sum    ----------- Y_{l,m}(th,ph) * unit_r  +  --------- Psi
        #                l,m   d r                                       r        l,m
        #
        # where Psi_lm are the vector spherical harmonics,
        #             __
        #  Psi    = r \/ Y    = [ m cot(th) Y     + exp(-i*ph) sqrt((l-m)*(l+m+1)) Y      ] * unit_th
        #     l,m         l,m                l,m                                    l,m+1
        #
        #                       + im/sin(th) Y    * unit_ph
        #                                     l,m
        #
        # and unit_r, unit_th and unit_ph are the unit vectors in spherical harmonics.

        splines = self._spherical_wave_expansion(
            function_values, calculate_derivative=True)
        radial_functions = splines['radial_functions']
        radial_derivatives = splines['radial_derivatives']

        # Coordinates of grids of all centers.
        x = self._x_all
        y = self._y_all
        z = self._z_all

        # build the total gradient from the gradients of the parts
        #  __            __
        #  \/ f = sum_I  \/  f^(I)
        dfdx_values = 0j*x
        dfdy_values = 0j*y
        dfdz_values = 0j*z
        
        for I in  range(0, self.Nat):
            # Cartesian coordinates relative to center I
            xI = x - self._atomic_coordinates[0,I]
            yI = y - self._atomic_coordinates[1,I]
            zI = z - self._atomic_coordinates[2,I]
            # spherical coordinates of the new grid relative to center I.
            rI,thI,phI = cartesian2spherical((xI,yI,zI))
            rhoI = np.sqrt(xI*xI+yI*yI)

            # The point rI=0 is set to a finite value to avoid dividing by zero.
            # This value does not matter since it only occurs in products with
            # with another factor of 0.
            rI[rI==0.0] = 1.0
            rhoI[rhoI==0.0] = 1.0

            # precalculate trigonometric functions
            # The point thI=0 is excluded.
            sin_thI = np.sin(thI)
            cos_thI = np.cos(thI)
            inv_sin_thI = 0*thI
            inv_sin_thI[thI!=0] = 1.0/sin_thI[thI!=0]
            cot_thI = 0*thI
            cot_thI[thI!=0] = cos_thI[thI!=0]/sin_thI[thI!=0]
            
            exp_miphI = np.exp(-1j*phI)
            
            # Transform to the radial coordinates z(r) on which the interpolation
            # splines are defined.
            rm = self._slater_radii_half[I]
            xr = (rI-rm)/(rI+rm)
            zr = np.arccos(xr) / np.pi

            # Iterate over spherical harmonics in blocks with the same l.
            sph_block_it = spherical_harmonics_block_it(thI,phI)
            for Yl_block in sph_block_it:
                for (l,m), Ylm in Yl_block.items():
                    # Fetch interpolation spline for the real and imaginary parts of
                    # the radial function fI_lm(r).
                    spline_lm_real, spline_lm_imag = radial_functions[I][(l,m)]
                    # Interpolate the radial function f^I_lm(r) on the full grid.
                    fI_lm = (
                                 interpolate.splev(zr, spline_lm_real, der=0, ext=0) \
                        + 1.0j * interpolate.splev(zr, spline_lm_imag, der=0, ext=0))
                    # Interpolate the derivative d(f^I_lm)/dr on the full grid.
                    spline_deriv_lm_real, spline_deriv_lm_imag = radial_derivatives[I][(l,m)]
                    dfIdr_lm = (
                                 interpolate.splev(zr, spline_deriv_lm_real, der=0, ext=0) \
                        + 1.0j * interpolate.splev(zr, spline_deriv_lm_imag, der=0, ext=0))
                    """
                    ### DEBUG
                    import matplotlib.pyplot as plt
                    sort_index = np.argsort(rI)
                    plt.plot(rI[sort_index], fI_lm[sort_index])
                    plt.plot(rI[sort_index], dfIdr_lm[sort_index])
                    plt.xlim(0.0, 10.0)
                    plt.title(f"l= {l} m= {m}")
                    plt.show()
                    ###
                    """

                    #
                    # vector spherical harmonics
                    #
                    # theta-component of vector spherical harmonic 
                    Psi_lm_theta = m*cot_thI * Ylm
                    if m < l:
                        # Here we need the spherical harmonic Y_{l,m+1}(th,ph).
                        Ylmp1 = Yl_block[(l,m+1)]
                        Psi_lm_theta += np.sqrt((l-m)*(l+m+1)) * exp_miphI * Ylmp1
                    # phi-component of vector spherical harmonic
                    Psi_lm_phi = 1j*m * inv_sin_thI * Ylm
                
                    # [r] component of gradient in the direction of the radial unit vector
                    # df^I_lm/dr Y_lm
                    grad_f_radial = dfIdr_lm * Ylm

                    dfdx_values += grad_f_radial * xI/rI
                    dfdy_values += grad_f_radial * yI/rI
                    dfdz_values += grad_f_radial * zI/rI

                    # [theta] component of gradient in the direction of the theta unit vector.
                    grad_f_theta = fI_lm/rI * Psi_lm_theta

                    dfdx_values += grad_f_theta * xI*zI/(rI*rhoI)
                    dfdy_values += grad_f_theta * yI*zI/(rI*rhoI)
                    dfdz_values += grad_f_theta * (-rhoI/rI)
                
                    # [phi] component of gradient in the direction of phi unit vector
                    grad_f_phi = fI_lm/rI * Psi_lm_phi

                    dfdx_values += grad_f_phi * (-yI)/rhoI
                    dfdy_values += grad_f_phi *  xI/rhoI
                    # no contribution to dfdz
                    
                # Up to (including) l = (Lmax-1)/2 the integration on
                # the Lebedev grid is exact.
                if l == (self.Lmax-1)/2:
                    break
                
        return (dfdx_values, dfdy_values, dfdz_values)

    def laplacian(self, function_values):
        """
        compute the Laplacian of a function f
        __2
        \/  f(r)

        numerically on a multicenter spherical grid using the grid values
        of the function.

        Parameters
        ----------
        function_values : np.ndarray of shape (npts,)
          The values of the function f(x,y,z) on the multicenter grid,
          which can be obtained with .evaluate(f).

        Returns
        -------
        laplacian_values : np.ndarray of shape (npts,)
          The values of the laplacian at the grid points.
        """
        splines = self._spherical_wave_expansion(
            function_values, calculate_laplacian=True)
        radial_functions = splines['radial_functions']
        radial_laplacians = splines['radial_laplacians']

        # Coordinates of grids of all centers.
        x = self._x_all
        y = self._y_all
        z = self._z_all
        
        # build the total Laplacian from the Laplacians of the parts
        #  __2           __2
        #  \/ f = sum_I  \/  f^(I)
        laplacian_values = 0j*x        
        
        for I in  range(0, self.Nat):
            # Cartesian coordinates relative to center I
            xI = x - self._atomic_coordinates[0,I]
            yI = y - self._atomic_coordinates[1,I]
            zI = z - self._atomic_coordinates[2,I]
            # spherical coordinates of the new grid relative to center I.
            rI,thI,phI = cartesian2spherical((xI,yI,zI))

            # Transform to the radial coordinates z(r) on which the interpolation
            # splines are defined.
            rm = self._slater_radii_half[I]
            xr = (rI-rm)/(rI+rm)
            zr = np.arccos(xr) / np.pi

            sph_it = spherical_harmonics_it(thI,phI)
            for Ylm,l,m in sph_it:
                # Fetch splines for radial Laplacians.
                spline_lapl_lm_real, spline_lapl_lm_imag = radial_laplacians[I][(l,m)]
                # interpolate on the big grid.
                LuI_lm =       interpolate.splev(zr, spline_lapl_lm_real, der=0, ext=0) \
                        + 1.0j*interpolate.splev(zr, spline_lapl_lm_imag, der=0, ext=0)

                lapI_lm = LuI_lm/rI
                laplacian_values += lapI_lm*Ylm
                
                # Up to (including) l = (Lmax-1)/2 the integration on
                # the Lebedev grid is exact.
                if m == -(self.Lmax-1)/2:
                    break

        return laplacian_values

