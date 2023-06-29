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
from becke.LebedevQuadrature import outerN, spherical_harmonics_it
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
        # These lists contain the Cartesian coordinates and weights around each center.
        self._x_coordinates = []
        self._y_coordinates = []
        self._z_coordinates = []
        self._volume_weights = []
        self._voronoi_weights = []
        # radial grids
        #  coverging [-1,1]
        self._transformed_radial_grids = []
        #  coverging [0,+oo]
        self._radial_grids = []
        # 1/2 Slater radii for each center
        self._slater_radii_half = []
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
        
        Nat = atomic_coordinates.shape[1]
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

            # cartesian coordinates of grid
            xI = (outerN(r, sc) + atomic_coordinates[0,I])
            yI = (outerN(r, ss) + atomic_coordinates[1,I])
            zI = (outerN(r, c ) + atomic_coordinates[2,I])

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

            # append coordinates relative to center I
            self._x_coordinates.append(xI)
            self._y_coordinates.append(yI)
            self._z_coordinates.append(zI)
            # radial grids
            self._transformed_radial_grids.append(xr)
            self._radial_grids.append(r)
            self._slater_radii_half.append(rm)
            # append weights from center I
            self._volume_weights.append(volume_weightsI)
            self._voronoi_weights.append(voronoi_weightsI)
            # Index where the grid for the next center starts.
            self._start_index.append(self._start_index[-1] + Npts)

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
        coords = np.vstack([
            np.hstack([x.flatten() for x in self._x_coordinates]),
            np.hstack([y.flatten() for y in self._y_coordinates]),
            np.hstack([z.flatten() for z in self._z_coordinates])
        ]).transpose()

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
        values_on_grids = []
        for x,y,z in zip(self._x_coordinates, self._y_coordinates, self._z_coordinates):
            # evaluate f on the grid points of the I'th center
            f_xyz = function(x.flatten(), y.flatten(), z.flatten())
            values_on_grids.append(f_xyz)
        # stack grids from different centers.
        function_values = np.hstack(values_on_grids)

        return function_values
    
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
        dfdx_value : np.ndarray of shape (npts,)
          The values of the df/dx(r) at the grid points.
        dfdy_value : np.ndarray of shape (npts,)
          The values of the df/dy(r) at the grid points.
        dfdz_value : np.ndarray of shape (npts,)
          The values of the df/dz(r) at the grid points.
        """
        pass

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
        laplacian_value : np.ndarray of shape (npts,)
          The values of the laplacian at the grid points.
        """
        lap = 0j*function_values
        # build the total Laplacian from the Laplacians of the parts
        #  __2           __2
        #  \/ f = sum_I  \/  f^(I)

        th,ph,angular_weights = self._angular_grid
        
        Nat = self._atomic_coordinates.shape[1]
        for I in  range(0, Nat):
            xI = self._x_coordinates[I]
            yI = self._y_coordinates[I]
            zI = self._z_coordinates[I]

            Nr, Nang = xI.shape

            k = np.array(range(1,Nr+1))
            zr = k/(Nr+1.0)
            xr = self._transformed_radial_grids[I]
            r = self._radial_grids[I]
            rm = self._slater_radii_half[I]
            
            # The function values are stored in a one-dimensional vectors.
            # The values belonging to the grid on center I are extracted
            # and put into a matrix of shape (Nr,Nang).
            fI = np.reshape(
                function_values[self._start_index[I]:self._start_index[I+1]],
                (Nr,Nang))
            
            # compute Laplacian
            #    (I)    __2  (I)
            # lap    =  \/  f    
            #

            # expand function f_I(r) into spherical harmonics
            #
            #  f_I(x,y,z) = sum_l sum_{m=-l}^{l}  f_lm(r) Y_{l,m}(th,ph)
            #
            # and apply Laplacian to each (l,m) component
            sph_it = spherical_harmonics_it(th,ph)
            for Ylm,l,m in sph_it:
                wYlm = outerN(np.ones(Nr), angular_weights*Ylm.conjugate())
                fI_lm = 4.0*np.pi * np.sum(fI*wYlm, axis=-1)

                omx = 1-xr
                opx = 1+xr
                # coefficients for Laplacian operator after coordinate transformation
                c0 = -l*(l+1)/rm**2 * (omx/opx)**2
                c1 = 1.0/(4.0*np.pi*rm**2) * omx**(2.5) * (1.0+opx) / opx**(1.5)
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
                lapI_lm = outerN(LuI_lm/r, Ylm)
            
                # contribution to Laplacian from center I
                lap[self._start_index[I]:self._start_index[I+1]] += lapI_lm.flatten()

                if m == -(self.Lmax-1)/2:
                    break

        return lap

