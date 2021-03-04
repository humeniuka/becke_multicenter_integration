#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Becke's scheme for multicenter integration, solution of Poisson's equation and 
inhomogeneous Schroedinger equation

References
----------
[1] A.Becke, "A multicenter numerical integration scheme for polyatomic molecules",
    J.Chem.Phys. 88, 2547 (1988)
[2] A.Becke, R.Dickson, "Numerical solution of Poisson's equation in polyatomic molecules",
    J.Chem.Phys. 89, 2993 (1988)
[3] A.Becke, R.Dickson, "Numerical solution of Schroedinger's equation in polyatomic molecules",
    J.Chem.Phys. 92, 3610 (1990)

Some useful information is also contained in
[4] T.Shiozaki, S.Hirata, "Grid-based numerical Hartree-Fock solutions of polyatomic molecules",
    Phys.Rev. A 76, 040503(R) (2007)

"""
import numpy as np
import numpy.linalg as la
from scipy import interpolate

from becke import settings
from becke.LebedevQuadrature import get_lebedev_grid, outerN, spherical_harmonics_it
from becke.lebedev_quad_points import LebedevGridPoints, Lebedev_Npts, Lebedev_Lmax
from becke.SphericalCoords import cartesian2spherical
from becke.AtomicData import bohr_to_angs, atom_names, slater_radii

def number_of_radial_points(Z):
    """
    select the number of radial grid points for the subintegral
    around an atom with atomic number Z
    """
    # Hydrogen atom receives an initial quota of 20 points
    Nr = 20
    # Each shell receives an additional 5 points
    if Z >= 2:
        Nr += 5
    if Z >= 11:
        Nr += 5
    if Z >= 19:
        Nr += 5
    if Z >= 37:
        Nr += 5
    if Z >= 55:
        Nr += 5
    if Z >= 87:
        Nr += 5
        
    return Nr

def select_angular_grid(lebedev_order):
    """find closest Lebedev grid of requested order"""
    n_lebedev = abs(np.array(Lebedev_Lmax) - lebedev_order).argmin()
    Lmax = Lebedev_Lmax[n_lebedev]
    if lebedev_order != Lmax:
        print( "No grid for order %s, using grid which integrates up to Lmax = %s exactly instead."
            % (lebedev_order, Lmax) )
    # angular grid
    th,ph,angular_weights = np.array(LebedevGridPoints[Lebedev_Npts[n_lebedev]]).transpose()
    
    return Lmax, (th,ph,angular_weights)

def print_grid_summary(atomlist,
                       lebedev_order=23, radial_grid_factor=1):
    """
    print a table with number of radial and angular grid points
    for each atom. The total number of grid points in all overlapping
    grids is also shown.

    Parameters
    ----------
    atomlist     :  list of tuples (Zat,(x,y,z)) with
                    molecular geometry

    Optional
    --------
    lebedev_order      : order Lmax of the Lebedev grid
    radial_grid_factor : the number of radial grid points is increased by this factor

    Returns
    -------
    grid_sizes         : list of tuples (Nr, Nang)
                         giving the number of radial and angular grid points for
                         each atom
    
    """
    print( " " )
    print( "   Size of grids" )
    print( " " )
    print( " Atom      #    radial points    angular points     radial x angular" )
    print( " -------------------------------------------------------------------" )
    Ntot = 0
    grid_sizes = []
    for i,(Z,pos) in enumerate(atomlist):
        # radial grid
        Nr = number_of_radial_points(Z)
        # increase number of grid points is requested
        Nr *= radial_grid_factor
        # angular grid
        Lmax, (th,ph,angular_weights) = select_angular_grid(lebedev_order)
        Nang = th.size
        Ntot += Nr*Nang
        
        grid_sizes.append( (Nr,Nang) )
        
        print( " %2s      %3.1d      %7.1d           %7.1d           %10.1d" % (atom_names[Z-1], i+1, Nr, Nang, Nr*Nang) )

    print( " -------------------------------------------------------------------" )
    print( " Total                                                %10.1d" % Ntot )
    print( "" )
    
    return grid_sizes

def multicenter_integration(f, atomic_coordinates, atomic_numbers, lebedev_order=23, radial_grid_factor=1):
    """
    compute the integral

             / 
         I = | f(x,y,z) dV
             / 

    numerically on a multicenter spherical grid using Becke's scheme 
   
    Parameters
    ----------
    f                  : callable, f(x,y,z) should evaluate the function at the 
                         grid points specified by x = [x0,x1,...,xn], y = [y0,y1,...yn]
                         and z = [z0,z1,...,zn]
    atomic_coordinates : numpy array with shape (3,Nat), atomic_coordinates[:,i] is the 
                         cartesian position of atom i
    atomic_numbers     : numpy array with shape (Nat)
    
    Optional
    --------
    lebedev_order      : order Lmax of the Lebedev grid
    radial_grid_factor : the number of radial grid points is increased by this factor

    Returns
    -------
    I       : value of the integral
    """
    # angular grid
    Lmax, (th,ph,angular_weights) = select_angular_grid(lebedev_order)
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

    plot_cutoff_profile = False
    if plot_cutoff_profile == True:
        import matplotlib.pyplot as plt
        mu = np.linspace(-1.0,1.0,100)
        for k in range(1,5):
            plt.plot(mu, s(mu,k=k), label=r"$k=%d$" % k)
        plt.legend()
        plt.show()
    
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

    integral = 0.0
    # atom-centered subintegral
    for I in  range(0, Nat):
        # radial grid
        Nr = number_of_radial_points(atomic_numbers[I])
        # increase number of grid points is requested
        Nr *= radial_grid_factor
        rm = 0.5*slater_radii[atomic_names[I]]

        k = np.array(range(1,Nr+1))
        # grid points on interval [-1,1]
        xr = np.cos(k/(Nr+1.0) * np.pi)
        # weights
        radial_weights = np.pi/(Nr+1.0) * np.sin(k/(Nr+1.0) * np.pi)**2
        # from variable transformation
        g = 2 * rm**3 * np.sqrt(((1+xr)/(1-xr)**3)**3)
        radial_weights *= g
        # radial grid points on interval [0,infinity]
        r = rm * (1+xr)/(1-xr)

        # cartesian coordinates of grid
        x = (outerN(r, sc) + atomic_coordinates[0,I]).flatten()
        y = (outerN(r, ss) + atomic_coordinates[1,I]).flatten()
        z = (outerN(r, c ) + atomic_coordinates[2,I]).flatten()
        weights = outerN(radial_weights, 4.0*np.pi * angular_weights).flatten()
        #
        Npts = Nr*Nang
        # distance between grid points and atom i
        dist = np.zeros((Npts, Nat))
        for i in range(0, Nat):
            dist[:,i] = np.sqrt(    (x - atomic_coordinates[0,i])**2   \
                                   +(y - atomic_coordinates[1,i])**2   \
                                   +(z - atomic_coordinates[2,i])**2 )

        # P_i(r) as defined in eqn. (13)
        P = np.ones((Npts,Nat))
        for i in range(0, Nat):
            for j in range(0, Nat):
                if i==j:
                    continue
                # mu_ij as defined in eqn. (11)
                mu = (dist[:,i]-dist[:,j])/R[i,j]
                nu = mu + a[i,j]*(1-mu**2)
                P[:,i] *= s(nu)
        Ptot = np.sum(P, axis=1)
    
        # weight function
        wr = P[:,I]/Ptot

        # evaluate function on the grid
        fI = wr * f(x,y,z)

        sub_integral = np.sum( weights * fI )
        #print( "I= %d    sub_integral= %s" % (I, sub_integral) )
        integral += sub_integral

    return integral

def poisson(atomlist, f):
    """
    solve the Poisson equation
      __2
      \/  V(r) = -4 pi f(r)

    numerically on a multicenter spherical grid

    Parameters
    ----------
    atomlist           : list of tuples (Zat, (x,y,z)) 
                         with molecular geometry
    f                  : callable, f(x,y,z) should evaluate the charge distribution at the 
                         grid points specified by x = [x0,x1,...,xn], y = [y0,y1,...yn]
                         and z = [z0,z1,...,zn]

    Returns
    -------
    V       : callable, V(x,y,z) evaluates the electrostatic potential generated
              by the charge distribution f.
    """    
    # Bring geometry data into a form understood by the module MolecularIntegrals
    atomic_numbers, atomic_coordinates = atomlist2arrays(atomlist)
    return multicenter_poisson(f, atomic_coordinates, atomic_numbers,
                               radial_grid_factor=settings.radial_grid_factor,
                               lebedev_order=settings.lebedev_order)


def multicenter_poisson(f, atomic_coordinates, atomic_numbers, lebedev_order=23, radial_grid_factor=1):
    """
    solve the Poisson equation
      __2
      \/  V(r) = -4 pi f(r)

    numerically on a multicenter spherical grid

    Parameters
    ----------
    f                  : callable, f(x,y,z) should evaluate the charge distribution at the 
                         grid points specified by x = [x0,x1,...,xn], y = [y0,y1,...yn]
                         and z = [z0,z1,...,zn]
    atomic_coordinates : numpy array with shape (3,Nat), atomic_coordinates[:,i] is the 
                         cartesian position of atom i
    atomic_numbers     : numpy array with shape (Nat)
    
    Optional
    --------
    lebedev_order      : order Lmax of the Lebedev grid
    radial_grid_factor : the number of radial grid points is increased by this factor

    Returns
    -------
    V       : callable, V(x,y,z) evaluates the electrostatic potential generated
              by the charge distribution f.
    """    
    # angular grid
    Lmax, (th,ph,angular_weights) = select_angular_grid(lebedev_order)
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

    radial_potentials = []
    for I in  range(0, Nat):
        # radial grid
        Nr = number_of_radial_points(atomic_numbers[I])
        # increase number of grid points is requested
        Nr *= radial_grid_factor
        rm = 0.5*slater_radii[atomic_names[I]]

        k = np.array(range(1,Nr+1))
        # grid points on interval [-1,1]
        zr = k/(Nr+1.0)
        xr = np.cos(zr * np.pi)
        # weights
        radial_weights = np.pi/(Nr+1.0) * np.sin(k/(Nr+1.0) * np.pi)**2
        # from variable transformation
        g = 2 * rm**3 * np.sqrt(((1+xr)/(1-xr)**3)**3)
        radial_weights *= g
        # radial grid points on interval [0,infinity]
        r = rm * (1+xr)/(1-xr)

        # cartesian coordinates of grid
        x = (outerN(r, sc) + atomic_coordinates[0,I])
        y = (outerN(r, ss) + atomic_coordinates[1,I])
        z = (outerN(r, c ) + atomic_coordinates[2,I])
        weights = outerN(radial_weights, 4.0*np.pi * angular_weights)
        #
        Npts = Nr*Nang
        # distance between grid points and atom i
        dist = np.zeros((Nr,Nang, Nat))
        for i in range(0, Nat):
            dist[:,:,i] = np.sqrt(  (x - atomic_coordinates[0,i])**2   \
                                   +(y - atomic_coordinates[1,i])**2   \
                                   +(z - atomic_coordinates[2,i])**2 )

        # P_i(r) as defined in eqn. (13)
        P = np.ones((Nr,Nang,Nat))
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
        wr = P[:,:,I]/Ptot

        # evaluate function on the grid
        fI = wr * f(x,y,z)

        # solve the Poisson equation
        # __2  (I)
        # \/  V    = - 4 pi f_I(r)
        #

        # total charge qI
        qI = np.sum( weights * fI)
        
        # expand charge distribution f_I(r) into spherical harmonics
        #
        #  f_I(x,y,z) = sum_l sum_{m=-l}^{l}  f_lm(r) Y_{l,m}(th,ph)
        #
        # and solve Poisson equation for each (l,m) component
        radial_potentials.append( {} )
        sph_it = spherical_harmonics_it(th,ph)
        for Ylm,l,m in sph_it:
            wYlm = outerN(np.ones(Nr), angular_weights*Ylm.conjugate())
            fI_lm = 4.0*np.pi * np.sum(fI*wYlm, axis=-1)

            # boundary conditions for u(z)
            #  z=0 <-> x=1  <-> r=+infinity
            #  z=1 <-> x=-1 <-> r=0
            if (l,m) == (0,0):
                u0 = np.sqrt(4.0*np.pi) * qI
            else:
                u0 = 0.0
            u1 = 0.0
            #
            omx = 1-xr
            opx = 1+xr
            #   l*(l+1)
            # - ------- u = c0 u
            #     r^2
            c0 = -l*(l+1)/rm**2 * (omx/opx)**2
            # The transformation of partial derivatives from
            # r-coordinates to z-coordinates is accomplished as
            #  d^2 u      d u      d^2 u
            #  ----- = c1 --- + c2 -----
            #  d r^2      d z      d z^2
            c1 = 1.0/(4.0*np.pi*rm**2) * omx**(2.5) * (1+opx) / opx**(1.5)
            c2 = 1.0/(4.0*np.pi**2*rm**2) * omx**3 / opx
            # source term
            #   -4 pi r fI_lm  = -4 pi rm (1+x)/(1-x) fI_lm
            source = -4.0*np.pi * rm * opx / omx * fI_lm

            u_lm = solve_radial_dgl(c0,c1,c2,source, u0, u1)
            # r*V(r) = u(r)
            VI_lm = u_lm/r
            
            # z-array is sorted in assending order,
            #   0 < z[0] < z[1] < ... < z[-1] < 1
            # while associated r-array is sorted in descending order
            #   infinity > r[0] > r[1] > ... > r[-1] > 0

            spline_lm_real = interpolate.splrep(zr, VI_lm.real, s=0)
            spline_lm_imag = interpolate.splrep(zr, VI_lm.imag, s=0)
            radial_potentials[-1][(l,m)] = spline_lm_real, spline_lm_imag

            # Up to (including) l = (Lmax-1)/2 the integration on
            # the Lebedev grid is exact. If higher spherical harmonics
            # were to be included, interpolated function can blow
            # up after several iterations.
            if m == -(Lmax-1)/2:
                break

    def electrostatic_potential_func(x,y,z):
        """
        function for evaluating the electrostatic potential V(x,y,z)
        """
        V = 0j*x
        # build the total potential from the solutions to the subproblems
        #  V = sum_n  V^(n)
        for I in range(0, Nat):
            xI = x - atomic_coordinates[0,I]
            yI = y - atomic_coordinates[1,I]
            zI = z - atomic_coordinates[2,I]
            # spherical coordinates
            rI,thI,phI = cartesian2spherical((xI,yI,zI))
            #
            sph_it = spherical_harmonics_it(thI,phI)

            rm = 0.5*slater_radii[atomic_names[I]]
            xr = (rI-rm)/(rI+rm)
            zr = np.arccos(xr) / np.pi

            for Ylm,l,m in sph_it:
                
                spline_lm_real, spline_lm_imag = radial_potentials[I][(l,m)]
                # interpolate
                VI_lm = interpolate.splev(zr, spline_lm_real, der=0, ext=0) \
                        + 1.0j*interpolate.splev(zr, spline_lm_imag, der=0, ext=0)
                V += VI_lm*Ylm

                if m == -(Lmax-1)/2:
                    break

        return V.real
    
    return electrostatic_potential_func

    
def solve_radial_dgl(c0,c1,c2, source, u0, u1):
    """
    solve the differential equation on the interval [0,1]

                        du         d^2 u
     c0(z) u(z) + c1(z) -- + c2(z) -----  = source(z)
                        dz         dz^2

    for u(z) subject to the boundary conditions

    u(z=0) = u0
    u(z=1) = u1

    Central finite difference formula for 1st and 2nd derivatives are taken from Bickley [1].


    Parameters
    ----------
    c0,c1,c2, source  :   values of c0(z), c1(z), c2(z), source(z) on an equidistant
                          grid z_i = i/(N+1)  i=1,...,N
    u0, u1            :   scalar value, boundary conditions u(z=0), u(z=1)

    Returns
    -------
    u                 :   solution u(z) on the equidistant grid

    References
    ----------
    [1] W. Bickley, "Formulae for Numerical Differentiation",
        The Mathematical Gazette, vol. 25, no. 263, pp. 19-27 (1941)

    """
    # convert the differential equation into an algebraic equation
    #   A.u = b
    N = len(c0)
    h = 1.0/(N+1)    # separation between equidistant points

    # operators d/dz and d^2/dz^2
    D1 = np.zeros((N,N))
    D2 = np.zeros((N,N))
    # terms from boundary conditions
    b1 = np.zeros(N)
    b2 = np.zeros(N)
    # non-centered five-point formulae for i=0
    D1[0,0:4] = np.array([                    -20.0, +36.0, -12.0, 2.0])/(24.0*h)
    b1[0]     =           -6.0/(24.0*h) * u0
    D2[0,0:4] = np.array([                    -20.0,  +6.0,  +4.0, -1.0])/(12.0*h**2)
    b2[0]     =           +11.0/(12.0*h**2) * u0
    # non-centered six-point formulae for i=1
    D1[1,0:5] = np.array([                    -60.0, -40.0, +120.0, -30.0, +4.0])/(120.0*h)
    b1[1]     =            6.0/(120.0*h) * u0
    D2[1,0:5] = np.array([                    +80.0,-150.0,  +80.0, -5.0,  0.0])/(60.0*h**2)
    b2[1]     =           -5.0/(60.0*h**2) * u0
    # centered seven-point formulae for i=2
    D1[2,0:6] = np.array([                     +108.0, -540.0, 0.0, 540.0, -108.0, 12.0])/(720.0*h)
    b1[2]     =           -12.0/(720.0*h) * u0
    D2[2,0:6] = np.array([                      -54.0, +540.0, -980.0, +540.0, -54.0, +4.0])/(360.0*h**2)
    b2[2]     =             4.0/(360.0*h**2) * u0
    # centered seven-point formulae for i=3,...,N-4
    for i in range(3, N-3):
        D1[i,i-3:i+4] = np.array([-12.0, 108.0, -540.0, 0.0, +540.0, -108.0, +12.0])/(720.0*h)
        D2[i,i-3:i+4] = np.array([  4.0, -54.0, +540.0, -980.0, +540.0, -54.0, +4.0])/(360.0*h**2)
    # centered seven-point formulae for i=N-3
    D1[N-3,N-6:] = np.array([-12.0, +108.0, -540.0, 0.0, 540.0, -108.0              ])/(720.0*h)
    b1[N-3]      =                                                       +12.0/(720.0*h) * u1
    D2[N-3,N-6:] = np.array([+4.0, -54.0, 540.0, -980.0, +540.0, -54.0              ])/(360.0*h**2)
    b2[N-3]      =                                                        +4.0/(360.0*h**2) * u1
    # non-centered six-point formulae for i=N-2
    D1[N-2,N-5:] = np.array([-4.0, +30.0, -120.0, +40.0, +60.0                  ])/(120.0*h)
    b1[N-2]      =                                                -6.0/(120.0*h) * u1
    D2[N-2,N-5:] = np.array([0.0, -5.0, +80.0, -150.0, +80.0                    ])/(60.0*h**2)
    b2[N-2]      =                                                -5.0/(60.0*h**2) * u1
    # non-centered five-point formulae for i=N-1
    D1[N-1,N-4:] = np.array([-2.0, +12.0, -36.0, +20.0            ])/(24.0*h)
    b1[N-1]      =                                      +6.0/(24.0*h) * u1
    D2[N-1,N-4:] = np.array([-1.0, +4.0, +6.0, -20.0              ])/(12.0*h**2)
    b2[N-1]      =                                      +11.0/(12.0*h**2) * u1

    # The differential equation is transformed into the following linear equation
    #
    #  sum_j [ c0(i) delta_ij + c1(i) D1(i,j) + c2(i) D2(i,j) ] u(j) = source(i) - c1(i)*b1(i) - c2(i)*b2(i)
    #
    
    # build matrix A on the left hand side of the equation
    A = np.zeros((N,N))
    for i in range(0, N):
        A[i,i] = c0[i]
        A[i,:] += c1[i]*D1[i,:] + c2[i]*D2[i,:]

    # right hand side
    rhs = source - c1*b1 - c2*b2
    # solve matrix equation
    u = la.solve(A, rhs)
    
    return u


def multicenter_laplacian_ORIGINAL(f, atomic_coordinates, atomic_numbers, lebedev_order=23, radial_grid_factor=1):
    """
    compute the action of the Laplace operator on a function f
      __2
      \/  f(r)

    numerically on a multicenter spherical grid.

    The algorithm is very similar to but simpler than `multicenter_poisson()`: 
    The function f is decomposed using the weight functions wI(r)

         f(r) = sum_I fI(r)             with  fI(r) = wI(r) f(r)

    Each part fI(r) is expanded into spherical harmonics, fI = sum_lm fI_lm(r) Y_lm(th,ph),
    and the Laplace operator is applied in spherical harmonics,
         __2       d^2      L
         \/  = 1/r --- r - --- .
                   dr^2    r^2
    At the end the Laplacians of the parts are summed
         __2           __2
         \/  f = sum_I \/ fI(r)

    Parameters
    ----------
    f                  : callable, f(x,y,z) should evaluate the function at the 
                         grid points specified by x = [x0,x1,...,xn], y = [y0,y1,...yn]
                         and z = [z0,z1,...,zn]
    atomic_coordinates : numpy array with shape (3,Nat), atomic_coordinates[:,i] is the 
                         cartesian position of atom i
    atomic_numbers     : numpy array with shape (Nat)
    
    Optional
    --------
    lebedev_order      : order Lmax of the Lebedev grid
    radial_grid_factor : the number of radial grid points is increased by this factor

    Returns
    -------
    lapf               : callable, lapf(x,y,z) evaluates the Laplacian of f
    """    
    # angular grid
    Lmax, (th,ph,angular_weights) = select_angular_grid(lebedev_order)
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

    radial_laplacians = []
    for I in  range(0, Nat):
        # radial grid
        Nr = number_of_radial_points(atomic_numbers[I])
        # increase number of grid points is requested
        Nr *= radial_grid_factor
        rm = 0.5*slater_radii[atomic_names[I]]

        k = np.array(range(1,Nr+1))
        # grid points on interval [-1,1]
        zr = k/(Nr+1.0)
        xr = np.cos(zr * np.pi)
        # weights
        radial_weights = np.pi/(Nr+1.0) * np.sin(k/(Nr+1.0) * np.pi)**2
        # from variable transformation
        g = 2 * rm**3 * np.sqrt(((1+xr)/(1-xr)**3)**3)
        radial_weights *= g
        # radial grid points on interval [0,infinity]
        r = rm * (1+xr)/(1-xr)

        # cartesian coordinates of grid
        x = (outerN(r, sc) + atomic_coordinates[0,I])
        y = (outerN(r, ss) + atomic_coordinates[1,I])
        z = (outerN(r, c ) + atomic_coordinates[2,I])
        weights = outerN(radial_weights, 4.0*np.pi * angular_weights)
        #
        Npts = Nr*Nang
        # distance between grid points and atom i
        dist = np.zeros((Nr,Nang, Nat))
        for i in range(0, Nat):
            dist[:,:,i] = np.sqrt(  (x - atomic_coordinates[0,i])**2   \
                                   +(y - atomic_coordinates[1,i])**2   \
                                   +(z - atomic_coordinates[2,i])**2 )

        # P_i(r) as defined in eqn. (13)
        P = np.ones((Nr,Nang,Nat))
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
        wr = P[:,:,I]/Ptot

        # evaluate function on the grid
        fI = wr * f(x,y,z)

        # compute Laplacian
        #    (I)    __2  (I)
        # lap    =  \/  f    
        #

        # expand function f_I(r) into spherical harmonics
        #
        #  f_I(x,y,z) = sum_l sum_{m=-l}^{l}  f_lm(r) Y_{l,m}(th,ph)
        #
        # and apply Laplacian to each (l,m) component
        radial_laplacians.append( {} )
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
            #lapI_lm = LuI_lm/r
            
            # z-array is sorted in assending order,
            #   0 < z[0] < z[1] < ... < z[-1] < 1
            # while associated r-array is sorted in descending order
            #   infinity > r[0] > r[1] > ... > r[-1] > 0

            # It seems better to spline LuI_lm instead of LuI_lm/r,
            # because the first function is smoother and has no
            # singularity at r=0.
            spline_lm_real = interpolate.splrep(zr, LuI_lm.real, s=0)
            spline_lm_imag = interpolate.splrep(zr, LuI_lm.imag, s=0)
            radial_laplacians[-1][(l,m)] = spline_lm_real, spline_lm_imag

            if m == -(Lmax-1)/2:
                break

    def laplacian_func(x,y,z):
        """
        function for evaluating the Laplacian of f
             __2
             \/  f
        """
        lap = 0j*x
        # build the total Laplacian from the Laplacians of the parts
        #  __2           __2
        #  \/ f = sum_n  \/  f^(n)
        for I in range(0, Nat):
            xI = x - atomic_coordinates[0,I]
            yI = y - atomic_coordinates[1,I]
            zI = z - atomic_coordinates[2,I]
            # spherical coordinates
            rI,thI,phI = cartesian2spherical((xI,yI,zI))
            #
            sph_it = spherical_harmonics_it(thI,phI)

            rm = 0.5*slater_radii[atomic_names[I]]
            xr = (rI-rm)/(rI+rm)
            zr = np.arccos(xr) / np.pi

            for Ylm,l,m in sph_it:
                
                spline_lm_real, spline_lm_imag = radial_laplacians[I][(l,m)]
                # interpolate
                LuI_lm = interpolate.splev(zr, spline_lm_real, der=0, ext=0) \
                        + 1.0j*interpolate.splev(zr, spline_lm_imag, der=0, ext=0)

                lapI_lm = LuI_lm/rI
                lap += lapI_lm*Ylm

                if m == -(Lmax-1)/2:
                    break

        return lap.real
    
    return laplacian_func

def laplacian(atomlist, f, cusps_separate=True):
    """
    compute the action of the Laplace operator on a function f
      __2
      \/  f(r)

    numerically on a multicenter spherical grid.

    The algorithm is very similar to but simpler than `multicenter_poisson()`: 
    The function f is decomposed using the weight functions wI(r)

         f(r) = sum_I fI(r)             with  fI(r) = wI(r) f(r)

    Each part fI(r) is expanded into spherical harmonics, fI = sum_lm fI_lm(r) Y_lm(th,ph),
    and the Laplace operator is applied in spherical harmonics,
         __2       d^2      L
         \/  = 1/r --- r - --- .
                   dr^2    r^2
    At the end the Laplacians of the parts are summed
         __2           __2
         \/  f = sum_I \/ fI(r)

    Parameters
    ----------
    atomlist           : list of tuples (Zat, (x,y,z)) 
                         with molecular geometry
    f                  : callable, f(x,y,z) should evaluate the function at the 
                         grid points specified by x = [x0,x1,...,xn], y = [y0,y1,...yn]
                         and z = [z0,z1,...,zn]
    
    Optional
    --------
    cusps_separate     : if True, Laplacians of `f` around the atoms are calculated
                         separately to avoid numerical artifacts that arise from the cusps
                         if `f` represents a wavefunction

    Returns
    -------
    lapf               : callable, lapf(x,y,z) evaluates the Laplacian of f
    """
    # Bring geometry data into a form understood by the module MolecularIntegrals
    atomic_numbers, atomic_coordinates = atomlist2arrays(atomlist)
    return multicenter_laplacian(f, atomic_coordinates, atomic_numbers,
                                 cusps_separate=cusps_separate,
                                 radial_grid_factor=settings.radial_grid_factor,
                                 lebedev_order=settings.lebedev_order)

def multicenter_laplacian(f, atomic_coordinates, atomic_numbers,
                          cusps_separate=True,
                          lebedev_order=23, radial_grid_factor=1):
    """
    compute the action of the Laplace operator on a function f
      __2
      \/  f(r)

    numerically on a multicenter spherical grid.

    The algorithm is very similar to but simpler than `multicenter_poisson()`: 
    The function f is decomposed using the weight functions wI(r)

         f(r) = sum_I fI(r)             with  fI(r) = wI(r) f(r)

    Each part fI(r) is expanded into spherical harmonics, fI = sum_lm fI_lm(r) Y_lm(th,ph),
    and the Laplace operator is applied in spherical harmonics,
         __2       d^2      L
         \/  = 1/r --- r - --- .
                   dr^2    r^2
    At the end the Laplacians of the parts are summed
         __2           __2
         \/  f = sum_I \/ fI(r)

    Parameters
    ----------
    f                  : callable, f(x,y,z) should evaluate the function at the 
                         grid points specified by x = [x0,x1,...,xn], y = [y0,y1,...yn]
                         and z = [z0,z1,...,zn]
    atomic_coordinates : numpy array with shape (3,Nat), atomic_coordinates[:,i] is the 
                         cartesian position of atom i
    atomic_numbers     : numpy array with shape (Nat)
    
    Optional
    --------
    cusps_separate     : if True, Laplacians of `f` around the atoms are calculated
                         separately to avoid numerical artifacts that arise from the cusps
                         if `f` represents a wavefunction
    lebedev_order      : order Lmax of the Lebedev grid
    radial_grid_factor : the number of radial grid points is increased by this factor

    Returns
    -------
    lapf               : callable, lapf(x,y,z) evaluates the Laplacian of f
    """
    Nat = atomic_coordinates.shape[1]
    atomic_names = [atom_names[Z-1] for Z in atomic_numbers]
    ###
    # Calculating the Laplacian is complicated by the fact that
    # electronic wavefunctions have cusps at the nuclear positions.
    # We split the wavefunction into a sum of spherically symmetric
    # wavefunctions g_i around each atom with cusps and the remainder.
    #             
    #   f = sum  g (|r-Ri|)  +  [ f  -  sum  g (|r-Rj|) ]
    #          i  i                j       j  j
    #
    # The Laplacian is computed separately for the g_i's
    # using the fact the for a spherically symmetric function
    #
    #  __2             d^2           
    #  \/  g (r) = 1/r ---- (r g (r)) 
    #       i          dr^2     i     
    # 
    # The remainder should be relatively smooth everywhere and pose
    # no problem to numerical differentiation.
    #
    
    # 1) First we calculate the spherical averages of f around each atom
    #    and spline the resulting functions g_i(z) and their Laplacian
    #    using z-coordinates.
    # list of splines g_i
    radial_avg = []
    # list of splines of the Laplacian of the spherical average 
    radial_lap_avg = []
    for I in  range(0, Nat):
        # center around which the average is performed
        Zat = atomic_numbers[I]
        pos = atomic_coordinates[:,I]
        atom = (Zat, pos)
        #
        gI_func = spherical_average_func(atom, f,
                                         lebedev_order=lebedev_order,
                                         radial_grid_factor=radial_grid_factor)

        # radial grid
        Nr = number_of_radial_points(atomic_numbers[I])
        # increase number of grid points is requested
        Nr *= radial_grid_factor
        rm = 0.5*slater_radii[atomic_names[I]]

        k = np.array(range(1,Nr+1))
        # grid points on interval [-1,1]
        zr = k/(Nr+1.0)
        xr = np.cos(zr * np.pi)
        # radial grid points on interval [0,infinity]
        r = rm * (1+xr)/(1-xr)

        # variable transformation
        omx = 1-xr
        opx = 1+xr

        gI = gI_func(r)
        
        # coefficients for Laplacian operator after coordinate transformation
        c0 = 0.0*omx  # -l*(l+1)/rm**2 * (omx/opx)**2  is zero since l=0 
        c1 = 1.0/(4.0*np.pi*rm**2) * omx**(2.5) * (1+opx) / opx**(1.5)
        c2 = 1.0/(4.0*np.pi**2*rm**2) * omx**3 / opx

        # apply Laplacian operator to spherically symmetric function
        #         1  d^2 
        # L gI = --- ---- ( r gI(r) )
        #         r  dr^2

        # Is better to interpolate r*L(gI) instead of L(gI)
        # in order to avoid the singularity at r=0
        #  r L gI = d^2/dr^2 (r gI)
        rLgI = radial_difop(c0,c1,c2, r*gI)
        
        # interpolate gI(z) and LgI(z) on a grid
        spline_avg = interpolate.splrep(zr, gI, s=0)
        radial_avg.append(spline_avg)

        spline_lap_avg = interpolate.splrep(zr, rLgI, s=0)
        radial_lap_avg.append( spline_lap_avg )

    ####
    
    # 2) Define weight functions for fuzzy Voronoi decomposition
    #
    # angular grid
    Lmax, (th,ph,angular_weights) = select_angular_grid(lebedev_order)
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

    # 3) Evaluate function f on the multicenter grid and perform a spherical
    #    wave decomposition

    # 
    radial_laplacians = []
    for I in  range(0, Nat):
        # radial grid
        Nr = number_of_radial_points(atomic_numbers[I])
        # increase number of grid points is requested
        Nr *= radial_grid_factor
        rm = 0.5*slater_radii[atomic_names[I]]

        k = np.array(range(1,Nr+1))
        # grid points on interval [-1,1]
        zr = k/(Nr+1.0)
        xr = np.cos(zr * np.pi)
        # weights
        radial_weights = np.pi/(Nr+1.0) * np.sin(k/(Nr+1.0) * np.pi)**2
        # from variable transformation
        g = 2 * rm**3 * np.sqrt(((1+xr)/(1-xr)**3)**3)
        radial_weights *= g
        # radial grid points on interval [0,infinity]
        r = rm * (1+xr)/(1-xr)

        # cartesian coordinates of grid
        x = (outerN(r, sc) + atomic_coordinates[0,I])
        y = (outerN(r, ss) + atomic_coordinates[1,I])
        z = (outerN(r, c ) + atomic_coordinates[2,I])
        weights = outerN(radial_weights, 4.0*np.pi * angular_weights)
        #
        Npts = Nr*Nang
        # distance between grid points and atom i
        dist = np.zeros((Nr,Nang, Nat))
        for i in range(0, Nat):
            dist[:,:,i] = np.sqrt(  (x - atomic_coordinates[0,i])**2   \
                                   +(y - atomic_coordinates[1,i])**2   \
                                   +(z - atomic_coordinates[2,i])**2 )

        # P_i(r) as defined in eqn. (13)
        P = np.ones((Nr,Nang,Nat))
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
        wr = P[:,:,I]/Ptot

        # evaluate function on the grid
        fI = wr * f(x,y,z)

        if cusps_separate:
            ####
            # subtract spherical averages around each atom
            #
            #  f_i   = w (r) [ f(r) - sum  g (|r-Rj|) ]
            #           i                j  j
            for j in range(0, Nat):
                # rJ = |r-Rj]
                rJ = dist[:,:,j]
                # transform to z-coordinates
                rmJ = 0.5*slater_radii[atomic_names[j]]
                xrJ = (rJ-rmJ)/(rJ+rmJ)
                zrJ = np.arccos(xrJ) / np.pi
                
                gJ_spline = radial_avg[j]
                fI -= wr * interpolate.splev(zrJ, gJ_spline, der=0, ext=0)
            ####
        
        # compute Laplacian
        #    (I)    __2  (I)
        # lap    =  \/  f    
        #

        # expand function f_I(r) into spherical harmonics
        #
        #  f_I(x,y,z) = sum_l sum_{m=-l}^{l}  f_lm(r) Y_{l,m}(th,ph)
        #
        # and apply Laplacian to each (l,m) component
        radial_laplacians.append( {} )
        sph_it = spherical_harmonics_it(th,ph)
        for Ylm,l,m in sph_it:
            wYlm = outerN(np.ones(Nr), angular_weights*Ylm.conjugate())
            fI_lm = 4.0*np.pi * np.sum(fI*wYlm, axis=-1)

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

            spline_lm_real = interpolate.splrep(zr, LuI_lm.real, s=0)
            spline_lm_imag = interpolate.splrep(zr, LuI_lm.imag, s=0)
            radial_laplacians[-1][(l,m)] = spline_lm_real, spline_lm_imag

            if m == -(Lmax-1)/2:
                break

    #                                   __2
    # 4) Define function for evaluating \/ f
    def laplacian_func(x,y,z):
        """
        function for evaluating the Laplacian of f
             __2
             \/  f
        """
        lap = 0j*x
        # build the total Laplacian from the Laplacians of the parts
        #  __2           __2
        #  \/ f = sum_I  \/  f^(I)
        for I in range(0, Nat):
            xI = x - atomic_coordinates[0,I]
            yI = y - atomic_coordinates[1,I]
            zI = z - atomic_coordinates[2,I]
            # spherical coordinates
            rI,thI,phI = cartesian2spherical((xI,yI,zI))
            #
            sph_it = spherical_harmonics_it(thI,phI)

            rm = 0.5*slater_radii[atomic_names[I]]
            xr = (rI-rm)/(rI+rm)
            zr = np.arccos(xr) / np.pi

            for Ylm,l,m in sph_it:
                
                spline_lm_real, spline_lm_imag = radial_laplacians[I][(l,m)]
                # interpolate
                LuI_lm = interpolate.splev(zr, spline_lm_real, der=0, ext=0) \
                        + 1.0j*interpolate.splev(zr, spline_lm_imag, der=0, ext=0)
                lapI_lm = LuI_lm/rI
                lap += lapI_lm*Ylm
                
                if cusps_separate:
                    #####
                    if l == 0 and m == 0:
                        # add back the Laplacian due to spherical
                        # averaged part g_i that has been subtracted previously:
                        #   __2    __2                           __2
                        #   \/ f = \/ [ f - sum_i g_i ]  + sum_i \/ g_i
                        #   
                        rLgI = interpolate.splev(zr, radial_lap_avg[I], der=0, ext=0)
                        lap += rLgI/rI
                    #####
                    
                if m == -(Lmax-1)/2:
                    break

        return lap.real
    
    return laplacian_func

def radial_difop(c0,c1,c2, u):
    """
    apply the differential operator L defined by the coefficients c0,c1 and c2
    to a function u(z) defined on a grid covering the range [0,1]

                                   du         d^2 u
       L u(z) = c0(z) u(z) + c1(z) -- + c2(z) -----
                                   dz         dz^2

    Central finite difference formula for 1st and 2nd derivatives are taken from Bickley [1].


    Parameters
    ----------
    c0,c1,c2          :   values of c0(z), c1(z), c2(z) on an equidistant
                          grid z_i = i/(N+1)  i=1,...,N
                          defining the operator L
    u                 :   values of u(z) on the grid

    Returns
    -------
    Lu                :   result of applying the operator L to u(z)

    References
    ----------
    [1] W. Bickley, "Formulae for Numerical Differentiation",
        The Mathematical Gazette, vol. 25, no. 263, pp. 19-27 (1941)

    """
    N = len(u)      # number of grid points
    h = 1.0/(N+1)    # separation between equidistant points
    
    # operators d/dz and d^2/dz^2
    D1 = np.zeros((N,N))
    D2 = np.zeros((N,N))
    # non-centered seven-point formulae for i=0
    # D^1 u_0
    D1[0,0:7] = np.array([ -1764.0, +4320.0,  -5400.0,  +4800.0, -2700.0,  +864.0, -120.0  ])/(720.0*h)
    # D^2 u_0
    D2[0,0:7] = np.array([ +1624.0, -6264.0, +10530.0, -10160.0, +5940.0, -1944.0, +274.0 ])/(360.0*h**2)
    # non-centered seven-point formulae for i=1
    # D^1 u_1
    D1[1,0:7] = np.array([ -120.0, -924.0, +1800.0, -1200.0, +600.0, -180.0, +24.0 ])/(720.0*h)
    # D^2 u_1
    D2[1,0:7] = np.array([ +274.0, -294.0, -510.0, +940.0, -570.0, +186.0, -26.0 ])/(360.0*h**2)
    # non-centered seven-point formulae for i=2
    # D^1 u_2
    D1[2,0:7] = np.array([ +24.0, -288.0, -420.0, +960.0, -360.0, +96.0, -12.0])/(720.0*h)
    # D^2 u_2
    D2[2,0:7] = np.array([ -26.0, +456.0, -840.0, +400.0,  +30.0, -24.0,  +4.0])/(360.0*h**2)
    # centered seven-point formulae for i=3,...,N-4
    for i in range(3, N-3):
        D1[i,i-3:i+4] = np.array([-12.0, +108.0, -540.0,    0.0, +540.0, -108.0, +12.0])/(720.0*h)
        D2[i,i-3:i+4] = np.array([ +4.0,  -54.0, +540.0, -980.0, +540.0,  -54.0,  +4.0])/(360.0*h**2)
    # non-centered seven-point formulae for i=N-3
    # D^1 u_{N-3}   ~   D^1 u_4
    D1[N-3,N-7:] = np.array([+12.0, -96.0, +360.0, -960.0, +420.0, +288.0, -24.0 ])/(720.0*h)
    # D^2 u_{N-3}   ~   D^2 u_4
    D2[N-3,N-7:] = np.array([ +4.0, -24.0,  +30.0, +400.0, -840.0, +456.0, -26.0 ])/(360.0*h**2)
    # non-centered seven-point formulae for i=N-2
    # D^1 u_{N-2}   ~   D^1 u_5
    D1[N-2,N-7:] = np.array([ -24.0, +180.0, -600.0, +1200.0, -1800.0, +924.0, +120.0])/(720.0*h)
    # D^2 u_{N-2}   ~   D^2 u_5
    D2[N-2,N-7:] = np.array([ -26.0, +186.0, -570.0,  +940.0,  -510.0, -294.0, +274.0])/(360.0*h**2)
    # non-centered seven-point formulae for i=N-1
    # D^1 u_{N-1}   ~   D^1 u_6
    D1[N-1,N-7:] = np.array([ +120.0,  -864.0, +2700.0,  -4800.0,  +5400.0, -4320.0, +1764.0])/(720.0*h)
    # D^2 u_{N-1}   ~   D^2 u_6
    D2[N-1,N-7:] = np.array([ +274.0, -1944.0, +5940.0, -10160.0, +10530.0, -6264.0, +1624.0])/(360.0*h**2)

    # finite difference formula converts differential operators into matrices
    Lu = c0*u + c1*np.dot(D1,u) + c2*np.dot(D2,u)

    return Lu

def spherical_average_func(atom, f,
                           lebedev_order=23, radial_grid_factor=1):
    """
    create a function avg(r) that evaluates the spherical average
    of f(x,y,z) around the center defined by atom (Zat,(X,Y,Z)).

                          /             /
        avg(r) = 1/(4 pi) | sin(th) dth | dph  f(r-Rc)
                          /             /

    Parameters
    ----------
    atom              :  tuple (Zat,(xc,yc,zc)) where Zat is the atomic
                         number used to select the integration grid
                         and Rc=(xc,yc,zc) are the coordinates of the center
                         which is taken as the origin
    f                 :  callable f(x,y,z)

    Optional
    --------
    lebedev_order      : order Lmax of the Lebedev grid
    radial_grid_factor : the number of radial grid points is increased by this factor


    Returns
    -------
    avg                : callable avg(r) that evaluates the spherical
                         average of f around the atom
    """
    # origin
    Zat,(xc,yc,zc) = atom
    # angular grid
    Lmax, (th,ph,angular_weights) = select_angular_grid(lebedev_order)
    Nang = len(th)
    sc = np.sin(th)*np.cos(ph)
    ss = np.sin(th)*np.sin(ph)
    c  = np.cos(th)

    # radial grid
    Nr = number_of_radial_points(Zat)
    # increase number of grid points is requested
    Nr *= radial_grid_factor
    rm = 0.5*slater_radii[atom_names[Zat-1]]

    k = np.array(range(1,Nr+1))
    # grid points on interval [-1,1]
    zr = k/(Nr+1.0)
    xr = np.cos(zr * np.pi)
    # radial grid points on interval [0,infinity]
    r = rm * (1+xr)/(1-xr)

    # cartesian coordinates of grid
    x = (outerN(r, sc) + xc)
    y = (outerN(r, ss) + yc)
    z = (outerN(r, c ) + zc)
    #
    Npts = Nr*Nang

    # evaluate function on the grid
    fI = f(x,y,z)
    
    sph_it = spherical_harmonics_it(th,ph)
    for Ylm,l,m in sph_it:
        wYlm = outerN(np.ones(Nr), angular_weights*Ylm.conjugate())
        # We only need the l=0,m=0 term
        if l == 0 and m == 0:
            fI_00 = 4.0*np.pi * np.sum(fI*wYlm, axis=-1)
            break

    spline_00_real = interpolate.splrep(zr, fI_00.real, s=0)
    spline_00_imag = interpolate.splrep(zr, fI_00.imag, s=0)
        
    def avg_func(r):
        xr = (r-rm)/(r+rm)
        zr = np.arccos(xr) / np.pi

        avg00 =        interpolate.splev(zr, spline_00_real, der=0, ext=0) \
               + 1.0j*interpolate.splev(zr, spline_00_imag, der=0, ext=0)
        # avg00 is the projection of f onto Y_00 = 1/sqrt(4*pi)
        # To average over the solid angle, another factor of
        # 1/sqrt(4*pi) is missing.
        avg = avg00 * np.sqrt(1.0/(4.0*np.pi))
        
        return avg.real

    return avg_func


def multicenter_operation(fs, op, 
                          atomic_coordinates, atomic_numbers,
                          lebedev_order=23, radial_grid_factor=1):
    """
    given a list of functions fs = [f1,f2,...,fn] perform the n-ary operation

         h = op(f1,f2,...,fn)

    to create a new function h. Examples for binary operations are
    addition

        op = lambda fs: a*fs[0]+b*fs[1]

    or multiplication

        op = lambda fs: fs[0]*fs[1]

    The new function h is defined by radial splines on the multicenter grid.
    
    At first it appears easier to simplify define a new function h as

        def h(x,y,z):
            return op(f1(x,y,z),f2(x,y,z),...)

    However, every evaluation of h requires the evaluation of all f1,f2,...,fn. 
    If we would use this approach to build up more complicated functions, the effort
    for evaluating the combined functions would grow with the number of elementary
    functions used to define them. By interpolating h over the grid, the effort
    for evaluating h remains the same as that for each fi. 


    Parameters
    ----------
    fs                :  list of callables, fs[i](x,y,z) should evaluate the i-th function
                         grid points specified by x = [x0,x1,...,xn], y = [y0,y1,...yn]
                         and z = [z0,z1,...,zn]
    op                 : operator, callable or lambda function, takes a list of numpy grids
                         as a single argument, e.g. op([f1(x,y,z),f2(x,y,z)]) for a binary operator.
    atomic_coordinates : numpy array with shape (3,Nat), atomic_coordinates[:,i] is the 
                         cartesian position of atom i
    atomic_numbers     : numpy array with shape (Nat)
    
    Optional
    --------
    lebedev_order      : order Lmax of the Lebedev grid
    radial_grid_factor : the number of radial grid points is increased by this factor

    Returns
    -------
    h                  : callable, h(x,y,z) evaluates the function h(x,y,z) = op([f1(x,y,z),f2(x,y,z),...])
    """    
    # angular grid
    Lmax, (th,ph,angular_weights) = select_angular_grid(lebedev_order)
    Nang = len(th)
    sc = np.sin(th)*np.cos(ph)
    ss = np.sin(th)*np.sin(ph)
    c  = np.cos(th)
    # for nuclear weight functions
    def s(mu, k=3):
        ff = mu
        for ik in range(0, k):
            ff = 1.5 * ff -0.5 * ff**3
        return 0.5*(1-ff)
    
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

    radial_functions = []
    for I in  range(0, Nat):
        # radial grid
        Nr = number_of_radial_points(atomic_numbers[I])
        # increase number of grid points is requested
        Nr *= radial_grid_factor
        rm = 0.5*slater_radii[atomic_names[I]]

        k = np.array(range(1,Nr+1))
        # grid points on interval [-1,1]
        zr = k/(Nr+1.0)
        xr = np.cos(zr * np.pi)
        # weights
        radial_weights = np.pi/(Nr+1.0) * np.sin(k/(Nr+1.0) * np.pi)**2
        # from variable transformation
        gg = 2 * rm**3 * np.sqrt(((1+xr)/(1-xr)**3)**3)
        radial_weights *= gg
        # radial grid points on interval [0,infinity]
        r = rm * (1+xr)/(1-xr)

        # cartesian coordinates of grid
        x = (outerN(r, sc) + atomic_coordinates[0,I])
        y = (outerN(r, ss) + atomic_coordinates[1,I])
        z = (outerN(r, c ) + atomic_coordinates[2,I])
        weights = outerN(radial_weights, 4.0*np.pi * angular_weights)
        #
        Npts = Nr*Nang
        # distance between grid points and atom i
        dist = np.zeros((Nr,Nang, Nat))
        for i in range(0, Nat):
            dist[:,:,i] = np.sqrt(  (x - atomic_coordinates[0,i])**2   \
                                   +(y - atomic_coordinates[1,i])**2   \
                                   +(z - atomic_coordinates[2,i])**2 )

        # P_i(r) as defined in eqn. (13)
        P = np.ones((Nr,Nang,Nat))
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
        wr = P[:,:,I]/Ptot

        # evaluate h = op(f1,f2,...) on the grid
        
        # First we need to evaluate the functions fi on the grid
        fs_xyz = []
        for f in fs:
            fs_xyz.append( f(x,y,z) )
        # Then we pass the values of the functions as arguments to the operator
        hI = wr * op(fs_xyz)

        radial_functions.append( {} )
        sph_it = spherical_harmonics_it(th,ph)
        for Ylm,l,m in sph_it:
            wYlm = outerN(np.ones(Nr), angular_weights*Ylm.conjugate())
            hI_lm = 4.0*np.pi * np.sum(hI*wYlm, axis=-1)

            spline_lm_real = interpolate.splrep(zr, hI_lm.real, s=0)
            spline_lm_imag = interpolate.splrep(zr, hI_lm.imag, s=0)
            radial_functions[-1][(l,m)] = spline_lm_real, spline_lm_imag

            if m == -(Lmax-1)/2:
                break

    def h_func(x,y,z):
        """
        function for evaluating h = op(f,g)
        """
        h = 0j*x
        # sum over centers
        #  h = sum_I  h^(I)
        for I in range(0, Nat):
            xI = x - atomic_coordinates[0,I]
            yI = y - atomic_coordinates[1,I]
            zI = z - atomic_coordinates[2,I]
            # spherical coordinates
            rI,thI,phI = cartesian2spherical((xI,yI,zI))
            #
            sph_it = spherical_harmonics_it(thI,phI)

            rm = 0.5*slater_radii[atomic_names[I]]
            xr = (rI-rm)/(rI+rm)
            zr = np.arccos(xr) / np.pi

            for Ylm,l,m in sph_it:
                
                spline_lm_real, spline_lm_imag = radial_functions[I][(l,m)]
                # interpolate
                hI_lm = interpolate.splev(zr, spline_lm_real, der=0, ext=0) \
                        + 1.0j*interpolate.splev(zr, spline_lm_imag, der=0, ext=0)
                h += hI_lm*Ylm

                if m == -(Lmax-1)/2:
                    break

        return h.real
    
    return h_func

def radial_component_func(atom, f, l, m,  
                          lebedev_order=23, radial_grid_factor=1):
    """
    create a function f_{l,m}(r) that evaluates the projection 
    of f(x,y,z) onto the real spherical harmonic Y_{l,m}(th,ph).
    The origin of the coordinate system and the integration grid
    is defined by atom (Zat,(X,Y,Z)).

                  / /
        f  (r) =  | | Y   (th,ph)  f(r,th,ph)  sin(th) dth dph  
         l,m      / /  l,m           

    Parameters
    ----------
    atom              :  tuple (Zat,(xc,yc,zc)) where Zat is the atomic
                         number used to select the integration grid
                         and xc,yc,zc are the coordinates of the center
                         which is taken as the origin
    f                 :  callable f(x,y,z)
    l,m               :  integers, -l <= m <= l, angular quantum numbers

    Optional
    --------
    lebedev_order      : order Lmax of the Lebedev grid
    radial_grid_factor : the number of radial grid points is increased by this factor


    Returns
    -------
    f_lm               : callable f_lm(r) that evaluates the radial part of the
                         l,m-component of f(x,y,z)
    """
    Zat,(xc,yc,zc) = atom
    # angular grid
    Lmax, (th,ph,angular_weights) = select_angular_grid(lebedev_order)
    Nang = len(th)
    sc = np.sin(th)*np.cos(ph)
    ss = np.sin(th)*np.sin(ph)
    c  = np.cos(th)

    # radial grid
    Nr = number_of_radial_points(Zat)
    # increase number of grid points is requested
    Nr *= radial_grid_factor
    rm = 0.5*slater_radii[atom_names[Zat-1]]

    k = np.array(range(1,Nr+1))
    # grid points on interval [-1,1]
    zr = k/(Nr+1.0)
    xr = np.cos(zr * np.pi)
    # radial grid points on interval [0,infinity]
    r = rm * (1+xr)/(1-xr)

    # cartesian coordinates of grid
    x = (outerN(r, sc) + xc)
    y = (outerN(r, ss) + yc)
    z = (outerN(r, c ) + zc)
    #
    Npts = Nr*Nang

    # evaluate function on the grid
    fI = f(x,y,z)
    
    sph_it = spherical_harmonics_it(th,ph)
    for Ylm,ll,mm in sph_it:
        # We only need the l=0,m=0 term
        if ll == l and mm == m:
            # real spherical harmonics
            if m < 0:
                Ylm_real = -np.sqrt(2.0) * Ylm.imag
            elif m > 0:
                Ylm_real =  np.sqrt(2.0) * (-1)**m * Ylm.real
            else:
                # m == 0
                Ylm_real = Ylm.real

            wYlm = outerN(np.ones(Nr), angular_weights*Ylm_real)
                
            fI_lm = 4.0*np.pi * np.sum(fI*wYlm, axis=-1)
            break

    spline_lm = interpolate.splrep(zr, fI_lm, s=0)
        
    def f_lm_func(r):
        xr = (r-rm)/(r+rm)
        zr = np.arccos(xr) / np.pi

        f_lm = interpolate.splev(zr, spline_lm, der=0, ext=0)
        
        return f_lm

    return f_lm_func


def atomlist2arrays(atomlist):
    """
    convert geometry specification to numpy arrays

    Parameters
    ==========
    atomlist        :   list of tuples (Z,[x,y,z]) for each atom,
                        molecular geometry

    Returns
    =======
    atomic_numbers     : numpy array with atomic numbers
    atomic_coordinates : numpy array of shape (3,Nat) 
                         with coordinates
    """
    Nat = len(atomlist)
    atomic_numbers = np.zeros(Nat, dtype=int)
    atomic_coordinates = np.zeros((3,Nat))
    for i in range(0, Nat):
        Z,pos = atomlist[i]
        atomic_numbers[i] = Z
        atomic_coordinates[:,i] = pos
    return atomic_numbers, atomic_coordinates


def multicenter_grids(atomlist,
                      kmax=3,
                      lebedev_order=23, radial_grid_factor=1):
    """
    compute grid points and weights of the multicenter grids for visualization
   
    Parameters
    ----------
    atomlist           : list of tuples (Zat,(xI,yI,zI)) with atomic numbers and 
                         atom positions, which define the multicenter grid
    
    Optional
    --------
    kmax               : How fuzzy should the Voronoi polyhedrons be? Larger kmax
                         means borders are fuzzier.
    lebedev_order      : order Lmax of the Lebedev grid
    radial_grid_factor : the number of radial grid points is increased by this factor

    Returns
    -------
    grid_points        : list of tuples (x,y,z) with positions of points in each grid,
                         grid_points[I][0] contains the x-positions of the points
                         belonging to the grid around atom I
    grid_weights       : list of numpy arrays, grid_weights[I][k] contains the weight
                         of the k-th point in the grid around atom I due to the fuzzy
                         Voronoi decomposition.
    grid_volumes       : list of numpy arrays, grid_volumes[I][k] contains the volume
                         element around the k-th point in the grid at atom I.
    """
    atomic_numbers, atomic_coordinates = atomlist2arrays(atomlist)
    # angular grid
    Lmax, (th,ph,angular_weights) = select_angular_grid(lebedev_order)
    Nang = len(th)
    sc = np.sin(th)*np.cos(ph)
    ss = np.sin(th)*np.sin(ph)
    c  = np.cos(th)
    # for nuclear weight functions
    def s(mu, k=kmax):
        f = mu
        for ik in range(0, k):
            f = 1.5 * f -0.5 * f**3
        return 0.5*(1-f)

    plot_cutoff_profile = False
    if plot_cutoff_profile == True:
        import matplotlib.pyplot as plt
        mu = np.linspace(-1.0,1.0,100)
        for k in range(1,5):
            plt.plot(mu, s(mu,k=k), label=r"$k=%d$" % k)
        plt.legend()
        plt.show()
    
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

    grid_points = []
    grid_weights = []
    grid_volumes = []
    # atom-centered subintegral
    for I in  range(0, Nat):
        # radial grid
        Nr = number_of_radial_points(atomic_numbers[I])
        # increase number of grid points is requested
        Nr *= radial_grid_factor
        rm = 0.5*slater_radii[atomic_names[I]]

        k = np.array(range(1,Nr+1))
        # grid points on interval [-1,1]
        xr = np.cos(k/(Nr+1.0) * np.pi)
        # weights
        radial_weights = np.pi/(Nr+1.0) * np.sin(k/(Nr+1.0) * np.pi)**2
        # from variable transformation
        g = 2 * rm**3 * np.sqrt(((1+xr)/(1-xr)**3)**3)
        radial_weights *= g
        # radial grid points on interval [0,infinity]
        r = rm * (1+xr)/(1-xr)

        # cartesian coordinates of grid
        x = (outerN(r, sc) + atomic_coordinates[0,I]).flatten()
        y = (outerN(r, ss) + atomic_coordinates[1,I]).flatten()
        z = (outerN(r, c ) + atomic_coordinates[2,I]).flatten()
        weights = outerN(radial_weights, 4.0*np.pi * angular_weights).flatten()
        #
        Npts = Nr*Nang
        # distance between grid points and atom i
        dist = np.zeros((Npts, Nat))
        for i in range(0, Nat):
            dist[:,i] = np.sqrt(    (x - atomic_coordinates[0,i])**2   \
                                   +(y - atomic_coordinates[1,i])**2   \
                                   +(z - atomic_coordinates[2,i])**2 )

        # P_i(r) as defined in eqn. (13)
        P = np.ones((Npts,Nat))
        for i in range(0, Nat):
            for j in range(0, Nat):
                if i==j:
                    continue
                # mu_ij as defined in eqn. (11)
                mu = (dist[:,i]-dist[:,j])/R[i,j]
                nu = mu + a[i,j]*(1-mu**2)
                P[:,i] *= s(nu)
        Ptot = np.sum(P, axis=1)
    
        # weight function due to partitioning of volume
        wr = P[:,I]/Ptot
        
        grid_points.append( (x.flatten(), y.flatten(), z.flatten()) )
        # The weights come from the fuzzy Voronoi partitioning 
        grid_weights.append( wr.flatten() )
        # The naming is a little bit confusing, the `weights` are
        # actually the volume elements dV_i around each point.
        grid_volumes.append( weights.flatten() )

    return grid_points, grid_weights, grid_volumes

def join_grids(points, weights, volumes):
    """
    combine the multicenter grids into a single grid so that we get
    a quadrature rule for integration
         /
         | f(x,y,z) dV = sum  w  f(x ,y ,z ) 
         /                  i  i    i  i  i

    Parameters
    ----------
    points, weights, volumes:  return values of `multicenter_grids`

    Returns
    -------
    x,y,z     :  1d numpy arrays with cartesian coordinates of grid points
    w         :  1d numpy array with weights
    """
    # weights of quadrature rule
    w = []
    # sampling points of quadrature rule
    x,y,z = [],[],[]
    # xI,yI,zI : grid points of spherical grid around atom I
    # wI : weights of spherical grid around atom I
    # vI : volume elements of spherical grid around atom I
    for (xI,yI,zI), wI, dVI in zip(points, weights, volumes):
        x += [xI]
        y += [yI]
        z += [zI]
        # The weights are the product of the weight function
        # (from fuzzy Voronoi decomposition of space) and the volume element.
        w += [wI*dVI]
    # join arrays
    w = np.hstack(w)
    x = np.hstack(x)
    y = np.hstack(y)
    z = np.hstack(z)

    return x,y,z, w
        

