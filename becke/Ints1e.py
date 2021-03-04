#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
compute matrix elements of single particle operators such as
 - the overlap (a|b)
 - the kinetic energy (a|T|b)
 - the nuclear attraction energy (a|sum_I -Z(I)/|r-R(I)| |b)
 - the dipole operator (a|e*r|b)
using Becke's multicenter integration scheme
"""
import numpy as np
import numpy.linalg as la

from becke.MulticenterIntegration import multicenter_integration, multicenter_laplacian, atomlist2arrays
# default parameters controlling the resolution of multicenter grid
from becke import settings

def integral(atomlist, f):
    """
    compute the integral

             / 
         I = | f(x,y,z) dV
             / 

    numerically on a multicenter spherical grid using Becke's scheme 
   
    Parameters
    ----------
    atomlist           : list of tuples (Zat, (x,y,z)) 
                         with molecular geometry
    f                  : callable, f(x,y,z) should evaluate the function at the 
                         grid points specified by x = [x0,x1,...,xn], y = [y0,y1,...yn]
                         and z = [z0,z1,...,zn]

    Returns
    -------
    I       : value of the integral
    """
    # Bring geometry data into a form understood by the module MolecularIntegrals
    atomic_numbers, atomic_coordinates = atomlist2arrays(atomlist)

    # compute integral on a multicenter grid
    integ = multicenter_integration(f, atomic_coordinates, atomic_numbers,
                                  radial_grid_factor=settings.radial_grid_factor,
                                  lebedev_order=settings.lebedev_order)

    return integ

def overlap(atomlist, bfA, bfB):
    """
    overlap between two basis functions
    
        (a|b)

    Parameters
    ----------
    atomlist       :  list of tuples (Z,[x,y,z]) with atomic numbers
                      and positions
    bfA, bfB       :  callables, atomic basis functions
                      e.g. bfA(x,y,z) etc.

    Returns
    -------
    Sab            :  float, overlap integral    
    """
    # Bring geometry data into a form understood by the module MolecularIntegrals
    atomic_numbers, atomic_coordinates = atomlist2arrays(atomlist)
    # Now we compute the integrals numerically on a multicenter grid.
    # 1. define integrand s = a b
    def s_integrand(x,y,z):
        return bfA(x,y,z).conjugate() * bfB(x,y,z)
    #                                    
    # 2. integrate density on a multicenter grid
    Sab = multicenter_integration(s_integrand, atomic_coordinates, atomic_numbers,
                                  radial_grid_factor=settings.radial_grid_factor,
                                  lebedev_order=settings.lebedev_order)

    return Sab
    
    
def kinetic(atomlist, bfA, bfB):
    """
    matrix element of kinetic energy
                   __2
         (a| - 1/2 \/  |b)

    Parameters
    ----------
    atomlist       :  list of tuples (Z,[x,y,z]) with atomic numbers
                      and positions
    bfA, bfB       :  callables, atomic basis functions
                      e.g. bfA(x,y,z) etc.

    Returns
    -------
    Tab            :  float, kinetic energy integral
    """
    # Bring data into a form understood by the module MolecularIntegrals
    atomic_numbers, atomic_coordinates = atomlist2arrays(atomlist)
    # Now we compute the integrals numerically on a multicenter grid.
    #                   __2
    # 1. find Laplacian \/ b
    lapB = multicenter_laplacian(bfB, atomic_coordinates, atomic_numbers,
                                 radial_grid_factor=settings.radial_grid_factor,
                                 lebedev_order=settings.lebedev_order)
    #                                __2
    # 2. define integrand t = -1/2 a \/ b
    def t_integrand(x,y,z):
        return -0.5 * bfA(x,y,z).conjugate() * lapB(x,y,z)
    #                                    
    # 3. integrate kinetic energy density on multicenter grid
    Tab = multicenter_integration(t_integrand, atomic_coordinates, atomic_numbers,
                                  radial_grid_factor=settings.radial_grid_factor,
                                  lebedev_order=settings.lebedev_order)

    return Tab

def nuclear(atomlist, bfA, bfB):
    """
    matrix element of nuclear attraction
                     (-Zk)
         (a| sum_k -------- |b)
                    |r-Rk|
    Parameters
    ----------
    atomlist       :  list of tuples (Z,[x,y,z]) with atomic numbers
                      and positions
    bfA, bfB       :  callables, atomic basis functions
                      e.g. bfA(x,y,z) etc.

    Returns
    -------
    Nab            :  float, integral of nuclear attraction
    """
    # Bring data into a form understood by the module MolecularIntegrals
    atomic_numbers, atomic_coordinates = atomlist2arrays(atomlist)
    # Now we compute the integrals numerically on a multicenter grid.
    #                          
    # 1. define integrand n = sum_k (-Zk)/|r-Rk| * a(r) * b(r)
    def nuc_integrand(x,y,z):
        # nuclear attraction potential
        nuc = 0.0*x
        for Zk,Rk in atomlist:
            # position of k-th nucleus
            X,Y,Z = Rk
            # Zk is the atomic number of k-th nucleus
            nuc -= Zk / np.sqrt( (x-X)**2 + (y-Y)**2 + (z-Z)**2 )
        # product of bra and ket wavefunctions
        rhoAB = bfA(x,y,z).conjugate() * bfB(x,y,z)
        nuc *= rhoAB
        
        return nuc
    #                                    
    # 2. integrate nuclear attraction energy density on multicenter grid
    Nab = multicenter_integration(nuc_integrand, atomic_coordinates, atomic_numbers,
                                  radial_grid_factor=settings.radial_grid_factor,
                                  lebedev_order=settings.lebedev_order)

    return Nab

def nuclear_repulsion(atomlist):
    """
    repulsion energy between nuclei
                          Z(i) Z(j)
       V_rep = sum  sum  -------------
                i   j>i   |R(i)-R(j)|

    Parameters
    ----------
    atomlist        :  list of tuples (Z,[x,y,z]) for each atom,
                       molecular geometry

    Returns
    -------
    Vrep            :  float, repulsion energy
    """
    atomic_numbers, atomic_coordinates = atomlist2arrays(atomlist)
    Nat = len(atomic_numbers)
    
    Vrep = 0.0
    for i in range(0, Nat):
        Zi = atomic_numbers[i]
        Ri = atomic_coordinates[:,i]
        for j in range(i+1,Nat):
            Zj = atomic_numbers[j]
            Rj = atomic_coordinates[:,j]
            
            Vrep += Zi*Zj / la.norm(Ri-Rj)
            
    return Vrep

def electronic_dipole(atomlist, bfA, bfB):
    """
    electric dipole between two basis functions
    
        (a|e*r|b)

    Parameters
    ----------
    atomlist       :  list of tuples (Z,[x,y,z]) with atomic numbers
                      and positions
    bfA, bfB       :  callables, atomic basis functions
                      e.g. bfA(x,y,z) etc.

    Returns
    -------
    Dab            :  numpy array with components [Dx,Dy,Dz] of dipole
                      matrix elements (a|e*x|b), (a|e*y|b), (a|e*z|b)
    """
    # Bring geometry data into a form understood by the module MolecularIntegrals
    atomic_numbers, atomic_coordinates = atomlist2arrays(atomlist)

    # 1. define integrand, xyz=0,1,2 selects the x,y or z-component
    def dipole_density(x,y,z, xyz=0):
        er = [x,y,z]
        return bfA(x,y,z) * er[xyz] * bfB(x,y,z)
    #                                    
    # 2. integrate density on a multicenter grid
    Dab = np.zeros(3, dtype=complex)
    for xyz in [0,1,2]:
        Dab[xyz] = multicenter_integration(lambda x,y,z: dipole_density(x,y,z, xyz=xyz),
                                           atomic_coordinates, atomic_numbers,
                                           radial_grid_factor=settings.radial_grid_factor,
                                           lebedev_order=settings.lebedev_order)
    return Dab.real
    
