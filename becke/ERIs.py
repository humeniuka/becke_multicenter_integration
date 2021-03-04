#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
compute electron repulsion integrals (ERI) 

  (ab|1/r12|cd)

between numerical atomic basis functions using Becke's multicenter integration scheme
"""
from becke.MulticenterIntegration import multicenter_poisson, multicenter_integration, atomlist2arrays
from becke import settings

import numpy as np
from scipy import interpolate

def electron_repulsion(atomlist, bfA, bfB, bfC, bfD):
    """
    compute the electron repulsion integral

      (ab|1/r12|cd)

    Parameters
    ==========
    atomlist            :  list of tuples (Z,[x,y,z]) with atomic numbers
                           and positions
    bfA, bfB, bfC, bfD  :  callables,
                           atomic basis functions, e.g. bfA(x,y,z) etc.

    Returns
    =======
    Iabcd               :  float, ERI
    """
    
    def rhoAB(x,y,z):
        return bfA(x,y,z) * bfB(x,y,z)
    def rhoCD(x,y,z):
        return bfC(x,y,z) * bfD(x,y,z)
    
    Iabcd = electron_repulsion_rho(atomlist, rhoAB, rhoCD)
    return Iabcd

def electron_repulsion_rho(atomlist, rhoAB, rhoCD):
    """
    compute the electron repulsion integral

      (ab|1/r12|cd)

    Parameters
    ==========
    atomlist            :  list of tuples (Z,[x,y,z]) with atomic numbers
                           and positions
    rhoAB, rhoCD        :  callables, rhoAB(x,y,z) computes the product a(x,y,z)*b(x,y,z)
                           and rhoCD(x,y,z) computes the product c(x,y,z)*d(x,y,z)

    Returns
    =======
    Iabcd               :  float
    """
    # bring data into a form understood by the module MolecularIntegrals
    atomic_numbers, atomic_coordinates = atomlist2arrays(atomlist)
    # Now we compute the integrals numerically on a multicenter grid.
    #
    # compute electrostatic Hartree term
    # (ab|1/r12|cd)
    # 1. solve the Poisson equation to get the electrostatic potential
    #    Vcd(r) due to the charge distribution c(r)*d(r)    
    Vcd = multicenter_poisson(rhoCD, atomic_coordinates, atomic_numbers,
                              radial_grid_factor=settings.radial_grid_factor,
                              lebedev_order=settings.lebedev_order)
    #
    # 2. integrate a(r)*b(r)*Vcd(r)
    def Iabcd_integrand(x,y,z):
        return rhoAB(x,y,z) * Vcd(x,y,z)

    # Coulomb integral 
    Iabcd = multicenter_integration(Iabcd_integrand, atomic_coordinates, atomic_numbers,
                                    radial_grid_factor=settings.radial_grid_factor,
                                    lebedev_order=settings.lebedev_order)

    return Iabcd
