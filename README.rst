
Numerical Molecular Integrals on Multicenter Grids (Becke's Integration Scheme)
-------------------------------------------------------------------------------
This python package computes molecular integrals numerically for arbitrary
basis functions using Becke's multicenter grids [1]_. 

* The following matrix elements are available:

   - overlap (a|b)
   - kinetic energy (a|T|b)
   - nuclear attraction energy: sum_I -Z(I) * (a|1/rI|b)
   - dipole operator (a|e*r|b)
   - electron repulsive integrals (ab|1/r12|cd)

* Arbitrary functions can be integrated numerically over space [1]_.  
* Apart from this, Poisson's equation and Laplace's equation can be solved numerically
  for arbitrary charge distributions or wavefunctions [2]_.

The code is rather slow and only intended for debugging electron integral routines.

Requirements
------------

Required python packages:

 * numpy, scipy, matplotlib
 * mpmath

Installation
------------
The package is installed with

.. code-block:: bash

   $ pip install -e .

in the top directory. To verify the proper functioning of the code
a set of tests should be run with

.. code-block:: bash

   $ cd tests
   $ python -m unittest

Getting Started
---------------

First we need to import `numpy` and the `becke` module:

.. code-block:: python
		
   import numpy as np
   import becke

  
The multicenter grid is defined by the molecular geometry. Space is partitioned into
fuzzy Voronoi polyhedra. Each atom is the center of a spherical grid and the grids of
all atoms are superimposed. The geometry is defined as a list of tuples `(Zat, (X,Y,Z))`
where `Zat` is the atomic number, and `X,Y,Z` are the cartesian coordinates of the atom
in bohr:

.. code-block:: python
		
   # H2 geometry
   atoms = [(1, (0.0, 0.0,-0.5)),
            (1, (0.0, 0.0,+0.5))]

The resolution of the multicenter grids is controlled via:

.. code-block:: python
		
   from becke import settings
   settings.radial_grid_factor = 3  # increase number of radial points by factor 3
   settings.lebedev_order = 23      # angular Lebedev grid of order 23

Wavefunctions are defined as python functions, which take three numpy arrays with the
x-, y- and z-coordinates as input.

.. code-block:: python
		
   # 1s orbital on first hydrogen sA
   def aoA(x,y,z):
      r = np.sqrt(x**2+y**2+(z+0.5)**2)
      return 1.0/np.sqrt(np.pi) * np.exp(-r)

   # 1s orbital on second hydrogen sB
   def aoB(x,y,z):
       r = np.sqrt(x**2+y**2+(z-0.5)**2)
       return 1.0/np.sqrt(np.pi) * np.exp(-r)

Now typical one- and two-electron integrals can be calculated for the atomic orbitals
by numerical integration:
       
.. code-block:: python

   # one-electron integrals
   print("(a|b)= ", becke.overlap(atoms, aoA, aoB) )
   print("(a|b)= ", becke.integral(atoms, lambda x,y,z: aoA(x,y,z)*aoB(x,y,z)) )
   print("(a|T|b)= ", becke.kinetic(atoms, aoA, aoB) )
   print("(a|V|b)= ", becke.nuclear(atoms, aoA, aoB) )
   print("(a|e*r|b)= ", becke.electronic_dipole(atoms, aoA, aoB) )

   # two-electron repulsion integrals
   print("(aa|bb)= ", becke.electron_repulsion(atoms, aoA, aoA, aoB, aoB) )
   print("(ab|ab)= ", becke.electron_repulsion(atoms, aoA, aoB, aoA, aoB) )

When computing the Laplacian or solving the Poisson equation, the return values
are functions themselves that allow to evaluate the Laplacian or electrostatic
potential on a grid `(x,y,z)`:
   
.. code-block:: python
   
   #                        __2
   # Laplacian lap(x,y,z) = \/ wfn
   lap = becke.laplacian(atoms, aoA)

The Laplacian can be used to compute the kinetic energy:

.. code-block:: python
		
   print("(a|T|a)= ", -0.5 * becke.integral(atoms, lambda x,y,z: aoA(x,y,z) * lap(x,y,z) ) )

The following code solves the Poisson equation for the electron density of the
hydrogen atom and plots the electrostatic potential along the z-axis:

.. code-block:: python

   s = becke.overlap(atoms, aoA, aoB)
   # lowest molecular orbital of hydrogen molecule
   def mo(x,y,z):
       return (aoA(x,y,z) + aoB(x,y,z))/np.sqrt(2*(1+s))

   print("(mo|mo)= ", becke.overlap(atoms, mo, mo) )
   
   # electrostatic potential due to electronic density
   v = becke.poisson(atoms, lambda x,y,z: mo(x,y,z)**2)

   import matplotlib.pyplot as plt
   r = np.linspace(-2.0, 2.0, 100)
   plt.plot(r, v(0*r,0*r,r), label=r"$V_{elec}$")

   plt.xlabel(r"z / $a_0$")
   plt.ylabel(r"electrostatic potential")
   plt.show()

   
----------
References
----------
.. [1] A.Becke, "A multicenter numerical integration scheme for polyatomic molecules",
    J.Chem.Phys. 88, 2547 (1988)
.. [2] A.Becke, R.Dickson, "Numerical solution of Poisson's equation in polyatomic molecules",
    J.Chem.Phys. 89, 2993 (1988)

Some useful information is also contained in

.. [3] T.Shiozaki, S.Hirata, "Grid-based numerical Hartree-Fock solutions of polyatomic molecules",
    Phys.Rev. A 76, 040503(R) (2007)
