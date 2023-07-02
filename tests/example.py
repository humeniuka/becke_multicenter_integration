import numpy as np
import becke

# The multicenter grid is defined by the molecular geometry. Space is partitioned into
# fuzzy Voronoi polyhedra. Each atom is the center of a spherical grid and the grids of
# all atoms are superimposed.

# H2 geometry
atoms = [(1, (0.0, 0.0,-0.5)),
         (1, (0.0, 0.0,+0.5))]

# Wavefunctions are defined as python functions, which take three numpy arrays with the
# x-, y- and z-coordinates as input.

# 1s orbital on first hydrogen sA
def aoA(x,y,z):
    r = np.sqrt(x**2+y**2+(z+0.5)**2)
    return 1.0/np.sqrt(np.pi) * np.exp(-r)

# 1s orbital on second hydrogen sB
def aoB(x,y,z):
    r = np.sqrt(x**2+y**2+(z-0.5)**2)
    return 1.0/np.sqrt(np.pi) * np.exp(-r)


# one-electron integrals
print("(a|b)= ", becke.overlap(atoms, aoA, aoB) )
print("(a|b)= ", becke.integral(atoms, lambda x,y,z: aoA(x,y,z)*aoB(x,y,z)) )
print("(a|T|b)= ", becke.kinetic(atoms, aoA, aoB) )
print("(a|V|b)= ", becke.nuclear(atoms, aoA, aoB) )
print("(a|e*r|b)= ", becke.electronic_dipole(atoms, aoA, aoB) )

# two-electron repulsion integrals
print("(aa|bb)= ", becke.electron_repulsion(atoms, aoA, aoA, aoB, aoB) )
print("(ab|ab)= ", becke.electron_repulsion(atoms, aoA, aoB, aoA, aoB) )

#                        __2
# Laplacian lap(x,y,z) = \/ wfn
lap = becke.laplacian(atoms, aoA)

# The Laplacian can be used to compute the kinetic energy
print("(a|T|a)= ", -0.5 * becke.integral(atoms, lambda x,y,z: aoA(x,y,z) * lap(x,y,z) ) )

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


# Operations can be performed directly on grid values.
from becke.BeckeMulticenterGrid import BeckeMulticenterGrid

grid = BeckeMulticenterGrid(atoms, radial_grid_factor=3, lebedev_order=23)

# Put the orbitals onto a grid.
aoA_values = grid.evaluate(aoA)
aoB_values = grid.evaluate(aoB)

# Product of orbitals on the grid.
prodAB_values = aoA_values * aoB_values

# Create an interpolation function prodAB(x,y,z) = aoA(x,y,z) * aoB(x,y,z)
# for plotting.
prodAB = grid.interpolate(prodAB_values)

plt.plot(r, prodAB(0*r,0*r,r).real)
plt.xlabel(r"z / $a_0$")
plt.ylabel(r"$\phi_A(r) \phi_B(r)$")
plt.show()

print("(a|b)= ", grid.integrate(prodAB_values))

laplacian_aoB_values = grid.laplacian(aoB_values)
print("(a|T|b)= ", -0.5*grid.integrate(aoA_values * laplacian_aoB_values))

# compute gradient of orbital B on the grid. The partial derivatives
# with respect to x,y and z are returned as separate numpy arrays.
daoBdx_values, daoBdy_values, daoBdz_values = grid.gradient(aoB_values)

# Interpolate grid values for plotting the z-component of the gradient.
daoBdz = grid.interpolate(daoBdz_values)

plt.plot(r, daoBdz(0*r,0*r,r).real)
plt.xlabel(r"z / $a_0$")
plt.ylabel(r"$d\phi_B/dz$")
plt.show()

print("(a|d/dz|b)= ", grid.integrate(aoA_values * daoBdz_values))
