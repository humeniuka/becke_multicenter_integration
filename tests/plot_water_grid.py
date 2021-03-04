#!/usr/bin/env python
"""
visualize Becke grid for the water molecule
"""
import matplotlib.pyplot as plt

from becke.MulticenterIntegration import multicenter_grids
from becke.AtomicData import element_name

# experimental geometry of water
#  r(OH) = 0.958 Ang, angle(H-O-H) = 104.4776 degrees
atomlist = [
    (8, (0.000000000000000,  0.000000000000000, -0.222540557483415)),
    (1, (0.000000000000000, +1.431214118579765,  0.886071388908105)),
    (1, (0.000000000000000, -1.431214118579765,  0.886071388908105))]

grid_points, grid_weights, grid_volumes = multicenter_grids(atomlist, kmax=20,
                                                            lebedev_order=41, radial_grid_factor=3)

# show cut along x=0
eps = 0.2
for i,((xi,yi,zi),wi) in enumerate(zip(grid_points, grid_weights)):
    # select points close to the yz-plane
    y2d = yi[(-eps <= xi) & (xi <= eps)]
    z2d = zi[(-eps <= xi) & (xi <= eps)]
    w2d = wi[(-eps <= xi) & (xi <= eps)]
    
    # show position of atom, on which the grid is centered
    Zat, pos = atomlist[i]
    plt.plot([pos[1]],[pos[2]], "o")
    plt.text(pos[1],pos[2], element_name(Zat).upper(),
             fontsize=20, 
             horizontalalignment='center', verticalalignment='center')
    # show grid points, the size of the marke is proportional to the weight
    plt.scatter(y2d,z2d,s=w2d*100, alpha=0.5)
    
plt.xlim((-2.5, 2.5))
plt.ylim((-2.0, 3.0))

plt.gca().axis("off")
plt.show()

