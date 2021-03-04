#!/usr/bin/env python
# coding: utf-8
"""spherical harmonics"""

import numpy as np
from mpmath import mpf # needed to evalulate ratios of large factorials

# adapted from Warren Weckesser for use with numpy
# http://projects.scipy.org/scipy/attachment/ticket/1296/assoc_legendre.py

def fact(n):
    """n!"""
    f = 1
    while(n>1):
        f *= n
        n -= 1
    return(f)
    
def fact2(k):
    """n!!"""
    if k <= 0:
        return 1
    f = k
    while k >= 3:
        k -= 2
        f *= k
    return f


# renormalized associated Legendre polynomials, 
# (2*l-1)!! Pr_(m,l) = P_(m,l)
# The double factorial can be cancelled 

def _increase_degree_renorm(P_lminus1_m, P_lminus2_m, x, l, m):
    """ 
    Define renormalized associated Legendre polynmials by (2*l-1)!! * Pr_(m,l) = P_(m,l)
    This avoids overflows because of the double factorial

    Pr_(m,l)(x) = (x * Pr_(m,l-1)(x) - (l-1+m)/((2*l-1)*(2*l-3)) * Pr_(m,l-2)(x))/(l-m)      formula (C)
    
    graphically the effect of formula C is depicted by
        P_(m,l-1) --(C)--> P_(m,l)
    C increases the degree but leaves the order.
    """
    return( (x*P_lminus1_m - (l-1+m)/((2*l-1.0)*(2*l-3.0))*P_lminus2_m)/float(l-m) )

def _initial_term_renorm(x, l):
    """
    P_(l,l)(x) = (-1)^l (1-x^2)^(l/2)          formula (R)

    Formula R creates a new starting point for an iteration over the degree
         --(R)--> P_(l,l)
    """
    pll = pow(-1,l)*pow( np.sqrt(1.0-pow(x,2)), l)
    return pll
    

def _assocLegendre_renorm_it(x):
    """create an iterator for the renormalized associate Legendre polynomials
    The iteration scheme can be visualized as a matrix where for each call to R a new row is
    inserted which is then filled from left to right by successive calls to C.
    --(R)--> P_(0,0) --(C)--> P_(0,1) --(C)--> P_(0,2) --(C)--> P_(0,3) --(C)--> ....
                     --(R)--> P_(1,1) --(C)--> P_(1,2) --(C)--> P_(1,3) --(C)--> ....
                                      --(R)--> P_(2,2) --(C)--> P_(2,3) --(C)--> ....
                                                          ....   .....        
    The iterator returns the polynomials in the following order:
      P_(0,0),
      P_(1,0),P_(1,1),
      P_(2,0),P_(2,1),P_(2,2),
      ...
      P_(l,0),P_(l,1),...,P_(l,m),...,P_(l,l)
      ...
    In order to obtain the unrenormalized polynomials the term P_(l,m) has to be 
    multiplied by (2*l-1)!!
    """
    if getattr(x, '__iter__', False): # got array
        xf = x.ravel()
        if max(xf) > 1.0 or min(xf) < -1.0:
            raise ValueError('require -1 <= x <= 1, but x=%f', xf)
    else:  # got number
        if x > 1.0 or x < -1.0:
            raise ValueError('require -1 <= x <= 1, but x=%f', x)
    p00 = np.ones(x.shape)
    # P_(0,0) is just 1.0
    yield p00
    lastPcol = [np.zeros(x.shape)]
    curPcol = [p00]
    l = 1
    while True:
        nextPcol = list(range(0, l))
        for m in range(0, l):
            nextPcol[m] = _increase_degree_renorm(curPcol[m], lastPcol[m], x, l, m)
            yield nextPcol[m]
        curPcol.append(np.zeros(x.shape))
        pll = _initial_term_renorm(x,l)
        yield pll
        nextPcol.append(pll)
        """add P_(l,l) to current column and prepare for the next iteration 
        which will generate P_(l+1,0), P_(l+1,1),...,P_(l+1,l+1)"""
        l += 1
        lastPcol = curPcol
        curPcol = nextPcol


# spherical harmonics

def spherical_harmonics_it(th, phi):
    """
    th is the polar coordinate of the points on the grid [0,pi]x[0,2pi]
    phi is the azimuthal coordinate of the points on the grid [0,pi]x[0,2pi]
    
    Returns an iterator to the values of the spherical harmonics
    on the grid [0,pi]x[0,2pi] in the following order
    Y_(0,0),
    Y_(1,0), Y_(1,+1), Y_(1,-1),
    Y_(2,0), Y_(2,+1), Y_(2,-1), Y_(2,+2), Y_(2,-2),
    .....
    i.e. not in the order -m, -(m-1),...,m-1,m  but  0,1,-1,2,-2,...,m-1,-(m-1),m,-m

    The iterator returns a tuple (Ylm, l, m)
    """
    x = np.cos(th)
    ap_it = _assocLegendre_renorm_it(x)
    l = 0
    while True:
        for m in range(0, l+1):
            Plm = next(ap_it)
            # factorials = sqrt( fact(l-m)/float(fact(l+m)) ) * fact2(2*l-1)
            # use high precision mpf type to calculate ratios of factorials
            factorials = float( np.sqrt( mpf(fact(l-m))/mpf(fact(l+m)) ) * mpf(fact2(2*l-1)) )
            N = np.sqrt((2*l+1)/(4.0*np.pi)) * factorials
            Ylm = N * Plm * np.exp(1.0j*m*phi) 
            yield (Ylm, l, m)
            if (m > 0):
                Yl_minus_m = pow(-1,m)*Ylm.conjugate()
                yield (Yl_minus_m, l, -m)
        l += 1

