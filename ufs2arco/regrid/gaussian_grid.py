"""Tools for working with Gaussian grids."""
from __future__ import absolute_import, division, print_function

import numpy as np
import numpy.linalg as la
from numpy.polynomial.legendre import legcompanion, legder, legval

def gaussian_latitudes(n):
    """Construct latitudes and latitude bounds for a Gaussian grid.

    Args:
        n (int): The Gaussian grid number (half the number of latitudes in the grid.
    Returns:
        latitudes (numpy.array): length ``n`` array of latitudes in degrees
        bounds2d (numpy.array): ``(n, 2)`` array of grid bounds
    """
    if abs(int(n)) != n:
        raise ValueError("n must be a non-negative integer")
    nlat = 2 * n
    # Create the coefficients of the Legendre polynomial and construct the
    # companion matrix:
    cs = np.array([0] * nlat + [1], dtype=int)
    cm = legcompanion(cs)
    # Compute the eigenvalues of the companion matrix (the roots of the
    # Legendre polynomial) taking advantage of the fact that the matrix is
    # symmetric:
    roots = la.eigvalsh(cm)
    roots.sort()
    # Improve the roots by one application of Newton's method, using the
    # solved root as the initial guess:
    fx = legval(roots, cs)
    fpx = legval(roots, legder(cs))
    roots -= fx / fpx
    # The roots should exhibit symmetry, but with a sign change, so make sure
    # this is the case:
    roots = (roots - roots[::-1]) / 2.0
    # Compute the Gaussian weights for each interval:
    fm = legval(roots, cs[1:])
    fm /= np.abs(fm).max()
    fpx /= np.abs(fpx).max()
    weights = 1.0 / (fm * fpx)
    # Weights should be symmetric and sum to two (unit weighting over the
    # interval [-1, 1]):
    weights = (weights + weights[::-1]) / 2.0
    weights *= 2.0 / weights.sum()
    # Calculate the bounds from the weights, still on the interval [-1, 1]:
    bounds1d = np.empty([nlat + 1])
    bounds1d[0] = -1
    bounds1d[1:-1] = -1 + weights[:-1].cumsum()
    bounds1d[-1] = 1
    # Convert the bounds to degrees of latitude on [-90, 90]:
    bounds1d = np.rad2deg(np.arcsin(bounds1d))
    bounds2d = np.empty([nlat, 2])
    bounds2d[:, 0] = bounds1d[:-1]
    bounds2d[:, 1] = bounds1d[1:]
    # Convert the roots from the interval [-1, 1] to latitude values on the
    # interval [-90, 90] degrees:
    latitudes = np.rad2deg(np.arcsin(roots))
    return latitudes, bounds2d
