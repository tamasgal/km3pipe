"""
===========
Convex Hull
===========

Convex hull of a random set of points.

Lifted from ``scipy.spatial.qhull.pyx``.
"""

import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

import km3pipe.style
km3pipe.style.use("km3pipe")

points = np.random.rand(30, 2)   # 30 random points in 2-D
hull = ConvexHull(points)

##############################################################################
# Plot it:

plt.plot(points[:,0], points[:,1], 'o')
for simplex in hull.simplices:
    plt.plot(points[simplex, 0], points[simplex, 1], 'k-')

##############################################################################
# We could also have directly used the vertices of the hull, which
# for 2-D are guaranteed to be in counterclockwise order:

plt.plot(points[hull.vertices,0], points[hull.vertices,1], 'r--', lw=2)
plt.plot(points[hull.vertices[0],0], points[hull.vertices[0],1], 'ro')
plt.show()
