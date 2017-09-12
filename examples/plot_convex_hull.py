"""
===========
Convex Hull
===========

Convex hull of a set of points, representing Dom x-y positions.

Derived from ``scipy.spatial.qhull.pyx``.
"""

import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

from km3pipe.core import Detector
from km3pipe.math import Polygon
import km3pipe.style
km3pipe.style.use("km3pipe")

filename = "data/orca_115strings_av23min20mhorizontal_18OMs_alt9mvertical_v1.detx"
detector = Detector(filename)
dom_pos = detector.dom_positions
xy = np.array([(pos[0], pos[1]) for _, pos in dom_pos.items()])
hull = ConvexHull(xy)

##############################################################################
# Plot it:

plt.plot(xy[:, 0], xy[:, 1], 'o')
for simplex in hull.simplices:
    plt.plot(xy[simplex, 0], xy[simplex, 1], 'k-')

##############################################################################
# We could also have directly used the vertices of the hull, which
# for 2-D are guaranteed to be in counterclockwise order:

plt.plot(xy[hull.vertices, 0], xy[hull.vertices, 1], 'r--', lw=2)
plt.plot(xy[hull.vertices[0], 0], xy[hull.vertices[0], 1], 'ro')
plt.show()
plt.savefig('foo.pdf')

##############################################################################
# Now let's draw a polygon inside, and color the points which are contained.

poly_vertices = np.array([
    (-60, 120),
    (80, 120),
    (110, 60),
    (110, -30),
    (70, -110),
    (-70, -110),
    (-90, -70),
    (-90, 60),
])
poly = Polygon(poly_vertices)
contain_mask = poly.contains(xy)
plt.clf()
plt.plot(xy[contain_mask, 0], xy[contain_mask, 1], 'yo')
plt.plot(xy[~contain_mask, 0], xy[~contain_mask, 1], 'bo')
plt.plot(poly_vertices[:, 0], poly_vertices[:, 1], 'k-')
plt.show()
plt.savefig('bar.pdf')
