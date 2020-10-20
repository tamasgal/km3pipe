#!/usr/bin/env python
# coding: utf-8 -*-
"""

==============
Sphere Cuts
==============

This example shows some spher(ical) cuts on a set of points in 3D
using ``kp.math.spherecut`` and ``kp.math.spherecutmask``.

"""
import km3pipe as kp
import matplotlib.pyplot as plt
import numpy as np


################
# Generating a few points randomly distributed in a cube
points = np.random.rand(2000, 3)

fig = plt.figure()
ax = fig.gca(projection="3d")

################
# This will select all points which are inside a sphere with radius 0.5
# and centered at (0.5, 0.5, 0.5):

selected_points = kp.math.spherecut([0.5, 0.5, 0.5], 0.0, 0.5, points)

ax.scatter(*selected_points.T)

################
# Sometimes it's better to use (boolean) masks which can later be used
# to combine multiple selections. In this example we create two
# masks and use them to not only highlight the selected points but
# also to hide those from the original dataset.
#
points = np.random.rand(1000, 3)

mask1 = kp.math.spherecutmask([0, 0, 0], 0.8, 1.0, points)
mask2 = kp.math.spherecutmask([0.8, 0.8, 0.8], 0, 0.2, points)

fig = plt.figure()
ax = fig.gca(projection="3d")
ax.scatter(*points[~(mask1 | mask2)].T, label="unselected points")
ax.scatter(*points[mask1].T, label="mask1")
ax.scatter(*points[mask2].T, label="mask2")

ax.legend()

fig.tight_layout()
plt.show()
