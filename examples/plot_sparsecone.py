# -*- coding: utf-8 -*-
"""

====
Cone
====

Sparse Cone
"""
from __future__ import absolute_import, print_function, division

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D    # noqa

from km3pipe.math import SparseCone
import km3pipe.style
km3pipe.style.use("moritz")

spike = [0, 0, 1]
bottom = [0, 0, 0]
angle = np.pi / 4
n_angles = 20
cone = SparseCone(spike, bottom, angle)
circ_samp = cone.sample_circle(n_angles=n_angles)
axis_samp = cone.sample_axis
samp = cone.sample(n_angles)

samp = np.array(samp)

##############################################################################
# plot the same in 3D because why not?

plt.clf()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(samp[:, 0], samp[:, 1], samp[:, 2], 'yo')
plt.show()
