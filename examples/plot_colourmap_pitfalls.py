# -*- coding: utf-8 -*-
"""

==================
Colormap pitfalls.
==================

Choosing the right colormap is important. A bad colormap like ``jet``
(standard in ROOT and matplotlib < 2.0) fools you into seeing structure
where there isn't any.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import km3pipe.style

km3pipe.style.use("km3pipe")

x = np.linspace(-np.pi, np.pi, 300)
xx, yy = np.meshgrid(x, x)
z = np.cos(xx) + np.cos(yy)

fig, axes = plt.subplots(2, 2)

for ax, colourmap in zip(axes.flatten(), ["gray", "jet", "cubehelix", "cividis"]):
    ax.set_title(colourmap)
    im = ax.imshow(z, cmap=plt.get_cmap(colourmap))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="7%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
