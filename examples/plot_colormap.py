# Code source: Óscar Nájera
# License: BSD 3 clause
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

x = np.linspace(-np.pi, np.pi, 300)
xx, yy = np.meshgrid(x, x)
z = np.cos(xx) + np.cos(yy)

plt.figure()
plt.imshow(z)
plt.colorbar()
plt.xlabel('$x$')
plt.ylabel('$y$')
