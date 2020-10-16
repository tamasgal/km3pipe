# -*- coding: utf-8 -*-
"""

==================
PMT Time Slewing
==================

Show different variants of PMT time slewing calculations.

Variant 3 is currently (as of 2020-10-16) what's also used in Jpp.

"""

# Author: Tamas Gal <tgal@km3net.de>
# License: BSD-3

import km3pipe as kp
import numpy as np
import matplotlib.pyplot as plt
kp.style.use()

tots = np.arange(256)
slews = {variant: kp.calib.slew(tots, variant=variant) for variant in (1, 2, 3)}

fig, ax = plt.subplots()

for variant, slew in slews.items():
    ax.plot(tots, slew, label=f"Variant {variant}")

ax.set_xlabel("ToT / ns")
ax.set_ylabel("time slewing / ns")
ax.legend()

fig.tight_layout()
plt.show()
