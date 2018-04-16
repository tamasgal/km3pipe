# Filename: topology.py
# pylint: disable=locally-disabled
"""
Topology related modules.

"""

import numpy as np

from km3pipe import Module


class TriggeredDUs(Module):
    """Check for triggered DUs."""

    def process(self, blob):
        triggered_hits = blob['Hits'].triggered_hits.serialise(to='pandas')
        self.calibration.apply(triggered_hits)
        dus = np.unique(triggered_hits['du'])
        n_dus = len(dus)

        blob['TriggeredDUs'] = dus

        if n_dus > 1:
            blob['multiple_du_event'] = True
        else:
            blob['single_du_event'] = True

        return blob
