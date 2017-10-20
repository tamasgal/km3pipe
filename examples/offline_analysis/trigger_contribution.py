#!/usr/bin/env python
# coding=utf-8
# vim: ts=4 sw=4 et
"""
=====================
Trigger Contributions
=====================

Shows the median trigger contribution for each DOM.
This script can be used to easily identify DOMs in a run, which are out
of sync.

"""
# Author: Tamas Gal <tgal@km3net.de>
# License: MIT
#!/usr/bin/env python
from __future__ import division
from collections import defaultdict
import sys
import numpy as np
import km3pipe as kp
from km3modules.common import StatusBar

if len(sys.argv) == 2:
    filename = sys.argv[1]
else:
    raise SystemExit("Usage: trigger_contribution.py FILENAME")

detector = kp.hardware.Detector(det_id=29)


class TriggerContributionCalculator(kp.Module):
    """Shows the median trigger contribution for each DOM"""
    def configure(self):
        self.trigger_contributions = defaultdict(list)

    def process(self, blob):
        hits = blob['Hits'].triggered_hits
        n_hits = len(hits)
        dom_ids = np.unique(hits.dom_id)
        for dom_id in dom_ids:
            trigger_contribution = np.sum(hits.dom_id == dom_id) / n_hits
            self.trigger_contributions[dom_id].append(trigger_contribution)
        return blob

    def finish(self):
        print("{}\n{:>12}  {:>4} {:>4}  {:>12}\n{}"
              .format("="*42, "DOM ID", "du", "floor", "trig. contr.", "-"*42))
        for dom_id, trigger_contribution in self.trigger_contributions.items():
            du, floor, _ = detector.doms[dom_id]
            mean_tc = np.median(trigger_contribution)
            print("{:>12}  {:>4} {:>4}  {:>12.2f}%"
                  .format(dom_id, du, floor, mean_tc*100))


pipe = kp.Pipeline()
pipe.attach(kp.io.jpp.JPPPump, filename=filename)
pipe.attach(StatusBar, every=5000)
pipe.attach(TriggerContributionCalculator)
pipe.drain()
