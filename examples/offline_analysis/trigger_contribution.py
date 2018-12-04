#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: ts=4 sw=4 et
"""
=====================
Trigger Contributions
=====================

Shows the median trigger contribution for each DOM.
This script can be used to easily identify DOMs in a run, which are out
of sync.

"""
from __future__ import absolute_import, print_function, division

from collections import defaultdict
import sys
import numpy as np
import km3pipe as kp
from km3modules.common import StatusBar

if len(sys.argv) == 2:
    filename = sys.argv[1]
else:
    raise SystemExit("Usage: trigger_contribution.py FILENAME")

det = kp.hardware.Detector(det_id=29)
log = kp.logger.get_logger('TriggerContribution')


class TriggerContributionCalculator(kp.Module):
    """Shows the mean trigger contribution for each DOM"""

    def configure(self):
        self.dus = self.get("dus")    # only select DOMs on these DUs
        self.trigger_contributions = defaultdict(list)
        self.n_events = 0

    def process(self, blob):
        hits = blob['Hits'].triggered_rows
        n_hits = len(hits)
        dom_ids = np.unique(hits.dom_id)
        for dom_id in dom_ids:
            trigger_contribution = np.sum(hits.dom_id == dom_id) / n_hits
            self.trigger_contributions[dom_id].append(trigger_contribution)
        self.n_events += 1
        return blob

    def finish(self):
        print(
            "{}\n{:>12}  {:>4} {:>4}  {:>12}\n{}".format(
                "=" * 42, "DOM ID", "du", "floor", "trig. contr.", "-" * 42
            )
        )
        summary = []
        for dom_id, trigger_contribution in self.trigger_contributions.items():
            du, floor = omkey(dom_id)
            mean_tc = np.sum(trigger_contribution) / self.n_events
            summary.append(((du, floor), dom_id, mean_tc))
        for (du, floor), dom_id, mean_tc in sorted(summary):
            print(
                "{:>12}  {:>4} {:>4}  {:>12.2f}%".format(
                    dom_id, du, floor, mean_tc * 100
                )
            )

        dom_ids = set(det.doms.keys())
        if self.dus is not None:
            log.warning(
                "Showing only DOMs which are on the following DUs: {}".format(
                    ', '.join(str(du) for du in self.dus)
                )
            )
            dom_ids = set(d for d in dom_ids if det.doms[d][0] in self.dus)

        inactive_doms = []
        for dom_id in set(dom_ids) - set(self.trigger_contributions.keys()):
            inactive_doms.append(dom_id)
        if inactive_doms:
            print("The following DOMs were inactive:")
            for dom_id in inactive_doms:
                print("{}_(DU{}-{})".format(dom_id, *omkey(dom_id)), end=' ')


def omkey(dom_id):
    """Returns (du, floor) for given DOM ID"""
    return det.doms[dom_id][0:2]


pipe = kp.Pipeline()
pipe.attach(kp.io.jpp.EventPump, filename=filename)
pipe.attach(StatusBar, every=5000)
pipe.attach(TriggerContributionCalculator, dus=[2])
pipe.drain()
