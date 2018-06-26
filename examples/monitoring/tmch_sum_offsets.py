#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: ts=4 sw=4 et
"""
=====================================================
Check consistency of monitoring channel and TDC times
=====================================================

This script picks a monitoring channel packet with no HRV flags for any PMT
and tries to match one of the following 50 summaryslices using the PMT rates.
The index of the summaryslice, the rates-diff and the time offsets are logged
in a CSV file.

.. code-block:: bash

    Usage:
        tmch_sum_offsets.py DOM_ID [-n N_TIMESLICES]
        tmch_sum_offsets.py (-h | --help)

    Options:
        DOM_ID           The DOM ID.
        -n N_TIMESLICES  The number of timeslices to investigate [default: 50].
        -h --help        Show this screen.

"""
from __future__ import absolute_import, print_function, division

import io
import os
import km3pipe as kp
from km3pipe.io.daq import TMCHData, DAQPreamble, DAQSummaryslice
import numpy as np

__author__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
VERSION = "1.0"

log = kp.logger.get_logger("tmch_sum_offsets")


class MonitoringChannelPicker(kp.Module):
    """Picks a monitoring channel packet with no HRV for a given DOM"""

    def configure(self):
        self.dom_id = self.require("dom_id")

    def process(self, blob):
        if not self.services['SummarysliceFound']():
            return blob
        tag = str(blob['CHPrefix'].tag)
        if tag == 'IO_MONIT':
            tmch_data = TMCHData(io.BytesIO(blob['CHData']))
            dom_id = tmch_data.dom_id
            if dom_id != self.dom_id or tmch_data.hrvbmp > 0:
                log.info("Skipping TMCH packet due to HRV flags")
                return blob
            blob['Candidate'] = tmch_data
            print("Next IO_MONIT candidate picked.")
        return blob


class SummarysliceMatcher(kp.Module):
    def configure(self):
        self.dom_id = self.require("dom_id")
        self.n_timeslices = self.get("n_timeslices", default=50)
        self.expose(self.io_sum_found, 'SummarysliceFound')
        self._candidate = None
        self._rates_io_monit = None
        self._reset()

        filename = "tmch_sum_offsets_{}.csv".format(self.dom_id)
        self.fobj = self._get_file_handler(filename)

    def _reset(self):
        self._io_sum_found = True
        self._diff = []
        self._summaries = []
        self.i = 0

    def _get_file_handler(self, filename):
        if not os.path.exists(filename):
            fobj = open(filename, 'w')
            fobj.write(
                "run dom_id nearest_idx diff "
                "tmch_timestamp tmch_ns "
                "summary_timestamp summary_ns\n"
            )
        else:
            fobj = open(filename, 'a')
        return fobj

    def process(self, blob):
        if 'Candidate' in blob:
            print("Searching for new IO_SUM candidate")
            self._io_sum_found = False
            self._candidate = blob['Candidate']
            self._rates_io_monit = np.array(self._candidate.pmt_rates)

        if self._candidate is None:
            return blob

        tag = str(blob['CHPrefix'].tag)
        if tag == 'IO_SUM':
            data = io.BytesIO(blob['CHData'])
            preamble = DAQPreamble(file_obj=data)    # noqa
            summary = DAQSummaryslice(file_obj=data)
            try:
                rates = np.array(summary.summary_frames[self.dom_id])
            except KeyError:
                print("No DOM data in summaryslice, skipping...")
                return blob
            self.i += 1
            diff = np.sum(np.abs(self._rates_io_monit) - np.abs(rates))
            self._diff.append(diff)
            self._summaries.append(summary)

            if self.i > self.n_timeslices:
                print("Trying to match a summaryslice.")
                idx_nearest = (np.abs(self._diff)).argmin()
                summary = self._summaries[idx_nearest]
                print(
                    "min", min(self._diff), "max", max(self._diff), "nearest",
                    self._diff[idx_nearest]
                )
                print(self._rates_io_monit)
                print(summary.summary_frames[self.dom_id])
                print(
                    "Time of IO_MONIT:", self._candidate.utc_seconds,
                    self._candidate.nanoseconds
                )
                print(
                    "Time of IO_SUM:", summary.header.time_stamp,
                    summary.header.ticks * 16
                )
                self.fobj.write(
                    "{} {} {} {} {} {} {} {}\n".format(
                        self._candidate.run, self.dom_id, idx_nearest,
                        self._diff[idx_nearest], self._candidate.utc_seconds,
                        self._candidate.nanoseconds, summary.header.time_stamp,
                        summary.header.ticks * 16
                    )
                )
                self._reset()
                self.fobj.flush()
        return blob

    def io_sum_found(self):
        """Service which tells if we are ready for the next TMCH packet"""
        return self._io_sum_found

    def finish(self):
        self.fobj.close()


def main():
    from docopt import docopt
    args = docopt(__doc__)

    dom_id = int(args['DOM_ID'])

    pipe = kp.Pipeline(timeit=True)
    pipe.attach(
        kp.io.CHPump,
        host='127.0.0.1',
        port=5553,
        tags='IO_SUM, IO_MONIT',
        timeout=60 * 60 * 24 * 7,
        max_queue=1000
    )
    pipe.attach(MonitoringChannelPicker, dom_id=dom_id)
    pipe.attach(
        SummarysliceMatcher, dom_id=dom_id, n_timeslices=int(args['-n'])
    )
    pipe.drain()


if __name__ == "__main__":
    main()
