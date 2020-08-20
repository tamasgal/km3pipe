#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: ts=4 sw=4 et
"""
==================
Monitoring Channel
==================

This application demonstrates how to access monitoring channel data.

"""

# Author: Tamas Gal <tgal@km3net.de>
# License: MIT
#!/usr/bin/env python
# vim: ts=4 sw=4 et
import io
import km3pipe as kp

__author__ = "Tamas Gal"
__email__ = "tgal@km3net.de"


def print_monitoring_data(blob):
    data = io.BytesIO(blob["CHData"])
    print(kp.io.daq.TMCHData(data))
    return blob


pipe = kp.Pipeline()
pipe.attach(kp.io.CHPump, host="localhost", tags="IO_MONIT")
pipe.attach(print_monitoring_data)
pipe.drain()
