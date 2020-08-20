#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: ts=4 sw=4 et
# Author: Tamas Gal <tgal@km3net.de>
# License: MIT
"""
==========
MSG reader
==========

This application demonstrates how to access MSGs from Ligier.

"""

import io
import km3pipe as kp

__author__ = "Tamas Gal"
__email__ = "tgal@km3net.de"


def print_msg(blob):
    data = io.BytesIO(blob["CHData"])
    print(data)
    return blob


pipe = kp.Pipeline()
pipe.attach(kp.io.CHPump, host="localhost", tags="MSG")
pipe.attach(print_msg)
pipe.drain()
