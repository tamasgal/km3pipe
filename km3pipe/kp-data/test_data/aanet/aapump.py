#!/usr/bin/env python
import sys
import km3pipe as kp

log = kp.logger.get_logger('km3pipe.io.aanet')
log.setLevel('INFO')

fname = sys.argv[-1]

p = kp.Pipeline()
p.attach(kp.io.AanetPump, filename=fname, ignore_hits=True)
p.attach(kp.io.HDF5Sink, filename=fname + '.h5')
p.drain()
