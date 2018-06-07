#!/usr/bin/env python
import km3pipe as kp

log = kp.logger.get_logger('km3pipe.io.aanet')
log.setLevel('DEBUG')

fname = 'small.root'

p = kp.Pipeline()
p.attach(kp.io.AanetPump, filename=fname)
p.attach(kp.io.HDF5Sink, filename=fname + '.h5')
p.drain(1)
