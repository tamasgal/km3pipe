#!/usr/bin/env python
import km3pipe as kp

log = kp.logger.get_logger('km3pipe.io.aanet')
log.setLevel('INFO')

fname = 'Corsika-74005_EPOS_NO_Charmed_VOLUMEDET_p_107.propa.km3v5r4.JTERun5009Eff05.JGandalf.aanet.root'

p = kp.Pipeline()
p.attach(kp.io.AanetPump, filename=fname, ignore_hits=True)
p.attach(kp.io.HDF5Sink, filename=fname + '.h5')
p.drain()

p = kp.Pipeline()
p.attach(kp.io.AanetPump, filename=fname, ignore_hits=False)
p.attach(kp.io.HDF5Sink, filename=fname + '.withhits.h5')
p.drain()
