#!/usr/bin/env python
import km3pipe as kp

log = kp.logger.get_logger('km3pipe.io.aanet')
log.setLevel('INFO')

fname = 'Corsika-74005_EPOS_NO_Charmed_VOLUMEDET_p_107.propa.km3v5r4.JTERun5009Eff05.JGandalf.aanet.root'

p = kp.Pipeline()
p.attach(kp.io.AanetPump, filename=fname)
p.attach(kp.io.HDF5Sink, filename=fname + '.h5')
p.drain()
