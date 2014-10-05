from __future__ import division, absolute_import, print_function

__author__ = 'tamasgal'

from km3pipe import Pipeline
from km3pipe.pumps import DAQPump

pipeline = Pipeline(cycles=5)
pipeline.attach(DAQPump, 'daq_pump',
                filename='/Users/tamasgal/Desktop/RUN-PPM_DU-00430-20140730-121124_detx.dat')
pipeline.drain()


