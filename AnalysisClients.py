# Author: Tamas Gal <tgal@km3net.de>                                                                  
# License: MIT 
# !/usr/bin/env python                                                                                  
# coding=utf-8                                                                                          
# vim: ts=4 sw=4 et

from __future__ import print_function
from km3pipe import Pipeline, Module
from km3pipe.core import Pump
from km3pipe.io.daq import TMCHRepump
from collections import defaultdict

import struct
from struct import unpack
import io
import km3pipe as kp

__author__ = "Tamas Gal  &  Alba Domi"
__email__ = "tgal@km3net.de  -  adomi@ge.infn.it"

N_DOMS = 18
N_DUS = 2

detector = kp.hardware.Detector(det_id=14)

class TMCHDatas(object):
    """Monitoring Channel data."""
    def __init__(self, file_obj):
        f = file_obj

        data_type = f.read(4)
        if data_type != b'TMCH':
            raise ValueError("Invalid datatype: {0}".format(data_type))

        self.run = unpack('>I', f.read(4))[0]
        self.udp_sequence_number = unpack('>I', f.read(4))[0] 
        self.utc_seconds = unpack('<I', f.read(4))[0]
        self.nanoseconds = unpack('>I', f.read(4))[0] * 16
        self.dom_id = unpack('>I', f.read(4))[0]
        self.dom_status_0 = unpack('>I', f.read(4))[0]  
        self.dom_status_1 = unpack('>I', f.read(4))[0]  
        self.dom_status_2 = unpack('>I', f.read(4))[0]  
        self.dom_status_3 = unpack('>I', f.read(4))[0]  
        self.pmt_rates = [r*10.0 for r in unpack('>' + 31*'I', f.read(31*4))]
        self.hrvbmp = unpack('>I', f.read(4))[0]  
        self.flags = unpack('>I', f.read(4))[0]  
        self.yaw, self.pitch, self.roll = unpack('>fff', f.read(12))
        self.ax, self.ay, self.az = unpack('>fff', f.read(12))
        self.gx, self.gy, self.gz = unpack('>fff', f.read(12))
        self.hx, self.hy, self.hz = unpack('>fff', f.read(12))
        self.temp = unpack('>H', f.read(2))[0] / 100.0
        self.humidity = unpack('>H', f.read(2))[0] / 100.0
        self.tdcfull = unpack('>I', f.read(4))[0]
        self.aesfull = unpack('>I', f.read(4))[0]
        self.flushc = unpack('>I', f.read(4))[0]  
        self.ts_duration_microseconds = unpack('>I', f.read(4))[0]  

    def __str__(self):
        return str(vars(self))

    def __repr__(self):
        return self.__str__()


class TMCHRepumps(Pump):
    """Takes a IO_MONIT raw dump and replays it."""
    def configure(self):
        filename = self.require("filename")
        self.fobj = open(filename, "rb")
    
    def process(self, blob):
        try:
            while True:
                datatype = self.fobj.read(4)
                if datatype == b'TMCH':
                    self.fobj.seek(-4, 1)
                    blob['TMCHDatas'] = TMCHDatas(self.fobj)
                    return blob
        except struct.error:
            raise StopIteration

    def finish(self):
        self.fobj.close()

class Times(Module):

    n = 1

    def configure(self): #this method is called when this module is attached
        self.times_16nanosec = defaultdict(list) 
        self.times_sec = defaultdict(list)

    def Time_Check(self, DOM_ID, present_time_ns, previous_time_ns):
        dt = present_time_ns - previous_time_ns
        if dt != 1e8 and dt != 0 and dt != -9e8 :
            dt *= 1e-6
            line = 0
            floor = 0
#            line, floor, n_pmts = detector.doms[DOM_ID]
            print("**ERROR!** Monitoring Frame duration for DOM " + str(DOM_ID)  + " (" + str(floor) + " DU " + str(line) + ") = " + str(dt) + " ms")
        return dt

    def number_of_UDP_packets(self, DOM_ID, UTC_time_sec, previous_time_s):
        t = UTC_time_sec - previous_time_s
        if t == 0:
            Times.n += 1
#            print(Times.n)
        else:
            print("...ooOOoo...ooOOoo...ooOOoo...Number of UDP packets = " + str(Times.n))
            Times.n = 1
            
    def process(self, blob): #this method is called for each blob
        tmch_data = blob['TMCHDatas']
        DOM_ID = tmch_data.dom_id
 #       print(DOM_ID)
        if DOM_ID != 808969132 and DOM_ID != 808469291 and DOM_ID != 808982053: 
            line, floor, n_pmts = detector.doms[DOM_ID]
            if line == 1 and floor == 2:
                print('******** New UDP packet ********')
                print('Analysing DOM ' + str(floor) + ' DU ' + str(line))
                print("UTC SEC: " + str(tmch_data.utc_seconds))
                print("NANOSEC: " + str(tmch_data.nanoseconds))
                self.times_16nanosec[DOM_ID].append(tmch_data.nanoseconds)
                Times.Time_Check(self, DOM_ID = DOM_ID, present_time_ns = tmch_data.nanoseconds, previous_time_ns = self.times_16nanosec[DOM_ID][0])
                self.times_16nanosec[DOM_ID][0] = tmch_data.nanoseconds
                if len(self.times_16nanosec[DOM_ID]) > 1:
                    del self.times_16nanosec[DOM_ID][1]

#        print("LEN " + str(len(self.times_16nanosec[DOM_ID])))
                    self.times_sec[DOM_ID].append(tmch_data.utc_seconds)
  #      print("Previous time = " + str(self.times_sec[DOM_ID][0]) + " , present time = " + str(tmch_data.utc_seconds))
                    Times.number_of_UDP_packets(self, DOM_ID = DOM_ID, UTC_time_sec = tmch_data.utc_seconds, previous_time_s = self.times_sec[DOM_ID][0])
                    self.times_sec[DOM_ID][0] = tmch_data.utc_seconds
                    if len(self.times_sec[DOM_ID]) > 1:
                        del self.times_sec[DOM_ID][1]
        return blob  #remember to always return the blob!
        
    def finish(self): #this method is called when the pipeline finishes draining
        print("finish")


pipe = Pipeline()
pipe.attach(TMCHRepumps, filename="/Users/Alba/Desktop/ΦSICA/KM3NeT/Monitoring/IO_MONIT_1465120119.dat")
#pipe.attach(TMCHRepumps, filename="/Users/Alba/Desktop/ΦSICA/KM3NeT/Monitoring/dump-monitoring-clb2017.dqd")
pipe.attach(Times, dom_id=808982053) 
pipe.drain(100)

