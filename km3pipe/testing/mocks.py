# coding=utf-8
# Filename: mocks.py
"""
Mocks, fakes and dummies of external libraries.

"""
from __future__ import division, absolute_import, print_function


__author__ = "Tamas Gal and Moritz Lotze"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal and Moritz Lotze"
__email__ = "tgal@km3net.de"
__status__ = "Development"


class FakeVec(object):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class FakeAanetHit(object):
    def __init__(self, channel_id, dir_x, dir_y, dir_z, dom_id, id, pmt_id,
                 pos_x, pos_y, pos_z, t0, t, tot, trig):
        # self.channel_id = chr(channel_id)
        self.channel_id = channel_id
        self.dir = FakeVec(dir_x, dir_y, dir_z)
        self.dom_id = dom_id
        self.id = id
        self.pmt_id = pmt_id
        self.pos = FakeVec(pos_x, pos_y, pos_z)
        self.t0 = t0
        self.t = t
        self.tot = tot
        self.trig = trig
