from __future__ import division, absolute_import, print_function

__author__ = 'tamasgal'


class Detector(object):
    def __init__(self, filename=None):
        self.det_file = None
        if filename:
            self.open_file(filename)

    def open_file(self, filename):
        pass

    @property
    def id(self):
        for line in self.det_file.readlines():
            return int(line.split()[0])
