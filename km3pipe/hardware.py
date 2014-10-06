from __future__ import division, absolute_import, print_function

__author__ = 'tamasgal'


class Detector(object):
    def __init__(self, filename=None):
        self.det_file = None
        self.det_id = None
        self.n_doms = None

        if filename:
            self.open_file(filename)

    def open_file(self, filename):
        pass

    def parse_header(self):
        """Extract information from the header of the detector file"""
        self.det_file.seek(0, 0)
        first_line = self.det_file.readline()
        self.det_id, self.n_doms = [int(i) for i in first_line.split()]

