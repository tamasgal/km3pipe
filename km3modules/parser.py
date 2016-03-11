# coding=utf-8
# Filename: parser.py
# pylint: disable=locally-disabled
"""
A collection of parsers.

"""
from __future__ import division, absolute_import, print_function

from km3pipe import Module
from km3pipe.pumps.daq import DAQPreamble, DAQSummaryslice

try:
    from cStringIO import StringIO
except ImportError:
    try:
        from StringIO import StringIO
    except ImportError:
        from io import StringIO


class CHParser(Module):
    def __init__(self, **context):
        super(self.__class__, self).__init__(**context)
        self.tags = self.get('tags') or []

    def process(self, blob):
        if not ('CHData' in blob and 'CHPrefix' in blob):
            return blob

        tag = str(blob['CHPrefix'].tag)

        if tag not in self.tags:
            return blob

        if tag == 'IO_SUM':
            data = blob['CHData']
            data_io = StringIO(data)
            preamble = DAQPreamble(file_obj=data_io)  # noqa
            summaryslice = DAQSummaryslice(file_obj=data_io)
            blob['DAQSummaryslice'] = summaryslice

        return blob
