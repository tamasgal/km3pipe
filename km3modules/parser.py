# coding=utf-8
# Filename: parser.py
# pylint: disable=locally-disabled
"""
A collection of parsers.

"""
from __future__ import division, absolute_import, print_function

from km3pipe import Module
from km3pipe.io.daq import DAQPreamble, DAQSummaryslice, DAQEvent
from km3pipe.common import StringIO


class CHParser(Module):
    """A parser for ControlHost data.

    This parser will choose the correct class to parse the binary data
    for given `tags`.

    """
    parse_map = {
        'IO_SUM': ['DAQSummaryslice', DAQSummaryslice],
        'IO_EVT': ['DAQEvent', DAQEvent],
        }

    def __init__(self, **context):
        super(self.__class__, self).__init__(**context)
        self.tags = self.get('tags') or []

    def process(self, blob):
        if not ('CHData' in blob and 'CHPrefix' in blob):
            return blob

        tag = str(blob['CHPrefix'].tag)

        if tag not in self.tags and tag not in self.parse_map:
            return blob

        data = blob['CHData']
        data_io = StringIO(data)
        preamble = DAQPreamble(file_obj=data_io)  # noqa

        blob_key, ParserClass = self.parse_map[tag]
        blob[blob_key] = ParserClass(file_obj=data_io)

        return blob
