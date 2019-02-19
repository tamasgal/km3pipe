# Filename: dataclasses.py
# pylint: disable=W0232,C0103,C0111
# vim:set ts=4 sts=4 sw=4 et syntax=python:
"""
Dataclasses for internal use. Heavily based on Numpy arrays.
"""
from __future__ import absolute_import, print_function, division

import numpy as np

__author__ = "Tamas Gal and Moritz Lotze"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Moritz Lotze"
__email__ = "mlotze@km3net.de"
__status__ = "Development"
__all__ = ('TEMPLATES', )

TEMPLATES = {
    'Direction': {
        'dtype': np.dtype([
            ('dir_x', '<f4'),
            ('dir_y', '<f4'),
            ('dir_z', '<f4'),
        ]),
        'h5loc': None,
        'split_h5': False,
        'h5singleton': False,
    },
    'Position': {
        'dtype': np.dtype([
            ('pos_x', '<f4'),
            ('pos_y', '<f4'),
            ('pos_z', '<f4'),
        ]),
        'h5loc': None,
        'split_h5': False,
        'h5singleton': False,
    },
    'EventInfo': {
        'dtype': np.dtype([
            ('det_id', '<i4'),
            ('frame_index', '<u4'),
            ('livetime_sec', '<u8'),
            ('mc_id', '<i4'),
            ('mc_t', '<f8'),
            ('n_events_gen', '<u8'),
            ('n_files_gen', '<u8'),
            ('overlays', '<u4'),
            ('trigger_counter', '<u8'),
            ('trigger_mask', '<u8'),
            ('utc_nanoseconds', '<u8'),
            ('utc_seconds', '<u8'),
            ('weight_w1', '<f8'),
            ('weight_w2', '<f8'),
            ('weight_w3', '<f8'),
            ('run_id', '<u8'),
            ('group_id', '<u8'),
        ]),
        'h5loc': '/event_info',
        'split_h5': False,
        'h5singleton': False,
    },
    'TimesliceHits': {
        'dtype': np.dtype([
            ('channel_id', 'u1'),
            ('dom_id', '<u4'),
            ('time', '<f8'),
            ('tot', 'u1'),
            ('group_id', '<u4'),
        ]),
        'h5loc': '/timeslice_hits',
        'split_h5': True,
        'h5singleton': False,
    },
    'Hits': {
        'dtype': np.dtype([
            ('channel_id', 'u1'),
            ('dom_id', '<u4'),
            ('time', '<f8'),
            ('tot', 'u1'),
            ('triggered', '?'),
            ('group_id', '<u4'),
        ]),
        'h5loc': '/hits',
        'split_h5': True,
        'h5singleton': False,
    },
    'CalibHits': {
        'dtype': np.dtype([
            ('channel_id', 'u1'),
            ('dir_x', '<f4'),
            ('dir_y', '<f4'),
            ('dir_z', '<f4'),
            ('dom_id', '<u4'),
            ('du', 'u1'),
            ('floor', 'u1'),
            ('pos_x', '<f4'),
            ('pos_y', '<f4'),
            ('pos_z', '<f4'),
            ('t0', '<f4'),
            ('time', '<f8'),
            ('tot', 'u1'),
            ('triggered', '?'),
            ('group_id', '<u4'),
        ]),
        'h5loc': '/hits',
        'split_h5': True,
        'h5singleton': False,
    },
    'McHits': {
        'dtype': np.dtype([
            ('a', '<f4'),
            ('origin', '<u4'),
            ('pmt_id', '<u4'),
            ('time', '<f8'),
            ('group_id', '<u4'),
        ]),
        'h5loc': '/mc_hits',
        'split_h5': True,
        'h5singleton': False,
    },
    'CalibMcHits': {
        'dtype': np.dtype([
            ('a', '<f4'),
            ('dir_x', '<f4'),
            ('dir_y', '<f4'),
            ('dir_z', '<f4'),
            ('origin', '<u4'),
            ('pmt_id', '<u4'),
            ('pos_x', '<f4'),
            ('pos_y', '<f4'),
            ('pos_z', '<f4'),
            ('time', '<f8'),
            ('group_id', '<u4'),
        ]),
        'h5loc': '/mc_hits',
        'split_h5': True,
        'h5singleton': False,
    },
    'Tracks': {
        'dtype': np.dtype([
            ('bjorkeny', '<f8'),
            ('dir_x', '<f8'),
            ('dir_y', '<f8'),
            ('dir_z', '<f8'),
            ('energy', '<f8'),
            ('id', '<u4'),
            ('interaction_channel', '<u4'),
            ('is_cc', '<u4'),    # TODO: consider bool ('?') for slicing
            ('length', '<f8'),
            ('pos_x', '<f8'),
            ('pos_y', '<f8'),
            ('pos_z', '<f8'),
            ('time', '<i4'),
            ('type', '<i4'),
            ('group_id', '<u4'),
        ]),
        'h5loc': '/tracks',
        'split_h5': False,
        'h5singleton': False,
    },
    'McTracks': {
        'dtype': np.dtype([
            ('bjorkeny', '<f8'),
            ('dir_x', '<f8'),
            ('dir_y', '<f8'),
            ('dir_z', '<f8'),
            ('energy', '<f8'),
            ('id', '<u4'),
            ('interaction_channel', '<u4'),
            ('is_cc', '<u4'),    # TODO: consider bool ('?') for slicing
            ('length', '<f8'),
            ('pos_x', '<f8'),
            ('pos_y', '<f8'),
            ('pos_z', '<f8'),
            ('time', '<i4'),
            ('type', '<i4'),
            ('group_id', '<u4'),
        ]),
        'h5loc': '/mc_tracks',
        'split_h5': False,
        'h5singleton': False,
    },
    'SummaryFrameInfo': {
        'dtype': np.dtype([
            ('dom_id', '<u4'),
            ('fifo_status', '<u4'),
            ('frame_id', '<u4'),
            ('frame_index', '<u4'),
            ('has_udp_trailer', '<u4'),
            ('high_rate_veto', '<u4'),
            ('max_sequence_number', '<u4'),
            ('n_packets', '<u4'),
            ('slice_id', '<u4'),
            ('utc_nanoseconds', '<u4'),
            ('utc_seconds', '<u4'),
            ('white_rabbit_status', '<u4'),
        ]),
        'h5loc': '/summary_slice_info',
        'split_h5': False,
        'h5singleton': False,
    },
    'SummarysliceInfo': {
        'dtype': np.dtype([
            ('frame_index', '<u4'),
            ('slice_id', '<u4'),
            ('timestamp', '<u4'),
            ('nanoseconds', '<u4'),
            ('n_frames', '<u4'),
        ]),
        'h5loc': '/todo',
        'split_h5': False,
        'h5singleton': False,
    },
    'TimesliceInfo': {
        'dtype': np.dtype([
            ('frame_index', '<u4'),
            ('slice_id', '<u4'),
            ('timestamp', '<u4'),
            ('nanoseconds', '<u4'),
            ('n_frames', '<u4'),
        ]),
        'h5loc': '/timeslice_info',
        'split_h5': False,
        'h5singleton': False,
    },
    'SummaryframeSeries': {
        'dtype': np.dtype([
            ('dom_id', '<u4'),
            ('max_sequence_number', '<u4'),
            ('n_received_packets', '<u4'),
            ('group_id', '<u4'),
        ]),
        'h5loc': '/todo',
        'split_h5': False,
        'h5singleton': False,
    },
    'TimesliceFrameInfo': {
        'dtype': np.dtype([
            ('det_id', '<i4'),
            ('run_id', '<u8'),
            ('sqnr', '<u8'),
            ('timestamp', '<u4'),
            ('nanoseconds', '<u4'),
            ('dom_id', '<u4'),
            ('dom_status', '<u4'),
            ('n_hits', '<u4'),
        ]),
        'h5loc': '/todo',
        'split_h5': False,
        'h5singleton': False,
    },
    'SummaryFrameSeries': {
        'dtype': np.dtype([
            ('dom_id', '<u4'),
            ('max_sequence_number', '<u4'),
            ('n_received_packets', '<u4'),
            ('group_id', '<u4'),
        ]),
        'h5loc': '/summary_frame_series',
        'split_h5': False,
        'h5singleton': False,
    }
}
