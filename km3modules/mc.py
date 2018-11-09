#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:set ts=4 sts=4 sw=4 et:
"""MC Helpers.
"""
from __future__ import absolute_import, print_function, division

import numpy as np

from km3pipe import Module
from km3pipe.mc import pdg2name
from km3pipe.math import zenith, azimuth

__author__ = "Michael Moser and Tamas Gal and Moritz Lotze"

NEUTRINOS = {
    'nu_e',
    'anu_e',
    'nu_mu',
    'anu_mu',
    'nu_tau',
    'anu_tau',
}


class McTruth(Module):
    """Extract MC info of 1st MC track.

    Parameters
    ----------
    most_energetic_primary: bool, default=True
    """

    def configure(self):
        self.most_energetic_primary = bool(self.get('most_energetic_primary')
                                           ) or True

    @classmethod
    def t2f(cls, row):
        return pdg2name(row['type'])

    @classmethod
    def is_nu(cls, flavor):
        return flavor in NEUTRINOS

    def process(self, blob):
        mc = blob['McTracks'].conv_to('pandas')
        if self.most_energetic_primary:
            mc.sort_values('energy', ascending=False, inplace=True)
        mc = mc.head(1)
        mc['zenith'] = zenith(mc[['dir_x', 'dir_y', 'dir_z']])
        mc['azimuth'] = azimuth(mc[['dir_x', 'dir_y', 'dir_z']])
        flavor = mc.apply(self.t2f, axis=1)
        mc['is_neutrino'] = flavor.apply(self.is_nu)
        blob['McTruth'] = mc
        return blob


def convert_mc_times_to_jte_times(times_mc, evt_timestamp_in_ns, evt_mc_time):
    """
    Function that converts MC times to JTE times.

    Parameters
    ----------
    times_mc : np.ndarray
        Time array with MC times.
    evt_timestamp_in_ns : int
        Total timestamp of the event in nanoseconds.
    evt_mc_time : int
        Mc time of the event in nanoseconds.

    Returns
    -------
    ndarray
        Converted time array with JTE times.
    """
    # needs to be cast to normal ndarray (not recarray), or else we
    # would get invalid type promotion
    times_mc = np.array(times_mc).astype(float)
    times_jte = times_mc - evt_timestamp_in_ns + evt_mc_time
    return times_jte


class MCTimeCorrector(Module):
    """
    Module that converts JTE hit times to MC times.
    Thus, the following tables need to be converted:
    - mc_tracks
    - mc_hits

    Parameters
    ----------
    mc_hits_key : str, optional
        Name of the mc_hits to convert (default: 'McHits').
    mc_tracks_key : str, optional
        Name of the mc_tracks to convert (default: 'McTracks').
    event_info_key : str, optional
        Name of the event_info to store this in (default: 'EventInfo').
    """

    def configure(self):
        # get event_info, hits and mc_tracks key ; define conversion func
        self.event_info_key = self.get('event_info_key', default='EventInfo')
        self.mc_tracks_key = self.get('mc_tracks_key', default='McTracks')
        self.mc_hits_key = self.get('mc_hits_key', default='McHits')

        self.convert_mc_times_to_jte_times = \
            np.frompyfunc(convert_mc_times_to_jte_times, 3, 1)

    def process(self, blob):
        # convert the mc times to jte times
        event_info = blob[self.event_info_key]
        mc_tracks = blob[self.mc_tracks_key]
        mc_hits = blob[self.mc_hits_key]
        timestamp_in_ns = event_info.timestamp * 1e9 + event_info.nanoseconds

        mc_tracks['time'] = self.convert_mc_times_to_jte_times(
            mc_tracks.time, timestamp_in_ns, event_info.mc_time
        )
        mc_hits['time'] = self.convert_mc_times_to_jte_times(
            mc_hits.time, timestamp_in_ns, event_info.mc_time
        )

        blob[self.mc_tracks_key] = mc_tracks
        blob[self.mc_hits_key] = mc_hits

        return blob
