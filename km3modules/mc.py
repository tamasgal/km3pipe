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


def convert_hits_jte_t_to_mc_t(hits_times_jte, evt_timestamp_in_ns, 
                               evt_mc_time):
    """
    Function that converts JTE times of hits to MC times.

    Parameters
    ----------
    hits_times_jte : np.ndarray
        Hits array with JTE times.
    evt_timestamp_in_ns : int
        Total timestamp of the event in nanoseconds.
    evt_mc_time : int
        Mc time of the event in nanoseconds.

    Returns
    -------
    ndarray
        Converted hits array with MC times.
    """
    # needs to be cast to normal ndarray (not recarray), or else we
    # would get invalid type promotion
    hits_times_jte = np.array(hits_times_jte).astype(float)
    hits_times_mc = hits_times_jte + evt_timestamp_in_ns - evt_mc_time
    return hits_times_mc
    
    
def convert_tracks_mc_t_to_jte_t(tracks_times_mc_t, evt_timestamp_in_ns, 
                                 evt_mc_time):
    """
    Function that converts MC times of mc_tracks to JTE times.

    Parameters
    ----------
    tracks_times_mc_t : np.ndarray
        Mc_tracks array with MC times.
    evt_timestamp_in_ns : int
        Total timestamp of the event in nanoseconds.
    evt_mc_time : int
        Mc time of the event in nanoseconds.

    Returns
    -------
    ndarray
        Converted mc_tracks array with JTE times.
    """
    # needs to be cast to normal ndarray (not recarray), or else we
    # would get invalid type promotion
    tracks_times_mc_t = np.array(tracks_times_mc_t).astype(float)
    tracks_times_jte_t = tracks_times_mc_t - evt_timestamp_in_ns + evt_mc_time
    return tracks_times_jte_t


class MCTimeCorrector(Module):
    """
    Module that converts JTE hit times to MC times.

    Parameters
    ----------
    hits_key : str, optional
        Name of the Hits to convert (default: 'Hits').
    mc_tracks_key : str, optional
        Name of the mc_tracks to convert (default: 'McTracks').
    event_info_key : str, optional
        Name of the EventInfo to store this in (default: 'EventInfo').
    """
    def __init__(self, convert_hits_jte_t_to_mc_t_flag,
                 convert_tracks_mc_t_to_jte_t_flag):
        Module.__init__(self)
        self.convert_hits_jte_t_to_mc_t_flag = \
        convert_hits_jte_t_to_mc_t_flag
        self.convert_tracks_mc_t_to_jte_t_flag = \
        convert_tracks_mc_t_to_jte_t_flag

    def configure(self):
        self.event_info_key = self.get('event_info_key', default='EventInfo')
        if self.convert_hits_jte_t_to_mc_t_flag:
            self.hits_key = self.get('hits_key', default='Hits')
            self.convert_hits_jte_t_to_mc_t = \
            np.frompyfunc(convert_hits_jte_t_to_mc_t, 3, 1)
        if self.convert_tracks_mc_t_to_jte_t_flag:
            self.mc_tracks_key = self.get('mc_tracks_key', default='McTracks')
            self.convert_tracks_mc_t_to_jte_t = \
            np.frompyfunc(convert_tracks_mc_t_to_jte_t, 3, 1)

    def process(self, blob):
        event_info = blob[self.event_info_key]
        timestamp_in_ns = event_info.timestamp * 1e9 + event_info.nanoseconds
        
        if self.convert_hits_jte_t_to_mc_t_flag:
            hits = blob[self.hits_key]
            hits['time'] = self.convert_hits_jte_t_to_mc_t(
                hits.time, timestamp_in_ns, event_info.mc_time)
            blob[self.hits_key] = hits
            
        if self.convert_tracks_mc_t_to_jte_t_flag:
            mc_tracks = blob[self.mc_tracks_key]
            mc_tracks['time'] = self.convert_tracks_mc_t_to_jte_t(
                mc_tracks.time, timestamp_in_ns, event_info.mc_time)
            blob[self.mc_tracks_key] = mc_tracks
            
        return blob
