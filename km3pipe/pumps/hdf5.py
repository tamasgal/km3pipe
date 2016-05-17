# coding=utf-8
# Filename: hdf5.py
# pylint: disable=C0103,R0903
# vim:set ts=4 sts=4 sw=4 et:
"""
Pumps for the EVT simulation dataformat.

"""
from __future__ import division, absolute_import, print_function

import os.path

import tables

from km3pipe import Pump, Module
from km3pipe.dataclasses import HitSeries, TrackSeries
from km3pipe.logger import logging

log = logging.getLogger(__name__)  # pylint: disable=C0103

__author__ = 'tamasgal'

POS_ATOM = tables.FloatAtom(shape=3)


class EventInfo(tables.IsDescription):
    det_id = tables.IntCol()
    event_id = tables.UIntCol()
    frame_index = tables.UIntCol()
    mc_id = tables.IntCol()
    mc_t = tables.Float64Col()
    overlays = tables.UInt8Col()
    run_id = tables.UIntCol()
    # timestamp = tables.Float64Col()
    trigger_counter = tables.UInt64Col()
    trigger_mask = tables.UInt64Col()


class Hit(tables.IsDescription):
    channel_id = tables.UInt8Col()
    dom_id = tables.UIntCol()
    event_id = tables.UIntCol()
    id = tables.UIntCol()
    pmt_id = tables.UIntCol()
    run_id = tables.UIntCol()
    time = tables.IntCol()
    tot = tables.UInt8Col()
    triggered = tables.BoolCol()


class Track(tables.IsDescription):
    dir = tables.FloatCol(shape=(3,))
    energy = tables.FloatCol()
    event_id = tables.UIntCol()
    id = tables.UIntCol()
    pos = tables.FloatCol(shape=(3,))
    run_id = tables.UIntCol()
    time = tables.IntCol()
    type = tables.IntCol()


class RecoTrack(tables.IsDescription):
    dir = tables.FloatCol(shape=(3,))
    energy = tables.FloatCol()
    event_id = tables.UIntCol()
    pos = tables.FloatCol(shape=(3,))
    run_id = tables.UIntCol()
    time = tables.IntCol()
    quality = tables.FloatCol()         # likelihood


class RecoLNSTrack(RecoTrack):
    beta = tables.FloatCol()
    n_fits = tables.UIntCol()
    max_likelihood = tables.FloatCol()
    n_compatible_solutions = tables.UIntCol()
    n_hits = tables.UIntCol()
    # shape might be buggy
    error_matrix = tables.FloatCol(shape=(15,))


class JGandalfTrack(RecoTrack):
    # shape might be buggy
    error_matrix = tables.FloatCol(shape=(15,))
    beta_0 = tables.FloatCol()
    beta_1 = tables.FloatCol()
    likelihood = tables.FloatCol()
    reduced_likelihood = tables.FloatCol()
    energy_uncorrected = tables.FloatCol()
    n_hits = tables.UIntCol()
    energy_old = tables.FloatCol()


class QStrategyTrack(RecoTrack):
    m_estimator = tables.FloatCol()
    r_final = tables.FloatCol()
    collected_charge = tables.FloatCol()
    m_prefit = tables.FloatCol()
    r_prefit = tables.FloatCol()
    intertia = tables.FloatCol()


class AAShowerFitTrack(RecoTrack):
    m_estimator = tables.FloatCol()
    t_vertex = tables.FloatCol()
    n_hits = tables.UIntCol()
    beta = tables.FloatCol()
    # shape = (x, y, z, E, theta, phi) x dito
    error_matrix = tables.FloatCol(shape=(6, 6))


class DusjTrack(RecoTrack):
    # fix those namings later
    # some of them will be thrown out anyway
    gold_parameter = tables.FloatCol()
    SmallInertia = tables.FloatCol()
    TimeResidualFWHM = tables.FloatCol()
    TimeResidualNumberOfHits = tables.FloatCol()
    YIntersepto1000_o50 = tables.FloatCol()
    DusjShowerRecoVertexFitLogLikelihood = tables.FloatCol()
    DusjShowerRecoVertexFitDegreesOfFreedom = tables.FloatCol()
    DusjShowerRecoVertexFitReducedLogLikelihood = tables.FloatCol()
    ReconstructedShowerEnergy = tables.FloatCol()
    DusjShowerRecoFinalFitLogLikelihood = tables.FloatCol()
    FitHorizontalDistanceToDetectorCenter = tables.FloatCol()
    FitNumberOfStrings = tables.FloatCol()
    FitQuadrupoleMoment = tables.FloatCol()
    FitTimeResidualChiSquare = tables.FloatCol()
    FitTotalCharge = tables.FloatCol()
    FitVerticalDistanceToDetectorCenter = tables.FloatCol()


class ThomasParameters(tables.IsDescription):
    # fix those namings later
    # some of them will be thrown out anyway
    # Slope parameters (each 11, several fit ranges)
    slope = tables.FloatCol()
    chi2 = tables.FloatCol()
    y_intersept = tables.FloatCol()
    # difference between current fit and fit to range [0-1000]
    y_intersept_diff = tables.UIntCol()
    # Time residual distribution
    TimeResidualNumberOfHits = tables.UIntCol()
    TimeResidualMean = tables.FloatCol()
    TimeResidualMedian = tables.FloatCol()
    TimeResidualRMS = tables.FloatCol()
    TimeResidualWidth15_85 = tables.FloatCol()
    GParameter = tables.FloatCol()
    GoldParameter = tables.FloatCol()
    # Tensor of inertia
    # shapes/dtype???
    RelativeInertia = tables.FloatCol()
    MiddleInertia = tables.FloatCol()
    SmallInertia = tables.FloatCol()
    BigInertia = tables.FloatCol()


class HDF5Sink(Module):
    def __init__(self, **context):
        """A Module to convert (KM3NeT) ROOT files to HDF5."""
        super(self.__class__, self).__init__(**context)
        self.filename = self.get('filename') or 'dump.h5'
        self.index = 1
        self.h5file = tables.open_file(self.filename, mode="w", title="KM3NeT")
        self.filters = tables.Filters(complevel=5)
        self.hits = self.h5file.create_table('/', 'hits',
                                             Hit, "Hits",
                                             filters=self.filters)
        self.mc_hits = self.h5file.create_table('/', 'mc_hits',
                                                Hit, "MC Hits",
                                                filters=self.filters)
        self.mc_tracks = self.h5file.create_table('/', 'mc_tracks',
                                                  Track, "MC Tracks",
                                                  filters=self.filters)
        self.event_info = self.h5file.create_table('/', 'event_info',
                                                   EventInfo, "Event Info",
                                                   filters=self.filters)

    def _write_hits(self, hits, hit_row):
        for hit in hits:
            hit_row['channel_id'] = hit.channel_id
            hit_row['dom_id'] = hit.dom_id
            hit_row['event_id'] = hits.event_id
            hit_row['id'] = hit.id
            hit_row['pmt_id'] = hit.pmt_id
            # hit_row['run_id'] = hit.run_id
            hit_row['time'] = hit.time
            hit_row['tot'] = hit.tot
            hit_row['triggered'] = hit.triggered
            hit_row.append()

    def _write_tracks(self, tracks, track_row):
        for track in tracks:
            track_row['dir'] = track.dir
            track_row['energy'] = track.energy
            track_row['event_id'] = tracks.event_id
            track_row['id'] = track.id
            track_row['pos'] = track.pos
            # track_row['run_id'] = track.run_id
            track_row['time'] = track.time
            track_row['type'] = track.type
            track_row.append()

    def _write_event_info_from_aanet(self, info, info_row):
        info_row['det_id'] = info.det_id
        info_row['event_id'] = info.id
        info_row['frame_index'] = info.frame_index
        info_row['mc_id'] = info.mc_id
        info_row['mc_t'] = info.mc_t
        info_row['overlays'] = info.overlays
        info_row['run_id'] = info.run_id
        # info_row['timestamp'] = info.timestamp
        info_row['trigger_counter'] = info.trigger_counter
        info_row['trigger_mask'] = info.trigger_mask
        info_row.append()

    def process(self, blob):
        hits = blob['Hits']
        self._write_hits(hits, self.hits.row)
        if 'MCHits' in blob:
            self._write_hits(blob['MCHits'], self.mc_hits.row)
        if 'MCTracks' in blob:
            self._write_tracks(blob['MCTracks'], self.mc_tracks.row)
        if 'Evt' in blob:
            self._write_event_info_from_aanet(blob['Evt'], self.event_info.row)

        if not self.index % 1000:
            self.hits.flush()
            self.mc_hits.flush()
            self.mc_tracks.flush()
            self.event_info.flush()

        self.index += 1
        return blob

    def finish(self):
        self.hits.cols.event_id.create_index()
        self.event_info.cols.event_id.create_index()
        self.mc_hits.cols.event_id.create_index()
        self.mc_tracks.cols.event_id.create_index()
        #self.event_info.cols.run_id.create_index()
        #self.mc_hits.cols.run_id.create_index()
        #self.mc_tracks.cols.run_id.create_index()
        self.hits.flush()
        self.event_info.flush()
        self.mc_hits.flush()
        self.mc_tracks.flush()
        self.h5file.close()


class HDF5Pump(Pump):
    """Provides a pump for KM3NeT HDF5 files"""
    def __init__(self, **context):
        super(self.__class__, self).__init__(**context)
        self.filename = self.get('filename')
        if os.path.isfile(self.filename):
            self.h5_file = tables.File(self.filename)
        else:
            raise IOError("No such file or directory: '{0}'"
                          .format(self.filename))
        self.index = None
        self._reset_index()

        try:
            event_info = self.h5_file.get_node('/', 'event_info')
            self.event_ids = event_info.cols.event_id[:]
        except tables.NoSuchNodeError:
            log.critical("No /event_info table found.")
            raise SystemExit

        self._n_events = len(self.event_ids)

    def process(self, blob):
        try:
            blob = self.get_blob(self.index)
        except KeyError:
            self._reset_index()
            raise StopIteration
        self.index += 1
        return blob

    def _get_hits(self, event_id, table_name='hits', where='/'):
        table = self.h5_file.get_node(where, table_name)
        rows = table.read_where('event_id == %d' % event_id)
        return HitSeries.from_table(rows, event_id)

    def _get_tracks(self, event_id, table_name='tracks', where='/'):
        table = self.h5_file.get_node(where, table_name)
        rows = table.read_where('event_id == %d' % event_id)
        return TrackSeries.from_table(rows, event_id)

    def _get_event_info(self, event_id, table_name='event_info', where='/'):
        table = self.h5_file.get_node(where, table_name)
        return table[event_id]

    def get_blob(self, index):
        event_id = self.event_ids[index]
        blob = {}
        blob['Hits'] = self._get_hits(event_id, table_name='hits')
        blob['MCHits'] = self._get_hits(event_id, table_name='mc_hits')
        blob['MCTracks'] = self._get_tracks(event_id, table_name='mc_tracks')
        blob['EventInfo'] = self._get_event_info(event_id,
                                                 table_name='event_info')
        return blob

    def finish(self):
        """Clean everything up"""
        self.h5_file.close()

    def _reset_index(self):
        """Reset index to default value"""
        self.index = 0

    def __len__(self):
        return self._n_events

    def __iter__(self):
        return self

    def next(self):
        """Python 2/3 compatibility for iterators"""
        return self.__next__()

    def __next__(self):
        if self.index >= self._n_events:
            self._reset_index()
            raise StopIteration
        blob = self.get_blob(self.index)
        self.index += 1
        return blob

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.get_blob(index)
        elif isinstance(index, slice):
            return self._slice_generator(index)
        else:
            raise TypeError("index must be int or slice")

    def _slice_generator(self, index):
        """A simple slice generator for iterations"""
        start, stop, step = index.indices(len(self))
        for i in range(start, stop, step):
            yield self.get_blob(i)
