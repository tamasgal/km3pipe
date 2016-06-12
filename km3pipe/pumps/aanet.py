# coding=utf-8
# Filename: aanet.py
# pylint: disable=locally-disabled
"""
Pump for the Aanet data format.

"""
from __future__ import division, absolute_import, print_function
import os.path

from km3pipe import Pump
from km3pipe.dataclasses import HitSeries, TrackSeries
from km3pipe.logger import logging

log = logging.getLogger(__name__)  # pylint: disable=C0103


class AanetPump(Pump):
    """A pump for binary Aanet files."""

    def __init__(self, **context):
        super(self.__class__, self).__init__(**context)

        self.filename = self.get('filename')
        self.filenames = self.get('filenames') or []
        self.indices = self.get('indices')
        self.additional = self.get('additional')
        if self.additional:
            self.id = self.get('id')
            self.return_without_match = self.get("return_without_match")

        if not self.filename and not self.filenames:
            raise ValueError("No filename(s) defined")

        if self.additional and not self.filename:
            log.error("additional file only implemeted for single files")

        if self.filename:
            if "[index]" in self.filename and self.indices:
                self._parse_filenames()
            else:
                self.filenames.append(self.filename)

        self.header = None
        self.blobs = self.blob_generator()
        if self.additional:
            import ROOT
            import aa  # noqa
            dummy_evt = ROOT.Evt()
            dummy_evt.frame_index = -1
            self.previous = {"Evt": dummy_evt}

    def _parse_filenames(self):
        prefix, suffix = self.filename.split('[index]')
        self.filenames += [prefix + str(i) + suffix for i in self.indices]

    def get_blob(self, index):
        NotImplementedError("Aanet currently does not support indexing.")

    def blob_generator(self):
        """Create a blob generator."""
        # pylint: disable:F0401,W0612
        import aa  # noqa
        from ROOT import EventFile

        for filename in self.filenames:
            print("Reading from file: {0}".format(filename))
            if not os.path.exists(filename):
                log.warn(filename + " not available: continue without it")
                continue

            try:
                event_file = EventFile(filename)
            except Exception:
                raise SystemExit("Could not open file")

            try:
                self.header = event_file.rootfile().Get("Header")
            except AttributeError:
                pass

            for event in event_file:
                if event.det_id <= 0:  # apply ZED correction
                    for track in event.mc_trks:
                        track.pos.z += 405.93
                blob = {'Evt': event,
                        'Hits': HitSeries.from_aanet(event.hits, event.id),
                        'MCHits': HitSeries.from_aanet(event.mc_hits, event.id),
                        'Reco': read_mini_dst(event, event.id),
                        'MCTracks': TrackSeries.from_aanet(event.mc_trks,
                                                           event.id),
                        'filename': filename,
                        'Header': self.header}
                yield blob
            del event_file

    def event_index(self, blob):
        if self.id:
            return blob["Evt"].id
        else:
            return blob["Evt"].frame_index

    def process(self, blob=None):
        if self.additional:
            new_blob = self.previous
            if self.event_index(new_blob) == blob["Evt"].frame_index:
                blob[self.additional] = new_blob
                return blob

            elif self.event_index(new_blob) > blob["Evt"].frame_index:
                if self.return_without_match:
                    return blob
                else:
                    return None

            else:
                while self.event_index(new_blob) < blob["Evt"].frame_index:
                    new_blob = next(self.blobs)

                self.previous = new_blob

                return self.process(blob)

        else:
            return next(self.blobs)

    def __iter__(self):
        return self

    def next(self):
        """Python 2/3 compatibility for iterators"""
        return self.__next__()

    def __next__(self):
        return next(self.blobs)


def read_mini_dst(aanet_event, event_id):
    pos_to_recname = {
        0: 'RecoLNS',
        1: 'JGandalf',
        2: 'AaShowerFit',
        3: 'QStrategy',
        4: 'Dusj',
    }
    recname_to_reader = {
        'RecoLNS': parse_recolns,
        'JGandalf': parse_jgandalf,
        'AaShowerFit': parse_aashowerfit,
        'QStrategy': parse_qstrategy,
        'Dusj': parse_dusj,
    }
    minidst = {}
    for k, trk in enumerate(aanet_event.trks):
        recname = pos_to_recname[k]
        reader = recname_to_reader[recname]
        minidst[recname] = reader(trk)
    #minidst['ThomasFeatures'] = parse_thomasfeatures(aanet_event.usr)
    minidst['event_id'] = event_id
    return minidst


def parse_track(trk):
    out = {}
    out['position'] = (trk.pos.x, trk.pos.y, trk.pos.z)
    out['direction'] = (trk.dir.x, trk.dir.y, trk.dir.z)
    out['time'] = trk.t
    out['energy'] = trk.E
    out['quality'] = trk.lik
    return out


def parse_thomasfeatures(aanet_usr):
    out = {}
    return out


def parse_recolns(aanet_trk):
    if not aanet_trk.rec_stage > -9999:
        return {}
    out = parse_track(aanet_trk)
    out['beta'] = aanet_trk.fitinf[0]
    out['n_fits'] = aanet_trk.fitinf[1]
    out['max_likelihood'] = aanet_trk.fitinf[2]
    out['n_compatible_solutions'] = aanet_trk.fitinf[3]
    out['n_hits'] = aanet_trk.fitinf[4]
    out['error_matrix'] = list(aanet_trk.error_matrix)
    return out


def parse_jgandalf(aanet_trk):
    if not aanet_trk.rec_stage > -9999:
        return {}
    out = parse_track(aanet_trk)
    return out


def parse_aashowerfit(aanet_trk):
    if not aanet_trk.rec_stage > -9999:
        return {}
    out = parse_track(aanet_trk)
    return out


def parse_qstrategy(aanet_trk):
    if not aanet_trk.rec_stage > -9999:
        return {}
    out = parse_track(aanet_trk)
    return out


def parse_dusj(aanet_trk):
    if not aanet_trk.rec_stage > -9999:
        return {}
    out = parse_track(aanet_trk)
    return out
