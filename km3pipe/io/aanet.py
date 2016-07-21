# Filename: aanet.py
# pylint: disable=locally-disabled
"""
Pump for the Aanet data format.

"""
from __future__ import division, absolute_import, print_function
import os.path

import numpy as np

from km3pipe import Pump
from km3pipe.dataclasses import HitSeries, TrackSeries, EventInfo
from km3pipe.logger import logging

log = logging.getLogger(__name__)  # pylint: disable=C0103

__author__ = "Tamas Gal, Thomas Heid and Moritz Lotze"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = ["Liam Quin & Javier Barrios Marti"]
__license__ = "MIT"
__maintainer__ = "Tamas Gal and Moritz Lotze"
__email__ = "tgal@km3net.de"
__status__ = "Development"


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

                if len(event.w) == 3:
                    w1, w2, w3 = event.w
                else:
                    w1 = w2 = w3 = np.nan

                blob = {'Evt': event,
                        'Hits': HitSeries.from_aanet(event.hits, event.id),
                        'MCHits': HitSeries.from_aanet(event.mc_hits, event.id),
                        'Reco': read_mini_dst(event, event.id),
                        'MCTracks': TrackSeries.from_aanet(event.mc_trks,
                                                           event.id),
                        'filename': filename,
                        'Header': self.header,
                        'EventInfo': EventInfo(
                            event.det_id,
                            event.id,
                            event.frame_index,
                            event.mc_id,
                            event.mc_t,
                            event.overlays,
                            event.run_id,
                            event.trigger_counter,
                            event.trigger_mask,
                            event.t.GetNanoSec(),
                            event.t.GetSec(),
                            w1,
                            w2,
                            w3,
                        ),
                       }
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
        minidst[recname]['event_id'] = event_id
    minidst['ThomasFeatures'] = parse_thomasfeatures(aanet_event.usr)
    return minidst


def parse_track(trk):
    out = {}
    out['pos_x'] = trk.pos.x
    out['pos_y'] = trk.pos.y
    out['pos_z'] = trk.pos.z
    out['dir_x'] = trk.dir.x
    out['dir_y'] = trk.dir.y
    out['dir_z'] = trk.dir.z
    out['time'] = trk.t
    out['energy'] = trk.E
    out['quality'] = trk.lik
    return out


def parse_thomasfeatures(aanet_usr):
    out = {}
    did_converge = len(aanet_usr) > 1

    Thomas_keys = ['Slopeo1000_o100',
                   'Slope50_1000',
                   'YIntersept0_1000',
                   'Slopeo1000_1000',
                   'Chi2200_1000',
                   'TimeResidualNumberOfHits',
                   'Chi2o1000_o20',
                   'YInterspotDiffo1000_o100',
                   'YIntersepto1000_o100',
                   'Slope200_1000',
                   'Chi2o1000_0',
                   'Slope100_1000',
                   'Slopeo1000_o20',
                   'YIntersepto1000_o10',
                   'Chi2100_1000',
                   'Chi2o1000_o100',
                   'YIntersepto1000_o50',
                   'Chi2o1000_1000',
                   'YIntersept25_1000',
                   'Chi225_1000',
                   'YInterspotDiffo1000_o20',
                   'YInterspotDiffo1000_o10',
                   'Chi2o1000_o10',
                   'YInterspotDiffo1000_o50',
                   'YIntersept50_1000',
                   'TimeResidualMean',
                   'Chi20_1000',
                   'Chi2o1000_o50',
                   'YIntersept200_1000',
                   'Slope0_1000',
                   'YIntersepto1000_1000',
                   'Slope25_1000',
                   'Slopeo1000_o10',
                   'TimeResidualMedian',
                   'YIntersepto1000_o20',
                   'Slopeo1000_o50',
                   'TimeResidualRMS',
                   'YInterspotDiffo1000_0',
                   'Slopeo1000_0',
                   'YIntersepto1000_0',
                   'YIntersept100_1000',
                   'Chi250_1000',
                   'TimeResidualWidth15_85',
                   'MiddleInertia',
                   'GParameter',
                   'SmallInertia',
                   'RelativeInertia',
                   'BigInertia',
                   'GoldParameter']

    out['did_converge'] = did_converge
    if not did_converge:
        for key in Thomas_keys:
            out[key] = np.nan
        return out

    for count, key in enumerate(Thomas_keys):
        out[key] = aanet_usr[count]
    return out


def parse_recolns(aanet_trk):
    out = parse_track(aanet_trk)
    did_converge = aanet_trk.rec_stage > -9999
    out['did_converge'] = did_converge

    recolns_keys = ['beta', 'n_fits', 'Lambda',
                    'n_compatible_solutions', 'Nhits', 'NhitsL0', 'NhitsL1']

    if not did_converge:
        for key in recolns_keys:
            out[key] = np.nan
        return out

    for count, key in enumerate(recolns_keys):
        out[key] = aanet_trk.usr[count]
    return out


def parse_jgandalf(aanet_trk):
    out = parse_track(aanet_trk)
    did_converge = aanet_trk.rec_stage > -9999
    out['did_converge'] = did_converge

    jgandalf_keys = ['Energy_f', 'Energy_can', 'Beta0',
                     'Beta1', 'Lik', 'Lik_reduced', 'NhitsL0', 'NhitsL1']

    if not did_converge:
        for key in jgandalf_keys:
            out[key] = np.nan
        return out

    for count, key in enumerate(jgandalf_keys):
        out[key] = aanet_trk.usr[count]
    return out


def parse_aashowerfit(aanet_trk):
    out = parse_track(aanet_trk)
    did_converge = aanet_trk.rec_stage > -9999
    out['did_converge'] = did_converge

    aashow_keys = ['NhitsAA', 'M_estimator', 'beta',
                   'NhitsL0', 'NhitsL1']

    if not did_converge:
        for key in aashow_keys:
            out[key] = np.nan
        return out
    for count, key in enumerate(aashow_keys):
        out[key] = aanet_trk.usr[count]
    return out


def parse_qstrategy(aanet_trk):
    out = parse_track(aanet_trk)
    did_converge = aanet_trk.rec_stage > -9999
    out['did_converge'] = did_converge
    qstrat_keys = ['MFinal', 'Charge', 'MPreFit',
                   'RPreFit', 'Inertia', 'NhitsL0', 'NhitsL1']

    if not did_converge:
        for key in qstrat_keys:
            out[key] = np.nan
        return out

    for count, key in enumerate(qstrat_keys):
        out[key] = aanet_trk.usr[count]
    return out


def parse_dusj(aanet_trk):
    out = parse_track(aanet_trk)
    did_converge = aanet_trk.rec_stage > -9999
    out['did_converge'] = did_converge

    dusj_keys = ['BigInertia', 'Chi2100_1000', 'Chi2o1000_1000',
                 'Chi2o1000_o10', 'Chi2o1000_o100', 'Chi2o1000_o50',
                 'DusjShowerRecoFinalFitMinimizerCalls',
                 'DusjShowerRecoFinalFitReducedLogLikelihood',
                 'DusjShowerRecoVertexFitLogLikelihood',
                 'DusjShowerRecoVertexFitMinimizerCalls',
                 'DusjShowerRecoVertexFitReducedLogLikelihood',
                 'FitConvergencePositionAzimuth',
                 'FitConvergencePositionTime',
                 'FitConvergencePositionX',
                 'FitConvergencePositionY',
                 'FitConvergencePositionZ',
                 'FitConvergencePositionZenith',
                 'FitVerticalDistanceToDetectorCenter',
                 'GParameter',
                 'GoldParameter',
                 'ShowerIdentifierHorizontalDistanceToDetectorCenter',
                 'ShowerIdentifierReducedChiSquare',
                 'ShowerIdentifierVerticalDistanceToDetectorCenter',
                 'Slopeo1000_o50',
                 'SmallInertia',
                 'TimeResidualMean',
                 'TimeResidualNumberOfHits',
                 'TimeResidualRMS',
                 'YIntersept25_1000',
                 'YIntersepto1000_1000',
                 'YInterspotDiffo1000_o50',
                 'NhitsL0',
                 'NhitsL1']

    if not did_converge:
        for key in dusj_keys:
            out[key] = np.nan
        return out

    for count, key in enumerate(dusj_keys):
        out[key] = aanet_trk.usr[count]
    return out
