#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim:set ts=4 sts=4 sw=4 et:
"""
Pump for the Aanet data format.

This is undoubtedly the ugliest module in the entire framework.
If you have a way to read aanet files via the Jpp interface,
your pull request is more than welcome!
"""
from __future__ import absolute_import, print_function, division

from collections import defaultdict
import os.path

import numpy as np

from km3pipe.core import Pump, Blob
from km3pipe.dataclasses import Table
from km3pipe.logger import get_logger

log = get_logger(__name__)    # pylint: disable=C0103

__author__ = "Moritz Lotze and Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = "Thomas Heid, Liam Quinn, Javier Barrios Martí"
__license__ = "MIT"
__maintainer__ = "Moritz Lotze and Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"

FITINF2NUM = {
    'JGANDALF_BETA0_RAD': 0,    # angular resolution [rad]
    'JGANDALF_BETA1_RAD': 1,    # angular resolution [rad]
    'JGANDALF_CHI2': 2,    # chi2
    'JGANDALF_NUMBER_OF_HITS': 3,    # number of hits
    'JENERGY_ENERGY': 4,    # uncorrected energy [GeV]
    'JENERGY_CHI2': 5,    # chi2
    'JGANDALF_LAMBDA': 6,    # control parameter
    'JGANDALF_NUMBER_OF_ITERATIONS': 7,    # number of iterations
    'JSTART_NPE_MIP': 8,    # number of photo-electrons
    # up to the barycentre
    'JSTART_NPE_MIP_TOTAL': 9,    # number of photo-electrons
    # along the whole track
    'JSTART_LENGTH_METRES': 10,    # distance between first
    # and last hits in metres
    'JVETO_NPE': 11,    # number of photo-electrons
    'JVETO_NUMBER_OF_HITS': 12,    # number of hits
    'JENERGY_MUON_RANGE_METRES': 13,    # range of a muon with the
    # reconstructed energy [m]
    'JENERGY_NOISE_LIKELIHOOD': 14,    # log likelihood of every hit
    # being K40
    'JENERGY_NDF': 15,    # number of degrees of freedom
    'JENERGY_NUMBER_OF_HITS': 16,    # number of hits
    'JCOPY_Z_M': 17,    # true vertex position along
    # track [m]
}

RECO2NUM = {
    'JMUONBEGIN': 0,    # Start muon fit applications
    'JMUONPREFIT': 1,    # JPrefit.cc
    'JMUONSIMPLEX': 2,    # JSimplex.cc
    'JMUONGANDALF': 3,    # JGandalf.cc
    'JMUONENERGY': 4,    # JEnergy.cc
    'JMUONSTART': 5,    # JStart.cc
    'JMUONEND': 6,    # Termination muon fit applications
    'LineFit': 7,    # An angular reco guess.
    # It could be a seed for JPrefit
    'JSHOWERBEGIN': 100,    # Start shower fit applications
    'JSHOWERPREFIT': 101,    # JShowerPrefit.cc
    'JSHOWEREND': 102,    # Termination shower fit applications
    'JPP_REC_TYPE': 4000,    # Jpp reconstruction type for AAnet
    'JUSERBEGIN': 1000,    # Start of user applications
    'JMUONVETO': 1001,    # JVeto.cc
    'JPRESIM': 1002,    # JPreSim_HTR.cc
    'JMUONPATH': 1003,    # JPath.cc
    'JMCEVT': 1004,    # JMCEvt.cc
    'KM3DeltaPos': 10000,    # This is not a fit this gives
    # position information only
}

FITINF2NAME = {v: k for k, v in FITINF2NUM.items()}
RECO2NAME = {v: k for k, v in RECO2NUM.items()}


class AanetPump(Pump):
    """A pump for binary Aanet files.

    Parameters
    ----------
    filename: str, optional
        Name of the file to open. If this parameter is not given, ``filenames``
        needs to be specified instead.
    ignore_hits: bool, optional [default=False]
        If true, don't read our the hits/mchits.
    """

    def configure(self):
        self.filename = self.require('filename')
        self.ignore_hits = bool(self.get('ignore_hits'))
        self.header = None
        self.blobs = self.blob_generator()
        self.group_id = 0

    def get_blob(self, index):
        NotImplementedError("Aanet currently does not support indexing.")

    def blob_generator(self):
        """Create a blob generator."""
        # pylint: disable:F0401,W0612
        import aa    # pylint: disablF0401        # noqa
        from ROOT import EventFile    # pylint: disablF0401

        filename = self.filename
        log.info("Reading from file: {0}".format(filename))
        if not os.path.exists(filename):
            log.warning(filename + " not available: continue without it")

        try:
            event_file = EventFile(filename)
        except Exception:
            raise SystemExit("Could not open file")

        log.info("Generating blobs through new aanet API...")
        self.header = self._parse_header(event_file.header)
        for event in event_file:
            log.debug('Reading event...')
            blob = self._read_event(event, filename)
            log.debug('Reading header...')
            blob["AaHeader"] = self.header
            self.group_id += 1
            yield blob
        del event_file

    def _parse_eventinfo(self, event):
        event_id = event.frame_index
        mc_id = event.frame_index - 1
        # run_id = self._get_run_id()
        wgt1, wgt2, wgt3, wgt4 = self._parse_wgts(event.w)
        tab_data = {
            'event_id': event_id,
            'mc_id': mc_id,
            'run_id': event.run_id,  # TODO: this may segfault in aanet (yeah!)
            'weight_w1': wgt1,
            'weight_w2': wgt2,
            'weight_w3': wgt3,
            'weight_w4': wgt4,
            'group_id': self.group_id,
        }
        tab_data['timestamp'] = event.t.GetSec()
        tab_data['nanoseconds'] = event.t.GetNanoSec()
        tab_data['mc_time'] = event.mc_t
        info = Table(tab_data, h5loc='/event_info', name='EventInfo')
        return info

    @staticmethod
    def _parse_wgts(wgt):
        if len(wgt) == 3:
            wgt1, wgt2, wgt3 = wgt
            wgt4 = np.nan
        elif len(wgt) == 4:
            # what the hell is w4?
            wgt1, wgt2, wgt3, wgt4 = wgt
        else:
            wgt1 = wgt2 = wgt3 = wgt4 = np.nan
        return wgt1, wgt2, wgt3, wgt4

    def _parse_tracks(self, tracks):
        out = defaultdict(list)
        for i, trk in enumerate(tracks):
            self.log.debug('Reading Track #{}...'.format(i))
            trk_type = trk.rec_type
            try:
                trk_name = RECO2NAME[trk_type]
            except KeyError:
                trk_name = "Generic_Track_#{}".format(i)
                self.log.warn(
                    "Unknown Reconstruction type! "
                    "Setting to '{}'".format(trk_name)
                )
            trk_dict = self._read_track(trk)
            out[trk_name].append(
                Table(
                    trk_dict,
                    h5loc='/reco/{}'.format(trk_name.lower()),
                    name=trk_name
                )
            )
        for key in out:
            name = out[key][0].name
            h5loc = out[key][0].h5loc
            out[key] = Table(
                np.concatenate(out[key]),
                name=name,
                h5loc=h5loc,
            )
        self.log.debug(out)
        return out

    def _read_track(self, trk):
        out = {}
        out['pos_x'] = trk.pos.x
        out['pos_y'] = trk.pos.y
        out['pos_z'] = trk.pos.z
        out['dir_x'] = trk.dir.x
        out['dir_y'] = trk.dir.y
        out['dir_z'] = trk.dir.z
        out['id'] = trk.id
        out['energy'] = trk.E
        out['time'] = trk.t
        out['length'] = trk.len
        out['likelihood'] = trk.lik
        out['rec_type'] = trk.rec_type
        out['group_id'] = self.group_id
        # TODO: hit_ids,
        # TODO: rec_stages,
        self.log.debug('Reading fitinf...')
        fitinf = self._parse_fitinf(trk.fitinf)
        out.update(fitinf)
        return out

    def _parse_fitinf(self, fitinf):
        # iterating empty ROOT vector causes segfaults!
        if len(fitinf) == 0:
            self.log.debug("Found empty fitinf, skipping...")
            return {}

        out = {}
        for i, elem in enumerate(fitinf):
            name = FITINF2NAME[i]
            self.log.debug("Reading fitinf #{} ('{}')...".format(i, name))
            out[name] = elem
        return out

    def _parse_mctracks(self, mctracks):
        out = defaultdict(list)
        for trk in mctracks:
            out['dir_x'].append(trk.dir.x)
            out['dir_y'].append(trk.dir.y)
            out['dir_z'].append(trk.dir.z)
            out['pos_x'].append(trk.pos.x)
            out['pos_y'].append(trk.pos.y)
            out['pos_z'].append(trk.pos.z)
            out['energy'].append(trk.E)
            out['time'].append(trk.t)
            out['type'].append(trk.type)
            out['id'].append(trk.id)
            out['length'].append(trk.len)
            out['bjorkeny'].append(trk.getusr('by'))
            out['interaction_channel'].append(trk.getusr('ichan'))
            out['is_cc'].append(trk.getusr('cc'))
        out['group_id'] = self.group_id
        return Table(out, name='McTracks', h5loc='/mc_tracks')

    def _parse_mchits(self, mchits):
        out = defaultdict(list)
        for hit in mchits:
            out['a'].append(hit.a)
            out['origin'].append(hit.origin)
            out['pmt_id'].append(hit.pmt_id)
            out['time'].append(hit.t)
        out['group_id'] = self.group_id
        return Table(out, name='McHits', h5loc='/mc_hits', split_h5=True)

    def _parse_hits(self, hits):
        out = defaultdict(list)
        for hit in hits:
            out['channel_id'].append(hit.channel_id)
            out['dom_id'].append(hit.dom_id)
            out['time'].append(hit.t)
            out['tot'].append(hit.tot)
            out['triggered'].append(hit.trig)
        out['group_id'] = self.group_id
        return Table(out, name='Hits', h5loc='/hits', split_h5=True)

    @staticmethod
    def _parse_header(header):
        tags = {}
        for key, taglist in header._hdr_dict():
            tags[key] = [k for k in taglist]
        out = {}
        for i, (key, entries) in enumerate(header):
            out[key] = {}
            for j, elem in enumerate(entries.split()):
                if key in tags:
                    try:
                        elem_name = tags[key][j]
                    except IndexError:
                        elem_name = '{}_{}'.format(key, j)
                        log.info(
                            "Can't infer field name, "
                            "setting to '{}'...".format(elem_name)
                        )
                else:
                    elem_name = '{}_{}'.format(key, j)
                    log.info(
                        "Can't infer field name, "
                        "setting to '{}'...".format(elem_name)
                    )
                out[key][elem_name] = elem
        return out

    def _read_event(self, event, filename):
        blob = Blob()
        if self.ignore_hits:
            self.log.debug('Skipping Hits...')
        else:
            self.log.debug('Reading Hits...')
            blob['Hits'] = self._parse_hits(event.hits)
            self.log.debug('Reading McHits...')
            blob['McHits'] = self._parse_mchits(event.mc_hits)
        self.log.debug('Reading McTracks...')
        blob['McTracks'] = self._parse_mctracks(event.mc_trks)
        self.log.debug('Reading EventInfo...')
        blob['EventInfo'] = self._parse_eventinfo(event)
        self.log.debug('Reading Tracks...')
        blob.update(self._parse_tracks(event.trks))
        return blob

    def process(self, blob=None):
        return next(self.blobs)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.blobs)
