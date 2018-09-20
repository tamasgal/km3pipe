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
from km3pipe.io.hdf5 import HDF5Header
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
    'JGANDALF_BETA0_RAD': 0,
    'JGANDALF_BETA1_RAD': 1,
    'JGANDALF_CHI2': 2,
    'JGANDALF_NUMBER_OF_HITS': 3,
    'JENERGY_ENERGY': 4,
    'JENERGY_CHI2': 5,
    'JGANDALF_LAMBDA': 6,
    'JGANDALF_NUMBER_OF_ITERATIONS': 7,
    'JSTART_NPE_MIP': 8,
    'JSTART_NPE_MIP_TOTAL': 9,
    'JSTART_LENGTH_METRES': 10,
    'JVETO_NPE': 11,
    'JVETO_NUMBER_OF_HITS': 12,
    'JENERGY_MUON_RANGE_METRES': 13,
    'JENERGY_NOISE_LIKELIHOOD': 14,
    'JENERGY_NDF': 15,
    'JENERGY_NUMBER_OF_HITS': 16,
    'JCOPY_Z_M': 17,
}

# jpp > 10.1 (trunk @10276)
AANET_RECTYPE_PLACEHOLDER = 4000

RECO2NUM = {
    'JMUONBEGIN': 0,
    'JMUONPREFIT': 1,
    'JMUONSIMPLEX': 2,
    'JMUONGANDALF': 3,
    'JMUONENERGY': 4,
    'JMUONSTART': 5,
    # JMUONEND @ 10.1, JLINEFIT @ trunk
    'JLINEFIT': 6,
    # 10.1 artifact, REMOVE IN FUTURE
    'LineFit': 7,
    'JMUONEND': 99,
    'JSHOWERBEGIN': 100,
    'JSHOWERPREFIT': 101,
    'JSHOWERPOSITIONFIT': 102,
    'JSHOWERCOMPLETEFIT': 103,
    'JSHOWEREND': 199,
    'JPP_REC_TYPE': AANET_RECTYPE_PLACEHOLDER,
    'JUSERBEGIN': 1000,
    'JMUONVETO': 1001,
    'JPRESIM': 1002,
    'JMUONPATH': 1003,
    'JMCEVT': 1004,
    'JUSEREND': 1099,
    'KM3DeltaPos': 10000,
}

FITINF2NAME = {v: k for k, v in FITINF2NUM.items()}
RECO2NAME = {v: k for k, v in RECO2NUM.items()}

IS_CC = {
    3: 0,    # False,
    2: 1,    # True,
    1: 0,    # False,
    0: 1,    # True,
}


class AanetPump(Pump):
    """A pump for binary Aanet files.

    Parameters
    ----------
    filename: str, optional
        Name of the file to open. If this parameter is not given, ``filenames``
        needs to be specified instead.
    ignore_hits: bool, optional [default=False]
        If true, don't read our the hits/mchits.
    bare: bool, optional [default=False]
        Do not create KM3Pipe specific data, just wrap the bare aanet API.
        This will only give you ``blob['evt']``.
    """

    def configure(self):
        self.filename = self.require('filename')
        self.ignore_hits = bool(self.get('ignore_hits'))
        self.bare = self.get('bare', default=False)
        self.raw_header = None
        self.header = None
        self.blobs = self.blob_generator()
        self.group_id = 0
        self._generic_dtypes_avail = {}

    def get_blob(self, index):
        NotImplementedError("Aanet currently does not support indexing.")

    def blob_generator(self):
        """Create a blob generator."""
        # pylint: disable:F0401,W0612
        import aa    # pylint: disablF0401        # noqa
        from ROOT import EventFile    # pylint: disable F0401

        filename = self.filename
        log.info("Reading from file: {0}".format(filename))
        if not os.path.exists(filename):
            log.warning(filename + " not available: continue without it")

        try:
            event_file = EventFile(filename)
        except Exception:
            raise SystemExit("Could not open file")

        log.info("Generating blobs through new aanet API...")

        if self.bare:
            log.info("Skipping data conversion, only passing bare aanet data")
            for event in event_file:
                yield Blob({'evt': event, 'event_file': event_file})
        else:
            log.info("Unpacking aanet header into dictionary...")
            hdr = self._parse_header(event_file.header)
            if not hdr:
                log.info("Empty header dict found, skipping...")
                self.raw_header = None
            else:
                log.info("Converting Header dict to Table...")
                self.raw_header = self._convert_header_dict_to_table(hdr)
                log.info("Creating HDF5Header")
                self.header = HDF5Header.from_table(self.raw_header)
            for event in event_file:
                log.debug('Reading event...')
                blob = self._read_event(event, filename)
                log.debug('Reading header...')
                blob["RawHeader"] = self.raw_header
                blob["Header"] = self.header
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
            'run_id': event.run_id,    # TODO: this may segfault in aanet
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
        log.info("Reading Tracks...")
        out = defaultdict(list)
        # iterating empty ROOT vector causes segfaults!
        if len(tracks) == 0:
            self.log.debug("Found empty tracks, skipping...")
            return out
        for i, trk in enumerate(tracks):
            self.log.debug('Reading Track #{}...'.format(i))
            trk_dict = self._read_track(trk)
            # set name + h5loc later, if the name is not available, we need
            # the dtype to make a new name
            tab = Table(trk_dict)

            trk_type = trk.rec_type
            try:
                trk_name = RECO2NAME[trk_type]
            except KeyError:
                trk_type = AANET_RECTYPE_PLACEHOLDER
            if trk_type == AANET_RECTYPE_PLACEHOLDER:
                # if we have a history available but no name (because JEvt.cc),
                # then use the concatenated history as the name.
                # If that is not available, enumerate the tracks by their
                # dtypes (since they have no other tagging)
                if len(trk.rec_stages) == 0:
                    self.log.info(
                        "Unknown Reconstruction type & no history available!"
                    )
                    trk_name = self._handle_generic(tab.dtype)
                else:
                    self.log.info(
                        "Unknown Reconstruction type! Using history..."
                    )
                    # iteration in reverse order segfaults, whyever...
                    stages = [k for k in trk.rec_stages]
                    trk_name = '__'.join([RECO2NAME[k] for k in stages[::-1]])
                    trk_name = 'JHIST__' + trk_name

            tab.name = trk_name
            tab.h5loc = '/reco/{}'.format(trk_name.lower())
            out[trk_name].append(tab)
        log.info("Merging tracks into table...")
        for key in out:
            log.debug("Merging '{}'...".format(key))
            name = out[key][0].name
            h5loc = out[key][0].h5loc
            out[key] = Table(
                np.concatenate(out[key]),
                name=name,
                h5loc=h5loc,
            )
        self.log.debug(out)
        return out

    # sometimes the reco name/tag is not correctly written
    # which means that multiple different fits with
    # different dtypes have the same name.
    # Keep track of the dtypes from unnamed tracks and just enumerate them
    # this is problematic since 2 unnamed tracks with same length
    # (e.g. fit A and B just write pos_x, pos_y, pos_z)
    # would get merged -- they would be thrown into a table containing
    # both A and B.
    # This needs to be fixed upstream obviously, so here we should just make
    # noise about it

    def _handle_generic(self, dt):
        pref = "GENERIC_TRACK"
        if dt in self._generic_dtypes_avail:
            nam = self._generic_dtypes_avail[dt]
            return nam
        cnt = len(self._generic_dtypes_avail)
        nam = '{}_{}'.format(pref, cnt)
        self.log.warn(
            "Unknown Reconstruction type! "
            "Setting to '{}'. This may lead to "
            "unrelated fit getting merged -- which is very likely not "
            "what you want! The only way to fix this is to put the "
            "proper numbers into the `rec_type` of your input file! "
            "For now, we will just count + enumerate all different "
            "datastructures but I do not have any information to tell "
            "them apart!".format(nam)
        )
        self._generic_dtypes_avail[dt] = nam
        return nam

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
        # iterating empty ROOT vector causes segfaults!
        if len(mctracks) == 0:
            self.log.debug("Found empty mctracks, skipping...")
            return out
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
            out['is_cc'].append(IS_CC[trk.getusr('cc')])
        out['group_id'] = self.group_id
        return Table(out, name='McTracks', h5loc='/mc_tracks')

    def _parse_mchits(self, mchits):
        out = defaultdict(list)
        # iterating empty ROOT vector causes segfaults!
        if len(mchits) == 0:
            self.log.debug("Found empty mchits, skipping...")
            return out
        for hit in mchits:
            out['a'].append(hit.a)
            out['origin'].append(hit.origin)
            out['pmt_id'].append(hit.pmt_id)
            out['time'].append(hit.t)
        out['group_id'] = self.group_id
        return Table(out, name='McHits', h5loc='/mc_hits', split_h5=True)

    def _parse_hits(self, hits):
        out = defaultdict(list)
        # iterating empty ROOT vector causes segfaults!
        if len(hits) == 0:
            self.log.debug("Found empty hits, skipping...")
            return out
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
        if len(header) == 0:
            return out
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

    # TODO: delete this method and use the function in io/hdf5.py
    @staticmethod
    def _convert_header_dict_to_table(header_dict):
        if not header_dict:
            log.warn("Can't convert empty header dict to table, skipping...")
            return
        tab_dict = defaultdict(list)
        log.debug("Param:   field_names    field_values    dtype")
        for parameter, data in header_dict.items():
            fields = []
            values = []
            types = []
            for field_name, field_value in data.items():
                fields.append(field_name)
                values.append(field_value)
                try:
                    _ = float(field_value)    # noqa
                    types.append('f4')
                except ValueError:
                    types.append('a{}'.format(len(field_value)))
            tab_dict['parameter'].append(parameter)
            tab_dict['field_names'].append(' '.join(fields))
            tab_dict['field_values'].append(' '.join(values))
            tab_dict['dtype'].append(' '.join(types))
            log.debug(
                "{}: {} {} {}".format(
                    tab_dict['parameter'][-1],
                    tab_dict['field_names'][-1],
                    tab_dict['field_values'][-1],
                    tab_dict['dtype'][-1],
                )
            )
        return Table(
            tab_dict, h5loc='/raw_header', name='RawHeader', h5singleton=True
        )

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
