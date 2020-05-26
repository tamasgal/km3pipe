#!/usr/bin/env python3

from ..core import Module, Blob
from ..dataclasses import Table
from .hdf5 import HDF5Header

import km3io
import numpy as np
from collections import defaultdict

USR_MC_TRACKS_KEYS = [b'energy_lost_in_can', b'bx', b'by', b'ichan', b'cc']


class OfflinePump(Module):
    def configure(self):
        self._filename = self.get("filename")

        self._reader = km3io.OfflineReader(self._filename)
        self.header = self._reader.header
        self.blobs = self._blob_generator()

    def process(self, blob=None):
        return next(self.blobs)

    def finish(self):
        self._reader.close()

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.blobs)

    def __getitem__(self, item):
        if not isinstance(item, int):
            raise TypeError("Only integer indices are supported.")
        return Blob({
            'event': self._reader.events[item],
            'header': self.header
        })

    def get_number_of_blobs(self):
        return len(self._reader.events)

    def _blob_generator(self):
        for event in self._reader.events:
            blob = Blob({'event': event, 'header': self.header})
            yield blob


class EventPump(Module):
    def configure(self):
        self._filename = self.require("filename")
        self._split_hits = self.get("split_hits", default=True)
        self.skip_hits = self.get("skip_hits", default=False)
        self.skip_mc_hits = self.get("skip_mc_hits", default=False)
        self.skip_mc_tracks = self.get("skip_mc_tracks", default=False)
        self.skip_header = self.get("skip_header", default=False)

        self._read_usr_data = self.get("read_usr_data", default=False)
        if self._read_usr_data:
            self.log.warning(
                "Reading usr-data will massively decrease the performance."
            )

        self._reader = km3io.OfflineReader(self._filename)

        self.header = None
        self.raw_header = None
        if not self.skip_header:
            if self._reader.header is not None:
                self.header = HDF5Header.from_km3io(self._reader.header)
                self.raw_header = self._generate_raw_header()

        self.blobs = self._blob_generator()

    def process(self, blob=None):
        return next(self.blobs)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.blobs)

    def __getitem__(self, item):
        if not isinstance(item, int):
            raise TypeError("Only integer indices are supported.")
        return self._parse_event(self._reader.events[item], Blob())

    def get_number_of_blobs(self):
        return len(self._reader.events)

    def _blob_generator(self):
        for event in self._reader.events:
            blob = Blob()

            self._parse_event(event, blob)
            yield blob

    def _parse_event(self, event, blob):
        blob["RawHeader"] = self.raw_header
        blob["Header"] = self.header

        blob['Event'] = event
        if not self.skip_hits and event.n_hits > 0:
            blob['Hits'] = self._parse_hits(event.hits)
        if not self.skip_mc_hits and event.n_mc_hits > 0:
            blob['McHits'] = self._parse_mc_hits(event.mc_hits)
        if not self.skip_mc_tracks and event.n_mc_tracks > 0:
            blob['McTracks'] = self._parse_mc_tracks(event.mc_tracks)
        return blob

    def _generate_raw_header(self):
        header = {}
        raw_header = {"field_names": [], "parameter": [], "field_values": []}
        for n, x in self._reader._fobj['Head'
                                       ]._map_3c_string_2c_string_3e_.items():
            header[n.decode("utf-8")] = x.decode("utf-8").strip()
        for attribute, fields in km3io.definitions.mc_header.items():
            values = header.get(attribute, '').split()
            if not values:
                continue
            raw_header["field_values"].append(" ".join(map(str, values)))
            raw_header["field_names"].append(" ".join(fields))
            raw_header["parameter"].append(attribute)
        return Table(
            raw_header,
            h5loc="/raw_header",
            name="RawHeader",
            h5singleton=True
        )

    def _parse_eventinfo(self, event):
        wgt1, wgt2, wgt3, wgt4 = self._parse_wgts(event.w)
        tab_data = {
            'event_id': event.id,
            'run_id': event.run_id,
            'weight_w1': wgt1,
            'weight_w2': wgt2,
            'weight_w3': wgt3,
            'weight_w4': wgt4,
            'timestamp': event.t_sec,
            'nanoseconds': event.t_ns,
            'mc_time': event.mc_t,
            'trigger_mask': event.trigger_mask,
            'trigger_counter': event.trigger_counter,
            'overlays': event.overlays,
            'det_id': event.det_id,
            'frame_index': event.frame_index,
            'mc_run_id': event.mc_run_id,
        }
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

    def _parse_usr_to_dct(self, mc_tracks):
        dct = defaultdict(list)
        for k in USR_MC_TRACKS_KEYS:
            dec_key = k.decode('utf_8')
            for i in range(mc_tracks.usr_names.shape[0]):
                value = np.nan
                if k in mc_tracks.usr_names[i]:
                    mask = mc_tracks.usr_names[i] == k
                    value = mc_tracks.usr[i][mask][0]
                dct[dec_key].append(value)
        return dct

    def _parse_mc_tracks(self, mc_tracks):
        dct = {
            'dir_x': mc_tracks.dir_x,
            'dir_y': mc_tracks.dir_y,
            'dir_z': mc_tracks.dir_z,
            'pos_x': mc_tracks.pos_x,
            'pos_y': mc_tracks.pos_y,
            'pos_z': mc_tracks.pos_z,
            'energy': mc_tracks.E,
            'time': mc_tracks.t,
            'type': mc_tracks.type,
            'id': mc_tracks.id,
            'length': mc_tracks.len
        }
        if self._read_usr_data:
            dct.update(self._parse_usr_to_dct(mc_tracks))
        return Table(dct, name='McTracks', h5loc='/mc_tracks', split_h5=True)

    def _parse_mc_hits(self, mc_hits):
        return Table({
            "a": mc_hits.a,
            "origin": mc_hits.origin,
            "pmt_id": mc_hits.pmt_id,
            "time": mc_hits.t
        },
                     h5loc="/mc_hits",
                     split_h5=self._split_hits,
                     name='McHits')

    def _parse_hits(self, hits):
        return Table({
            "channel_id": hits.channel_id,
            "dom_id": hits.dom_id,
            "time": hits.t,
            "tot": hits.tot,
            "triggered": hits.trig
        },
                     h5loc="/hits",
                     split_h5=self._split_hits,
                     name='Hits')
