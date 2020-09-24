#!/usr/bin/env python3
from collections import defaultdict

import numpy as np

import km3pipe as kp

USR_MC_TRACKS_KEYS = [b"energy_lost_in_can", b"bx", b"by", b"ichan", b"cc"]


class HitsTabulator(kp.Module):
    """
    Create `kp.Table` from hits provided by `km3io`.

    Parameters
    ----------
    kind: str
      The kind of hits to tabulate:
        "offline": the hits in an offline file
        "online": snapshot and triggered hits (will be combined)
        "mc": MC hits
    split: bool (default: True)
      Defines whether the hits should be split up into individual arrays
      in a single group (e.g. hits/dom_id, hits/channel_id) or stored
      as a single HDF5Compound array (e.g. hits).
    """

    def configure(self):
        self.kind = self.require("kind")
        self.split = self.get("split", default=True)

    def process(self, blob):
        self.cprint(blob)
        if self.kind == "offline":
            hits = blob["event"].hits
            blob["Hits"] = kp.Table(
                {
                    "channel_id": hits.channel_id,
                    "dom_id": hits.dom_id,
                    "time": hits.t,
                    "tot": hits.tot,
                    "triggered": hits.trig,
                },
                h5loc="/hits",
                split_h5=self.split,
                name="Hits",
            )

        if self.kind == "mc":
            mc_hits = blob["event"].mc_hits
            blob["McHits"] = kp.Table(
                {
                    "a": mc_hits.a,
                    "origin": mc_hits.origin,
                    "pmt_id": mc_hits.pmt_id,
                    "time": mc_hits.t,
                },
                h5loc="/mc_hits",
                split_h5=self._split_hits,
                name="McHits",
            )
        return blob


class MCTracksTabulator(kp.Module):
    """
    Create `kp.Table` from MC tracks provided by `km3io`.

    Parameters
    ----------
    split: bool (default: True)
      Defines whether the tracks should be split up into individual arrays
      in a single group (e.g. tracks/dom_id, hits/channel_id) or stored
      as a single HDF5Compound array (e.g. hits).
    read_usr_data: bool (default: False)
      Parses usr-data which is originally meant for user stored values, but
      was abused by generator software to store properties. This issue will
      be sorted out hopefully soon as it dramatically decreases the processing
      performance and usability.
    """

    def configure(self):
        self.split = self.get("split", default=True)

        self._read_usr_data = self.get("read_usr_data", default=False)
        if self._read_usr_data:
            self.log.warning(
                "Reading usr-data will massively decrease the performance."
            )

    def process(self, blob):
        mc_tracks = blob["event"].mc_tracks
        blob["McTracks"] = self._parse_mc_tracks(mc_tracks)
        return blob

    def _parse_usr_to_dct(self, mc_tracks):
        dct = defaultdict(list)
        for k in USR_MC_TRACKS_KEYS:
            dec_key = k.decode("utf_8")
            for i in range(mc_tracks.usr_names.shape[0]):
                value = np.nan
                if k in mc_tracks.usr_names[i]:
                    mask = mc_tracks.usr_names[i] == k
                    value = mc_tracks.usr[i][mask][0]
                dct[dec_key].append(value)
        return dct

    def _parse_mc_tracks(self, mc_tracks):
        dct = {
            "dir_x": mc_tracks.dir_x,
            "dir_y": mc_tracks.dir_y,
            "dir_z": mc_tracks.dir_z,
            "pos_x": mc_tracks.pos_x,
            "pos_y": mc_tracks.pos_y,
            "pos_z": mc_tracks.pos_z,
            "energy": mc_tracks.E,
            "time": mc_tracks.t,
            "type": mc_tracks.type,
            "id": mc_tracks.id,
            "length": mc_tracks.len,
        }
        if self._read_usr_data:
            dct.update(self._parse_usr_to_dct(mc_tracks))
        return kp.Table(dct, name="McTracks", h5loc="/mc_tracks", split_h5=self.split)


class EventInfoTabulator(kp.Module):
    """
    Create `kp.Table` from event information provided by `km3io`.

    """
    def process(self, blob):
        blob["EventInfo"] = self._parse_eventinfo(blob["event"])
        return blob

    def _parse_eventinfo(self, event):
        wgt1, wgt2, wgt3, wgt4 = self._parse_wgts(event.w)
        tab_data = {
            "event_id": event.id,
            "run_id": event.run_id,
            "weight_w1": wgt1,
            "weight_w2": wgt2,
            "weight_w3": wgt3,
            "weight_w4": wgt4,
            "timestamp": event.t_sec,
            "nanoseconds": event.t_ns,
            "mc_time": event.mc_t,
            "trigger_mask": event.trigger_mask,
            "trigger_counter": event.trigger_counter,
            "overlays": event.overlays,
            "det_id": event.det_id,
            "frame_index": event.frame_index,
            "mc_run_id": event.mc_run_id,
        }
        info = kp.Table(tab_data, h5loc="/event_info", name="EventInfo")
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


class OfflineHeaderTabulator(kp.Module):
    def process(self, blob):
        blob['RawHeader'] = kp.io.hdf5.header2table(blob['header'])
        return blob
