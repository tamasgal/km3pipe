#!/usr/bin/env python3
from collections import defaultdict

import numpy as np
import awkward as ak

import km3pipe as kp
import km3io

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
        self.with_calibration = self.get("with_calibration", default=False)
        self.split = self.get("split", default=True)

    def process(self, blob):
        if self.kind == "offline":
            n = blob["event"].n_hits
            if n == 0:
                return blob
            hits = blob["event"].hits

            hits_data = {
                "channel_id": hits.channel_id,
                "dom_id": hits.dom_id,
                "time": hits.t,
                "tot": hits.tot,
                "triggered": hits.trig,
            }

            if self.with_calibration:
                hits_data["pos_x"] = hits.pos_x
                hits_data["pos_y"] = hits.pos_y
                hits_data["pos_z"] = hits.pos_z
                hits_data["dir_x"] = hits.dir_x
                hits_data["dir_y"] = hits.dir_y
                hits_data["dir_z"] = hits.dir_z
                hits_data["tdc"] = hits.tdc

            blob["Hits"] = kp.Table(
                hits_data,
                h5loc="/hits",
                split_h5=self.split,
                name="Hits",
            )

        if self.kind == "mc":
            n = blob["event"].n_mc_hits
            if n == 0:
                return blob
            mc_hits = blob["event"].mc_hits
            blob["McHits"] = kp.Table(
                {
                    "a": mc_hits.a,
                    "origin": mc_hits.origin,
                    "pmt_id": mc_hits.pmt_id,
                    "time": mc_hits.t,
                },
                h5loc="/mc_hits",
                split_h5=self.split,
                name="McHits",
            )

        if self.kind == "online":
            raise NotImplementedError(
                "The extraction of online (DAQ) hits is not implemented yet."
            )
        return blob


class MCTracksTabulator(kp.Module):
    """
    Create `kp.Table` from MC tracks provided by `km3io`.

    Parameters
    ----------
    split: bool (default: False)
      Defines whether the tracks should be split up into individual arrays
      in a single group (e.g. mc_tracks/by, mc_tracks/origin) or stored
      as a single HDF5Compound array (e.g. mc_tracks).
    read_usr_data: bool (default: False)
      Parses usr-data which is originally meant for user stored values, but
      was abused by generator software to store properties. This issue will
      be sorted out hopefully soon as it dramatically decreases the processing
      performance and usability.
    """

    def configure(self):
        self.split = self.get("split", default=False)

        self._read_usr_data = self.get("read_usr_data", default=False)
        if self._read_usr_data:
            self.log.warning(
                "Reading usr-data will massively decrease the performance."
            )

    def process(self, blob):
        n = blob["event"].n_mc_tracks
        if n == 0:
            return blob

        mc_tracks = blob["event"].mc_tracks
        blob["McTracks"] = self._parse_mc_tracks(mc_tracks)
        return blob

    def _parse_usr_to_dct(self, mc_tracks):
        dct = defaultdict(list)
        for k in USR_MC_TRACKS_KEYS:
            dec_key = k.decode("utf_8")
            for i in range(len(mc_tracks.usr_names)):
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
            "pdgid": mc_tracks.pdgid,
            "id": mc_tracks.id,
            "length": mc_tracks.len,
        }
        if self._read_usr_data:
            dct.update(self._parse_usr_to_dct(mc_tracks))
        return kp.Table(dct, name="McTracks", h5loc="/mc_tracks", split_h5=self.split)


class RecoTracksTabulator(kp.Module):
    """
    Create `kp.Table` from recostruced tracks provided by `km3io`.

    Parameters
    ----------
    best_tracks: bool (default: False)
      Additionally determine best track.
    split: bool (default: False)
      Defines whether the tracks should be split up into individual arrays
      in a single group (e.g. reco/tracks/dom_id, reco/tracks/channel_id) or stored
      as a single HDF5Compound array (e.g. reco/tracks).
    """

    def configure(self):

        self.split = self.get("split", default=False)
        self.best_tracks = self.get("best_tracks", default=False)
        self.aashower_legacy = self.get("aashower_legacy", default=False)

        self._best_track_fmap = {
            km3io.definitions.reconstruction.JMUONPREFIT: (
                km3io.tools.best_jmuon,
                "best_jmuon",
            ),
            km3io.definitions.reconstruction.JSHOWERPREFIT: (
                km3io.tools.best_jshower,
                "best_jshower",
            ),
            km3io.definitions.reconstruction.DUSJSHOWERPREFIT: (
                km3io.tools.best_dusjshower,
                "best_dusjshower",
            ),
            km3io.definitions.reconstruction.AASHOWERFITPREFIT: (
                km3io.tools.best_aashower,
                "best_aashower",
            ),
        }

    def process(self, blob):
        n_tracks = blob["event"].n_tracks
        # we first check if there are any tracks, otherwise the other calls will raise
        if n_tracks == 0:
            return blob

        all_tracks = blob["event"].tracks

        if self.aashower_legacy == True:

            all_tracks.rec_stages = np.where(
                all_tracks.rec_type
                == km3io.definitions.reconstruction.AANET_RECONSTRUCTION_TYPE,
                all_tracks.rec_stages + 300,
                all_tracks.rec_stages,
            )

        # put all tracks into the blob
        self._put_tracks_into_blob(blob, all_tracks, "tracks", n_tracks)

        # select the best track using the km3io tools
        if self.best_tracks:

            # check if it contains any of the specific reco types (can be several)
            for stage, (best_track, reco_name) in self._best_track_fmap.items():
                if stage in all_tracks.rec_stages:
                    tracks = best_track(all_tracks)
                    self._put_tracks_into_blob(blob, tracks, reco_name, 1)

        return blob

    def _put_tracks_into_blob(self, blob, tracks, reco_identifier, n_tracks):

        """
        Put a certain type of "tracks" in the blob and give specific name.

        Parameters
        ----------
        tracks : awkward array
            The tracks object to be put in the blob eventually. Can be only best tracks.
        identifier : string
            A string to name the kp table.
        n_tracks : int
            The number of tracks from before. Use to distinguish between best and all tracks.

        """

        reco_tracks = dict(
            pos_x=tracks.pos_x,
            pos_y=tracks.pos_y,
            pos_z=tracks.pos_z,
            dir_x=tracks.dir_x,
            dir_y=tracks.dir_y,
            dir_z=tracks.dir_z,
            E=tracks.E,
            rec_type=tracks.rec_type,
            t=tracks.t,
            likelihood=tracks.lik,
            length=tracks.len,  # do all recos have this?
        )

        if n_tracks != 1:
            reco_tracks.update(
                id=tracks.id,
                idx=np.arange(n_tracks),
            )

        n_columns = max(km3io.definitions.fitparameters.values()) + 1
        fitinf_array = np.ma.filled(
            ak.to_numpy(ak.pad_none(tracks.fitinf, target=n_columns, axis=-1)),
            fill_value=np.nan,
        ).astype("float32")
        fitinf_split = np.split(fitinf_array, fitinf_array.shape[-1], axis=-1)

        if n_tracks == 1:
            for fitparam, idx in km3io.definitions.fitparameters.items():
                reco_tracks[fitparam] = fitinf_split[idx][0]

        else:
            for fitparam, idx in km3io.definitions.fitparameters.items():
                reco_tracks[fitparam] = fitinf_split[idx][:, 0]

        blob["Reco_" + reco_identifier] = kp.Table(
            reco_tracks,
            h5loc=f"/reco/" + reco_identifier,
            name="Reco " + reco_identifier,
            split_h5=self.split,
        )

        # write out the rec stages only once with all tracks
        if n_tracks != 1:

            _rec_stage = np.array(ak.flatten(tracks.rec_stages)._layout)
            _counts = ak.count(tracks.rec_stages, axis=1)
            _idx = np.repeat(np.arange(n_tracks), _counts)

            blob["RecStages"] = kp.Table(
                dict(rec_stage=_rec_stage, idx=_idx),
                # Just to save space, we specify smaller dtypes.
                # We assume there will be never more than 32767
                # reco tracks for a single reconstruction type.
                dtypes=[("rec_stage", np.int16), ("idx", np.uint16)],
                h5loc=f"/reco/rec_stages",
                name="Reconstruction Stages",
                split_h5=self.split,
            )


class EventInfoTabulator(kp.Module):
    """
    Create `kp.Table` from event information provided by `km3io`.

    """

    def process(self, blob):

        # get the sim program
        if blob["header"]:
            if "simul" in blob["header"].keys():
                sim_program = blob["header"].simul.program
            else:  # not existent for real data
                sim_program = None
        else:
            sim_program = None

        blob["EventInfo"] = self._parse_eventinfo(blob["event"], sim_program)
        return blob

    def _parse_eventinfo(self, event, sim_program):
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

        if sim_program != None:

            # unfold the info in the w2list
            w2list_dict = self._unfold_w2list(event.w2list, sim_program)
            tab_data.update(w2list_dict)

        info = kp.Table(tab_data, h5loc="/event_info", name="EventInfo")
        return info

    def _unfold_w2list(self, w2list, sim_program):

        w2list_dict = {}
        definitions_dict = {}

        if sim_program.lower() == "gseagen":
            definitions_dict = km3io.definitions.w2list_gseagen
        elif sim_program.lower() == "genhen":
            definitions_dict = km3io.definitions.w2list_genhen
        # for cases like sim_program == "MUPAGE", the w2list is empty

        for key, idx in definitions_dict.items():
            w2list_dict[key] = np.nan if idx >= len(w2list) else w2list[idx]

        return w2list_dict

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
        if blob["header"]:
            blob["RawHeader"] = kp.io.hdf5.header2table(blob["header"])
        return blob
