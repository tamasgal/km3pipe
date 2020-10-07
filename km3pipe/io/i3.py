#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
I3Pump for ANTARES DATA.
"""
import numpy as np
import km3pipe as kp
from km3pipe.logger import get_logger

from icecube import icetray, dataclasses, dataio, antares_common
from icecube.dataclasses import *

__author__ = "Nicole Geisselbrecht"
__copyright__ = "Copyright 2016, Nicole Geisselbrecht and the KM3NeT collaboration."
__credits__ = "Tamas Gal"
__license__ = "MIT"
__maintainer__ = "Nicole Geisselbrecht"
__email__ = "ngeisselbrecht@km3net.de"
__status__ = "Development"

log = get_logger(__name__)  # pylint: disable=C0103


class I3Pump(kp.Module):
    """
    Gets information of an ANTARES I3 file and converts it to HDF5.

    Attributes
    ----------
    filename : str
        Path to the input I3 file.

    """

    def configure(self):
        self.filename = self.require("filename")
        self.fobj = dataio.I3File(self.filename)
        self.blobs = self.blob_generator()
        self.omGeoMap = None
        self.cog = None

        self._hit_buffer = np.empty((10000, 14))
        self._mchit_buffer = np.empty((10000, 12))

    def process(self, blob):
        """

        Parameters
        ----------
        blob : dict
            Empty dictionary.


        Returns
        -------
        blob : dict
            Dictionary which contains information of the I3 file.

        """

        frame = next(self.blobs)

        # EVENT HEADER INFO
        if "I3EventHeader" in frame:
            header = frame["I3EventHeader"]
            blob["EventInfo"] = self._read_event_info(frame, header)

        # MC INFO
        if "AntMCTree" and "I3MCWeightDict" in frame:

            weights = kp.Table(
                {
                    "w2": self.get_antares_w2(frame),
                    "w3": self.get_antares_w3(frame),
                    "nb_gen_events": self.get_nb_generated_events(frame),
                },
                h5loc="/mc_info",
            )

            blob["McInfo"] = weights

            nu = self.get_mc_primary(frame)
            muon = self.get_mc_highest_energy_muon(frame)

            if nu is not None:
                blob["NuInfo"] = self._read_particle_info(nu, "/nu")
            if muon is not None:
                blob["MuonInfo"] = self._read_particle_info(muon, "/muon")

        # MC HITS
        if "MCHits" in frame:
            mchits = frame["MCHits"]
            blob["McHits"] = self._read_mchits(frame, mchits)

        # AAFIT
        if "AafitFinalFit" in frame:
            aafit = frame["AafitFinalFit"]
            if aafit.fit_status == dataclasses.I3Particle.FitStatus.OK:
                blob["AafitInfo"] = self._read_aafit(frame, aafit)

        # BBFIT
        if "BBFitTrack" in frame:
            bbfit_track = frame["BBFitTrack"]
            if bbfit_track.fit_status == dataclasses.I3Particle.FitStatus.OK:
                bbfit_track_chi2 = -1.0
                if "BBFitTChi2" in frame:
                    bbfit_track_chi2 = frame["BBFitTChi2"]

                blob["BbfitTrackInfo"] = self._read_bbfit(
                    bbfit_track, bbfit_track_chi2, "/reco/bbfit_track"
                )

        if "BBFitBright" in frame:
            bbfit_bright = frame["BBFitBright"]
            if bbfit_bright.fit_status == dataclasses.I3Particle.FitStatus.OK:
                bbfit_bright_chi2 = -1.0
                if "BBFitBChi2" in frame:
                    bbfit_bright_chi2 = frame["BBFitBChi2"]

                blob["BbfitBrightInfo"] = self._read_bbfit(
                    bbfit_bright, bbfit_bright_chi2, "/reco/bbfit_bright"
                )

        # GRIDFIT
        if "GridFit_FinalFitResult" in frame:
            gridfit = frame["GridFit_FinalFitResult"]
            if gridfit.fit_status == dataclasses.I3Particle.FitStatus.OK:
                blob["GridFitInfo"] = self._read_gridfit(gridfit, "/reco/gridfit")

        # HIT INFO
        if "CalibratedPulses" in frame:
            pulses = frame["CalibratedPulses"]
            blob["HitsInfo"] = self._read_hits(frame, pulses)

        return blob

    @staticmethod
    def _read_event_info(frame, header):
        """
        Reads event information of the I3 file and returns it as dict.
        """
        triggercounter = -1
        if "AntaresTriggerCounter" in frame:
            trig = frame["AntaresTriggerCounter"]
            triggercounter = trig.value

        triggermask = 0
        if "AntTriggerHierarchy" in frame:
            trigger_types = {
                "ANT_TRIGGER_3D": 1,
                "ANT_TRIGGER_1D": 2,
                "ANT_TRIGGER_1D_WITH_PREFIT": 3,
                "ANT_TRIGGER_1D_MIXED": 4,
                "ANT_TRIGGER_1D_MIXED_WITH_PREFIT": 5,
                "ANT_TRIGGER_3D_SCAN": 6,
                "ANT_TRIGGER_3D_RECURSIVE": 7,
                "ANT_TRIGGER_OB": 8,
                "ANT_TRIGGER_MINIMUM_BIAS": 9,
                "ANT_TRIGGER_1S": 10,
                "ANT_TRIGGER_3S": 11,
                "ANT_TRIGGER_3S_SCAN": 12,
                "ANT_TRIGGER_TIMESTAMP_0": 13,
                "ANT_TRIGGER_T3": 14,
                "ANT_TRIGGER_T2": 15,
                "ANT_TRIGGER_TQ": 16,
                "ANT_TRIGGER_GC": 30,
                "ANT_MERGED": 32,
            }
            tree = frame["AntTriggerHierarchy"]
            for item in tree:
                trig_type = str(item.Key.Type)
                if trig_type in trigger_types:
                    triggermask = triggermask + 2 ** (trigger_types[trig_type] - 1)

        event_info = kp.Table(
            {
                "runID": header.RunID,
                "eventID": header.EventID,
                "triggercounter": triggercounter,
                "startTimeUnix": str(header.StartTime.GetUnixTime()),
                "startTime": str(header.StartTime),
                "triggermask": triggermask,
            },
            h5loc="/event_info",
        )
        return event_info

    @staticmethod
    def _read_particle_info(particle, h5loc):
        """
        Reads particle information of the I3 file and returns it as dict.

        """
        particle_mc = kp.Table(
            {
                "dir_x": particle.GetDir().GetX(),
                "dir_y": particle.GetDir().GetY(),
                "dir_z": particle.GetDir().GetZ(),
                "pos_x": particle.GetPos().X,
                "pos_y": particle.GetPos().Y,
                "pos_z": particle.GetPos().Z,
                "energy": particle.GetEnergy(),
                "pdgencoding": particle.GetPdgEncoding(),
            },
            h5loc=h5loc,
        )
        return particle_mc

    @staticmethod
    def _read_aafit(frame, aafit):
        """
        Reads aafit information of the I3 file and returns it as dict.
        """
        lambda_value = -1
        beta_value = -1
        if "AafitLambdaFinalFit" in frame:
            lambda_frame = frame["AafitLambdaFinalFit"]
            lambda_value = lambda_frame.value

        if "AafitErrorEstimateFinalFit" in frame:
            beta_frame = frame["AafitErrorEstimateFinalFit"]
            beta_value = beta_frame.value

        aafit_pos = aafit.GetPos()
        aafit_dir = aafit.GetDir()

        aafit_info = kp.Table(
            {
                "dir_x": aafit_dir.GetX(),
                "dir_y": aafit_dir.GetY(),
                "dir_z": aafit_dir.GetZ(),
                "pos_x": aafit_pos.X,
                "pos_y": aafit_pos.Y,
                "pos_z": aafit_pos.Z,
                "lambda": lambda_value,
                "beta": beta_value,
            },
            h5loc="/reco/aafit",
        )
        return aafit_info

    @staticmethod
    def _read_bbfit(bbfit, bbfit_chi2, h5loc):
        """
        Reads bbfit information of the I3 file and returns it as dict.
        """
        bbfit_pos = bbfit.GetPos()
        bbfit_dir = bbfit.GetDir()

        bbfit_info = kp.Table(
            {
                "dir_x": bbfit_dir.GetX(),
                "dir_y": bbfit_dir.GetY(),
                "dir_z": bbfit_dir.GetZ(),
                "pos_x": bbfit_pos.X,
                "pos_y": bbfit_pos.Y,
                "pos_z": bbfit_pos.Z,
                "chi2": bbfit_chi2.value,
            },
            h5loc=h5loc,
        )
        return bbfit_info

    @staticmethod
    def _read_gridfit(gridfit, h5loc):
        """
        Reads gridfit information of the I3 file and returns it as dict.
        """
        gridfit_pos = gridfit.GetPos()
        gridfit_dir = gridfit.GetDir()
        gridfit_info = kp.Table(
            {
                "dir_x": gridfit_dir.GetX(),
                "dir_y": gridfit_dir.GetY(),
                "dir_z": gridfit_dir.GetZ(),
                "pos_x": gridfit_pos.X,
                "pos_y": gridfit_pos.Y,
                "pos_z": gridfit_pos.Z,
                "lik": gridfit.GetLik(),
            },
            h5loc=h5loc,
        )
        return gridfit_info

    def _read_hits(self, frame, pulses):
        """Creates a hits Table from calibrated pulses"""
        idx = 0
        if pulses is not None:
            for omkey, hitseries in pulses:
                omgeo = self.omGeoMap[omkey]
                rate = self.get_rate(frame, omkey)
                for hit in hitseries:

                    if idx == len(self._hit_buffer):
                        self.increase_hits_buffer_size()
                    floor = 1 + (omkey.om - 1) / 3
                    pmt = (omkey.om + 2) % 3

                    self._hit_buffer[idx][0] = hit.ID
                    self._hit_buffer[idx][1] = omkey.string
                    self._hit_buffer[idx][2] = floor
                    self._hit_buffer[idx][3] = pmt
                    self._hit_buffer[idx][4] = omgeo.position.X - self.cog.X
                    self._hit_buffer[idx][5] = omgeo.position.Y - self.cog.Y
                    self._hit_buffer[idx][6] = omgeo.position.Z - self.cog.Z
                    self._hit_buffer[idx][7] = omgeo.orientation.GetDirX()
                    self._hit_buffer[idx][8] = omgeo.orientation.GetDirY()
                    self._hit_buffer[idx][9] = omgeo.orientation.GetDirZ()
                    self._hit_buffer[idx][10] = hit.Time
                    self._hit_buffer[idx][11] = hit.Charge
                    self._hit_buffer[idx][12] = rate
                    self._hit_buffer[idx][13] = hit.IsTriggered()
                    idx += 1

            buf = self._hit_buffer[:idx]

            hits = kp.Table(
                {
                    "id": buf[:, 0].astype(int),
                    "line": buf[:, 1].astype(np.uint8),
                    "floor": buf[:, 2].astype(np.uint8),
                    "pmt": buf[:, 3].astype(np.uint32),
                    "x": buf[:, 4],
                    "y": buf[:, 5],
                    "z": buf[:, 6],
                    "dx": buf[:, 7],
                    "dy": buf[:, 8],
                    "dz": buf[:, 9],
                    "time": buf[:, 10].astype(np.float32),
                    "charge": buf[:, 11],
                    "rate": buf[:, 12],
                    "triggered": buf[:, 13],
                },
                h5loc="/hits",
                split_h5=True,
            )
            return hits

    def increase_hits_buffer_size(self):
        """
        Increases the size of the hit buffer.
        """
        current_buffer_size = len(self._hit_buffer)
        new_buffer_size = int(current_buffer_size * 1.3)
        new_buffer = np.empty((new_buffer_size, 14))
        new_buffer[:current_buffer_size] = self._hit_buffer
        self._hit_buffer = new_buffer

    def _read_mchits(self, frame, mchits):
        """Creates a mchits Table from MCHits"""
        idx = 0
        if mchits is not None:
            for omkey, hitseries in mchits:
                om_geo = self.omGeoMap[omkey]
                rate = self.get_rate(frame, omkey)
                for hit in hitseries:
                    if idx == len(self._mchit_buffer):
                        self.increase_mchits_buffer_size()
                    floor = 1 + (omkey.om - 1) / 3
                    pmt = (omkey.om + 2) % 3
                    self._mchit_buffer[idx][0] = hit.HitID
                    self._mchit_buffer[idx][1] = omkey.string
                    self._mchit_buffer[idx][2] = floor
                    self._mchit_buffer[idx][3] = pmt
                    self._mchit_buffer[idx][4] = om_geo.position.X - self.cog.X
                    self._mchit_buffer[idx][5] = om_geo.position.Y - self.cog.Y
                    self._mchit_buffer[idx][6] = om_geo.position.Z - self.cog.Z
                    self._mchit_buffer[idx][7] = om_geo.orientation.GetDirX()
                    self._mchit_buffer[idx][8] = om_geo.orientation.GetDirY()
                    self._mchit_buffer[idx][9] = om_geo.orientation.GetDirZ()
                    self._mchit_buffer[idx][10] = hit.Time
                    self._mchit_buffer[idx][11] = rate
                    idx += 1

            mcbuf = self._mchit_buffer[:idx]

            mchits = kp.Table(
                {
                    "id": mcbuf[:, 0].astype(int),
                    "line": mcbuf[:, 1].astype(np.uint8),
                    "floor": mcbuf[:, 2].astype(np.uint8),
                    "pmt": mcbuf[:, 3].astype(np.uint32),
                    "x": mcbuf[:, 4],
                    "y": mcbuf[:, 5],
                    "z": mcbuf[:, 6],
                    "dx": mcbuf[:, 7],
                    "dy": mcbuf[:, 8],
                    "dz": mcbuf[:, 9],
                    "time": mcbuf[:, 10].astype(np.float32),
                    "rate": mcbuf[:, 11],
                },
                h5loc="/mchits",
                split_h5=True,
            )
            return mchits

    def increase_mchits_buffer_size(self):
        """
        Increases the size of the mchit buffer.
        """
        current_mcbuffer_size = len(self._mchit_buffer)
        new_mcbuffer_size = int(current_mcbuffer_size * 1.3)
        new_mcbuffer = np.empty((new_mcbuffer_size, 12))
        new_mcbuffer[:current_mcbuffer_size] = self._mchit_buffer
        self._mchit_buffer = new_mcbuffer

    @staticmethod
    def get_mc_highest_energy_muon(frame):
        mctree = frame.Get("AntMCTree")
        mctracksinice = mctree.GetInIce()
        mctrack = None
        highest_energy = -1.0

        for track in mctracksinice:
            if (track.GetType() == dataclasses.I3Particle.ParticleType.MuPlus) or (
                track.GetType() == dataclasses.I3Particle.ParticleType.MuMinus
            ):
                if track.GetEnergy() > highest_energy:
                    highest_energy = track.GetEnergy()
                    mctrack = track

        return mctrack

    @staticmethod
    def get_mc_primary(frame):
        mctree = frame.Get("AntMCTree")
        primaries = mctree.GetPrimaries()
        p = None
        highest_energy = -1.0

        for primary in primaries:
            if primary.GetEnergy() > highest_energy:
                highest_energy = primary.GetEnergy()
                p = primary

        return p

    @staticmethod
    def get_antares_w3(frame):
        """
        Gets weight w3 for event.
        """
        kgen_hen_year = 3.156e7
        weight_dict = frame.Get("I3MCWeightDict")
        w3 = weight_dict["GlobalWeight"] * (
            kgen_hen_year * icetray.I3Units.second
        )  # [1/yr]
        return w3

    @staticmethod
    def get_antares_w2(frame):
        """
        Gets weight w2 for event.
        """
        kgen_hen_year = 3.156e7
        weight_dict = frame.Get("I3MCWeightDict")
        # from OneWeight in units: [GeV.cm^2.s.sr]
        # to W2 in Antares units:  [GeV.m^2.(s/yr).sr]
        w2 = (
            weight_dict["OneWeight"]
            * (icetray.I3Units.centimeter2 / icetray.I3Units.meter2)
            * kgen_hen_year
        )
        return w2

    @staticmethod
    def get_nb_generated_events(frame):
        weight_dict = frame.Get("I3MCWeightDict")
        return weight_dict["NEvents"]

    @staticmethod
    def get_rate(frame, omkey):

        rate = -1.0

        if "OMCondition" in frame:
            condition_map = frame["OMCondition"]
            condition = condition_map[omkey]
            rate = condition.GetRate() / icetray.I3Units.kilohertz

        return rate

    def blob_generator(self):
        """
        Loop over frames in I3 file.
        Reads out geometry information of geometry frames and stores it.
        Yields physics frames.
        """
        for frame in self.fobj:

            # GEOMETRY
            if frame.Stop == icetray.I3Frame.Geometry:
                geo = frame["I3Geometry"]
                self.omGeoMap = geo.omgeo
                self.cog = geo.GetDetectorCOG()

            # PHYSICS
            elif frame.Stop == icetray.I3Frame.Physics:
                yield frame

    def finish(self):
        print("finished")
