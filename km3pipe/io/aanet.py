#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim:set ts=4 sts=4 sw=4 et:
"""
Pump for the Aanet data format.

This is undoubtedly the ugliest module in the entire framework.
If you have a way to read aanet files via the Jpp interface,
your pull request is more than welcome!
"""

from collections import defaultdict
import itertools
import subprocess
import os.path

import numpy as np
import km3io

from km3pipe.core import Pump, Blob
from km3pipe.io.hdf5 import HDF5Header
from km3pipe.dataclasses import Table
from km3pipe.logger import get_logger

log = get_logger(__name__)    # pylint: disable=C0103

__author__ = "Moritz Lotze, Tamas Gal and Johannes Schumann"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = "Thomas Heid, Liam Quinn, Javier Barrios Martí"
__license__ = "MIT"
__maintainer__ = "Moritz Lotze, Tamas Gal and Johannes Schumann"
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

FITINFDUSJ2NUM = {
    'DifferencesFirstAndSecondVertexFit_deltaTime': 0,
    'DifferencesFirstAndSecondVertexFit_distance': 1,
    'FinalShowerHits_0dist60_L1cc_SingleHits_emisAng40_trackShowerTres_N': 2,
    'FinalShowerHits_0dist60_L1cc_SingleHits_emisAng40_trackShowerTres_difference': 3,
    'FinalShowerHits_0dist60_L1cc_SingleHits_emisAng40_trackShowerTres_meanDifference': 4,
    'Fork_muonSuppression_decision': 5,
    'HitPatternCharge_finalShowerHits_10degAroundCherAngle_Charge_correlationCoefficient': 6,
    'HitPatternCharge_finalShowerHits_10degAroundCherAngle_Charge_scalarProduct': 7,
    'HitPatternCharge_finalShowerHits_10degAroundCherAngle_Charge_sum_Xcharge2': 8,
    'HitPatternCharge_finalShowerHits_10degAroundCherAngle_Charge_sum_Xcharge_times_charge': 9,
    'HitPatternCharge_finalShowerHits_10degAroundCherAngle_Charge_sum_XhitProb2': 10,
    'HitPatternCharge_finalShowerHits_10degAroundCherAngle_Charge_sum_XhitProb_times_charge': 11,
    'HitPatternCharge_finalShowerHits_10degAroundCherAngle_Charge_sum_XhitProb_times_hit': 12,
    'HitPatternCharge_finalShowerHits_10degAroundCherAngle_Charge_sum_charge2': 13,
    'HitPatternCharge_finalShowerHits_10degAroundCherAngle_Charge_sum_charge2__forXhitProb': 14,
    'HitPatternCharge_finalShowerHits_10degAroundCherAngle_Charge_sum_hit2': 15,
    'HitPatternCharge_finalShowerHits_10degAroundCherAngle_HitProbCharge_scalarProduct': 16,
    'HitPatternCharge_finalShowerHits_10degAroundCherAngle_HitProb_scalarProduct': 17,
    'HitPatternCharge_finalShowerHits_10degAroundCherAngle_NXcharge': 18,
    'HitPatternCharge_finalShowerHits_10degAroundCherAngle_Ncharge': 19,
    'HitPatternCharge_finalShowerHits_10degAroundCherAngle_Nhits': 20,
    'HitPatternTres_finalShowerHits_10degAroundCherAngle_AbsTres_Mean': 21,
    'HitPatternTres_finalShowerHits_10degAroundCherAngle_AbsTres_weightedMean': 22,
    'HitPatternTres_finalShowerHits_10degAroundCherAngle_Nhits': 23,
    'HitPatternTres_finalShowerHits_10degAroundCherAngle_Tres_Mean': 24,
    'HitPatternTres_finalShowerHits_10degAroundCherAngle_Tres_sum_XhitProb': 25,
    'HitPatternTres_finalShowerHits_10degAroundCherAngle_Tres_weightedMean': 26,
    'L0AroundL1HitSelection__weight_withoutSelfSquared': 27,
    'L0AroundL1HitSelection_weight': 28,
    'L0AroundL1HitSelection_weight_withoutSelfSquared': 29,
    'MuonSuppression_decision': 30,
    'MuonSuppression_deltaTresQ20Q80': 31,
    'MuonSuppression_enoughHits': 32,
    'Trigger_3L1Dmax52_FinalShowerHits_0dist80': 33,
    'Trigger_3L1Dmax52_FinalShowerHits_0dist80m25tres75': 34,
    'Trigger_MX8hitsDmax46_FinalShowerHits_0dist80': 35,
    'Trigger_MX8hitsDmax46_FinalShowerHits_0dist80m25tres75': 36,
    'best_DusjOrcaUsingProbabilitiesFinalFit_BjorkenY': 37,
    'best_DusjOrcaUsingProbabilitiesFinalFit_FitResult_FinalLLHValues_Nom': 38,
    'best_DusjOrcaUsingProbabilitiesFinalFit_FitResult_FinalLLHValues_Npmt': 39,
    'best_DusjOrcaUsingProbabilitiesFinalFit_FitResult_FinalLLHValues_Npmt_maxDeltaT10ns': 40,
    'best_DusjOrcaUsingProbabilitiesFinalFit_FitResult_FinalLLHValues_azimuth': 41,
    'best_DusjOrcaUsingProbabilitiesFinalFit_FitResult_FinalLLHValues_energy': 42,
    'best_DusjOrcaUsingProbabilitiesFinalFit_FitResult_FinalLLHValues_energyErrorDown_bestLLH': 43,
    'best_DusjOrcaUsingProbabilitiesFinalFit_FitResult_FinalLLHValues_energyErrorUp_bestLLH': 44,
    'best_DusjOrcaUsingProbabilitiesFinalFit_FitResult_FinalLLHValues_energy_bestLLH': 45,
    'best_DusjOrcaUsingProbabilitiesFinalFit_FitResult_FinalLLHValues_llhBestSinglePMTperDOM__sum_forNoSignal': 46,
    'best_DusjOrcaUsingProbabilitiesFinalFit_FitResult_FinalLLHValues_llhBestSinglePMTperDOM_sum': 47,
    'best_DusjOrcaUsingProbabilitiesFinalFit_FitResult_FinalLLHValues_llhSinglePMT_sum': 48,
    'best_DusjOrcaUsingProbabilitiesFinalFit_FitResult_FinalLLHValues_llhSinglePMT_sum_forNoSignal': 49,
    'best_DusjOrcaUsingProbabilitiesFinalFit_FitResult_FinalLLHValues_llh_sum': 50,
    'best_DusjOrcaUsingProbabilitiesFinalFit_FitResult_FinalLLHValues_multiplicity': 51,
    'best_DusjOrcaUsingProbabilitiesFinalFit_FitResult_FinalLLHValues_sumExpFromPoissonOMhits': 52,
    'best_DusjOrcaUsingProbabilitiesFinalFit_FitResult_FinalLLHValues_sumExpOMhits': 53,
    'best_DusjOrcaUsingProbabilitiesFinalFit_FitResult_FinalLLHValues_sumMeasuredOMhits': 54,
    'best_DusjOrcaUsingProbabilitiesFinalFit_FitResult_FinalLLHValues_sumMeasuredPMThits': 55,
    'best_DusjOrcaUsingProbabilitiesFinalFit_FitResult_FinalLLHValues_zenith': 56,
    'best_DusjOrcaUsingProbabilitiesFinalFit_FitResult_aroundCherAngle_FinalLLHValues_azimuth': 57,
    'best_DusjOrcaUsingProbabilitiesFinalFit_FitResult_aroundCherAngle_FinalLLHValues_energy': 58,
    'best_DusjOrcaUsingProbabilitiesFinalFit_FitResult_aroundCherAngle_FinalLLHValues_llh_diff_forE0p8': 59,
    'best_DusjOrcaUsingProbabilitiesFinalFit_FitResult_aroundCherAngle_FinalLLHValues_llh_diff_forE1p2': 60,
    'best_DusjOrcaUsingProbabilitiesFinalFit_FitResult_aroundCherAngle_FinalLLHValues_llh_overAllNorm': 61,
    'best_DusjOrcaUsingProbabilitiesFinalFit_FitResult_aroundCherAngle_FinalLLHValues_llh_sum': 62,
    'best_DusjOrcaUsingProbabilitiesFinalFit_FitResult_aroundCherAngle_FinalLLHValues_llh_sum_forNoSignal': 63,
    'best_DusjOrcaUsingProbabilitiesFinalFit_FitResult_aroundCherAngle_FinalLLHValues_llh_total': 64,
    'best_DusjOrcaUsingProbabilitiesFinalFit_FitResult_aroundCherAngle_FinalLLHValues_meanNomOverAllNorm': 65,
    'best_DusjOrcaUsingProbabilitiesFinalFit_FitResult_aroundCherAngle_FinalLLHValues_multiplicity': 66,
    'best_DusjOrcaUsingProbabilitiesFinalFit_FitResult_aroundCherAngle_FinalLLHValues_premiumEventFraction': 67,
    'best_DusjOrcaUsingProbabilitiesFinalFit_FitResult_aroundCherAngle_FinalLLHValues_pull': 68,
    'best_DusjOrcaUsingProbabilitiesFinalFit_FitResult_aroundCherAngle_FinalLLHValues_relativeWeightForOverAllNorm': 69,
    'best_DusjOrcaUsingProbabilitiesFinalFit_FitResult_aroundCherAngle_FinalLLHValues_sigmaNomOverAllNorm': 70,
    'best_DusjOrcaUsingProbabilitiesFinalFit_FitResult_aroundCherAngle_FinalLLHValues_sumExpFromPoissonOMhits': 71,
    'best_DusjOrcaUsingProbabilitiesFinalFit_FitResult_aroundCherAngle_FinalLLHValues_sumExpOMhits': 72,
    'best_DusjOrcaUsingProbabilitiesFinalFit_FitResult_aroundCherAngle_FinalLLHValues_sumMeasuredOMhits': 73,
    'best_DusjOrcaUsingProbabilitiesFinalFit_FitResult_aroundCherAngle_FinalLLHValues_sumMeasuredPMThits': 74,
    'best_DusjOrcaUsingProbabilitiesFinalFit_FitResult_aroundCherAngle_FinalLLHValues_zenith': 75,
    'best_DusjOrcaUsingProbabilitiesFinalFit_OUTVicinityNumber': 76,
    'best_FirstDusjOrcaVertexFit_OUTVicinityNumber': 77,
    'best_FirstDusjOrcaVertexFit_OUTVicinityWithTimeResidualToSeedNumber': 78,
    'best_SecondDusjOrcaVertexFit_OUTFiducalNumber': 79,
    'best_SecondDusjOrcaVertexFit_OUTVicinityNumber': 80,
    'deltaTres_Q20_Q80_ClusteredL2ORV1L1HitSelection_SingleHits_N': 81,
    'deltaTres_Q20_Q80_ClusteredL2ORV1L1HitSelection_SingleHits_difference': 82,
    'deltaTres_Q20_Q80_FinalShowerHits_0dist60_L1cc_SingleHits_N': 83,
    'deltaTres_Q20_Q80_FinalShowerHits_0dist60_L1cc_SingleHits_difference': 84,
    'geoCoverage_R130h160_angle20_lmin30_best_DusjOrcaUsingProbabilitiesFinalFit_FitResult': 85,
    'geoCoverage_R130h160_angle45_lmin30_best_DusjOrcaUsingProbabilitiesFinalFit_FitResult': 86,
    'geoCoverage_R130h160_angle60_lmin30_best_DusjOrcaUsingProbabilitiesFinalFit_FitResult': 87,
    'geoCoverage_R130h160_angle75_lmin30_best_DusjOrcaUsingProbabilitiesFinalFit_FitResult': 88
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
    'JDUSJBEGIN': 200,
    'JDUSJPREFIT': 201,
    'JDUSJPOSITIONFIT': 202,
    'JDUSJCOMPLETEFIT': 203,
    'JDUSJEND': 299,
    'JPP_REC_TYPE': AANET_RECTYPE_PLACEHOLDER,
    'JUSERBEGIN': 1000,
    'JMUONVETO': 1001,
    'JPRESIM': 1002,
    'JMUONPATH': 1003,
    'JMCEVT': 1004,
    'JUSEREND': 1099,
    'KM3DeltaPos': 10000,
}

JHIST_CHAINS = {
    'JMUON': [
        'JMUONGANDALF', 'JMUONENERGY', 'JMUONPREFIT', 'JMUONSIMPLEX',
        'JMUONSTART'
    ],
    'JSHOWER': ['JSHOWERPREFIT', 'JSHOWERPOSITIONFIT', 'JSHOWERCOMPLETEFIT'],
    'JDUSJ': ['JDUSJPREFIT', 'JDUSJPOSITIONFIT', 'JDUSJCOMPLETEFIT'],
}

FITINF2NAME = {v: k for k, v in FITINF2NUM.items()}
FITINFDUSJ2NAME = {v: k for k, v in FITINFDUSJ2NUM.items()}
RECO2NAME = {v: k for k, v in RECO2NUM.items()}

IS_CC = {
    3: 0,    # False,
    2: 1,    # True,
    1: 0,    # False,
    0: 1,    # True,
}

FIELDNAMES = {
    'can': b'zmin zmax r',
    'coord_origin': b'x y z',
    'cut_in': b'Emin Emax cosTmin cosTmax',
    'cut_nu': b'Emin Emax cosTmin cosTmax',
    'cut_primary': b'Emin Emax cosTmin cosTmax',
    'cut_seamuon': b'Emin Emax cosTmin cosTmax',
    'genhencut': b'gDir Emin',
    'genvol': b'zmin zmax r volume numberOfEvents',
    'livetime': b'numberOfSeconds errorOfSeconds',
    'norma': b'primaryFlux numberOfPrimaries',
    'physics': b' program',
    'propag': b'propag_0 propag_1 propag_2',
    'seabottom': b'seabottom_0',
    'seed': b'program level iseed seed_3 seed_4',
    'simul': b' program version date',
    'spectrum': b'alpha',
    'start_run': b'run_id'
}


class OfflinePump(Pump):
    """A pump for binary Aanet files.

    Parameters
    ----------
    filename: str, optional
        Name of the file to open. 
    ignore_hits: bool, optional [default=False]
        If true, don't read our the hits/mchits.
    bare: bool, optional [default=False]
        Do not create KM3Pipe specific data, just wrap the bare aanet API.
        This will only give you ``blob['evt']``.
    """
    def configure(self):
        self.filename = self.require('filename')
        self.index_start = self.get('index_start', default=1)
        self.ignore_hits = bool(self.get('ignore_hits'))
        self.bare = self.get('bare', default=False)

        self.log.info("Opening {0}".format(self.filename))
        self.r = km3io.OfflineReader(self.filename)

        self.group_id = 0
        self.num_blobs = self.get_number_of_blobs()
        # Meta
        self.cprint("Reading metadata using 'JPrintMeta'")
        meta_parser = MetaParser(filename=self.filename)
        self.meta = meta_parser.get_table()
        if self.meta is None:
            self.log.warning(
                "No metadata found, this means no data provenance!"
            )
        # Header
        self.header = None
        self.raw_header = None
        if self.r.header is not None:
            self.header = HDF5Header.from_km3io(self.r.header)
            self.raw_header = self._generate_raw_header()
        self._generic_dtypes_avail = {}
        self.file_index = int(self.index_start)
        self.blobs = self.blob_generator()

    def _generate_raw_header(self):
        header = {}
        raw_header = {"field_names": [], "parameter": [], "field_values": []}
        for n, x in self.r._fobj['Head']._map_3c_string_2c_string_3e_.items():
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

    def get_blob(self, index):
        if index < 0 or index >= self.num_blobs:
            raise IndexError()
        if self.bare:
            log.info("Skipping data conversion, only passing bare aanet data")
            return Blob({
                'evt': self.r.events[index],
                'event_file': self.filename
            })
        else:
            log.debug('Reading event...')
            blob = self._read_event(index)
            blob["RawHeader"] = self.raw_header
            blob["Header"] = self.header
            # blob['evt'] = self.r.events[i]
            # blob['event_file'] = self.r
            if self.meta is not None:
                blob['Meta'] = meta
            self.group_id += 1
            return blob

    def get_number_of_blobs(self):
        return len(self.r.events)

    def blob_generator(self):
        """Create a blob generator."""
        log.info("Generating blobs through new aanet API...called km3io ;)")
        for i in range(self.num_blobs):
            yield self.get_blob(i)

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
            'group_id': self.group_id,
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

    def _parse_tracks(self, tracks):
        log.info("Reading Tracks...")
        track_dict = defaultdict(list)
        # iterating empty ROOT vector causes segfaults!
        if len(tracks) == 0:
            self.log.debug("Found empty tracks, skipping...")
            return {}

        for i, trk in enumerate(tracks):
            self.log.debug('Reading Track #{}...'.format(i))
            trk_dict = self._read_track(trk)
            # set name + h5loc later, if the name is not available, we need
            # the dtype to make a new name

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
                    self.log.error("Unknown reco type & no history!")
                    trk_name = "UnknownTrack{}".format(i)
                else:
                    self.log.info("Unknown recoo type! Using history...")
                    stages_num = [s for s in trk.rec_stages]
                    stages = [RECO2NAME[s] for s in stages_num]

                    is_chain = False
                    for chain, default_stages in JHIST_CHAINS.items():
                        # chain is something like JMUON
                        if (RECO2NUM[chain + 'BEGIN'] < min(stages_num)) and (
                                RECO2NUM[chain + 'END'] > max(stages_num)):
                            if chain == "JDUSJ":
                                self.log.info("Adding missing Dusj parameters")
                                for dusj_param in FITINFDUSJ2NUM:
                                    if dusj_param not in trk_dict:
                                        trk_dict[dusj_param] = np.nan
                            self.log.info(
                                "Found {}, adding stage flags".format(chain)
                            )
                            trk_name = chain
                            for stage in default_stages:
                                if stage in stages:
                                    trk_dict[stage] = True
                                else:
                                    trk_dict[stage] = False
                            is_chain = True
                            break

                    if not is_chain:
                        self.log.info("Unknown chain, using stages as name")
                        trk_name = '__'.join([s for s in stages[::-1]])
                        trk_name = 'JHIST__' + trk_name

            # tab.h5loc = '/reco/{}'.format(trk_name.lower())
            track_dict[trk_name].append(trk_dict)

        return self._merge_tracks(track_dict)

    def _merge_tracks(self, track_dict):
        log.info("Merging tracks into table...")
        out = {}
        for track_name, tracks in track_dict.items():
            self.log.debug("Merging '{}'...".format(track_name))
            cols = set(itertools.chain(*[t.keys() for t in tracks]))
            track_data = defaultdict(list)
            for track in tracks:
                for col in cols:
                    if col in track:
                        track_data[col].append(track[col])
                    else:
                        track_data[col].append(np.nan)

            out[track_name] = Table(
                track_data,
                h5loc='/reco/{}'.format(track_name.lower()),
                name=track_name
            )
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
        out['hit_ids'] = trk.hit_ids
        out['rec_stages'] = trk.rec_stages
        self.log.debug('Reading fitinf...')

        isDusj = False
        if len(trk.rec_stages) > 0:
            if (min(trk.rec_stages) >= RECO2NUM['JDUSJBEGIN']) and (max(
                    trk.rec_stages) <= RECO2NUM['JDUSJEND']):
                isDusj = True
        if isDusj:
            fitinf = self._parse_fitinf_dusj(trk.fitinf)
        else:
            fitinf = self._parse_fitinf(trk.fitinf)

        out.update(fitinf)
        return out

    def _parse_fitinf(self, fitinf):
        if len(fitinf) == 0:
            self.log.debug("Found empty fitinf, skipping...")
            return {}

        out = {}
        for i, elem in enumerate(fitinf):
            name = FITINF2NAME[i]
            self.log.debug("Reading fitinf #{} ('{}')...".format(i, name))
            out[name] = elem
        return out

    def _parse_fitinf_dusj(self, fitinf):
        if len(fitinf) == 0:
            self.log.debug("Found empty fitinf, skipping...")
            return {}

        out = {}
        for i, elem in enumerate(fitinf):
            name = FITINFDUSJ2NAME[i]
            self.log.debug("Reading fitinf #{} ('{}')...".format(i, name))
            out[name] = elem
        return out

    def _parse_mctracks(self, mctracks):
        out = defaultdict(list)
        if len(mctracks.id) == 0:
            self.log.debug("Found empty mctracks, skipping...")
            return out
        out['dir_x'] = mctracks.dir_x
        out['dir_y'] = mctracks.dir_y
        out['dir_z'] = mctracks.dir_z
        out['pos_x'] = mctracks.pos_x
        out['pos_y'] = mctracks.pos_y
        out['pos_z'] = mctracks.pos_z
        out['energy'] = mctracks.E
        out['time'] = mctracks.t
        out['type'] = mctracks.type
        out['id'] = mctracks.id
        out['length'] = mctracks.len
        # out['bjorkeny'].append(trk.getusr('by'))
        # out['interaction_channel'].append(trk.getusr('ichan'))
        # try:
        #     is_cc = IS_CC[trk.getusr('cc')]
        # except Exception:
        #     # see git.km3net.de/km3py/km3pipe/issues/112
        #     # and http://trac.km3net.de/ticket/222
        #     # self.log.error(
        #     #     "Invalid value ({}) for the 'cc' usr-parameter in the "
        #     #     "MC track. 'is_cc' is now set to 0 (False).".format(
        #     #         trk.getusr('cc')
        #     #     )
        #     # )
        #     is_cc = [0]
        # finally:
        #     out['is_cc'].append(is_cc)
        out['group_id'] = self.group_id
        return Table(out, name='McTracks', h5loc='/mc_tracks')

    def _parse_mchits(self, mchits):
        out = defaultdict(list)
        # iterating empty ROOT vector causes segfaults!
        if len(mchits.id) == 0:
            self.log.debug("Found empty mchits, skipping...")
            return out
        out['a'] = mchits.a
        out['origin'] = mchits.origin
        out['pmt_id'] = mchits.pmt_id
        out['time'] = mchits.t
        out['group_id'] = self.group_id
        return Table(out, name='McHits', h5loc='/mc_hits', split_h5=True)

    def _parse_hits(self, hits):
        out = defaultdict(list)
        if len(hits.id) == 0:
            self.log.debug("Found empty hits, skipping...")
            return out
        out['channel_id'] = hits.channel_id
        out['dom_id'] = hits.dom_id
        out['time'] = hits.t
        out['tot'] = hits.tot
        out['triggered'] = hits.trig
        out['group_id'] = self.group_id
        return Table(out, name='Hits', h5loc='/hits', split_h5=True)

    def _read_event(self, index):
        blob = Blob()
        if self.ignore_hits:
            self.log.debug('Skipping Hits...')
        else:
            self.log.debug('Reading Hits...')
            blob['Hits'] = self._parse_hits(self.r[index].hits)
            self.log.debug('Reading McHits...')
            blob['McHits'] = self._parse_mchits(self.r[index].mc_hits)
        self.log.debug('Reading McTracks...')
        blob['McTracks'] = self._parse_mctracks(self.r[index].mc_tracks)
        self.log.debug('Reading EventInfo...')
        blob['EventInfo'] = self._parse_eventinfo(self.r[index].events)
        self.log.debug('Reading Tracks...')
        blob.update(self._parse_tracks(self.r[index].tracks))
        return blob

    def process(self, blob=None):
        return next(self.blobs)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.blobs)


class MetaParser(object):
    """A class which parses the JPrintMeta output for a given filename"""
    def __init__(self, filename=None, string=None):
        self.log = get_logger(__name__ + '.' + self.__class__.__name__)
        self.meta = []
        if filename is not None:
            try:
                string = subprocess.check_output([
                    'JPrintMeta', '-f', filename
                ])
            except IOError:
                self.log.error(
                    "Could not access meta information. Is Jpp loaded?"
                )
            try:
                self.parse_string(string)
            except IndexError:
                self.log.error("The Jpp metadata could not be parsed.")

    def parse_string(self, string):
        """Parse ASCII output of JPrintMeta"""
        self.log.info("Parsing ASCII data")

        if not string:
            self.log.warning("Empty metadata")
            return

        lines = string.splitlines()
        application_data = []

        application = lines[0].split()[0]
        self.log.debug("Reading meta information for '%s'" % application)

        for line in lines:
            if application is None:
                self.log.debug(
                    "Reading meta information for '%s'" % application
                )
                application = line.split()[0]
            application_data.append(line)
            if line.startswith(application + b' Linux'):
                self._record_app_data(application_data)
                application_data = []
                application = None

    def _record_app_data(self, data):
        """Parse raw metadata output for a single application

        The usual output is:
        ApplicationName RevisionNumber
        ApplicationName ROOT_Version
        ApplicationName KM3NET
        ApplicationName ./command/line --arguments --which --can
        contain
        also
        multiple lines
        and --addtional flags
        etc.
        ApplicationName Linux ... (just the `uname -a` output)
        """
        name, revision = data[0].split()
        root_version = data[1].split()[1]
        command = b'\n'.join(data[3:]).split(b'\n' + name + b' Linux')[0]
        self.meta.append({
            'application_name': np.string_(name),
            'revision': np.string_(revision),
            'root_version': np.string_(root_version),
            'command': np.string_(command)
        })

    def get_table(self, name='Meta', h5loc='/meta'):
        """Convert metadata to a KM3Pipe Table.

        Returns `None` if there is no data.

        Each column's dtype will be set to a fixed size string (numpy.string_)
        with the length of the longest entry, since writing variable length
        strings does not fit the current scheme.
        """
        if not self.meta:
            return None

        data = defaultdict(list)
        for entry in self.meta:
            for key, value in entry.items():
                data[key].append(value)
        dtypes = []
        for key, values in data.items():
            max_len = max(map(len, values))
            dtype = 'S{}'.format(max_len)
            dtypes.append((key, dtype))
        tab = Table(
            data, dtype=dtypes, h5loc=h5loc, name='Meta', h5singleton=True
        )
        return tab
