#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pump for the Aanet data format.

This is undoubtedly the ugliest module in the entire framework.
If you have a way to read aanet files via the Jpp interface,
your pull request is more than welcome!
"""
from __future__ import division, absolute_import, print_function

from collections import defaultdict
import os.path

import numpy as np
from scipy.stats import iqr

from km3pipe.core import Pump, Blob
from km3pipe.dataclasses import (RawHitSeries, McHitSeries,
                                 TrackSeries, EventInfo,
                                 KM3Array)
from km3pipe.logger import logging
from km3pipe.math import mad

log = logging.getLogger(__name__)  # pylint: disable=C0103

__author__ = "Tamas Gal, Thomas Heid and Moritz Lotze"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = ["Liam Quinn & Javier Barrios Martí"]
__license__ = "MIT"
__maintainer__ = "Tamas Gal and Moritz Lotze"
__email__ = "tgal@km3net.de"
__status__ = "Development"

FITINF_ENUM = [
    'beta0',
    'beta1',
    'chi2',
    'n_hits',
    'jenergy_energy',
    'jenergy_chi2',
    'lambda',
    'n_iter',
    'jstart_npe_mip',
    'jstart_npe_mip_total',
    'jstart_length',
    # 'jveto_npe',
    # 'jveto_nhits'
]


class AanetPump(Pump):
    """A pump for binary Aanet files.

    Parameters
    ----------
    filename: str, optional
        Name of the file to open. If this parameter is not given, ``filenames``
        needs to be specified instead.
    filenames: list(str), optional
        List of files to open.
    aa_fmt: string, optional (default: 'gandalf_new')
        Subformat of aanet in the file. Possible values:
        ``'minidst', 'jevt_jgandalf', 'gandalf_new', 'generic_track',
        'ancient_recolns'``
    use_aart_sps_lib: bool, optional [default=False]
        Use Aart's local SPS copy (@ Lyon) via ``gSystem.Load``?
    apply_zed_correction: bool, optional [default=False]
        correct ~400m offset in mc tracks.
    old_mc_id: bool, optional [default: False]
        Whether to read MC if from `evt.mc_id`. Newer aanet
        processings have `mc_id = evt.frame_index - 1` instead.
    missing: numeric, optional [default: 0]
        Filler for missing values.
    skip_header: bool, optional [default=False]
    correct_mc_times: bool, optional [default=False]
        convert hit times from JTE to MC time
    ignore_hits: bool, optional [default=False]
        If true, don't read our the hits/mchits.
    ignore_run_id_from_header: bool, optional [default=False]
        Ignore run ID from header, take from event instead;
        often, the event.run_id is overwritten with the default (1).
    """
    def configure(self):

        self.filename = self.get('filename')
        self.filenames = self.get('filenames') or []
        self.indices = self.get('indices')
        self.additional = self.get('additional')
        self.format = self.get('aa_fmt') or 'gandalf_new'
        self.use_id = self.get('use_id') or False
        self.use_aart_sps_lib = self.get('use_aart_sps_lib') or False
        self.old_mc_id = self.get('old_mc_id') or False
        self.apply_zed_correction = self.get('apply_zed_correction') or False
        self.skip_header = self.get('skip_header') or False
        self.missing = self.get('missing') or 0
        self.correct_mc_times = self.get('correct_mc_times', default=False)
        self.ignore_hits = bool(self.get('ignore_hits'))
        self.ignore_run_id_from_header = bool(
                self.get('ignore_run_id_from_header'))

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

        import ROOT # noqa
        self.aalib = self.get('aa_lib')
        if self.aalib:
            ROOT.gSystem.Load(self.aalib) # noqa
        elif self.use_aart_sps_lib:
            aapath = '/sps/km3net/users/heijboer/aanet/libaa.so'
            ROOT.gSystem.Load(aapath) # noqa
        else:
            import aa  # noqa

        self.header = None
        self.aanet_header = None
        self.blobs = self.blob_generator()
        self.i = 0

        if self.additional:
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
        from ROOT import EventFile

        found_any_files = False
        for filename in self.filenames:
            log.info("Reading from file: {0}".format(filename))
            if not os.path.exists(filename):
                log.warning(filename + " not available: continue without it")
                continue

            if not found_any_files:
                found_any_files = True

            try:
                event_file = EventFile(filename)
            except Exception:
                raise SystemExit("Could not open file")

            # OLD HEADER CRAZINESS
            # http://jenkins.km3net.de/job/aanet_trunk/doxygen/Head_8hh_source.html
            livetime = 0
            livetime_err = 0
            ngen = 0
            nfilgen = 0

            try:
                log.info("Reading header...")
                self.header = event_file.rootfile().Get("Head")
            except (ValueError, UnicodeEncodeError, TypeError):
                log.warning(filename + ": can't read header.")
            try:
                log.info("Reading livetime...")
                lt_line = self.header.get_line('livetime')
                if len(lt_line) > 0:
                    livetime, livetime_err = lt_line.split()
                # livetime = self.header.get_field('livetime', 0)
                # livetime_err = self.header.get_field('livetime', 1)
                self.livetime_err = float(livetime_err)
                self.livetime = float(livetime)
            except (ValueError, UnicodeEncodeError, AttributeError):
                log.warning(filename + ": can't read livetime.")
                self.livetime = 0
            try:
                log.info("Reading genvol...")
                ngen = self.header.get_field('genvol', 4)
                self.ngen = float(ngen)
            except (ValueError, UnicodeEncodeError, AttributeError):
                log.warning(filename + ": can't read ngen.")
                self.ngen = 0
            try:
                log.info("Reading number of generated files...")
                # nfligen = ?????
                self.nfilgen = float(nfilgen)
            except (ValueError, UnicodeEncodeError):
                log.warning(filename + ": can't read nfilgen.")
                self.nfilgen = 0
            try:
                log.info("Reading run id...")
                self.header_run_id = self.header.get_field('start_run', 0)
            except (ValueError, UnicodeEncodeError, AttributeError):
                log.warning(filename + ": can't read ngen.")
                self.header_run_id = None
            # END OF OLD HEADER CRAZINESS

            # NEW HEADER CRAZINESS
            if not self.skip_header and (self.format != 'ancient_recolns'):
                log.info("Retrieving aanet header...")
                self.aanet_header = get_aanet_header(event_file)

            if self.format == 'ancient_recolns':
                log.info("Generating blobs through old aanet API...")
                while event_file.next():
                    event = event_file.evt
                    blob = self._read_event(event, filename)
                    blob["Header"] = self.aanet_header
                    yield blob
            else:
                log.info("Generating blobs through new aanet API...")
                for event in event_file:
                    blob = self._read_event(event, filename)
                    blob["Header"] = self.aanet_header
                    yield blob
            del event_file

        if not found_any_files:
            msg = "None of the input files available. Exiting."
            try:
                raise FileNotFoundError(msg)
            except NameError:
                # python2
                raise IOError(msg)

    def _read_event(self, event, filename):
        try:
            if self.apply_zed_correction:  # if event.det_id <= 0:  # apply ZED correction      # noqa
                for track in event.mc_trks:
                    track.pos.z += 405.93
        except AttributeError:
            pass

        if len(event.w) == 3:
            w1, w2, w3 = event.w
        elif len(event.w) == 4:
            # what the hell is w4?
            w1, w2, w3, w4 = event.w
        else:
            w1 = w2 = w3 = np.nan

        blob = Blob()
        blob['Evt'] = event
        if self.use_id:
            event_id = event.id
        else:
            event_id = self.i
        self.i += 1
        if self.format != 'ancient_recolns' and not self.ignore_hits:
            try:
                hits = RawHitSeries.from_aanet(event.hits, event_id)
                if self.correct_mc_times:
                    def converter(t):
                        ns = event.t.GetSec() * 1e9 + event.t.GetNanoSec()
                        return t + ns - event.mc_t
                    uconverter = np.frompyfunc(converter, 1, 1)
                    hits._arr["time"] = uconverter(hits.time)
                blob['Hits'] = hits
            except AttributeError:
                log.warning("No hits found.")
            try:
                mc_hits = McHitSeries.from_aanet(event.mc_hits, event_id)
                blob['McHits'] = mc_hits
            except AttributeError:
                log.warning("No MC hits found.")

        blob['McTracks'] = TrackSeries.from_aanet(event.mc_trks, event_id)

        blob['filename'] = filename
        try:
            if self.old_mc_id:
                mc_id = event.mc_id
            else:
                mc_id = event.frame_index - 1
        except AttributeError:
            mc_id = 0

        if not self.ignore_run_id_from_header \
                and self.header_run_id is not None:
            run_id = self.header_run_id
        else:
            run_id = event.run_id
        try:
            ei_data = np.array((
                event.det_id,
                event.frame_index,
                self.livetime,      # livetime_sec
                mc_id,       # mc_id,
                event.mc_t,
                self.ngen,      # n_events_gen
                self.nfilgen,   # n_files_gen
                event.overlays,
                # event.run_id,
                event.trigger_counter,
                event.trigger_mask,
                event.t.GetNanoSec(),
                event.t.GetSec(),
                w1, w2, w3,
                run_id,
                event_id), dtype=EventInfo.dtype)
            blob['EventInfo'] = EventInfo(ei_data)
        except AttributeError:
            ei_data = np.array((0,   # det_id
                               event.frame_index,
                               self.livetime,   # livetime_sec
                               mc_id,   # mc_id
                               0,   # mc_t
                               self.ngen,   # n_events_gen
                               self.nfilgen,   # n_files_gen
                               0,   # overlays
                               0,   # trigger_counter
                               0,   # trigger_mask
                               0,   # nanose
                               0,   # sec
                               w1, w2, w3,
                               run_id,
                               event_id), dtype=EventInfo.dtype)
            blob['EventInfo'] = EventInfo(ei_data)

        if len(event.trks) > 0:
            if self.format == 'minidst':
                recos = read_mini_dst(event, event_id)
                for recname, reco in recos.items():
                    blob[recname] = reco
            if self.format in ('jevt_jgandalf', 'gandalf', 'jgandalf'):
                track, dtype = parse_jevt_jgandalf(
                        event, event_id, self.missing)
                if track:
                    blob['Gandalf'] = KM3Array.from_dict(
                        track, dtype, h5loc='/reco')
            if self.format in ('gandalf_new', 'jgandalf_new'):
                track, dtype = parse_jgandalf_new(
                        event, event_id, self.missing)
                if track:
                    blob['Gandalf'] = KM3Array.from_dict(
                        track, dtype, h5loc='/reco')
            if self.format == 'generic_track':
                track, dtype = parse_generic_event(
                        event, event_id, self.missing)
                if track:
                    blob['Track'] = KM3Array.from_dict(
                        track, dtype, h5loc='/reco')
            if self.format in ('ancient_recolns', 'orca_recolns'):
                track, dtype = parse_ancient_recolns(
                        event, event_id, self.missing)
                if track:
                    blob['OrcaRecoLns'] = KM3Array.from_dict(
                        track, dtype, h5loc='/reco')
        else:
            log.debug("Event #{} does not have any tracks!".format(event_id))
        return blob

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


def get_aanet_header(event_file):
    """Returns a dict of the header entries.

    http://trac.km3net.de/browser/dataformats/aanet/trunk/evt/Head.hh

    """
    header = event_file.header
    desc = ("cut_primary cut_seamuon cut_in cut_nu:Emin Emax cosTmin cosTmax\n"
            "generator physics simul: program version date time\n"
            "seed:program level iseed\n"
            "PM1_type_area:type area TTS\n"
            "PDF:i1 i2\n"
            "model:interaction muon scattering numberOfEnergyBins\n"
            "can:zmin zmax r\n"
            "genvol:zmin zmax r volume numberOfEvents\n"
            "merge:time gain\n"
            "coord_origin:x y z\n"
            "genhencut:gDir Emin\n"
            "k40: rate time\n"
            "norma:primaryFlux numberOfPrimaries\n"
            "livetime:numberOfSeconds errorOfSeconds\n"
            "flux:type key file_1 file_2\n"
            "spectrum:alpha\n"
            "start_run:run_id")
    d = {}
    for line in desc.split("\n"):
        fields, values = [s.split() for s in line.split(':')]
        for field in fields:
            for value in values:
                if field == "physics" and value == "date":  # segfaults
                    continue
                d[field + '_' + value] = header.get_field(field, value)
    return d


def parse_ancient_recolns(aanet_event, event_id, missing=0):
    # the final recontructed track is in f.evt.trks, at the index 5
    # trks[0] ==> prefit track
    # trks[1] ==> m-estimator track
    # trks[2] ==> "original" pdf track
    # trks[3] ==> full pdf track, final reconstructed direction
    # trks[4] ==> + vertex fit and muon length fit
    # trks[5] ==> + bjorken-y fit and neutrino energy fit
    try:
        out = {}
        out['quality'] = aanet_event.trks[3].lik
        out['n_hits_used'] = aanet_event.trks[3].fitinf[0]
        out['energy_muon'] = aanet_event.trks[4].E
        out['energy_neutrino'] = aanet_event.trks[5].E
        out['bjorken_y'] = aanet_event.trks[5].By_rec
        out['pos_x'] = aanet_event.trks[5].pos.x
        out['pos_y'] = aanet_event.trks[5].pos.y
        out['pos_z'] = aanet_event.trks[5].pos.z
        out['dir_x'] = aanet_event.trks[5].dir.x
        out['dir_y'] = aanet_event.trks[5].dir.y
        out['dir_z'] = aanet_event.trks[5].dir.z

        sigma2_theta = aanet_event.trks[3].error_matrix[18] \
            if aanet_event.trks[3].error_matrix[18] > 0 else missing
        out['sigma2_theta'] = sigma2_theta
        sigma2_phi = aanet_event.trks[3].error_matrix[24] \
            if aanet_event.trks[3].error_matrix[24] > 0 else missing
        out['sigma2_phi'] = sigma2_phi
        sin_theta = np.sin(np.arccos(aanet_event.trks[3].dir.z))
        out['sin_theta'] = sin_theta
        out['beta'] = np.sqrt(
            sin_theta * sin_theta * sigma2_phi + sigma2_theta)
    except IndexError:
        keys = {'quality', 'n_hits_used', 'pos_x', 'pos_y', 'pos_z',        # noqa
                'dir_x', 'dir_y', 'dir_z', 'energy_muon', 'energy_neutrino',        # noqa
                'bjorken_y', 'beta', 'sigma2_theta', 'sigma2_phi',      # noqa
                'sin_theta', }    # noqa
        out = {key: missing for key in keys}
    dt = [(key, float) for key in sorted(out.keys())]
    out['event_id'] = event_id
    dt.append(('event_id', '<u4'))
    dt = np.dtype(dt)
    return out, dt


def parse_jevt_jgandalf(aanet_event, event_id, missing=0):
    try:
        track = aanet_event.trks[0]     # this might throw IndexError
        map = {}
        map['id'] = track.id
        map['pos_x'] = track.pos.x
        map['pos_y'] = track.pos.y
        map['pos_z'] = track.pos.z
        map['dir_x'] = track.dir.x
        map['dir_y'] = track.dir.y
        map['dir_z'] = track.dir.z
        map['time'] = track.t
        map['type'] = track.type
        map['rec_type'] = track.rec_type
        map['rec_stage'] = track.rec_stage
        map['beta0'] = track.fitinf[0]
        map['beta1'] = track.fitinf[1]
        map['lik'] = track.fitinf[2]
        map['lik_red'] = track.fitinf[3]
        map['energy'] = track.fitinf[4]
    except IndexError:
        keys = {'id', 'pos_x', 'pos_y', 'pos_z', 'dir_x', 'dir_y', 'dir_z',
                'time', 'type', 'rec_type', 'rec_stage', 'beta0', 'beta1',
                'lik', 'lik_red', 'energy', }
        map = {key: missing for key in keys}
    dt = [(key, float) for key in sorted(map.keys())]
    map['event_id'] = event_id
    dt.append(('event_id', '<u4'))
    dt = np.dtype(dt)
    return map, dt


def parse_jgandalf_new(aanet_event, event_id, missing=0):
    """Read JGandalf (ORCA at least) files from MXtrigger on."""

    def get_track_spread(tracks):
        """Grab metrics from all tracks which pass muon length fit."""
        variates = defaultdict(list)
        out = {}
        for track in tracks:
            if track.fitinf[10] < 0:
                continue
            variates['dir_x'].append(track.dir.x)
            variates['dir_y'].append(track.dir.y)
            variates['dir_z'].append(track.dir.z)
            variates['pos_x'].append(track.pos.x)
            variates['pos_y'].append(track.pos.y)
            variates['pos_z'].append(track.pos.z)
            for k in range(6):
                variates[FITINF_ENUM[k]].append(track.fitinf[k])
        for k, v in variates.items():
            out['spread_' + k + '_std'] = np.std(v)
            out['spread_' + k + '_mean'] = np.mean(v)
            out['spread_' + k + '_median'] = np.median(v)
            out['spread_' + k + '_mad'] = mad(v)
            out['spread_' + k + '_iqr'] = iqr(v)
        return out

    posdir = ['pos_x', 'pos_y', 'pos_z', 'dir_x', 'dir_y', 'dir_z', ]
    metrics = ['std', 'mean', 'median', 'mad', 'iqr']
    spread_keys = [
            'spread_{}_{}'.format(var, metric)
            for var in posdir + FITINF_ENUM[:6]
            for metric in metrics
    ]
    spread_keys.append('upgoing_vs_downgoing')
    all_keys = posdir + spread_keys + FITINF_ENUM + posdir + [
            'time', 'type', 'rec_type', 'rec_stage']
    try:
        track = aanet_event.trks[0]
        outmap = {}
        outmap['pos_x'] = track.pos.x
        outmap['pos_y'] = track.pos.y
        outmap['pos_z'] = track.pos.z
        outmap['dir_x'] = track.dir.x
        outmap['dir_y'] = track.dir.y
        outmap['dir_z'] = track.dir.z
        outmap['time'] = track.t
        outmap['type'] = track.type
        outmap['rec_type'] = track.rec_type
        outmap['rec_stage'] = track.rec_stage
        for i, key in enumerate(FITINF_ENUM):
            outmap[key] = track.fitinf[i]

        spread_tracks = get_track_spread(aanet_event.trks)
        outmap.update(spread_tracks)
        outmap['upgoing_vs_downgoing'] = upgoing_vs_downgoing(aanet_event.trks)
    except IndexError:
        log.warn("Could not read Gandalf track. Zeroing row...")
        outmap = {key: missing for key in all_keys}
    dt = [(key, float) for key in sorted(outmap.keys())]
    outmap['event_id'] = event_id
    dt.append(('event_id', '<u4'))
    dt = np.dtype(dt)
    return outmap, dt


def upgoing_vs_downgoing(tracks):
    """Compare upgoing vs downgoing hypothesis.

    upvsdown = (lik_upgoing - lik_downgoing)/(lik_upgoing + lik_downgoing)
    """
    upgoing = [
            track.fitinf[2]/track.fitinf[3] for track in tracks
            if track.dir.z >= 0
            ]
    downgoing = [
            track.fitinf[2]/track.fitinf[3] for track in tracks
            if track.dir.z < 0
            ]
    if not upgoing:
        return -np.inf
    if not downgoing:
        return np.inf

    upgoing_chi2 = np.nanmin(upgoing)
    downgoing_chi2 = np.nanmin(downgoing)
    out = (upgoing_chi2 - downgoing_chi2) / (upgoing_chi2 + downgoing_chi2)
    return out


def parse_generic_event(aanet_event, event_id):
    map = {}
    try:
        track = aanet_event.trks[0]
    except IndexError:
        # TODO: don't return empty map
        return map, None
    map['id'] = track.id
    map['pos_x'] = track.pos[0]
    map['pos_y'] = track.pos[1]
    map['pos_z'] = track.pos[2]
    map['dir_x'] = track.dir[0]
    map['dir_y'] = track.dir[1]
    map['dir_z'] = track.dir[2]
    map['energy'] = track.E
    map['time'] = track.t
    map['lik'] = track.lik
    map['type'] = track.type
    map['rec_type'] = track.rec_type
    map['rec_stage'] = track.rec_stage
    for k, entry in enumerate(track.fitinf):
        map['fitinf_{}'.format(k)] = entry
    for k, entry in enumerate(track.error_matrix):
        map['error_matrix_{}'.format(k)] = entry
    dt = [(key, float) for key in sorted(map.keys())]
    map['event_id'] = event_id
    dt.append(('event_id', '<u4'))
    dt = np.dtype(dt)
    return map, dt


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
    if len(aanet_event.trks) == 0:
        return minidst
    for k, trk in enumerate(aanet_event.trks):
        recname = pos_to_recname[k]
        reader = recname_to_reader[recname]

        reco_map, dtype = reader(trk)
        minidst[recname] = KM3Array.from_dict(
            reco_map, dtype, h5loc='/reco')

    thomas_map, dtype = parse_thomasfeatures(
        aanet_event.usr, aanet_event.usr_names)
    minidst['ThomasFeatures'] = KM3Array.from_dict(
        thomas_map, dtype, h5loc='/reco')
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


def parse_thomasfeatures(aanet_usr, aanet_usr_names, event_id=0):
    out = {}
    did_converge = len(aanet_usr) > 1

    Thomas_keys = []
    for e in aanet_usr_names:
        Thomas_keys.append(e)

    dtype = [(key, float) for key in Thomas_keys + list(out.keys())]

    out['did_converge'] = did_converge
    dtype.append(('did_converge', bool))
    out['event_id'] = event_id
    dtype.append(('event_id', '<u4'))
    dtype = np.dtype(dtype)

    if not did_converge:
        for key in Thomas_keys:
            out[key] = np.nan
    else:
        for count, key in enumerate(Thomas_keys):
            out[key] = aanet_usr[count]
    return out, dtype


def parse_recolns(aanet_trk, event_id=0):
    out = parse_track(aanet_trk)
    did_converge = aanet_trk.rec_stage > -9999

    recolns_keys = ['beta', 'n_fits', 'Lambda',
                    'n_compatible_solutions', 'Nhits', 'NhitsL0', 'NhitsL1']
    dtype = [(key, float) for key in recolns_keys + list(out.keys())]
    out['did_converge'] = did_converge
    dtype.append(('did_converge', bool))
    out['event_id'] = event_id
    dtype.append(('event_id', '<u4'))
    dtype = np.dtype(dtype)

    if not did_converge:
        for key in recolns_keys:
            out[key] = np.nan
    else:
        for count, key in enumerate(recolns_keys):
            out[key] = aanet_trk.usr[count]
    return out, dtype


def parse_jgandalf(aanet_trk, event_id=0):
    out = parse_track(aanet_trk)
    did_converge = aanet_trk.rec_stage > -9999

    jgandalf_keys = ['Energy_f', 'Energy_can', 'Beta0',
                     'Beta1', 'Lik', 'Lik_reduced', 'NhitsL0', 'NhitsL1']
    dtype = [(key, float) for key in jgandalf_keys + list(out.keys())]
    out['did_converge'] = did_converge
    dtype.append(('did_converge', bool))
    out['event_id'] = event_id
    dtype.append(('event_id', '<u4'))
    dtype = np.dtype(dtype)

    if not did_converge:
        for key in jgandalf_keys:
            out[key] = np.nan
    else:
        for count, key in enumerate(jgandalf_keys):
            out[key] = aanet_trk.usr[count]
    return out, dtype


def parse_aashowerfit(aanet_trk, event_id=0):
    out = parse_track(aanet_trk)
    did_converge = aanet_trk.rec_stage > -9999

    aashow_keys = ['NhitsAA', 'M_estimator', 'beta',
                   'NhitsL0', 'NhitsL1']
    dtype = [(key, float) for key in aashow_keys + list(out.keys())]
    out['did_converge'] = did_converge
    dtype.append(('did_converge', bool))
    out['event_id'] = event_id
    dtype.append(('event_id', '<u4'))
    dtype = np.dtype(dtype)

    if not did_converge:
        for key in aashow_keys:
            out[key] = np.nan
    else:
        for count, key in enumerate(aashow_keys):
            out[key] = aanet_trk.usr[count]
    return out, dtype


def parse_qstrategy(aanet_trk, event_id=0):
    out = parse_track(aanet_trk)
    did_converge = aanet_trk.rec_stage > -9999

    qstrat_keys = sorted(['MFinal', 'Charge', 'MPreFit',
                          'RPreFit', 'Inertia', 'NhitsL0', 'NhitsL1'])
    dtype = [(key, float) for key in qstrat_keys + list(out.keys())]
    out['did_converge'] = did_converge
    dtype.append(('did_converge', bool))
    out['event_id'] = event_id
    dtype.append(('event_id', '<u4'))
    dtype = np.dtype(dtype)

    if not did_converge:
        for key in qstrat_keys:
            out[key] = np.nan
    else:
        for count, key in enumerate(qstrat_keys):
            out[key] = aanet_trk.usr[count]
    return out, dtype


def parse_dusj(aanet_trk, event_id=0, missing=0):
    out = parse_track(aanet_trk)
    did_converge = aanet_trk.rec_stage > -9999

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
    dtype = [(key, float) for key in dusj_keys + list(out.keys())]
    out['did_converge'] = did_converge
    dtype.append(('did_converge', bool))
    out['event_id'] = event_id
    dtype.append(('event_id', '<u4'))
    dtype = np.dtype(dtype)

    if not did_converge:
        for key in dusj_keys:
            out[key] = missing
    else:
        for count, key in enumerate(dusj_keys):
            out[key] = aanet_trk.usr[count]
    return out, dtype
