#!/usr/bin/env python
"""
Pump for the Aanet data format.

This is undoubtedly the ugliest module in the entire framework.
If you have a way to read aanet files via the Jpp interface,
your pull request is more than welcome!
"""
from collections import defaultdict
import os.path

import numpy as np

from km3pipe.core import Pump, Blob
from km3pipe.dataclasses import Table
from km3pipe.logger import get_logger

log = get_logger(__name__)  # pylint: disable=C0103

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


class HeaderParser():
    example = {
        'XSecFile': ' /afs/in2p3.fr/home/throng/km3net/src/gSeaGen/v4r1/dat/gxspl-seawater.xml',
        'can': ' -117.2 139.5 205.402',
        'coord_origin': ' 0 0 0 ',
        'cut_in': ' 0 0 0 0',
        'cut_nu': ' 3 100 -1 1',
        'cut_primary': ' 0 0 0 0',
        'cut_seamuon': ' 0 0 0 0',
        'depth': '2475.000',
        'detector': ' /afs/in2p3.fr/throng/km3net/detectors/orca_115strings_av23min20mhorizontal_18OMs_alt9mvertical_v1.det',
        'drawing': 'surface',
        'end_event': '',
        'fgIsA': '',
        'flux': '14 FUNC 1E-9*pow(x,-2)',
        'flux_1': '-14 FUNC 1E-9*pow(x,-2)',
        'genhencut': ' 0 0',
        'genvol': ' 180.91 662.486 611.189 1.48e+09 9.1e+07',
        'livetime': ' 0 0',
        'muon_desc_file': ' ',
        'norma': ' 0 0',
        'physics': ' gSeaGen 4.1 160729 120347',
        'physics_1': 'GENIE 2.10.2 160729  120347',
        'prop_code': 'PropaMuon',
        'seed': 'gSeaGen 1200002',
        'simul': '    ',
        'source_mode': 'DIFFUSE',
        'spectrum': ' -3',
        'start_run': ' 2',
        'target': ' ',
        'tgen': '31556926.000000'
    }


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
    apply_zed_correction: bool, optional [default=False]
        correct ~400m offset in mc tracks.
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

        self.filename = self.require('filename')
        self.header = None
        self.aanet_header = None
        self.blobs = self.blob_generator()
        self.i = 0

    def get_blob(self, index):
        NotImplementedError("Aanet currently does not support indexing.")

    def blob_generator(self):
        """Create a blob generator."""
        # pylint: disable:F0401,W0612
        import aa
        from ROOT import EventFile

        filename = self.filename
        log.info("Reading from file: {0}".format(filename))
        if not os.path.exists(filename):
            log.warning(filename + " not available: continue without it")

        try:
            event_file = EventFile(filename)
        except Exception:
            raise SystemExit("Could not open file")

        log.info("Generating blobs through new aanet API...")
        for event in event_file:
            log.debug('Reading event...')
            blob = self._read_event(event, filename)
            log.debug('Reading header...')
            blob["Header"] = self.aanet_header
            yield blob
        del event_file

    def _parse_wgts(self, w):
        if len(w) == 3:
            w1, w2, w3 = w
            w4 = np.nan
        elif len(w) == 4:
            # what the hell is w4?
            w1, w2, w3, w4 = w
        else:
            w1 = w2 = w3 = w4 = np.nan
        return w1, w2, w3, w4

    def _parse_mctracks(self):
        raise NotImplementedError

    def _parse_mchits(self):
        raise NotImplementedError

    def _parse_hits(self):
        raise NotImplementedError

    def _parse_eventinto(self):
        raise NotImplementedError

    def get_sun_id(self):
        raise NotImplementedError
        # run_id = self.header_run_id
        # else:
        #     run_id = event.run_id
        # if run_id == '':
        #     run_id = -1

    def _read_event(self, event, filename):
        blob = Blob()
        group_id = self.i
        self.i += 1
        w1, w2, w3, w4 = self._parse_wgts(self, event.w)
        hits = self._parse_hits(event.hits, group_id)
        mchits = self._parse_mchits(event.mchits, group_id)
        mctracks = self._parse_mctracks(event.mc_trks, group_id)
        mc_id = event.frame_index - 1
        run_id = self._get_run_id()
        eventinfo = self._parse_eventinfo(evt)
        trks = self._parse_tracks(event)

    def event_index(self, blob):
        if self.id:
            return blob["Evt"].id
        else:
            return blob["Evt"].frame_index

    def process(self, blob=None):
        return next(self.blobs)

    def __iter__(self):
        return self

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
