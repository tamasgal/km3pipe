#!/usr/bin/env python3
# vim:set ts=4 sts=4 sw=4 et:
"""Feature Extractors.

TODO Clean this up & move funcs to main tools lib.
"""
from __future__ import division, absolute_import, print_function

import numpy as np
import pandas as pd
from scipy.linalg import eigvals
from scipy.stats import (
    chisquare, kurtosis, skew, entropy, iqr, mode, kurtosistest, normaltest,
    skewtest, tmean, tvar, tstd,
)
from statsmodels.robust.scale import mad

from km3pipe import Module
from km3pipe.io import map2df
from km3pipe.dataclasses import ArrayTaco


def bimod(skew, kurt, n, is_fisher=True):
    """Test a distribution for bimodality."""
    if not is_fisher:
        kurt = kurt - 3
    return np.divide(
        np.square(skew) + 1,
        kurt + np.divide(3 * np.square(n - 1), (n - 2) * (n - 3))
    )


def uniform_chi2(x):
    """Test a distribution for uniformity."""
    chi2, _ = chisquare(x, f_exp=None)
    return chi2


def idr(x, perc=10):
    up, lo = np.percentile(x, [100-perc, perc])
    return up - lo


def tensor_of_inertia(self, pos_x, pos_y, pos_z, weight=1):
    """Tensor of Intertia.

    Adapted from Thomas Heid' EventID (ROOT implementation).
    """
    toi = np.zeros((3, 3), dtype=float)
    toi[0][0] += np.square(pos_y) * np.square(pos_z)
    toi[0][1] += (-1) * pos_x * pos_y
    toi[0][2] += (-1) * pos_x * pos_z
    toi[1][0] += (-1) * pos_x * pos_y
    toi[1][1] += np.squar(pos_x) + np.square(pos_z)
    toi[1][2] += (-1) * pos_y * pos_z
    toi[2][0] += (-1) * pos_x * pos_z
    toi[2][1] += (-1) * pos_z * pos_y
    toi[2][2] += np.square(pos_x) + np.square(pos_y)
    toi *= weight
    return eigvals(toi)


class TrackMuxer(Module):
    """Compare outputs of different fits.

    Like, angular difference between 2 fit directions.

    TODO, WORK IN PROGRESS
    """
    __name__ = 'TrackMuxer'

    def __init__(self, **kwargs):
        super(self.__class__, self).__init__(**kwargs)
        self.tracks = self.get('tracks') or []

    def process(self, blob):
        blob[self.__name__] = self.__call__(blob)
        return blob

    def __call__(self, blob):
        out = {}
        for track in self.tracks:
            out[track] = self.mux(blob[track])
        return out

    def mux(self, track):
        return None

    def fit_trk(self, trk, topo_dict=None, prefix=''):
        if topo_dict is None:
            topo_dict = {}
        if prefix is not '':
            prefix = prefix + '_'
        azimuth, zenith = self._make_angles(trk)
        topo_dict[prefix + 'zenith'] = zenith
        topo_dict[prefix + 'azimuth'] = azimuth
        for metr in ['var', 'iqr', 'idr']:
            topo_dict[prefix + 'zenith_times_r_' + metr] = \
                zenith * topo_dict['r_' + metr]
        return topo_dict

    def _azimuth(self, trk):
        return np.arctan2(
            trk['dir_x'],
            trk['dir_y']
        )

    def _zenith(self, trk):
        return np.arccos(trk['dir_z'])

    # time residuals compared to vertex
    # space residuals


class TrawlerMod(Module):
    """Compute summary statistics on hits.

    Wrapper Module.

    TODO Merge this.
    """
    __name__ = 'Trawler'

    def __init__(self, **kwargs):
        super(self.__class__, self).__init__(**kwargs)
        self.trawler = Trawler()

    def process(self, blob):
        map = self.trawler(blob['Hits'])
        # blob[self.__name__] = ArrayTaco.from_dict(map)
        blob[self.__name__] = map2df(map)
        return blob


class Trawler():
    """Compute summary statistics on hits.

    The hits need to have a Geometry applied, i.e. have `pos` etc fields.
    """
    def __call__(self, hits):
        out = {}

        # time has arbitrary offset, so center it.
        hits['time'] -= np.median(hits['time'])

        n_hits = len(hits)
        out['n_hits'] = n_hits
        r2 = np.sqrt(
            np.square(hits.pos_x) +
            np.square(hits.pos_y)
        )
        r3 = np.sqrt(
            np.square(hits.pos_x) +
            np.square(hits.pos_y) +
            np.square(hits.pos_z)
        )
        z = hits.pos_z
        out['tot_sum'] = np.sum(hits.tot)
        out['tot_max'] = np.max(hits.tot)
        out['tot_max_over_nhits'] = out['tot_max'] / n_hits
        out['tot_max_rel'] = np.max(hits.tot) / np.sum(hits.tot)
        out['tot_sum_over_nhits'] = out['tot_sum'] / n_hits
        out['tot_sum_over_max'] = out['tot_sum'] / out['tot_max']
        out['z_totmax'] = hits.pos_z[np.argmax(hits.tot)]

        out.update(self.stats(z, 'z'))
        out.update(self.stats(r2, 'r2'))
        out.update(self.stats(r3, 'r3'))
        out.update(self.stats(hits.time, 'time'))

        # first pulses / truncated: variance etc

        # time windows (n_win = 3)
        for deno in ('tot_sum', 'n_hits', 'time_iqr', 'time_idr', 'time_var'):
            for meas in ('var', 'cog', 'idr', 'iqr'):
                for nume in ('r2', 'r3', 'z'):
                    feat = nume + '_' + meas + '_over_' + deno
                    out[feat] = out[nume + '_' + meas] / out[deno]
        return out

    @classmethod
    def stats(cls, x, prefix='', wgt=None):
        if wgt is None:
            wgt = np.ones_like(x)
        n_hits = len(x)
        o = {}
        o['med'] = np.median(x)
        o['mean'] = np.average(x)
        o['var'] = np.var(x)
        o['skew'] = skew(x)
        o['kurt'] = kurtosis(x, fisher=True)
        o['mad'] = mad(x)
        o['sum'] = np.sum(x)
        o['iqr'] = iqr(x)
        o['idr'] = idr(x)
        o['uni_chi2'] = uniform_chi2(x)
        o['bimod'] = bimod(o['skew'], o['kurt'], n_hits)
        o['entropy'] = entropy(x)
        o['cog'] = np.average(x, weights=wgt)
        o = {prefix + '_' + key: val for key, val in o.items()}     # noqa
        return o
