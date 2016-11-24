#!/usr/bin/env python3
# vim:set ts=4 sts=4 sw=4 et:
"""Simple reconstruction algorithms.

This module defines a base class, ``Reconstruction``.
"""
from __future__ import division, absolute_import, print_function

import numpy as np
import pandas as pd     # noqa
from scipy.linalg import eigvals
from scipy.stats import (
    chisquare, kurtosis, skew, entropy, iqr, mode, kurtosistest, normaltest,
    skewtest, tmean, tvar, tstd,
)
from statsmodels.robust.scale import mad

from km3pipe import Module
from km3pipe.dataclasses import KM3Array, KM3DataFrame     # noqa
from km3pipe.tools import azimuth, zenith, dist, unit_vector


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
    """Tensor of Inertia.

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


class Reconstruction(Module):
    """Reconstruction base class.

    Parameters
    ----------
    hit_sel: str, default='Hits'
        Blob key of the hits to run the fit on.
    key_out: str, default='GenericReco'
        Key to write into.
    """

    def __init__(self, **kwargs):
        super(Reconstruction, self).__init__(**kwargs)
        self.hit_sel = self.get('hit_sel') or 'Hits'
        self.key_out = self.get('key_out') or 'GenericReco'

    def process(self, blob):
        hits = blob[self.hit_sel].serialise(to='pandas')
        reco = self.fit(hits)
        if reco is not None:
            blob[self.key_out] = KM3DataFrame.deserialise(
                reco, h5loc='/reco', fmt='dict')
        return blob

    def fit(self, hits):
        return


class SvdFit(Reconstruction):
    """Primitive Linefit using Singular Value Decomposition.

    Parameters
    ----------
    n_doms_min: int, default=8
        Minimum number of Doms with hits.  Events with fewer will not
        be reconstructed.
    hit_sel: str, default='Hits'
        Blob key of the hits to run the fit on.
    key_out: str, default='SvdFit'
        Key to write into.

    Returns
    -------
    out: None (fit failed), or KM3DataFrame
        The km3df keys are: ['pos_x,y,z', 'dir_x,y,z', 'phi', 'theta'].
    """
    def __init__(self, **kwargs):
        super(SvdFit, self).__init__(**kwargs)
        self.hit_sel = self.get('hit_sel') or 'Hits'
        self.key_out = self.get('key_out') or 'SvdFit'
        self.n_doms_min = self.get('n_doms_min') or 8
        self.n_doms_min = int(self.n_doms_min)

    def fit(self, hits):
        dus = np.unique(hits['dom_id'])
        n_dus = len(dus)

        if n_dus < 8:
            return

        pos = hits[['pos_x', 'pos_y', 'pos_z']]
        center = pos.mean(axis=0)

        _, _, v = np.linalg.svd(pos - center)

        reco_dir = np.array(v[0])
        reco_pos = center
        len = dist(pos.iloc[-1], pos.iloc[0])
        dur = hits.time.iloc[-1] - hits.time.iloc[0]
        velo = len/dur

        out = {
            'pos_x': reco_pos['pos_x'],
            'pos_y': reco_pos['pos_y'],
            'pos_z': reco_pos['pos_z'],
            'dir_x': reco_dir[0],
            'dir_y': reco_dir[1],
            'dir_z': reco_dir[2],
            'phi': azimuth(reco_dir),
            'theta': zenith(reco_dir),
            'speed': velo,
        }
        return out


class LineFit(Reconstruction):
    """Fit a linear track, with direction and velocity.

    Parameters
    ----------
    hit_sel: str, default='Hits'
        Blob key of the hits to run the fit on.
    key_out: str, default='LineFit'
        Key to write into.

    Returns
    -------
    out: None (fit failed), or KM3DataFrame
        The km3df keys are: ['pos_x,y,z', 'dir_x,y,z', 'phi', 'theta', '
        speed'].
    """
    def __init__(self, **kwargs):
        super(LineFit, self).__init__(**kwargs)
        self.hit_sel = self.get('hit_sel') or 'Hits'
        self.key_out = self.get('key_out') or 'LineFit'

    def fit(self, hits):
        pos = hits[['pos_x', 'pos_y', 'pos_z']]

        reco_dir = unit_vector(pos.iloc[-1] - pos.iloc[0])

        out = {
            'dir_x': reco_dir[0],
            'dir_y': reco_dir[1],
            'dir_z': reco_dir[2],
            'phi': azimuth(reco_dir),
            'theta': zenith(reco_dir),
            'speed': velo,
        }
        return out


class Trawler(Reconstruction):
    """Compute summary statistics on hits.
    """
    def __init__(self, **kwargs):
        super(Trawler, self).__init__(**kwargs)
        self.hit_sel = self.get('hit_sel') or 'Hits'
        self.key_out = self.get('key_out') or 'Trawler'

    def fit(self, hits):
        out = {}

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
        for deno in ('tot_sum', 'n_hits', 'tot_max'):
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
