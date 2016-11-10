"""Simple reconstruction algorithms.

This module defines a base class, ``Reconstruction``.
"""
import numpy as np
import pandas as pd     # noqa

from km3pipe.dataclasses import ArrayTaco, KM3DataFrame     # noqa
from km3pipe.tools import azimuth, zenith
from km3pipe import Module


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
        self.n_doms_min = int(self.get('n_doms_min')) or 8

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
        out = {
            'pos_x': reco_pos['pos_x'],
            'pos_y': reco_pos['pos_y'],
            'pos_z': reco_pos['pos_z'],
            'dir_x': reco_dir[0],
            'dir_y': reco_dir[1],
            'dir_z': reco_dir[2],
            'phi': azimuth(reco_dir),
            'theta': zenith(reco_dir),
        }
        return out


class LineFit(Reconstruction):
    """Fit a linear track, with direction and velocity.

    Inspired by ``icecube.linefit``.

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
        dus = np.unique(hits['dom_id'])
        n_dus = len(dus)

        if n_dus < 8:
            return

        pos = hits[['pos_x', 'pos_y', 'pos_z']]
        center = pos.mean(axis=0)

        _, _, v = np.linalg.svd(pos - center)

        reco_dir = np.array(v[0])
        reco_pos = center
        out = {
            'pos_x': reco_pos['pos_x'],
            'pos_y': reco_pos['pos_y'],
            'pos_z': reco_pos['pos_z'],
            'dir_x': reco_dir[0],
            'dir_y': reco_dir[1],
            'dir_z': reco_dir[2],
            'phi': azimuth(reco_dir),
            'theta': zenith(reco_dir),
            'speed': None,
        }
        return out
