import km3pipe as kp
import numpy as np
import pandas as pd     # noqa
from km3pipe.dataclasses import ArrayTaco


class PrimFitter(kp.Module):
    """Primitive Linefit using Singular Value Decomposition.

    Parameters
    ----------
    hit_sel: str, default='Hits'
        Blob key of the hits to run the fit on.
        This assumes the key exists. Ensure this via::

            >>> pipe.attach(Module, only_if='MyBlobKey')
    """
    def __init__(self, **kwargs):
        super(self.__class__, self).__init__(**kwargs)
        self.hit_sel = self.get('hit_sel') or 'Hits'

    def process(self, blob):
        out = {}

        # has detector applied already
        hits = blob[self.hit_sel].serialise(to='pandas')
        dus = np.unique(hits['dom_id'])
        n_dus = len(dus)

        if n_dus < 8:
            return

        # moved to HitSelector module
        # fh = hits.drop_duplicates(subset='dom_id')
        # fh = hits.first_hits

        pos = fh[['pos_x', 'pos_y', 'pos_z']]
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
        }

        blob['PrimFitter'] = ArrayTaco.from_dict(out, h5loc='/reco')
        return blob
