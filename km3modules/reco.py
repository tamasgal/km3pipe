import km3pipe as kp
import numpy as np
import pandas as pd     # noqa
from km3pipe.dataclasses import ArrayTaco


class PrimFitter(kp.Module):
    def process(self, blob):
        out = {}

        # has detector applied already
        hits = blob['Hits'].serialise(to='pandas')
        dus = np.unique(hits['dom_id'])
        n_dus = len(dus)

        if n_dus < 8:
            return

        fh = hits.drop_duplicates(subset='dom_id')

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

        # muon = blob['MCTracks'].highest_energetic_muon
        # blob['Muon'] = muon
        # print("Incoming muon:      {0}".format(muon))
        # reco_muon = kp.dataclasses.Track(reco_dir, 0, 0, reco_pos, 0, 0)
        # print("Reconstructed muon: {0}".format(reco_muon))
        # angular_diff = 180 * (angle_between(muon.dir, reco_muon.dir) / np.pi)
        # print("Angular difference: {0}".format(angular_diff))

        blob['PrimFitter'] = ArrayTaco.from_dict(out, h5loc='/reco')
        return blob
