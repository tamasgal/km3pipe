import km3pipe as kp
import numpy as np
import pandas as pd


class PrimFitter(kp.Module):
    def process(self, blob):
        out = {}

        # has detector applied already
        hits = pd.DataFrame.from_records(blob['Hits'].__array__())
        dus = np.unique(hits['dom_id'])
        n_dus = len(dus)

        if n_dus < 8:
            return

        fh = hits.drop_duplicates(subset='dom_id')
        print(fh.columns)

        data = fh['pos_x', 'pos_y', 'pos_z']
        center = data.mean(axis=0)

        _, _, v = np.linalg.svd(data - center)
        reco_dir = v[0]
        reco_pos = center
        out['direction'] = reco_dir
        out['position'] = reco_pos

        # muon = blob['MCTracks'].highest_energetic_muon
        # blob['Muon'] = muon
        # print("Incoming muon:      {0}".format(muon))
        # reco_muon = kp.dataclasses.Track(reco_dir, 0, 0, reco_pos, 0, 0)
        # print("Reconstructed muon: {0}".format(reco_muon))
        # angular_diff = 180 * (angle_between(muon.dir, reco_muon.dir) / np.pi)
        # print("Angular difference: {0}".format(angular_diff))

        blob['PrimFitter'] = out
        return blob
