import km3pipe as kp
import numpy as np

class PrimFitter(kp.Module):
    def process(self, blob):
        out = {}

        # has detector applied already
        du_hits = blob['Hits']
        dus = du_hits.keys()
        n_dus = len(dus)

        if n_dus < 8:
            return

        first_hits = [hits[0] for hits in du_hits.values()]

        data = np.array([hit.pos for hit in first_hits])
        center = data.mean(axis=0)

        _, _, v = np.linalg.svd(data - center)
        reco_dir = v[0]
        reco_pos = center
        out['direction'] = reco_dir
        out['position'] = reco_pos

        #muon = blob['MCTracks'].highest_energetic_muon
        #blob['Muon'] = muon
        #print("Incoming muon:      {0}".format(muon))
        #reco_muon = kp.dataclasses.Track(reco_dir, 0, 0, reco_pos, 0, 0)
        #print("Reconstructed muon: {0}".format(reco_muon))
        #angular_diff = 180 * (angle_between(muon.dir, reco_muon.dir) / np.pi)
        #print("Angular difference: {0}".format(angular_diff))

        if not blob['Reco']:
            blob['Reco'] = {}
        blob['Reco']['PrimFitter'] = out
        return blob
