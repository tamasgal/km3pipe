from km3pipe import Module


class HitCounter(Module):
    def process(self, blob):
        print("Number of hits: " + str(len(blob['hits'])))
        return blob

class FirstHit(Module):
    def process(self, blob):
        hits = blob['hits']
        print("First hit: " + str(hits[0]))
        print("Second hit: " + str(hits[1]))
        return blob

class PrintTrks(Module):
    def process(self, blob):
        trks = blob['mc_trks']
        for trk in trks:
            print(trk)
        return blob
