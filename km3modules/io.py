#!/usr/bin/env python3
import km3pipe as kp


class HitsTabulator(kp.Module):
    """
    Create `kp.Table` from hits provided by `km3io`.

    Parameters
    ----------
    kind: str (default="offline")
      The kind of hits to tabulate:
        "offline": the hits in an offline file
        "online": snapshot and triggered hits (will be combined)
        "mc": MC hits
    split_hits: bool (default: True)
      Defines whether the hits should be split up into individual arrays
      in a single group (e.g. hits/dom_id, hits/channel_id) or stored
      as a single HDF5Compound array (e.g. hits).
    """
    def configure(self):
        self.kind = self.get("kind", default="offline")
        self.split_hits = self.get("split_hits", default=True)

    def process(self, blob):
        self.cprint(blob)
        if self.kind == "offline":
            hits = blob['event'].hits
            blob["Hits"] = kp.Table(
                {
                    "channel_id": hits.channel_id,
                    "dom_id": hits.dom_id,
                    "time": hits.t,
                    "tot": hits.tot,
                    "triggered": hits.trig,
                },
                h5loc="/hits",
                split_h5=True,
                name="Hits",
            )
        return blob
