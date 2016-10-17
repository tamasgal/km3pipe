# Read Files for PID

```python
#!/usr/bin/env python

from km3pipe.io import H5Chain

n_events = {
    'numuon_cc.h5': 240000,
    'nuelec_cc.h5': 120000,
    'nuelec_nc.h5': 120000,
    'atmuon.h5': 240000,
}
files = list(n_events.keys())
with H5Chain(files) as c:
    reco = c(n_events)['/reco']
    mc_tracks = c(n_events)['/mc_tracks']

# print first 5 events
print(reco[:5])
```

