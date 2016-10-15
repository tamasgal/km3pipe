# Readout with Pandas

```python
In [1]: import pandas as pd

In [2]: h5 = pd.HDFStore('aagandalf.JTE.KM3Sim.gseagen.elec-CC.1-5GeV-2.7E5-1bin-1.0gspec.ORCA115_9m_2016.1.root.h5')

In [3]: h5['/reco/j_evt_j_gandalf'][:2]
Out[3]:
      beta0     beta1     dir_x     dir_y     dir_z  energy   id        lik  \
0  0.007782  0.005475 -0.253622  0.652733 -0.713874     0.0  1.0 -27.362383
1  0.008053  0.005336 -0.652266  0.740139  0.163536     0.0  1.0 -20.944176

    lik_red      pos_x      pos_y       pos_z  rec_stage  rec_type  \
0 -0.346359 -10.075008 -53.669576   75.250049     1001.0    1001.0
1 -0.207368  10.115741  51.891898  142.377145     1001.0    1001.0

           time  type  event_id
0  4.999969e+07   0.0         1
1  4.999837e+07   0.0         4
```

