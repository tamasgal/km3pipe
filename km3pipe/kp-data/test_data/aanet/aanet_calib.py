#!/usr/bin/env python
# pylint: disable=locally-disabled,C0111,R0904,C0301,C0103,W0212

import sys

for mod in ['aa', 'ROOT']:
    if mod in sys.modules:
        del sys.modules[mod]

import aa    # noqa
from ROOT import Det, EventFile, TH2D

det = Det("KM3NeT_00000007_02122015_zest_DR_PMT.detx")
EventFile.read_timeslices = True
f = EventFile('small.root')
P = 50000    # 50 micro-seconds
h2 = TH2D("h2", "h2", 200, 0, P, 60, 0, 800)
for i, evt in enumerate(f):
    if i > 100:
        break
    if not i % 20:
        print(f.index, f.evt.hits.size())
    if f.index < 1000:
        continue
    det.apply(evt)

    for h in evt.hits:
        h2.Fill(h.t % P, h.pos.z)
    if f.index == 110:
        break
h2.Draw("colz")
