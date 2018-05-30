#!/usr/bin/env python
# pylint: disable=locally-disabled,C0111,R0904,C0301,C0103,W0212

import sys

from km3pipe import Pipeline
from km3pipe.io import AanetPump
from km3modules.common import Dump

for mod in ['aa', 'ROOT']:
    if mod in sys.modules:
        del sys.modules[mod]

import aa  # noqa


fname = 'small.root'

p = Pipeline()
p.attach(AanetPump, filename=fname)
p.attach(Dump)
p.drain()
