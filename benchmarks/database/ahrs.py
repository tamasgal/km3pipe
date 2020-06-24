#!/usr/bin/env python3

import km3pipe as kp

assert '3.4.3.4/AHRS/1.551' == kp.db.clbupi2ahrsupi('3.4.3.2/V2-2-1/2.551')
assert '3.4.3.4/AHRS/1.76' == kp.db.clbupi2ahrsupi('3.4.3.2/V2-2-1/2.76')
