#!/usr/bin/env python3

import km3pipe as kp

assert '3.4.3.4/AHRS/1.551'    == kp.db.clbupi2compassupi('3.4.3.2/V2-2-1/2.551')
assert '3.4.3.4/AHRS/1.76'     == kp.db.clbupi2compassupi('3.4.3.2/V2-2-1/2.76')
assert '3.4.3.4/LSM303/3.1106' == kp.db.clbupi2compassupi('3.4.3.2/V2-2-1/3.1013')
assert '3.4.3.4/LSM303/3.948'  == kp.db.clbupi2compassupi('3.4.3.2/V2-2-1/3.855')


