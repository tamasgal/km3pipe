#!/usr/bin/env python
"""
Example to demonstrate how to modify log levels of KM3Pipe modules.

"""
import km3pipe as kp

core_log = kp.logger.get_logger("km3pipe.core")
core_log.setLevel("DEBUG")


def foo(blob):
    print("Module called")
    return blob


pipe = kp.Pipeline()
pipe.attach(foo)
pipe.drain(5)
