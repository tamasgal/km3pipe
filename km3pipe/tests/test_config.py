# coding=utf-8
# Filename: test_config.py
"""
Test suite for configuration related functions and classes.

"""
from __future__ import division, absolute_import, print_function

from km3pipe.testing import TestCase, StringIO
from km3pipe.config import Config


CONFIGURATION = StringIO("\n".join((
    "[DB]",
    "username=foo",
    "password=narf",
    )))


class TestConfig(TestCase):
    def test_init(self):
        config = Config(None)  # noqa
        config._read_configuration()

    def test_whatever(self):
        pass
