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
    def setUp(self):
        self.config = Config(None)
        self.config._read_from_file(CONFIGURATION)
        CONFIGURATION.seek(0, 0)

    def test_db_credentials(self):
        self.assertEqual('foo', self.config.db_credentials[0])
        self.assertEqual('narf', self.config.db_credentials[1])

    def test_check_for_updates_defaults_to_true(self):
        self.assertTrue(self.config.check_for_updates)

    def test_time_zone_defaults_to_utc(self):
        self.assertEqual('UTC', self.config.time_zone._tzname)

    def test_slack_token_raises_error_by_default(self):
        with self.assertRaises(ValueError):
            self.config.slack_token
