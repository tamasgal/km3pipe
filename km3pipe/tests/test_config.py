# Filename: test_config.py
"""
Test suite for configuration related functions and classes.

"""

from km3pipe.testing import TestCase, StringIO
from km3pipe.config import Config

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"


CONFIGURATION = StringIO("\n".join((
    "[DB]",
    "username=foo",
    "password=narf",
    "timeout=10",
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

    def test_get_retrieves_correct_value(self):
        self.assertEqual("foo", self.config.get("DB", "username"))

    def test_get_returns_none_if_section_not_found(self):
        self.assertTrue(self.config.get("a", "b") is None)

    def test_get_returns_none_if_option_not_found(self):
        self.assertTrue(self.config.get("DB", "a") is None)

    def test_get_returns_default_if_option_not_found(self):
        self.assertEqual('b', self.config.get("DB", "a", default='b'))

    def test_get_returns_float_if_option_is_numberlike(self):
        self.assertTrue(isinstance(self.config.get("DB", "timeout"), float))

    def test_create_irods_session_returns_none_if_irods_module_missing(self):
        session = self.config.create_irods_session()
        self.assertTrue(session is None)
