# Filename: config.py
# pylint: disable=locally-disabled
"""
Tools for global configuration.

"""
from __future__ import absolute_import, print_function, division

import os
import pytz
from configparser import ConfigParser, Error, NoOptionError, NoSectionError
import getpass

from .logger import get_logger

try:
    input = raw_input
except NameError:
    pass

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal and Moritz Lotze"
__email__ = "tgal@km3net.de"
__status__ = "Development"

log = get_logger(__name__)    # pylint: disable=C0103

CONFIG_PATH = os.path.expanduser('~/.km3net')


class Config(object):
    def __init__(self, config_path=CONFIG_PATH):
        """Configuration manager for KM3NeT stuff"""
        self.config = ConfigParser()
        self._time_zone = None
        self._config_path = config_path

        if config_path is not None:
            self._init_from_path(config_path)

    def _init_from_path(self, path):
        if not os.path.exists(path):
            log.info("No configuration found at '{0}'".format(path))
            return
        self._read_from_path(path)

    def _read_from_path(self, path):
        """Read configuration from file path"""
        with open(path) as config_file:
            self._read_from_file(config_file)

    def _read_from_file(self, file_obj):
        self.config.read_file(file_obj)

    def set(self, section, key, value):
        if section not in self.config.sections():
            self.config.add_section(section)
        self.config.set(section, key, value)
        with open(self._config_path, 'w') as f:
            self.config.write(f)

    def get(self, section, key, default=None):
        try:
            value = self.config.get(section, key)
            try:
                return float(value)
            except ValueError:
                return value
        except (NoOptionError, NoSectionError):
            return default

    def create_irods_session(self):
        try:
            from irods.session import iRODSSession
        except ImportError:
            log.error(
                "Please install the iRODS Python client:\n\n"
                "    pip install git+git://github.com/irods/"
                "python-irodsclient.git\n"
            )
            return
        try:
            host = self.config.get('iRODS', 'host')
            port = self.config.get('iRODS', 'port')
            user = self.config.get('iRODS', 'user')
            zone = self.config.get('iRODS', 'zone')
        except Error:
            log.error("iRODS connection details missing from ~/.km3net")
            return

        try:
            password = self.config.get('iRODS', 'password')
        except Error:
            password = input("Please enter your iRODS password: ")

        return iRODSSession(
            host=host, port=port, user=user, password=password, zone=zone
        )

    @property
    def db_credentials(self):
        """Return username and password for the KM3NeT WebDB."""
        try:
            username = self.config.get('DB', 'username')
            password = self.config.get('DB', 'password')
        except Error:
            username = input("Please enter your KM3NeT DB username: ")
            password = getpass.getpass("Password: ")
        return username, password

    @property
    def db_session_cookie(self):
        try:
            return self.config.get('DB', 'session_cookie')
        except Error:
            return None

    @property
    def db_url(self):
        try:
            return self.config.get('DB', 'url')
        except Error:
            return None

    @property
    def slack_token(self):
        """Return slack token for chat bots."""
        try:
            token = self.config.get('Slack', 'token')
        except Error:
            raise ValueError("No Slack token defined in configuration file.")
        else:
            return token

    @property
    def rba_url(self):
        """Return the RainbowAlga URL."""
        try:
            url = self.config.get('RainbowAlga', 'url')
        except Error:
            return None
        else:
            return url

    @property
    def check_for_updates(self):
        try:
            return self.config.getboolean('General', 'check_for_updates')
        except Error:
            return True

    @property
    def time_zone(self):
        if not self._time_zone:
            try:
                time_zone = self.config.get('General', 'time_zone')
            except Error:
                time_zone = 'UTC'
            self._time_zone = pytz.timezone(time_zone)
        return self._time_zone
