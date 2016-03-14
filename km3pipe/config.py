# coding=utf-8
# Filename: config.py
# pylint: disable=locally-disabled
"""
Tools for global configuration.

"""
from __future__ import division, absolute_import, print_function

import os
import stat
import pytz
try:
    from configparser import ConfigParser, Error
except ImportError:
    from six.moves.configparser import ConfigParser, Error

import getpass
try:
    input = raw_input
except NameError:
    pass

from km3pipe.logger import logging

__author__ = 'tamasgal'

log = logging.getLogger(__name__)  # pylint: disable=C0103

CONFIG_PATH = os.path.expanduser('~/.km3net')


class Config(object):
    def __init__(self, config_path=CONFIG_PATH):
        """Configuration manager for KM3NeT stuff"""
        self.config = ConfigParser()
        self._time_zone = None

        if config_path is not None:
            self._init_from_path(config_path)

    def _init_from_path(self, path):
        if not os.path.exists(path):
            log.warn("No configuration found at '{0}'".format(path))
            return
        self._check_config_file_permissions(path)
        self._read_from_path(path)

    def _read_from_path(self, path):
        """Read configuration from file path"""
        with open(path) as config_file:
            self._read_from_file(config_file)

    def _check_config_file_permissions(self, path):
        """Make sure that the configuration file is 0600"""
        allowed_modes = ['0600', '0o600']
        if oct(stat.S_IMODE(os.lstat(path).st_mode)) not in allowed_modes:
            log.critical("Your config file is readable to others!\n" +
                         "Execute `chmod 0600 {0}`".format(path))

    def _read_from_file(self, file_obj):
        self.config.readfp(file_obj)

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
    def slack_token(self):
        """Return slack token for chat bots."""
        try:
            token = self.config.get('Slack', 'token')
        except Error:
            raise ValueError("No Slack token defined in configuration file.")
        else:
            return token

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
