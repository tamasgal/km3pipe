# coding=utf-8
# Filename: config.py
# pylint: disable=locally-disabled
"""
Tools for global configuration.

"""
from __future__ import division, absolute_import, print_function

import os
import stat
from six.moves.configparser import ConfigParser, NoSectionError
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
    def __init__(self):
        """Configuration manager for KM3NeT stuff"""
        self.config = ConfigParser()
        try:
            self._read_configuration()
        except IOError:
            log.warn("No configuration found at '{0}'".format(CONFIG_PATH))
        else:
            self._check_config_file_permissions()

    def _check_config_file_permissions(self):
        """Make sure that the configuration file is 0600"""
        allowed_modes = ['0600', '0o600']
        if oct(stat.S_IMODE(os.lstat(CONFIG_PATH).st_mode)) in allowed_modes:
            return True
        else:
            log.critical("Your config file is readable to others!\n" +
                         "Please execute `chmod 0600 {0}`".format(CONFIG_PATH))
            return False

    def _read_configuration(self):
        """Parse configuration file"""
        self.config.readfp(open(CONFIG_PATH))

    @property
    def db_credentials(self):
        """Return username and password for the KM3NeT WebDB."""
        try:
            username = self.config.get('DB', 'username')
            password = self.config.get('DB', 'password')
        except NoSectionError:
            username = input("Please enter your KM3NeT DB username: ")
            password = getpass.getpass("Password: ")
        return username, password
