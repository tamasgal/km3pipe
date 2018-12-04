# Filename: logger.py
# pylint: disable=locally-disabled,C0103
"""
The logging facility.

"""
from __future__ import absolute_import, print_function, division

from hashlib import sha256
import socket
import logging

from .tools import colored, supports_color

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"

loggers = {}    # this holds all the registered loggers
# logging.basicConfig()

DEPRECATION = 45
logging.addLevelName(DEPRECATION, "DEPRECATION")


def deprecation(self, message, *args, **kws):
    self._log(DEPRECATION, message, args, **kws)


logging.Logger.deprecation = deprecation

if supports_color():
    logging.addLevelName(
        logging.INFO,
        "\033[1;32m%s\033[1;0m" % logging.getLevelName(logging.INFO)
    )
    logging.addLevelName(
        logging.DEBUG,
        "\033[1;34m%s\033[1;0m" % logging.getLevelName(logging.DEBUG)
    )
    logging.addLevelName(
        logging.WARNING,
        "\033[1;33m%s\033[1;0m" % logging.getLevelName(logging.WARNING)
    )
    logging.addLevelName(
        logging.ERROR,
        "\033[1;31m%s\033[1;0m" % logging.getLevelName(logging.ERROR)
    )
    logging.addLevelName(
        logging.CRITICAL,
        "\033[1;101m%s\033[1;0m" % logging.getLevelName(logging.CRITICAL)
    )
    logging.addLevelName(DEPRECATION, "\033[1;35m%s\033[1;0m" % 'DEPRECATION')


class LogIO(object):
    """Read/write logging information.
    """

    def __init__(
            self, node, stream, url='pi2089.physik.uni-erlangen.de', port=28777
    ):
        self.node = node
        self.stream = stream
        self.url = url
        self.port = port
        self.sock = None
        self.connect()

    def send(self, message, level='info'):
        message_string = "+log|{0}|{1}|{2}|{3}\r\n" \
                         .format(self.stream, self.node, level, message)
        try:
            self.sock.send(message_string)
        except socket.error:
            print("Lost connection, reconnecting...")
            self.connect()
            self.sock.send(message_string)

    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.url, self.port))


def get_logger(name):
    """Helper function to get a logger"""
    if name in loggers:
        return loggers[name]
    logger = logging.getLogger(name)
    logger.propagate = False
    pre1, suf1 = hash_coloured_escapes(name) if supports_color() else ('', '')
    pre2, suf2 = hash_coloured_escapes(name + 'salt')  \
                 if supports_color() else ('', '')
    formatter = logging.Formatter(
        '%(levelname)s {}+{}+{} '
        '%(name)s: %(message)s'.format(pre1, pre2, suf1)
    )
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    loggers[name] = logger

    return logger


def available_loggers():
    """Return a list of avialable logger names"""
    return list(logging.Logger.manager.loggerDict.keys())


def set_level(name, level):
    """Set the log level for given logger"""
    get_logger(name).setLevel(level)


def get_printer(name, color=None, ansi_code=None, force_color=False):
    """Return a function which prints a message with a coloured name prefix"""

    if force_color or supports_color():
        if color is None and ansi_code is None:
            cpre_1, csuf_1 = hash_coloured_escapes(name)
            cpre_2, csuf_2 = hash_coloured_escapes(name + 'salt')
            name = cpre_1 + '+' + cpre_2 + '+' + csuf_1 + ' ' + name
        else:
            name = colored(name, color=color, ansi_code=ansi_code)

    prefix = name + ': '

    def printer(text):
        print(prefix + str(text))

    return printer


def hash_coloured(text):
    """Return a ANSI coloured text based on its hash"""
    ansi_code = int(sha256(text.encode('utf-8')).hexdigest(), 16) % 230
    return colored(text, ansi_code=ansi_code)


def hash_coloured_escapes(text):
    """Return the ANSI hash colour prefix and suffix for a given text"""
    ansi_code = int(sha256(text.encode('utf-8')).hexdigest(), 16) % 230
    prefix, suffix = colored('SPLIT', ansi_code=ansi_code).split('SPLIT')
    return prefix, suffix
