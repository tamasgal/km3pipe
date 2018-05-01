# Filename: logger.py
# pylint: disable=locally-disabled,C0103
"""
The logging facility.

"""
from hashlib import sha256
import socket
import logging

from .tools import colored

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"


logging.basicConfig()

logging.addLevelName(logging.INFO, "\033[1;32m%s\033[1;0m" %
                     logging.getLevelName(logging.INFO))
logging.addLevelName(logging.DEBUG, "\033[1;34m%s\033[1;0m" %
                     logging.getLevelName(logging.DEBUG))
logging.addLevelName(logging.WARNING, "\033[1;33m%s\033[1;0m" %
                     logging.getLevelName(logging.WARNING))
logging.addLevelName(logging.ERROR, "\033[1;31m%s\033[1;0m" %
                     logging.getLevelName(logging.ERROR))
logging.addLevelName(logging.CRITICAL, "\033[1;101m%s\033[1;0m" %
                     logging.getLevelName(logging.CRITICAL))

CONSOLE_HANDLER = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname)s] %(name)s: %(message)s')
CONSOLE_HANDLER.setFormatter(formatter)
# logging.getLogger('').addHandler(CONSOLE_HANDLER)


class LogIO(object):
    """Read/write logging information.
    """

    def __init__(self, node, stream,
                 url='pi2089.physik.uni-erlangen.de',
                 port=28777):
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


def get(name):
    """Helper function to get a logger"""
    logger = logging.getLogger(name)
    logger.addHandler(CONSOLE_HANDLER)
    return logger


def set_level(name, level):
    """Set the log level for given logger"""
    logger = get(name)
    logger.setLevel(level)


def get_printer(name, color=None, ansi_code=None):
    """Return a function which prints a message with a coloured name prefix"""
    if color is None and ansi_code is None:
        ansi_code = int(sha256(name.encode('utf-8')).hexdigest(), 16) % 230

    prefix = colored(name + ': ', color=color, ansi_code=ansi_code)

    def printer(text):
        print(prefix + text)

    return printer
