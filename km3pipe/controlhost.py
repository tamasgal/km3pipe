# Filename: controlhost.py
"""
A set of classes and tools wich uses the ControlHost protocol.

"""
import socket
import struct
import time

from .logger import get_logger

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"

log = get_logger(__name__)

BUFFER_SIZE = 1024


class Client(object):
    """The ControlHost client"""

    _valid_modes = ["any", "all"]

    def __init__(self, host, port=5553):
        self.host = host
        self.port = port
        self.socket = None
        self.tags = []
        self.valid_tags = []

    def subscribe(self, tag, mode="any"):
        if mode not in self._valid_modes:
            raise ValueError(
                "Possible subscription modes are: {}".format(
                    ", ".join(self._valid_modes)
                )
            )
        log.info("Subscribing to %s in mode %s", tag, mode)
        full_tag = self._full_tag(tag, mode)
        if full_tag not in self.tags:
            self.tags.append(full_tag)
        for t in tag.split():
            if t not in self.valid_tags:
                self.valid_tags.append(t)
        self._update_subscriptions()

    def unsubscribe(self, tag, mode="any"):
        try:
            self.tags.remove(self._full_tag(tag, mode))
            self.valid_tags.remove(tag)
        except ValueError:
            pass
        else:
            self._update_subscriptions()

    def _full_tag(self, tag, mode):
        mode_flag = " {} ".format("w" if mode == "any" else "a")
        full_tag = mode_flag + tag
        return full_tag

    def _update_subscriptions(self):
        log.debug("Subscribing to tags: %s", self.tags)
        if not self.socket:
            self._connect()
        tags = "".join(self.tags).encode("ascii")
        message = Message(b"_Subscri", tags)
        self.socket.send(message.data)
        message = Message(b"_Always")
        self.socket.send(message.data)

    def put_message(self, tag, data):
        """Send data to the ligier with a given tag"""
        if not self.socket:
            self._connect()
        msg = Message(tag, data)
        self.socket.send(msg.data)

    def get_message(self):
        while True:
            log.info("     Waiting for control host Prefix")
            try:
                data = self.socket.recv(Prefix.SIZE)
                timestamp = time.time()
                log.info("    raw prefix data received: '%s'", data)
                if data == b"":
                    raise EOFError
                prefix = Prefix(data=data, timestamp=timestamp)
            except (UnicodeDecodeError, OSError, struct.error):
                log.error("Failed to construct Prefix, reconnecting.")
                self._reconnect()
                continue

            try:
                prefix_tag = str(prefix.tag)
            except UnicodeDecodeError:
                log.error("The tag could not be decoded. Reconnecting.")
                self._reconnect()
                continue

            if prefix_tag not in self.valid_tags:
                log.error(
                    "Invalid tag '%s' received, ignoring the message \n"
                    "and reconnecting.\n"
                    "  -> valid tags are: %s",
                    prefix_tag,
                    self.valid_tags,
                )
                self._reconnect()
                continue
            else:
                break

        message = b""
        log.info("       got a Prefix with %d bytes.", prefix.length)
        while len(message) < prefix.length:
            log.info("          message length: %d", len(message))
            log.info("            (getting next part)")
            buffer_size = min((BUFFER_SIZE, (prefix.length - len(message))))
            try:
                message += self.socket.recv(buffer_size)
            except OSError:
                log.error("Failed to construct message.")
                raise BufferError
        log.info("     ------ returning message with %d bytes", len(message))
        return prefix, message

    def _connect(self):
        """Connect to JLigier"""
        log.debug("Connecting to JLigier")

        s = socket.socket()
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.connect((self.host, self.port))
        self.socket = s

    def _disconnect(self):
        """Close the socket"""
        log.debug("Disconnecting from JLigier")
        if self.socket:
            self.socket.close()

    def _reconnect(self):
        """Reconnect to JLigier and subscribe to the tags."""
        log.debug("Reconnecting to JLigier...")
        self._disconnect()
        self._connect()
        self._update_subscriptions()

    def __enter__(self):
        self._connect()
        return self

    def __exit__(self, exit_type, value, traceback):
        self._disconnect()


class Message(object):
    """The representation of a ControlHost message."""

    def __init__(self, tag, message=b""):
        try:
            message = message.encode()
        except AttributeError:
            pass
        try:
            tag = tag.encode()
        except AttributeError:
            pass
        self.prefix = Prefix(tag, len(message))
        self.message = message

    @property
    def data(self):
        return self.prefix.data + self.message


class Tag(object):
    """Represents the tag in a ControlHost Prefix."""

    SIZE = 8

    def __init__(self, data=None):
        self._data = b""
        self.data = data

    @property
    def data(self):
        """The byte data"""
        return self._data

    @data.setter
    def data(self, value):
        """Set the byte data and fill up the bytes to fit the size."""
        if not value:
            value = b""
        if len(value) > self.SIZE:
            raise ValueError("The maximum tag size is {}".format(self.SIZE))
        self._data = value
        while len(self._data) < self.SIZE:
            self._data += b"\x00"

    def __str__(self):
        return self.data.decode(encoding="UTF-8").strip("\x00")

    def __len__(self):
        return len(self._data)


class Prefix(object):
    """The prefix of a ControlHost message."""

    SIZE = 16

    def __init__(self, tag=None, length=None, data=None, timestamp=None):
        if data:
            self.data = data
        else:
            self.tag = Tag(tag)
            self.length = length
        if timestamp is None:
            self.timestamp = time.time()
        else:
            self.timestamp = timestamp

    @property
    def data(self):
        return self.tag.data + struct.pack(">i", self.length) + b"\x00" * 4

    @data.setter
    def data(self, value):
        self.tag = Tag(data=value[: Tag.SIZE])
        self.length = struct.unpack(">i", value[Tag.SIZE : Tag.SIZE + 4])[0]

    def __str__(self):
        return "ControlHost Prefix with tag '{0}' ({1} bytes of data)".format(
            self.tag, self.length
        )
