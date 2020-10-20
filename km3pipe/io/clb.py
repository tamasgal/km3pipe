# Filename: clb.py
"""
Pumps for the CLB data formats.

"""

import struct
from struct import unpack

import numpy as np

from thepipe import Blob, Module
from km3pipe.dataclasses import Table
from km3pipe.sys import ignored

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"


class CLBPump(Module):
    """A pump for binary CLB files.

    Parameters
    ----------
    file: str
        filename or file-like object.

    """

    pmt_dt = np.dtype([("channel_id", np.uint8), ("time", ">i"), ("tot", np.uint8)])

    def configure(self):
        self.file = self.require("file")
        if isinstance(self.file, str):
            self.file = open(self.file, "rb")
        self._packet_positions = []

        self._determine_packet_positions()

        self.blobs = self.blob_generator()

    def _determine_packet_positions(self):
        """Record the file pointer position of each frame"""
        self.cprint("Scanning UDP packets...")
        self.file.seek(0, 0)
        with ignored(struct.error):
            while True:
                pointer_position = self.file.tell()
                length = unpack("<i", self.file.read(4))[0]
                self._packet_positions.append(pointer_position)
                self.file.seek(length, 1)
        self.file.seek(0, 0)

    def __len__(self):
        return len(self._packet_positions)

    def seek_to_packet(self, index):
        """Move file pointer to the packet with given index."""
        pointer_position = self._packet_positions[index]
        self.file.seek(pointer_position, 0)

    def blob_generator(self):
        """Generate next blob in file"""
        for _ in range(len(self)):
            yield self.extract_blob()

    def extract_blob(self):
        try:
            length = unpack("<i", self.file.read(4))[0]
        except struct.error:
            raise StopIteration

        blob = Blob()

        blob["PacketInfo"] = Table(
            {
                "data_type": b"".join(unpack("cccc", self.file.read(4))).decode(),
                "run": unpack(">i", self.file.read(4))[0],
                "udp_sequence": unpack(">i", self.file.read(4))[0],
                "timestamp": unpack(">I", self.file.read(4))[0],
                "ns_ticks": unpack(">I", self.file.read(4))[0],
                "dom_id": unpack(">i", self.file.read(4))[0],
                "dom_status": unpack(">I", self.file.read(4))[0],
            },
            h5loc="/packet_info",
            split_h5=True,
            name="UDP Packet Info",
        )

        remaining_length = length - 7 * 4
        pmt_data = []

        count = remaining_length // self.pmt_dt.itemsize

        pmt_data = np.fromfile(self.file, dtype=self.pmt_dt, count=count)

        blob["Hits"] = Table(pmt_data, h5loc="/hits", split_h5=True)
        return blob

    def __getitem__(self, index):
        """Return blob at given index."""
        self.seek_to_packet(index)
        return self.extract_blob()

    def process(self, blob):
        return next(self.blobs)

    def __iter__(self):
        self.file.seek(0, 0)
        self.blobs = self.blob_generator()
        return self

    def __next__(self):
        return next(self.blobs)

    def finish(self):
        """Clean everything up"""
        self.file.close()
