import struct

from km3pipe import Pump
from km3pipe.logger import get_logger

log = get_logger(__name__)


class DAQPump(Pump):
    """A pump for binary DAQ files."""

    def __init__(self, **context):
        super(self.__class__, self).__init__(**context)
        self.filename = self.get('filename')
        self.frame_positions = []

        if self.filename:
            self.open_file(self.filename)
            self.determine_frame_positions()
        else:
            log.warn("No filename specified. Take care of the file handling!")

    def next_frame(self):
        #print("Preamble:")
        length, data_type = struct.unpack('<ii', self.blob_file.read(8))
        print length, data_type
        self.blob_file.seek(length-8, 1)
        raw_input()

    def determine_frame_positions(self):
        """Record the file pointer position of each frame"""
        self.rewind_file()
        try:
            while True:
                pointer_position = self.blob_file.tell()
                length, data_type = struct.unpack('<ii', self.blob_file.read(8))
                self.blob_file.seek(length-8, 1)
                self.frame_positions.append(pointer_position)
        except struct.error:
            pass
        self.rewind_file()
        print("Found {0} frames.".format(len(self.frame_positions)))

    def process(self, blob):
        #print self.next_frame()
        return blob

    def finish(self):
        self.blob_file.close()




# def foo():
#     print "Preamble:"
#     print struct.unpack('<ii', file.read(8))
#     print "Header:"
#     print struct.unpack('<iiq', file.read(16))
#     print "Subheader:"
#     print struct.unpack('<i', file.read(4))
#     print "Summary frames:"
#     print struct.unpack('i' + 'c'*31, file.read(35))
#     print struct.unpack('i' + 'c'*31, file.read(35))
#     print struct.unpack('i' + 'c'*31, file.read(35))
#
#     print "Header:"
#     print struct.unpack('<iii', file.read(12))
#     print "Timestamp"
#     print struct.unpack('<Q', file.read(8))

