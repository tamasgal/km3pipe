import struct

filename = '/Users/tamasgal/Desktop/RUN-PPM_DU-00430-20140730-121124_detx.dat'

from km3pipe import Module
from km3pipe.logger import get_logger

log = get_logger(__name__)


class DAQPump(Module):
    def __init__(self, **context):
        super(self.__class__, self).__init__(**context)
        self.filename = self.get('filename')
        self.daq_file = None

    def next_frame(file):
        print("Preamble:")
        length, data_type = struct.unpack('<ii', file.read(8))
        print length, data_type
        file.read(length-8)
        raw_input()

    def open_file(self):
        try:
            self.daq_file = open(self.filename, 'rb')
        except TypeError:
            log.error("Please specify a valid filename.")
            raise SystemExit
        except IOError as e:
            log.error(e)
            raise SystemExit

    def finish(self):
        self.daq_file.close()




def foo():
    print "Preamble:"
    print struct.unpack('<ii', file.read(8))
    print "Header:"
    print struct.unpack('<iiq', file.read(16))
    print "Subheader:"
    print struct.unpack('<i', file.read(4))
    print "Summary frames:"
    print struct.unpack('i' + 'c'*31, file.read(35))
    print struct.unpack('i' + 'c'*31, file.read(35))
    print struct.unpack('i' + 'c'*31, file.read(35))

    print "Header:"
    print struct.unpack('<iii', file.read(12))
    print "Timestamp"
    print struct.unpack('<Q', file.read(8))

