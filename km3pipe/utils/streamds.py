# coding=utf-8
"""
Access the KM3NeT StreamDS DataBase service.

Usage:
    streamds
    streamds list
    streamds info STREAM
    streamds STREAM [PARAMETERS...]
    streamds (-h | --help)
    streamds --version

Options:
    STREAM      Name of the stream.
    PARAMETERS  List of parameters separated by space (e.g. detid=29).
    -h --help   Show this screen.

"""
from __future__ import division, absolute_import, print_function

import km3pipe as kp

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2017, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"

log = kp.logger.get("StreamDS")


def print_streams():
    """Print all available streams with their full description"""
    sds = kp.db.StreamDS()
    sds.print_streams()


def print_info(stream):
    """Print the information about a stream"""
    sds = kp.db.StreamDS()
    sds.help(stream)


def get_data(stream, parameters):
    """Retrieve data for given stream and parameters, or None if not found"""
    sds = kp.db.StreamDS()
    if stream not in sds.streams:
        log.error("Stream '{}' not found in the database.".format(stream))
        return
    fun = getattr(sds, stream)
    params = {}
    if parameters:
        for parameter in parameters:
            if '=' not in parameter:
                log.error("Invalid parameter syntax '{}'\n"
                          "The correct syntax is 'parameter=value'"
                          .format(parameter))
                continue
            key, value = parameter.split('=')
            params[key] = value
    data = fun(**params)
    if data is not None:
        print(data)
    else:
        sds.help(stream)


def available_streams():
    """Show a short list of available streams."""
    sds = kp.db.StreamDS()
    print("Available streams: ")
    print(', '.join(sorted(sds.streams)))


def main():
    from docopt import docopt
    args = docopt(__doc__)

    if args['info']:
        print_info(args['STREAM'])
    elif args['list']:
        print_streams()
    elif args['STREAM']:
        get_data(args['STREAM'], args['PARAMETERS'])
    else:
        available_streams()
