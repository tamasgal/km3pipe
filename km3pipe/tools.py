# Filename: tools.py
# pylint: disable=C0103
"""
Some unsorted, frequently used logic.

"""
from __future__ import absolute_import, print_function, division

import base64
import collections
import os
import re
import socket
import subprocess
import sys

import numpy as np

__author__ = "Tamas Gal and Moritz Lotze"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = ["Konstantin Lepa <konstantin.lepa@gmail.com> for termcolor"]
__license__ = "MIT"
__maintainer__ = "Tamas Gal and Moritz Lotze"
__email__ = "tgal@km3net.de"
__status__ = "Development"


def ifiles(irods_path):
    """Return a list of filenames for given iRODS path (recursively)"""
    raw_output = subprocess.check_output(
        "ils -r --bundle {0}"
        "    | grep 'Bundle file:'"
        "    | awk '{{print $3}}'".format(irods_path),
        shell=True
    )
    filenames = raw_output.decode('ascii').strip().split("\n")
    return filenames


def iexists(irods_path):
    """Returns True of iRODS path exists, otherwise False"""
    try:
        subprocess.check_output(
            'ils {}'.format(irods_path),
            shell=True,
            stderr=subprocess.PIPE,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def token_urlsafe(nbytes=32):
    """Return a random URL-safe text string, in Base64 encoding.

    This is taken and slightly modified from the Python 3.6 stdlib.

    The string has *nbytes* random bytes.  If *nbytes* is ``None``
    or not supplied, a reasonable default is used.

    >>> token_urlsafe(16)  #doctest:+SKIP
    'Drmhze6EPcv0fN_81Bj-nA'

    """
    tok = os.urandom(nbytes)
    return base64.urlsafe_b64encode(tok).rstrip(b'=').decode('ascii')


def insert_prefix_to_dtype(arr, prefix):
    new_cols = [prefix + '_' + col for col in arr.dtype.names]
    arr.dtype.names = new_cols
    return arr


def prettyln(text, fill='-', align='^', prefix='[ ', suffix=' ]', length=69):
    """Wrap `text` in a pretty line with maximum length."""
    text = '{prefix}{0}{suffix}'.format(text, prefix=prefix, suffix=suffix)
    print(
        "{0:{fill}{align}{length}}".format(
            text, fill=fill, align=align, length=length
        )
    )


def irods_filepath(det_id, run_id):
    """Generate the iRODS filepath for given detector (O)ID and run ID"""
    data_path = "/in2p3/km3net/data/raw/sea"
    from km3pipe.db import DBManager
    if not isinstance(det_id, int):
        dts = DBManager().detectors
        det_id = int(dts[dts.OID == det_id].SERIALNUMBER.values[0])
    return data_path + "/KM3NeT_{0:08}/{2}/KM3NeT_{0:08}_{1:08}.root" \
           .format(det_id, run_id, run_id//1000)


def unpack_nfirst(seq, nfirst):
    """Unpack the nfrist items from the list and return the rest.

    >>> a, b, c, rest = unpack_nfirst((1, 2, 3, 4, 5), 3)
    >>> a, b, c
    (1, 2, 3)
    >>> rest
    (4, 5)

    """
    iterator = iter(seq)
    for _ in range(nfirst):
        yield next(iterator, None)
    yield tuple(iterator)


def split(string, callback=None, sep=None):
    """Split the string and execute the callback function on each part.

    >>> string = "1 2 3 4"
    >>> parts = split(string, int)
    >>> parts
    [1, 2, 3, 4]

    """
    if callback is not None:
        return [callback(i) for i in string.split(sep)]
    else:
        return string.split(sep)


def namedtuple_with_defaults(typename, field_names, default_values=[]):
    """Create a namedtuple with default values

    >>> Node = namedtuple_with_defaults('Node', 'val left right')
    >>> Node()
    Node(val=None, left=None, right=None)
    >>> Node = namedtuple_with_defaults('Node', 'val left right', [1, 2, 3])
    >>> Node()
    Node(val=1, left=2, right=3)
    >>> Node = namedtuple_with_defaults('Node', 'val left right', {'right':7})
    >>> Node()
    Node(val=None, left=None, right=7)
    >>> Node(4)
    Node(val=4, left=None, right=7)
    """
    the_tuple = collections.namedtuple(typename, field_names)
    the_tuple.__new__.__defaults__ = (None, ) * len(the_tuple._fields)
    if isinstance(default_values, collections.Mapping):
        prototype = the_tuple(**default_values)
    else:
        prototype = the_tuple(*default_values)
    the_tuple.__new__.__defaults__ = tuple(prototype)
    return the_tuple


def remain_file_pointer(function):
    """Remain the file pointer position after calling the decorated function

    This decorator assumes that the last argument is the file handler.

    """

    def wrapper(*args, **kwargs):
        """Wrap the function and remain its parameters and return values"""
        file_obj = args[-1]
        old_position = file_obj.tell()
        return_value = function(*args, **kwargs)
        file_obj.seek(old_position, 0)
        return return_value

    return wrapper


def itervalues(d):
    return iter(d.values())


def iteritems(d):
    return iter(d.items())


def decamelise(text):
    """Convert CamelCase to lower_and_underscore."""
    s = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', text)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s).lower()


def camelise(text, capital_first=True):
    """Convert lower_underscore to CamelCase."""

    def camelcase():
        if not capital_first:
            yield str.lower
        while True:
            yield str.capitalize

    if istype(text, 'unicode'):
        text = text.encode('utf8')
    c = camelcase()
    return "".join(next(c)(x) if x else '_' for x in text.split("_"))


ATTRIBUTES = dict(
    list(
        zip([
            'bold', 'dark', '', 'underline', 'blink', '', 'reverse',
            'concealed'
        ], list(range(1, 9)))
    )
)
del ATTRIBUTES['']

ATTRIBUTES_RE = '\033\[(?:%s)m' % '|'  \
                .join(['%d' % v for v in ATTRIBUTES.values()])

HIGHLIGHTS = dict(
    list(
        zip([
            'on_grey', 'on_red', 'on_green', 'on_yellow', 'on_blue',
            'on_magenta', 'on_cyan', 'on_white'
        ], list(range(40, 48)))
    )
)

HIGHLIGHTS_RE = '\033\[(?:%s)m' % '|'  \
                .join(['%d' % v for v in HIGHLIGHTS.values()])

COLORS = dict(
    list(
        zip([
            'grey',
            'red',
            'green',
            'yellow',
            'blue',
            'magenta',
            'cyan',
            'white',
        ], list(range(30, 38)))
    )
)

COLORS_RE = '\033\[(?:%s)m' % '|'.join(['%d' % v for v in COLORS.values()])

RESET = '\033[0m'
RESET_RE = '\033\[0m'


def colored(text, color=None, on_color=None, attrs=None, ansi_code=None):
    """Colorize text, while stripping nested ANSI color sequences.

    Author:  Konstantin Lepa <konstantin.lepa@gmail.com> / termcolor

    Available text colors:
        red, green, yellow, blue, magenta, cyan, white.
    Available text highlights:
        on_red, on_green, on_yellow, on_blue, on_magenta, on_cyan, on_white.
    Available attributes:
        bold, dark, underline, blink, reverse, concealed.
    Example:
        colored('Hello, World!', 'red', 'on_grey', ['blue', 'blink'])
        colored('Hello, World!', 'green')
    """
    if os.getenv('ANSI_COLORS_DISABLED') is None:
        if ansi_code is not None:
            return "\033[38;5;{}m{}\033[0m".format(ansi_code, text)
        fmt_str = '\033[%dm%s'
        if color is not None:
            text = re.sub(COLORS_RE + '(.*?)' + RESET_RE, r'\1', text)
            text = fmt_str % (COLORS[color], text)
        if on_color is not None:
            text = re.sub(HIGHLIGHTS_RE + '(.*?)' + RESET_RE, r'\1', text)
            text = fmt_str % (HIGHLIGHTS[on_color], text)
        if attrs is not None:
            text = re.sub(ATTRIBUTES_RE + '(.*?)' + RESET_RE, r'\1', text)
            for attr in attrs:
                text = fmt_str % (ATTRIBUTES[attr], text)
        return text + RESET
    else:
        return text


def cprint(text, color=None, on_color=None, attrs=None):
    """Print colorize text.

    Author:  Konstantin Lepa <konstantin.lepa@gmail.com> / termcolor

    It accepts arguments of print function.
    """
    print((colored(text, color, on_color, attrs)))


def issorted(arr):
    """Check if array is sorted."""
    return np.all(np.diff(arr) >= 0)


def lstrip(text):
    """Remove leading whitespace from each line of a multiline string."""
    return '\n'.join(l.lstrip() for l in text.lstrip().split('\n'))


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def is_coherent(seq):
    """Find out if list of subsequent integers is complete.

    Adapted from https://stackoverflow.com/questions/18131741/python-find-out-whether-a-list-of-integers-is-coherent

    ```
    is_coherent([1, 2, 3, 4, 5]) -> True
    is_coherent([1,    3, 4, 5]) -> False
    ```
    """
    return np.array_equal(seq, range(seq[0], int(seq[-1] + 1)))


class AnyBar():
    """A lightweight interface to the AnyBar macOS app."""

    def __init__(self, port=1738, address='localhost'):
        self.port = port
        self.address = address
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def change(self, color):
        self.sock.sendto(color.encode('utf-8'), (self.address, self.port))


def zero_pad(m, n=1):
    """Pad a matrix with zeros, on all sides."""
    return np.pad(m, (n, n), mode='constant', constant_values=[0])


def istype(obj, typename):
    """Drop-in replacement for `isinstance` to avoid imports"""
    return type(obj).__name__ == typename


def supports_color():
    """Checks if the terminal supports color."""
    supported_platform = sys.platform != 'win32' or 'ANSICON' in os.environ
    is_a_tty = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()

    if not supported_platform or not is_a_tty:
        return False

    return True


def get_jpp_revision(via_command='JPrint'):
    """Retrieves the Jpp revision number"""
    try:
        output = subprocess.check_output([via_command, '-v'],
                                         stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        if e.returncode == 1:
            output = e.output
        else:
            return None
    except OSError:
        return None
    revision = output.decode().split('\n')[0].split()[1].strip()
    return revision
