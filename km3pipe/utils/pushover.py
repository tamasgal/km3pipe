# Filename: pushover.py
"""
Send a push message to a device using Pushover.net API.

Usage:
    pushover MESSAGE...
    pushover (-h | --help)
    pushover --version

Options:
    MESSAGE     The message to send.
    -h --help   Show this screen.

"""

from km3pipe.config import Config
from km3pipe import version

try:
    import http.client as httplib
    from urllib.parse import urlencode
except ImportError:
    import httplib
    from urllib import urlencode

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2017, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"


def main():
    from docopt import docopt
    args = docopt(__doc__, version=version)

    config = Config()

    token = config.get("Pushover", "token")
    user_key = config.get("Pushover", "user_key")
    if token is None or user_key is None:
        print(
            "Please define your 'token' and 'user_key' in the "
            "'Pushover' section of your ~/.km3net configuration."
        )
        exit(1)

    conn = httplib.HTTPSConnection("api.pushover.net:443")
    conn.request(
        "POST", "/1/messages.json",
        urlencode({
            "token": token,
            "user": user_key,
            "message": ' '.join(args["MESSAGE"]),
        }), {"Content-type": "application/x-www-form-urlencoded"}
    )

    conn.getresponse()
