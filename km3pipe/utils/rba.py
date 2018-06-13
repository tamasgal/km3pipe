# Filename: rba.py
"""
RainbowAlga online display.

Usage:
    rba console FILE DETX
    rba -t TOKEN -n EVENT_ID [-u URL] FILE
    rba (-h | --help)
    rba --version

Options:
    FILE       Input file.
    -h --help  Show this screen.

"""

from cmd import Cmd

import km3pipe as kp
from km3pipe.srv import srv_event

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"


def rba():
    pass


class RBAPrompt(Cmd):
    def __init__(self, filename, detx):
        Cmd.__init__(self)
        self.filename = filename
        self.pump = kp.io.GenericPump(filename)
        self.cal = kp.calib.Calibration(filename=detx)
        self.prompt = "> "
        self.current_idx = 0
        self.token = None

    # Override methods in Cmd object
    def preloop(self):
        """Initialization before prompting user for commands.

        Despite the claims in the Cmd documentaion, Cmd.preloop() is not a stub
        """
        Cmd.preloop(self)    # sets up command completion
        self._hist = []    # No history yet
        self._locals = {}    # Initialize execution namespace for user
        self._globals = {}

    def default(self, line):
        """Called on an input line when the command prefix is not recognized.
           In that case we execute the line as Python code.
        """
        pass
        # try:
        #    exec(line) in self._locals, self._globals
        # except Exception as e:
        #    print(e.__class__, ":", e)

    def do_show(self, args):
        args = args.split(' ')
        if len(args) < 2:
            print("usage: show TOKEN EVENT_ID")
            return
        self.token = args[0]
        event = int(args[1])
        self.srv_event(event)

    def do_n(self, args):
        self.current_idx += 1
        self.srv_event(self.current_idx)

    def do_p(self, args):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.srv_event(self.current_idx)

    def srv_event(self, event):
        print("Serving event #{0}".format(event))
        hits = self.cal.apply(self.pump[event]["Hits"].triggered_rows)
        srv_event(self.token, hits)
        self.current_idx = event

    def do_file(self, args):
        print(self.filename)

    def do_hello(self, args):
        print("Sup?")

    def do_quit(self, args):
        self.pump.finish()
        print("Bye.")
        raise SystemExit

    def do_EOF(self, args):
        """Exit on system end of file character"""
        return self.do_quit(args)


def start_console(filename, detx):
    prompt = RBAPrompt(filename, detx)
    prompt.cmdloop('Entering RainbowAlga console...')


def main():
    from docopt import docopt
    args = docopt(__doc__)

    if args["console"]:
        start_console(args["FILE"], args["DETX"])


#    rba(arguments['FILE'])
