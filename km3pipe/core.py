# Filename: core.py
# pylint: disable=locally-disabled
"""
The core of the KM3Pipe framework.

"""
from __future__ import absolute_import, print_function, division

from collections import deque, OrderedDict
import inspect
import signal
import gzip
import time
from timeit import default_timer as timer
import types

import numpy as np

from .sys import peak_memory_usage, ignored
from .logger import get_logger, get_printer
from .time import Timer
from .tools import AnyBar

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = ["Thomas Heid"]
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"

log = get_logger(__name__)    # pylint: disable=C0103
# log.setLevel(logging.DEBUG)

STAT_LIMIT = 100000


class Pipeline(object):
    """The holy pipeline which holds everything together.

    If initialised with timeit=True, all modules will be monitored, otherwise
    only the overall statistics and modules with `timeit=True` will be
    shown.

    Parameters
    ----------
    timeit: bool, optional [default=False]
        Display time profiling statistics for the pipeline?
    """

    def __init__(self, blob=None, timeit=False, anybar=False):
        if anybar:
            self.anybar = AnyBar()
            self.anybar.change("blue")
        else:
            self.anybar = None

        self.init_timer = Timer("Pipeline and module initialisation")
        self.init_timer.start()

        self.modules = []
        self.services = {}
        self.calibration = None
        self.blob = blob or Blob()
        self.timeit = timeit
        self._timeit = {
            'init': timer(),
            'init_cpu': time.clock(),
            'cycles': deque(maxlen=STAT_LIMIT),
            'cycles_cpu': deque(maxlen=STAT_LIMIT)
        }
        self._cycle_count = 0
        self._stop = False
        self._finished = False

    def attach(self, module_factory, name=None, **kwargs):
        """Attach a module to the pipeline system"""
        if self.anybar: self.anybar.change("yellow")

        fac = module_factory

        if name is None:
            name = fac.__name__

        log.info("Attaching module '{0}'".format(name))

        if (inspect.isclass(fac) and issubclass(fac, Module)) or \
                name == 'GenericPump':
            log.debug("Attaching as regular module")
            module = fac(name=name, **kwargs)
            if hasattr(module, "services"):
                for service_name, obj in module.services.items():
                    self.services[service_name] = obj
            module.services = self.services
        else:
            if isinstance(fac, types.FunctionType):
                log.debug("Attaching as function module")
            else:
                log.critical(
                    "Don't know how to attach module '{0}'!\n"
                    "But I'll do my best".format(name)
                )
            module = fac
            module.name = name
            module.timeit = self.timeit

        # Special parameters
        if 'only_if' in kwargs:
            module.only_if = kwargs['only_if']
        else:
            module.only_if = None

        if 'every' in kwargs:
            module.every = kwargs['every']
        else:
            module.every = 1

        self._timeit[module] = {
            'process': deque(maxlen=STAT_LIMIT),
            'process_cpu': deque(maxlen=STAT_LIMIT),
            'finish': 0,
            'finish_cpu': 0
        }

        if hasattr(module, 'get_detector'):    # Calibration-like module
            self.calibration = module
            if module._should_apply:
                self.modules.append(module)
        else:    # normal module
            module.calibration = self.calibration
            self.modules.append(module)

    def attach_bundle(self, modules):
        for mod in modules:
            self.modules.append(mod)

    def _drain(self, cycles=None):
        """Activate the pump and let the flow go.

        This will call the process() method on each attached module until
        a StopIteration is raised, usually by a pump when it reached the EOF.

        A StopIteration is also raised when self.cycles was set and the
        number of cycles has reached that limit.

        """
        log.info("Now draining...")
        if not cycles:
            log.info("No cycle count, the pipeline may be drained forever.")

        if self.calibration:
            log.info("Setting up the detector calibration.")
            for module in self.modules:
                module.detector = self.calibration.get_detector()

        try:
            while not self._stop:
                cycle_start = timer()
                cycle_start_cpu = time.clock()

                log.debug("Pumping blob #{0}".format(self._cycle_count))
                self.blob = Blob()

                for module in self.modules:
                    if self.blob is None:
                        log.debug(
                            "Skipping {0}, due to empty blob."
                            .format(module.name)
                        )
                        continue
                    if module.only_if is not None and \
                            module.only_if not in self.blob:
                        log.debug(
                            "Skipping {0}, due to missing required key"
                            "'{1}'.".format(module.name, module.only_if)
                        )
                        continue

                    if (self._cycle_count + 1) % module.every != 0:
                        log.debug(
                            "Skipping {0} (every {1} iterations)."
                            .format(module.name, module.every)
                        )
                        continue

                    log.debug("Processing {0} ".format(module.name))
                    start = timer()
                    start_cpu = time.clock()
                    self.blob = module(self.blob)
                    if self.timeit or module.timeit:
                        self._timeit[module]['process'] \
                            .append(timer() - start)
                        self._timeit[module]['process_cpu'] \
                            .append(time.clock() - start_cpu)
                self._timeit['cycles'].append(timer() - cycle_start)
                self._timeit['cycles_cpu'
                             ].append(time.clock() - cycle_start_cpu)
                self._cycle_count += 1
                if cycles and self._cycle_count >= cycles:
                    raise StopIteration
        except StopIteration:
            log.info("Nothing left to pump through.")
        return self.finish()

    def drain(self, cycles=None):
        """Execute _drain while trapping KeyboardInterrupt"""
        if self.anybar: self.anybar.change("orange")

        self.init_timer.stop()
        log.info("Trapping CTRL+C and starting to drain.")
        signal.signal(signal.SIGINT, self._handle_ctrl_c)
        with ignored(KeyboardInterrupt):
            return self._drain(cycles)

    def finish(self):
        """Call finish() on each attached module"""
        if self.anybar: self.anybar.change("purple")

        finish_blob = Blob()
        for module in self.modules:
            if hasattr(module, 'pre_finish'):
                log.info("Finishing {0}".format(module.name))
                start_time = timer()
                start_time_cpu = time.clock()
                finish_blob[module.name] = module.pre_finish()
                self._timeit[module]['finish'] = timer() - start_time
                self._timeit[module]['finish_cpu'] = \
                    time.clock() - start_time_cpu
            else:
                log.info("Skipping function module {0}".format(module.name))
        self._timeit['finish'] = timer()
        self._timeit['finish_cpu'] = time.clock()
        self._print_timeit_statistics()
        self._finished = True

        if self.anybar: self.anybar.change("green")
        return finish_blob

    def _handle_ctrl_c(self, *args):
        """Handle the keyboard interrupts."""
        if self.anybar: self.anybar.change("exclamation")

        if self._stop:
            print("\nForced shutdown...")
            raise SystemExit
        if not self._stop:
            hline = 42 * '='
            print(
                '\n' + hline + "\nGot CTRL+C, waiting for current cycle...\n"
                "Press CTRL+C again if you're in hurry!\n" + hline
            )
            self._stop = True

    def _print_timeit_statistics(self):

        if self._cycle_count < 1:
            return

        def calc_stats(values):
            """Return a tuple of statistical values"""
            return [f(values) for f in (np.mean, np.median, min, max, np.std)]

        def timef(seconds):
            """Return a string of formatted time value for given seconds"""
            time = seconds
            if time > 180:
                time /= 60
                unit = 'min'
            else:
                unit = 's'
            return "{0:.6f}{1}".format(time, unit)

        def statsf(prefix, values):
            stats = "  mean: {0}  medi: {1}  min: {2}  max: {3}  std: {4}"
            values = [timef(v) for v in values]
            return "  " + prefix + stats.format(*values)

        cycles = self._timeit['cycles']
        n_cycles = len(cycles)

        cycles_cpu = self._timeit['cycles_cpu']
        overall = self._timeit['finish'] - self._timeit['init']
        overall_cpu = self._timeit['finish_cpu'] - self._timeit['init_cpu']
        memory = peak_memory_usage()

        print(60 * '=')
        print(
            "{0} cycles drained in {1} (CPU {2}). Memory peak: {3:.2f} MB"
            .format(
                self._cycle_count, timef(overall), timef(overall_cpu), memory
            )
        )
        if self._cycle_count > n_cycles:
            print(
                "Statistics are based on the last {0} cycles."
                .format(n_cycles)
            )
        if cycles:
            print(statsf('wall', calc_stats(cycles)))
        if cycles_cpu:
            print(statsf('CPU ', calc_stats(cycles_cpu)))

        for module in self.modules:
            if not module.timeit and not self.timeit:
                continue
            finish_time = self._timeit[module]['finish']
            finish_time_cpu = self._timeit[module]['finish_cpu']
            process_times = self._timeit[module]['process']
            process_times_cpu = self._timeit[module]['process_cpu']
            print(
                module.name + " - process: {0:.3f}s (CPU {1:.3f}s)"
                " - finish: {2:.3f}s (CPU {3:.3f}s)".format(
                    sum(process_times), sum(process_times_cpu), finish_time,
                    finish_time_cpu
                )
            )
            if len(process_times) > 0:
                print(statsf('wall', calc_stats(process_times)))
            if len(process_times_cpu) > 0:
                print(statsf('CPU ', calc_stats(process_times_cpu)))


class Module(object):
    """The module which can be attached to the pipeline"""

    def __init__(self, name=None, **parameters):
        log.debug("Initialising {0}".format(name))
        self._name = name
        self.parameters = parameters
        self.only_if = None
        self.every = 1
        self.detector = None
        if self.__module__ == '__main__':
            self.logger_name = self.__class__.__name__
        else:
            self.logger_name = self.__module__ + '.' + self.__class__.__name__
        log.debug("Setting up logger '{}'".format(self.logger_name))
        self.log = get_logger(self.logger_name)
        self.print = get_printer(self.logger_name)
        self.timeit = self.get('timeit') or False
        self._timeit = {
            'process': deque(maxlen=STAT_LIMIT),
            'process_cpu': deque(maxlen=STAT_LIMIT),
            'finish': 0,
            'finish_cpu': 0
        }
        self.services = {}
        self.configure()

    def configure(self):
        """Configure module, like instance variables etc."""
        pass

    def expose(self, obj, name):
        """Expose an object as a service to the Pipeline"""
        self.services[name] = obj

    @property
    def name(self):
        """The name of the module"""
        return self._name

    def add(self, name, value):
        """Add the parameter with the desired value to the dict"""
        self.parameters[name] = value

    def get(self, name, default=None):
        """Return the value of the requested parameter or `default` if None."""
        value = self.parameters.get(name)
        if value is None:
            return default
        return value

    def require(self, name):
        """Return the value of the requested parameter or raise an error."""
        value = self.get(name)
        if value is None:
            raise TypeError(
                "{0} requires the parameter '{1}'.".format(
                    self.__class__, name
                )
            )
        return value

    def process(self, blob):    # pylint: disable=R0201
        """Knead the blob and return it"""
        return blob

    def finish(self):
        """Clean everything up."""
        return

    def pre_finish(self):
        """Do the last few things before calling finish()"""
        return self.finish()

    def __call__(self, *args, **kwargs):
        """Run process if directly called."""
        log.info("Calling process")
        return self.process(*args, **kwargs)


class Pump(Module):
    """The pump with basic file or socket handling."""

    def __init__(self, *args, **kwargs):
        self.blob_file = None
        if args:
            log.warning(
                "Non-keywords argument passed. Please use keyword "
                "arguments to supress this warning. I will assume the "
                "first argument to be the `filename`."
            )
            Module.__init__(self, filename=args[0], **kwargs)
        else:
            Module.__init__(self, **kwargs)

    def open_file(self, filename):
        """Open the file with filename"""
        try:
            if filename.endswith('.gz'):
                self.blob_file = gzip.open(filename, 'rb')
            else:
                self.blob_file = open(filename, 'rb')
        except TypeError:
            log.error("Please specify a valid filename.")
            raise SystemExit
        except IOError as error_message:
            log.error(error_message)
            raise SystemExit

    def process(self, blob):
        """Create a blob"""
        raise NotImplementedError("The pump has no process() method!")

    def rewind_file(self):
        """Put the file pointer to position 0"""
        self.blob_file.seek(0, 0)

    def close_file(self):
        """Close file."""
        if self.blob_file:
            self.blob_file.close()

    def finish(self):
        pass

    def pre_finish(self):
        """Clean up open file or socket-handlers."""
        out = self.finish()
        self.close_file()
        return out

    def close(self):
        self.finish()

    def next(self):
        """Python 2 compatibility for iterators"""
        return self.__next__()

    def __enter__(self, *args, **kwargs):
        self.configure(*args, **kwargs)
        return self

    def __exit__(self, *args, **kwargs):
        self.finish()


class Blob(OrderedDict):
    """A simple (ordered) dict with a fancy name. This should hold the data."""
    pass


class Run(object):
    """A simple container for event info, hits, tracks and calibration.
    """

    def __init__(self, **tables):
        for key, val in tables.items():
            setattr(self, key, val)
