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
import os
import sys
import time
from timeit import default_timer as timer
import types

import toml
import numpy as np

from .sys import peak_memory_usage, ignored
from .logger import get_logger, get_printer
from .time import Timer
from .tools import AnyBar

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = ["Thomas Heid", "Johannes Schumann"]
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"

log = get_logger(__name__)    # pylint: disable=C0103
# log.setLevel(logging.DEBUG)

STAT_LIMIT = 100000
MODULE_CONFIGURATION = 'pipeline.toml'
RESERVED_ARGS = set(['every', 'only_if'])

if sys.version_info >= (3, 3):
    process_time = time.process_time
else:
    process_time = time.clock

# Consistent `isistance(..., basestring)` for Python 2 and 3
try:
    basestring
except NameError:
    basestring = str


class ServiceManager(object):
    """
    Main service manager
    """

    def __init__(self):
        self._services = {}

    def register(self, name, service):
        """
        Service registration

        Args:
            name: Name of the provided service
            service: Reference to the service
        """
        self._services[name] = service

    def get_missing_services(self, services):
        """
        Check if all required services are provided

        Args:
            services: List with the service names which are required
        Returns:
            List with missing services
        """
        required_services = set(services)
        provided_services = set(self._services.keys())
        missing_services = required_services.difference(provided_services)

        return sorted(missing_services)

    def __getitem__(self, name):
        return self._services[name]

    def __getattr__(self, name):
        return self._service[name]

    def __contains__(self, name):
        return name in self._services


class Pipeline(object):
    """The holy pipeline which holds everything together.

    If initialised with timeit=True, all modules will be monitored, otherwise
    only the overall statistics and modules with `timeit=True` will be
    shown.

    Parameters
    ----------
    timeit: bool, optional [default=False]
        Display time profiling statistics for the pipeline?
    configfile: str, optional
        Path to a configuration file (TOML format) which contains parameters
        for attached modules.
    """

    def __init__(self, blob=None, timeit=False, configfile=None, anybar=False):
        self.log = get_logger(self.__class__.__name__)
        self.print = get_printer(self.__class__.__name__)

        if anybar:
            self.anybar = AnyBar()
            self.anybar.change("blue")
        else:
            self.anybar = None

        if configfile is None and os.path.exists(MODULE_CONFIGURATION):
            configfile = MODULE_CONFIGURATION

        if configfile is not None:
            self.print(
                "Reading module configuration from '{}'".format(configfile)
            )
            self.log.warning(
                "Keep in mind that the module configuration file has "
                "precedence over keyword arguments in the attach method!"
            )
            with open(configfile, 'r') as fobj:
                self.module_configuration = toml.load(fobj)
        else:
            self.module_configuration = {}

        self.init_timer = Timer("Pipeline and module initialisation")
        self.init_timer.start()

        self.modules = []
        self.services = ServiceManager()
        self.required_services = {}
        self.calibration = None
        self.blob = blob or Blob()
        self.timeit = timeit
        self._timeit = {
            'init': timer(),
            'init_cpu': process_time(),
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
            if name in self.module_configuration:
                log.debug(
                    "Applying pipeline configuration file for module '%s'" %
                    name
                )
                for key, value in self.module_configuration[name].items():
                    if key in kwargs:
                        self.log.info(
                            "Overwriting parameter '%s' in module '%s' from "
                            "the pipeline configuration file." % (key, name)
                        )
                    kwargs[key] = value
            module = fac(name=name, **kwargs)
            if hasattr(module, "provided_services"):
                for service_name, obj in module.provided_services.items():
                    self.services.register(service_name, obj)
            if hasattr(module, "required_services"):
                updated_required_services = {}
                updated_required_services.update(self.required_services)
                updated_required_services.update(module.required_services)
                self.required_services = updated_required_services
            module.services = self.services
            module.pipeline = self
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
            required_keys = kwargs['only_if']
            if isinstance(required_keys, basestring):
                required_keys = [required_keys]
            module.only_if = set(required_keys)
        else:
            module.only_if = set()

        if 'blob_keys' in kwargs:
            module.blob_keys = kwargs['blob_keys']
        else:
            module.blob_keys = None

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
            self.log.deprecation(
                "Calibration-like modules will not be supported in future "
                "versions of KM3Pipe. Please use services instead.\n"
                "If you are attaching the `calib.Calibration` as a module, "
                "switch to `calib.CalibrationService` and use the "
                "`self.services['calibrate']()` method in your modules "
                "to apply calibration.\n\n"
                "This means:\n\n"
                "    pipe.attach(kp.calib.CalibrationService, ...)\n\n"
                "And inside the attached modules, you can apply the "
                "calibration with e.g.:\n\n"
                "    cal_hits = self.services['calibrate'](blob['Hits'])\n"
            )
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
                cycle_start_cpu = process_time()

                log.debug("Pumping blob #{0}".format(self._cycle_count))
                self.blob = Blob()

                for module in self.modules:
                    if self.blob is None:
                        log.debug(
                            "Skipping {0}, due to empty blob.".format(
                                module.name
                            )
                        )
                        continue
                    if module.only_if and not module.only_if.issubset(set(
                            self.blob.keys())):
                        log.debug(
                            "Skipping {0}, due to missing required key"
                            "'{1}'.".format(module.name, module.only_if)
                        )
                        continue

                    if (self._cycle_count + 1) % module.every != 0:
                        log.debug(
                            "Skipping {0} (every {1} iterations).".format(
                                module.name, module.every
                            )
                        )
                        continue

                    if module.blob_keys is not None:
                        blob_to_send = Blob({
                            k: self.blob[k]
                            for k in module.blob_keys
                            if k in self.blob
                        })
                    else:
                        blob_to_send = self.blob

                    log.debug("Processing {0} ".format(module.name))
                    start = timer()
                    start_cpu = process_time()
                    new_blob = module(blob_to_send)
                    if self.timeit or module.timeit:
                        self._timeit[module]['process'] \
                            .append(timer() - start)
                        self._timeit[module]['process_cpu'] \
                            .append(process_time() - start_cpu)

                    if module.blob_keys is not None:
                        if new_blob is not None:
                            for key in new_blob.keys():
                                self.blob[key] = new_blob[key]
                    else:
                        self.blob = new_blob

                self._timeit['cycles'].append(timer() - cycle_start)
                self._timeit['cycles_cpu'].append(
                    process_time() - cycle_start_cpu
                )
                self._cycle_count += 1
                if cycles and self._cycle_count >= cycles:
                    raise StopIteration
        except StopIteration:
            log.info("Nothing left to pump through.")
        return self.finish()

    def _check_service_requirements(self):
        """Final comparison of provided and required modules"""
        missing = self.services.get_missing_services(
            self.required_services.keys()
        )
        if missing:
            self.log.critical(
                "Following services are required and missing: {}".format(
                    ', '.join(missing)
                )
            )
            return False
        return True

    def drain(self, cycles=None):
        """Execute _drain while trapping KeyboardInterrupt"""
        if not self._check_service_requirements():
            self.init_timer.stop()
            return self.finish()

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
                start_time_cpu = process_time()
                finish_blob[module.name] = module.pre_finish()
                self._timeit[module]['finish'] = timer() - start_time
                self._timeit[module]['finish_cpu'] = \
                    process_time() - start_time_cpu
            else:
                log.info("Skipping function module {0}".format(module.name))
        self._timeit['finish'] = timer()
        self._timeit['finish_cpu'] = process_time()
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
            "{0} cycles drained in {1} (CPU {2}). Memory peak: {3:.2f} MB".
            format(
                self._cycle_count, timef(overall), timef(overall_cpu), memory
            )
        )
        if self._cycle_count > n_cycles:
            print(
                "Statistics are based on the last {0} cycles.".
                format(n_cycles)
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
        self._processed_parameters = []
        self.only_if = set()
        self.every = 1
        self.detector = None
        if self.__module__ == '__main__':
            self.logger_name = self.__class__.__name__
        else:
            self.logger_name = self.__module__ + '.' + self.__class__.__name__
        if name != self.logger_name:
            self.logger_name += '.{}'.format(name)
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
        self.services = ServiceManager()
        self.provided_services = {}
        self.required_services = {}
        self.configure()
        self._check_unused_parameters()

    def configure(self):
        """Configure module, like instance variables etc."""
        pass

    def expose(self, obj, name):
        """Expose an object as a service to the Pipeline"""
        self.provided_services[name] = obj

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
        self._processed_parameters.append(name)
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

    def require_service(self, name, why=''):
        self.required_services[name] = why

    def process(self, blob):    # pylint: disable=R0201
        """Knead the blob and return it"""
        return blob

    def finish(self):
        """Clean everything up."""
        return

    def pre_finish(self):
        """Do the last few things before calling finish()"""
        return self.finish()

    def _check_unused_parameters(self):
        """Check if any of the parameters passed in are ignored"""
        all_params = set(self.parameters.keys())
        processed_params = set(self._processed_parameters)
        unused_params = all_params - processed_params - RESERVED_ARGS

        if unused_params:
            self.log.warning(
                "The following parameters were ignored: {}".format(
                    ', '.join(sorted(unused_params))
                )
            )

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

    def __init__(self, *args, **kwargs):
        OrderedDict.__init__(self, *args, **kwargs)
        self.log = get_logger("Blob")

    def __str__(self):
        if len(self) == 0:
            return "Empty blob"
        padding = max(len(k) for k in self.keys()) + 3
        s = ["Blob ({} entries):".format(len(self))]
        for key, value in self.items():
            s.append(
                " '{}'".format(key).ljust(padding) +
                " => {}".format(repr(value))
            )
        return "\n".join(s)

    def __getitem__(self, key):
        try:
            val = OrderedDict.__getitem__(self, key)
        except KeyError:
            self.log.error(
                "No key named '{}' found in Blob. \n"
                "Available keys: {}".format(key, ', '.join(self.keys()))
            )
            raise
        return val


class Run(object):
    """A simple container for event info, hits, tracks and calibration.
    """

    def __init__(self, **tables):
        for key, val in tables.items():
            setattr(self, key, val)
