# coding=utf-8
# cython: profile=True
# Filename: core.pyx
# cython: embedsignature=True
# pylint: disable=locally-disabled
"""
The core of the KM3Pipe framework.

"""
from __future__ import division, absolute_import, print_function

from collections import deque, OrderedDict
import inspect
import signal
import gzip
import time
from timeit import default_timer as timer
import types

import numpy as np
import pandas as pd

from .sys import peak_memory_usage, ignored
from .hardware import Detector
from .dataclasses import (CRawHitSeries, HitSeries, RawHitSeries,
                          CMcHitSeries, McHitSeries)
from .tools import deprecated
from .logger import logging
from .time import Timer

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = ["Thomas Heid"]
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"

log = logging.getLogger(__name__)  # pylint: disable=C0103
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

    def __init__(self, blob=None, timeit=False):
        self.init_timer = Timer("Pipeline and module initialisation")
        self.init_timer.start()

        self.modules = []
        self.services = {}
        self.calibration = None
        self.blob = blob or Blob()
        self.timeit = timeit
        self._timeit = {'init': timer(), 'init_cpu': time.clock(),
                        'cycles': deque(maxlen=STAT_LIMIT),
                        'cycles_cpu': deque(maxlen=STAT_LIMIT)}
        self._cycle_count = 0
        self._stop = False
        self._finished = False

    def attach(self, module_factory, name=None, **kwargs):
        """Attach a module to the pipeline system"""
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
                log.critical("Don't know how to attach module '{0}'!\n"
                             "But I'll do my best".format(name))
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

        self._timeit[module] = {'process': deque(maxlen=STAT_LIMIT),
                                'process_cpu': deque(maxlen=STAT_LIMIT),
                                'finish': 0,
                                'finish_cpu': 0}

        if hasattr(module, 'get_detector'):  # Calibration-like module
            self.calibration = module
            if module._should_apply:
                self.modules.append(module)
        else:  # normal module
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
                        log.debug("Skipping {0}, due to empty blob."
                                  .format(module.name))
                        continue
                    if module.only_if is not None and \
                            module.only_if not in self.blob:
                        log.debug("Skipping {0}, due to missing required key"
                                  "'{1}'.".format(module.name, module.only_if))
                        continue

                    if (self._cycle_count + 1) % module.every != 0:
                        log.debug("Skipping {0} (every {1} iterations)."
                                  .format(module.name, module.every))
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
                self._timeit['cycles_cpu'].append(time.clock() -
                                                  cycle_start_cpu)
                self._cycle_count += 1
                if cycles and self._cycle_count >= cycles:
                    raise StopIteration
        except StopIteration:
            log.info("Nothing left to pump through.")
        self.finish()

    def drain(self, cycles=None):
        """Execute _drain while trapping KeyboardInterrupt"""
        self.init_timer.stop()
        log.info("Trapping CTRL+C and starting to drain.")
        signal.signal(signal.SIGINT, self._handle_ctrl_c)
        with ignored(KeyboardInterrupt):
            self._drain(cycles)

    def finish(self):
        """Call finish() on each attached module"""
        for module in self.modules:
            if hasattr(module, 'pre_finish'):
                log.info("Finishing {0}".format(module.name))
                start_time = timer()
                start_time_cpu = time.clock()
                module.pre_finish()
                self._timeit[module]['finish'] = timer() - start_time
                self._timeit[module]['finish_cpu'] = \
                    time.clock() - start_time_cpu
            else:
                log.info("Skipping function module {0}".format(module.name))
        self._timeit['finish'] = timer()
        self._timeit['finish_cpu'] = time.clock()
        self._print_timeit_statistics()
        self._finished = True

    def _handle_ctrl_c(self, *args):
        """Handle the keyboard interrupts."""
        if self._stop:
            print("\nForced shutdown...")
            raise SystemExit
        if not self._stop:
            hline = 42*'='
            print('\n' + hline + "\nGot CTRL+C, waiting for current cycle...\n"
                  "Press CTRL+C again if you're in hurry!\n" + hline)
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

        print(60*'=')
        print("{0} cycles drained in {1} (CPU {2}). Memory peak: {3:.2f} MB"
              .format(self._cycle_count,
                      timef(overall), timef(overall_cpu), memory))
        if self._cycle_count > n_cycles:
            print("Statistics are based on the last {0} cycles."
                  .format(n_cycles))
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
            print(module.name +
                  " - process: {0:.3f}s (CPU {1:.3f}s)"
                  " - finish: {2:.3f}s (CPU {3:.3f}s)"
                  .format(sum(process_times), sum(process_times_cpu),
                          finish_time, finish_time_cpu))
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
        self.timeit = self.get('timeit') or False
        self._timeit = {'process': deque(maxlen=STAT_LIMIT),
                        'process_cpu': deque(maxlen=STAT_LIMIT),
                        'finish': 0,
                        'finish_cpu': 0}
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
            raise TypeError("{0} requires the parameter '{1}'."
                            .format(self.__class__, name))
        return value

    def process(self, blob):  # pylint: disable=R0201
        """Knead the blob and return it"""
        return blob

    def finish(self):
        """Clean everything up."""
        pass

    def pre_finish(self):
        """Do the last few things before calling finish()"""
        self.finish()

    def __call__(self, *args, **kwargs):
        """Run process if directly called."""
        log.info("Calling process")
        return self.process(*args, **kwargs)


class Pump(Module):
    """The pump with basic file or socket handling."""

    def __init__(self, *args, **kwargs):
        self.blob_file = None
        if args:
            log.warning("Non-keywords argument passed. Please use keyword "
                        "arguments to supress this warning. I will assume the "
                        "first argument to be the `filename`.")
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
        self.finish()
        self.close_file()


class Blob(OrderedDict):
    """A simple (ordered) dict with a fancy name. This should hold the data."""
    pass


class Geometry(object):
    def __init__(self, *args, **kwargs):
        log.error("The 'Geometry' class has been renamed to 'Calibration'!")


class Calibration(Module):
    """A very simple, preliminary Module which gives access to the calibration.

    Parameters
    ----------
    apply: bool, optional [default=False]
        Apply the calibration to the hits (add position/direction/t0)?
    filename: str, optional [default=None]
        DetX file with detector description.
    det_id: int, optional
        .detx ID of detector (when retrieving from database).
    t0set: optional
        t0set (when retrieving from database).
    calibration: optional
        calibration (when retrieving from database).
    """
    def configure(self):
        self._should_apply = self.get('apply') or False
        self.filename = self.get('filename') or None
        self.det_id = self.get('det_id') or None
        self.t0set = self.get('t0set') or None
        self.calibration = self.get('calibration') or None
        self.detector = self.get('detector') or None
        self._pos_dom_channel = None
        self._dir_dom_channel = None
        self._t0_dom_channel = None
        self._pos_pmt_id = None
        self._dir_pmt_id = None
        self._t0_pmt_id = None

        if self.filename or self.det_id:
            if self.filename is not None:
                self.detector = Detector(filename=self.filename)
            if self.det_id:
                self.detector = Detector(det_id=self.det_id,
                                         t0set=self.t0set,
                                         calibration=self.calibration)

        if self.detector is not None:
            log.debug("Creating lookup tables")
            self._create_dom_channel_lookup()
            self._create_pmt_id_lookup()
        else:
            log.critical("No detector information loaded.")

    def process(self, blob, key='Hits'):
        if self._should_apply:
            self.apply(blob[key])
        return blob

    def get_detector(self):
        """Return the detector"""
        return self.detector

    def apply(self, hits):
        """Add x, y, z, t0 (and du, floor if DataFrame) columns to hit.

        When applying to ``RawHitSeries`` or ``McHitSeries``, a ``HitSeries``
        will be returned with the calibration information added.
        
        """
        if isinstance(hits, (HitSeries, list)):
            self._apply_to_hitseries(hits)
        elif isinstance(hits, pd.DataFrame):
            self._apply_to_table(hits)
        elif isinstance(hits, RawHitSeries):
            return self._apply_to_rawhitseries(hits)
        elif isinstance(hits, McHitSeries):
            return self._apply_to_mchitseries(hits)
        else:
            raise TypeError("Don't know how to apply calibration to '{0}'."
                            .format(hits.__class__.__name__))

    def _apply_to_hitseries(self, hits):
        """Add x, y, z and t0 offset to hit series"""
        for idx, hit in enumerate(hits):
            try:
                pmt = self.detector.get_pmt(hit.dom_id, hit.channel_id)
            except (KeyError, AttributeError):
                pmt = self.detector.pmt_with_id(hit.pmt_id)
            hits.pos_x[idx] = pmt.pos[0]
            hits.pos_y[idx] = pmt.pos[1]
            hits.pos_z[idx] = pmt.pos[2]
            hits.dir_x[idx] = pmt.dir[0]
            hits.dir_y[idx] = pmt.dir[1]
            hits.dir_z[idx] = pmt.dir[2]
            hits._arr['t0'][idx] = pmt.t0
            hits._arr['time'][idx] += pmt.t0
            # hit.a = hit.tot

    def _apply_to_rawhitseries(self, hits):
        """Create a HitSeries from RawHitSeries and add pos, dir and t0.
        
        Note that existing arrays like tot, dom_id, channel_id etc. will be
        copied by reference for better performance.
        
        """
        n = len(hits)
        cal = np.empty((n, 9))
        for i in range(n):
            lookup = self._calib_by_dom_and_channel
            calib = lookup[hits._arr['dom_id'][i]][hits._arr['channel_id'][i]]
            cal[i] = calib
        h = np.empty(n, CRawHitSeries.dtype)
        h['channel_id'] = hits.channel_id
        h['dir_x'] = cal[:, 3]
        h['dir_y'] = cal[:, 4]
        h['dir_z'] = cal[:, 5]
        h['dom_id'] = hits.dom_id
        h['du'] = cal[:, 7]
        h['floor'] = cal[:, 8]
        h['pos_x'] = cal[:, 0]
        h['pos_y'] = cal[:, 1]
        h['pos_z'] = cal[:, 2]
        h['t0'] = cal[:, 6]
        h['time'] = hits.time + cal[:, 6]
        h['tot'] = hits.tot
        h['triggered'] = hits.triggered
        h['event_id'] = hits._arr['event_id']
        return CRawHitSeries(h, hits.event_id)

    def _apply_to_mchitseries(self, hits):
        """Create a HitSeries from McHitSeries and add pos, dir and t0.
        
        Note that existing arrays like a, origin, pmt_id will be copied by
        reference for better performance.

        The attributes ``a`` and ``origin`` are not implemented yet.
        
        """
        n = len(hits)
        cal = np.empty((n, 9))
        for i in range(n):
            lookup = self._calib_by_pmt_id
            calib = lookup[hits._arr['pmt_id'][i]]
        h = np.empty(n, CMcHitSeries.dtype)
        h['channel_id'] = np.zeros(n, dtype=int)
        h['dir_x'] = cal[:, 3]
        h['dir_y'] = cal[:, 4]
        h['dir_z'] = cal[:, 5]
        h['du'] = cal[:, 7]
        h['floor'] = cal[:, 8]
        h['pmt_id'] = hits._arr['pmt_id']
        h['pos_x'] = cal[:, 0]
        h['pos_y'] = cal[:, 1]
        h['pos_z'] = cal[:, 2]
        h['t0'] = cal[:, 6]
        h['time'] = hits.time + cal[:, 6]
        h['tot'] = np.zeros(n, dtype=int)
        h['triggered'] = np.zeros(n, dtype=bool)
        h['event_id'] = hits._arr['event_id']
        return CMcHitSeries(h, hits.event_id)

    def _apply_to_table(self, table):
        """Add x, y, z and du, floor columns to hit table"""
        def get_pmt(hit):
            return self.detector.get_pmt(hit['dom_id'], hit['channel_id'])

        table['pos_x'] = table.apply(lambda h: get_pmt(h).pos.x, axis=1)
        table['pos_y'] = table.apply(lambda h: get_pmt(h).pos.y, axis=1)
        table['pos_z'] = table.apply(lambda h: get_pmt(h).pos.z, axis=1)
        table['dir_x'] = table.apply(lambda h: get_pmt(h).dir.x, axis=1)
        table['dir_y'] = table.apply(lambda h: get_pmt(h).dir.y, axis=1)
        table['dir_z'] = table.apply(lambda h: get_pmt(h).dir.z, axis=1)
        table['time'] += table.apply(lambda h: get_pmt(h).t0, axis=1)
        table['t0'] = table.apply(lambda h: get_pmt(h).t0, axis=1)
        table['du'] = table.apply(lambda h: get_pmt(h).omkey[0], axis=1)
        table['floor'] = table.apply(lambda h: get_pmt(h).omkey[1], axis=1)

    def _create_dom_channel_lookup(self):
        data = {}
        for dom_id, pmts in self.detector._pmts_by_dom_id.items():
            for pmt in pmts:
                if dom_id not in data:
                    data[dom_id] = {}
                data[dom_id][pmt.channel_id] = np.array((pmt.pos[0],
                                                        pmt.pos[1],
                                                        pmt.pos[2],
                                                        pmt.dir[0],
                                                        pmt.dir[1],
                                                        pmt.dir[2],
                                                        pmt.t0,
                                                        pmt.omkey[0],
                                                        pmt.omkey[1]))
        self._calib_by_dom_and_channel = data

    def _create_pmt_id_lookup(self):
        data = {}
        for pmt_id, pmt in self.detector._pmts_by_id.items():
            data[pmt_id] = np.array((pmt.pos[0],
                                     pmt.pos[1],
                                     pmt.pos[2],
                                     pmt.dir[0],
                                     pmt.dir[1],
                                     pmt.dir[2],
                                     pmt.t0,
                                     pmt.omkey[0],
                                     pmt.omkey[1],
                                     ))
        self._calib_by_pmt_id = data

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Calibration: det_id({0})".format(self.det_id)


class Run(object):
    """A simple container for event info, hits, tracks and calibration.
    """
    def __init__(self, **tables):
        for key, val in tables.items():
            setattr(self, key, val)
