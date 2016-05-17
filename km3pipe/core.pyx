# coding=utf-8
# cython: profile=True
# Filename: core.pyx
# cython: embedsignature=True
# pylint: disable=locally-disabled
"""
The core of the KM3Pipe framework.

"""
from __future__ import division, absolute_import, print_function

import signal
import gzip
import time
from timeit import default_timer as timer

import numpy as np
import pandas as pd

from km3pipe.tools import peak_memory_usage, ignored
from km3pipe.hardware import Detector
from km3pipe.dataclasses import Position, Direction, HitSeries
from km3pipe.logger import logging

__author__ = 'tamasgal'

log = logging.getLogger(__name__)  # pylint: disable=C0103


class Pipeline(object):
    """The holy pipeline which holds everything together.

    If initialised with timeit=True, all modules will be monitored, otherwise
    only the overall statistics and modules with `timeit=True` will be
    shown.
    """

    def __init__(self, blob=None, timeit=False):
        self.modules = []
        self.geometry = None
        self.blob = blob or Blob()
        self.timeit = timeit
        self._timeit = {'init': timer(), 'init_cpu': time.clock(),
                        'cycles': [], 'cycles_cpu': []}
        self._cycle_count = 0
        self._stop = False
        self._finished = False

    def attach(self, module_class, name=None, **kwargs):
        """Attach a module to the pipeline system"""
        if not name:
            name = module_class.__name__
        module = module_class(name=name, **kwargs)
        log.info("Attaching module '{0}'".format(name))
        try:
            module.get_detector()
            self.geometry = module
            if module._should_apply:
                self.modules.append(module)
        except AttributeError:
            if len(self.modules) < 1 and not isinstance(module, Pump):
                log.error("The first module to attach to the pipeline should "
                          "be a Pump!")
            self.modules.append(module)

    def _drain(self, cycles=None):
        """Activate the pump and let the flow go.

        This will call the process() method on each attached module until
        a StopIteration is raised, usually by a pump when it reached the EOF.

        A StopIteration is also raised when self.cycles was set and the
        number of cycles has reached that limit.

        """
        if not cycles:
            log.info("No cycle count, the pipeline may be drained forever.")

        if self.geometry:
            log.info("Setting up the detector geometry.")
            for module in self.modules:
                module.detector = self.geometry.get_detector()

        try:
            while not self._stop:
                cycle_start = timer()
                cycle_start_cpu = time.clock()
                self._cycle_count += 1

                log.debug("Pumping blob #{0}".format(self._cycle_count))
                pump = self.modules[0]
                start = timer()
                start_cpu = time.clock()
                self.blob = pump.process(self.blob)
                pump._timeit['process'].append(timer() - start)
                pump._timeit['process_cpu'].append(time.clock() - start_cpu)

                for module in self.modules[1:]:
                    if self.blob is None:
                        log.debug("Skipping {0}, due to empty blob."
                                  .format(module.name))
                        continue
                    log.debug("Processing {0} ".format(module.name))
                    start = timer()
                    start_cpu = time.clock()
                    self.blob = module.process(self.blob)
                    if self.timeit or module.timeit:
                        module._timeit['process'] \
                            .append(timer() - start)
                        module._timeit['process_cpu'] \
                            .append(time.clock() - start_cpu)
                self._timeit['cycles'].append(timer() - cycle_start)
                self._timeit['cycles_cpu'].append(time.clock() -
                                                  cycle_start_cpu)
                if cycles and self._cycle_count >= cycles:
                    raise StopIteration
        except StopIteration:
            log.info("Nothing left to pump through.")
        self.finish()

    def drain(self, cycles=None):
        """Execute _drain while trapping KeyboardInterrupt"""
        log.info("Now draining...")
        signal.signal(signal.SIGINT, self._handle_ctrl_c)
        with ignored(KeyboardInterrupt):
            self._drain(cycles)

    def finish(self):
        """Call finish() on each attached module"""
        for module in self.modules:
            log.info("Finishing {0}".format(module.name))
            start_time = timer()
            start_time_cpu = time.clock()
            module.pre_finish()
            module._timeit['finish'] = timer() - start_time
            module._timeit['finish_cpu'] = time.clock() - start_time_cpu
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
            return "{0:.2f}{1}".format(time, unit)

        def statsf(prefix, values):
            stats = "  mean: {0}  medi: {1}  min: {2}  max: {3}  std: {4}"
            values = [timef(v) for v in values]
            return "  " + prefix + stats.format(*values)

        cycles = self._timeit['cycles']
        n_cycles = len(cycles)
        if n_cycles < 1:
            return
        cycles_cpu = self._timeit['cycles_cpu']
        overall = self._timeit['finish'] - self._timeit['init']
        overall_cpu = self._timeit['finish_cpu'] - self._timeit['init_cpu']
        memory = peak_memory_usage()

        print(80*'=')
        print("{0} cycles drained in {1} (CPU {2}). Memory peak: {3:.2f} MB"
              .format(n_cycles, timef(overall), timef(overall_cpu), memory))
        print(statsf('wall', calc_stats(cycles)))
        print(statsf('CPU ', calc_stats(cycles_cpu)))

        for module in self.modules:
            if not module.timeit and not self.timeit:
                continue
            finish_time = module._timeit['finish']
            finish_time_cpu = module._timeit['finish_cpu']
            process_times = module._timeit['process']
            process_times_cpu = module._timeit['process_cpu']
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
        self.detector = None
        self.timeit = self.get('timeit') or False
        self._timeit = {'process': [], 'finish': 0,
                        'process_cpu': [], 'finish_cpu': 0}

    @property
    def name(self):
        """The name of the module"""
        return self._name

    def add(self, name, value):
        """Add the parameter with the desired value to the dict"""
        self.parameters[name] = value

    def get(self, name):
        """Return the value of the requested parameter"""
        return self.parameters.get(name)

    def process(self, blob):  # pylint: disable=R0201
        """Knead the blob and return it"""
        return blob

    def finish(self):
        """Clean everything up."""
        pass

    def pre_finish(self):
        """Do the last few things before calling finish()"""
        self.finish()


class Pump(Module):
    """The pump with basic file or socket handling."""

    def __init__(self, **context):
        Module.__init__(self, **context)
        self.blob_file = None

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

    def pre_finish(self):
        """Clean up open file or socket-handlers."""
        Module.finish(self)
        self.close_file()


class Blob(dict):
    """A simple dict with a fancy name. This should hold the data."""
    pass


class Geometry(Module):
    """A very simple, preliminary Module which gives access to the geometry"""
    def __init__(self, **context):
        super(self.__class__, self).__init__(**context)
        self._should_apply = self.get('apply') or False
        self.filename = self.get('filename') or None
        self.det_id = self.get('det_id') or None

        if self.filename or self.det_id:
            if self.filename is not None:
                self.detector = Detector(filename=self.filename)
            if self.det_id:
                self.detector = Detector(det_id=self.det_id)
        else:
            raise ValueError("Define either a filename or a detector ID.")

    def process(self, blob, key='Hits'):
        if self._should_apply:
            self.apply(blob[key])
        return blob

    def get_detector(self):
        """Return the detector"""
        return self.detector

    def apply(self, hits):
        """Add x, y, z (and du, floor if DataFrame) columns to hit"""
        if isinstance(hits, (HitSeries, list)):
            self._apply_to_hitseries(hits)
        elif isinstance(hits, pd.DataFrame):
            self._apply_to_table(hits)
        else:
            raise TypeError("Don't know how to apply geometry to '{0}'."
                            .format(hits.__class__.__name__))

    def _apply_to_hitseries(self, hits):
        """Add x, y and z to hit series"""
        for hit in hits:
            try:
                pmt = self.detector.get_pmt(hit.dom_id, hit.channel_id)
            except KeyError:
                pmt = self.detector.pmt_with_id(hit.pmt_id)
            hit.pos = Position(pmt.pos)
            hit.dir = Direction(pmt.dir)
            hit.t0 = pmt.t0
            hit.time += pmt.t0
            hit.a = hit.tot

    def _apply_to_table(self, table):
        """Add x, y, z and du, floor columns to hit table"""
        def get_pmt(hit):
            return self.detector.get_pmt(hit['dom_id'], hit['channel_id'])

        table['x'] = table.apply(lambda h: get_pmt(h).pos.x, axis=1)
        table['y'] = table.apply(lambda h: get_pmt(h).pos.y, axis=1)
        table['z'] = table.apply(lambda h: get_pmt(h).pos.z, axis=1)
        table['du'] = table.apply(lambda h: get_pmt(h).omkey[0], axis=1)
        table['floor'] = table.apply(lambda h: get_pmt(h).omkey[1], axis=1)


class AanetGeometry(Module):
    """AAnet based Geometry using Det()"""
    def __init__(self, **context):
        import aa  # noqa
        from ROOT import Det
        super(self.__class__, self).__init__(**context)
        filename = self.get('filename')
        self.detector = Det(filename)

    def get_detector(self):
        """Return the detector"""
        return self.detector
