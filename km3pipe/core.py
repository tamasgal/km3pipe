# coding=utf-8
# Filename: core.py
# pylint: disable=locally-disabled
"""
The core of the KM3Pipe framework.

"""
from __future__ import division, absolute_import, print_function

__author__ = 'tamasgal'

from km3pipe.hardware import Detector

import logging
from km3pipe.logger import logging

log = logging.getLogger(__name__)  # pylint: disable=C0103


class Pipeline(object):
    """The holy pipeline which holds everything together"""

    def __init__(self, blob=None):
        self.modules = []
        self.geometry = None
        self.blob = blob or Blob()
        self._cycle_count = 0

    def attach(self, module_class, name='Unnamed', **kwargs):
        """Attach a module to the pipeline system"""
        module = module_class(name=name, **kwargs)
        log.info("Attaching module '{0}'".format(name))
        try:
            module.get_detector()
            self.geometry = module
        except AttributeError:
            self.modules.append(module)

    def drain(self, cycles=None):
        """Activate the pump and let the flow go.

        This will call the process() method on each attached module until
        a StopIteration is raised, usually by a pump when it reached the EOF.

        A StopIteration is also raised when self.cycles was set and the
        number of cycles has reached that limit.

        """
        log.info("No cycle count set, the pipeline may be drained forever.")
        if self.geometry:
            log.info("Setting up the detector geometry.")
            for module in self.modules:
                module.detector = self.geometry.get_detector()

        try:
            while True:
                self._cycle_count += 1
                log.debug("Pumping blob #{0}".format(self._cycle_count))
                self.blob = self.modules[0].process(self.blob)
                for module in self.modules[1:]:
                    if self.blob is None:
                        log.debug("Skipping {0}, due to empty blob.".format(module.name))
                        continue
                    log.debug("Processing {0} ".format(module.name))
                    self.blob = module.process(self.blob)
                if cycles and self._cycle_count >= cycles:
                    raise StopIteration
        except StopIteration:
            log.info("Nothing left to pump through.")
        self.finish()

    def finish(self):
        """Call finish() on each attached module"""
        for module in self.modules:
            log.info("Finishing {0}".format(module.name))
            module.pre_finish()


class Module(object):
    """The module which can be attached to the pipeline"""

    def __init__(self, name=None, **parameters):
        log.debug("Initialising {0}".format(name))
        self._name = name
        self.parameters = parameters
        self.detector = None

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
            self.blob_file = open(filename, 'rb')
        except TypeError:
            log.error("Please specify a valid filename.")
            raise SystemExit
        except IOError as error_message:
            log.error(error_message)
            raise SystemExit

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
        filename = self.get('filename')
        self.detector = Detector(filename)

    def get_detector(self):
        return self.detector