from __future__ import division, absolute_import, print_function

__author__ = 'tamasgal'

from time import sleep

import logging
logging.addLevelName( logging.INFO, "\033[1;32m%s\033[1;0m" % logging.getLevelName(logging.INFO))
logging.addLevelName( logging.DEBUG, "\033[1;34m%s\033[1;0m" % logging.getLevelName(logging.DEBUG))
logging.addLevelName( logging.WARNING, "\033[1;33m%s\033[1;0m" % logging.getLevelName(logging.WARNING))
logging.addLevelName( logging.ERROR, "\033[1;31m%s\033[1;0m" % logging.getLevelName(logging.ERROR))

# create logger
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create formatter
#formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#formatter = logging.Formatter('[\x1b[32m%(levelname)s\033[0m] %(name)s: %(message)s')
formatter = logging.Formatter('[%(levelname)s] %(name)s: %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
log.addHandler(ch)

class Pipeline(object):
    """The holy pipeline which holds everything together"""

    def __init__(self, blob=None):
        self.modules = []
        self.blob = blob or Blob()

    def attach(self, module_class, name, **kwargs):
        """Attach a module to the pipeline system"""
        log.debug("Attaching module '{0}'".format(name))
        self.modules.append(module_class(name=name, **kwargs))

    def drain(self):
        """Activate the pump and let the flow go"""
        try:
            while True:
                for module in self.modules:
                    log.warning("Processing {0} ".format(module.name))
                    self.blob = module.process(self.blob)
        except StopIteration:
            log.info("Nothing left to pump through.")
        self.finish()

    def finish(self):
        for module in self.modules:
            log.info("Finishing {0}".format(module.name))
            module.finish()

class Module(object):
    """The module which can be attached to the pipeline"""

    def __init__(self, name=None, **parameters):
        log.info("Initialising {0}".format(name))
        self._name = name
        self.parameters = parameters

    @property
    def name(self):
        """The name of the module"""
        return self._name

    def add(self, name, value, help_text):
        """Add the parameter with the desired value to the dict"""
        self.parameters[name] = value

    def get(self, name):
        """Return the value of the requested parameter"""
        return self.parameters.get(name)

    def process(self, blob):
        """Knead the blob and return it"""
        return blob

    def finish(self):
        pass


class Blob(dict):
    """A simple dict with a fancy name."""
    pass
