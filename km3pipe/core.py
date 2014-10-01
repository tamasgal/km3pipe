from __future__ import division, absolute_import, print_function

__author__ = 'tamasgal'

from time import sleep

import logging
# create logger
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

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
        log.info("> Attaching module '{0}'".format(name))
        self.modules.append(module_class(name=name, **kwargs))

    def drain(self):
        """Activate the pump and let the flow go"""
        try:
            while True:
                for module in self.modules:
                    self.blob = module.process(self.blob)
        except StopIteration:
            print("Pipeline empty. Switching off the pump.")


class Module(object):
    """The module which can be attached to the pipeline"""

    def __init__(self, name=None, **parameters):
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
