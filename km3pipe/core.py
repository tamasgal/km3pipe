from __future__ import division, absolute_import, print_function

__author__ = 'tamasgal'

class Pipeline(object):
    """The holy pipeline which holds everything together"""

    def __init__(self):
        self.modules = []
        self.blob = Blob()

    def attach(self, module_class, name, **kwargs):
        """Attach a module to the pipeline system"""
        self.modules.append(module_class(name=name, **kwargs))

    def drain(self):
        """Activate the pump and let the flow go"""
        for module in self.modules:
            module.process(self.blob)


class Module(object):
    """The module which can be attached to the pipeline"""

    def __init__(self, name=None, **parameters):
        self._name = name
        self.parameters = parameters

    @property
    def name(self):
        return self._name

    def add(self, name, value, help_text):
        self.parameters[name] = value

    def get(self, name):
        return self.parameters.get(name)

    def process(self, blob):
        """Knead the blob and return it"""
        return blob


class Blob(dict):
    pass
