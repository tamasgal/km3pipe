__author__ = 'tamasgal'

class Pipeline(object):
    """The holy pipeline which holds everything together"""

    def __init__(self):
        self.modules = []

    def attach(self, module):
        """Attach a module to the pipeline system"""
        self.modules.append(module)

    def drain(self):
        """Activate the pump and let the flow go"""
        pass



class Module(object):
    """The module which can be attached to the pipeline"""

    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name


class Blob(object):
    pass
