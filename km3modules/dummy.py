"""Dummy

This is designed to fail
"""
import numpy as np
import pandas as pd     # noqa

from km3pipe import Module

class Dummy(Module):
    """Dummy base class.

    """

    def __init__(self, **kwargs):
        super(self.__class__, self).__init__(**kwargs)
        self.hit_sel = self.get('hit_sel') or 'Hits'
        self.key_out = self.get('key_out') or 'PrimFit'

    def process(self, blob):
        print(self.fit())
        return blob

    def fit(self):
        return 42


class SubDummy(Dummy):
    """Primitive Sub-Subclass

    """
    def __init__(self, **kwargs):
        super(self.__class__, self).__init__(**kwargs)
        self.hit_sel = self.get('hit_sel') or 'Hits'
        self.key_out = self.get('key_out') or 'PrimFit'

    def fit(self):
        return 4200
