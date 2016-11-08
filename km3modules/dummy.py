"""Dummy

This is designed to fail
"""

from km3pipe import Module


class Dummy(Module):
    """Dummy base class.

    """

    def __init__(self, **kwargs):
        super(Dummy, self).__init__(**kwargs)
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
        super(SubDummy, self).__init__(**kwargs)
        self.hit_sel = self.get('hit_sel') or 'Hits'
        self.key_out = self.get('key_out') or 'PrimFit'

    def fit(self):
        return 4200
