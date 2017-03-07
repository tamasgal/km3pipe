#!/usr/bin/env python3
# vim:set ts=4 sts=4 sw=4 et:
"""MC Helpers.
"""
from __future__ import division, absolute_import, print_function

from km3pipe import Module
from km3pipe.mc import pdg2name, name2pdg
from km3pipe.math import zenith, azimuth


NEUTRINOS = {'nu_e', 'anu_e', 'nu_mu', 'anu_mu', }      # noqa


class McTruth(Module):
    """Extract MC info of 1st MC track.

    Parameters
    ----------
    most_energetic_primary: bool, default=True
    """
    def __init__(self, **context):
        super(self.__class__, self).__init__(**context)
        self.most_energetic_primary = bool(self.get('most_energetic_primary')) or True

    @classmethod
    def t2f(cls, row):
        return pdg2name(row['type'])

    @classmethod
    def is_nu(cls, flavor):
        return flavor in NEUTRINOS

    def process(self, blob):
        mc = blob['McTracks'].conv_to('pandas')
        if self.most_energetic_primary:
            mc.sort_values('energy', ascending=False, inplace=True)
        mc = mc.head(1)
        mc['zenith'] = zenith(mc[['dir_x', 'dir_y', 'dir_z']])
        mc['azimuth'] = azimuth(mc[['dir_x', 'dir_y', 'dir_z']])
        flavor = mc.apply(self.t2f, axis=1)
        mc['is_neutrino'] = flavor.apply(self.is_nu)
        blob['McTruth'] = mc
        return blob
