# Filename: mc.py
# pylint: disable=C0103
"""
Monte Carlo related things.

"""
from __future__ import absolute_import, print_function, division

import numpy as np

from .logger import get_logger

__author__ = "Tamas Gal and Moritz Lotze"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal and Moritz Lotze"
__email__ = "tgal@km3net.de"
__status__ = "Development"

log = get_logger(__name__)    # pylint: disable=C0103


def geant2pdg(geant_code):
    """Convert GEANT particle ID to PDG"""
    conversion_table = {
        1: 22,    # photon
        2: -11,    # positron
        3: 11,    # electron
        5: -13,    # muplus
        6: 13,    # muminus
        7: 111,    # pi0
        8: 211,    # piplus
        9: -211,    # piminus
        10: 130,    # k0long
        11: 321,    # kplus
        12: -321,    # kminus
        13: 2112,    # neutron
        14: 2212,    # proton
        16: 310,    # kaon0short
        17: 221,    # eta
    }
    try:
        return conversion_table[geant_code]
    except KeyError:
        return 0


_PDG2NAME = {
    1: 'd',
    2: 'u',
    3: 's',
    4: 'c',
    5: 'b',
    6: 't',
    11: 'e-',
    -11: 'e+',
    12: 'nu_e',
    -12: 'anu_e',
    13: 'mu-',
    -13: 'mu+',
    14: 'nu_mu',
    -14: 'anu_mu',
    15: 'tau-',
    -15: 'tau+',
    16: 'nu_tau',
    -16: 'anu_tau',
    22: 'photon',
    111: 'pi0',
    130: 'K0L',
    211: 'pi-',
    -211: 'pi+',
    310: 'K0S',
    311: 'K0',
    321: 'K+',
    -321: 'K-',
    2112: 'n',
    2212: 'p',
    -2212: 'p-',
}

_NAME2PDG = {val: key for key, val in _PDG2NAME.items()}    # noqa


def pdg2name(pdg_id):
    """Convert PDG ID to human readable names"""
    # pylint: disable=C0330
    try:
        return _PDG2NAME[pdg_id]
    except KeyError:
        return "N/A"


def name2pdg(name):
    try:
        return _NAME2PDG[name]
    except KeyError:
        return 0


def most_energetic(df):
    """Grab most energetic particle from mc_tracks dataframe."""
    idx = df.groupby(['event_id'])['energy'].transform(max) == df['energy']
    return df[idx].reindex()


def leading_particle(df):
    """Grab leading particle (neutrino, most energetic bundle muon).

    Note: selecting the most energetic mc particle does not always select
    the neutrino! In some sub-percent cases, the post-interaction
    secondaries can have more energy than the incoming neutrino!

    aanet convention: mc_tracks[0] = neutrino
    so grab the first row

    if the first row is not unique (neutrinos are unique), it's a muon bundle
    grab the most energetic then
    """
    leading = df.groupby('event_id', as_index=False).first()
    unique = leading.type.unique()

    if len(unique) == 1 and unique[0] == 0:
        leading = most_energetic(df)
    return leading


def get_flavor(pdg_types):
    """Build a 'flavor' from the 'type' column."""
    import pandas as pd
    return pd.Series(pdg_types).apply(pdg2name)


def _p_eq_nu(pdg_type):
    return np.abs(pdg_type) in {12, 14, 16}


def _p_eq_mu(pdg_type):
    return pdg_type == -13


def is_neutrino(pdg_types):
    """flavor string -> is_neutrino"""
    import pandas as pd
    return pd.Series(pdg_types).apply(_p_eq_nu)


def is_muon(pdg_types):
    """flavor string -> is_neutrino"""
    import pandas as pd
    return pd.Series(pdg_types).apply(_p_eq_mu)
