# coding=utf-8
# Filename: style.py
# pylint: disable=locally-disabled
"""
The KM3Pipe style definitions.

"""
from __future__ import division, absolute_import, print_function

import seaborn as sns


style_definition = {'grid.color': '.85',
                    'grid.linestyle': u'--',
                    'text.color': '.15',
                    'xtick.color': '.15',
                    'xtick.direction': u'in',
                    'xtick.major.size': 5.0,
                    'xtick.minor.size': 2.0,
                    'ytick.color': '.15',
                    'ytick.direction': u'in',
                    'ytick.major.size': 5.0,
                    'ytick.minor.size': 2.0,
                    'axes.labelcolor': '.45',
                    'font.sans-serif': [u'Helvetica Neue',
                                        u'Helvetica', u'Arial',
                                        u'Liberation Sans',
                                        u'Bitstream Vera Sans',
                                        u'sans-serif']}

sns.set_style('whitegrid', style_definition)
# sns.set_palette("husl")

colors = ["coral", "turquoise blue", "orangey yellow", "avocado",
          "neon purple", "steel grey", "marine"]
sns.set_palette(sns.xkcd_palette(colors))


def set_context(context):
    contexts = {
            'notebook': {'font_scale': 1.3,
                         'rc': {'line.linewidth': 2.5}},
            'paper':    {'font_scale': 1.0,
                         'rc': {}},
            'poster':   {'font_scale': 1.0,
                         'rc': {}},
            'talk':     {'font_scale': 1.5,
                         'rc': {}}
            }
    sns.set_context(context,
                    font_scale=contexts[context]['font_scale'],
                    rc=contexts[context]['rc'])
