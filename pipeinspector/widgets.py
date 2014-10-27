# coding=utf-8
# Filename: widgets.py
"""
GUI elements for the pipeinspector.

"""
from __future__ import division, absolute_import, print_function

import urwid
import math


class BlobSelector(urwid.WidgetWrap):
    def __init__ (self, description):
        self.content = description
        self.item = [
            urwid.AttrWrap(urwid.Text('%s' % description), 'blob', 'focus'),
        ]
        w = urwid.Columns(self.item)
        self.__super.__init__(w)

    def selectable(self):
        return True

    def keypress(self, size, key):
        return key


class BlobWidget(urwid.Pile):
    signals = ['next_blob']
    def __init__(self):
        self.width = 25
        self.size = (0,)
        self.index = 0
        urwid.Pile.__init__(self, [urwid.Text('', wrap='clip'), 
                                   urwid.Text('', wrap='clip'), 
                                   urwid.Text('', wrap='clip')])

    def goto_blob(self, position):
        self.index = position
        self.draw()

    def previous_blob(self, step=1):
        self.index -= step
        if self.index <= 0:
            self.index = 0
        self.draw()

    def next_blob(self, step=1):
        self.index += step
        self.draw()

    def draw(self):
        self.widget_list[0].set_text(self._make_blob_icons(self.index))
        self.widget_list[1].set_text(self._make_ruler(self.index))
        self.widget_list[2].set_text(self._make_scale_labels(self.index))

    def render(self, size, focus):
        self.size = size
        self.draw()
        return urwid.Pile.render(self, size, focus)

    def _make_ruler(self, start):
        if start <= 10:
            start = 0
        else:
            start -= 10
        segment = "|    '    "
        repeat = int(math.ceil(self.width / len(segment)) + 1)
        ruler = segment * repeat
        slice_start = (start % len(segment))
        slice_end = (start % len(segment)) + self.width
        return ruler[slice_start:slice_end]

    def _make_scale_labels(self, start):
        if start <= 10:
            start = 0
        else:
            start -= 10
        lowest_tick = int(math.floor(start / 10) * 10)
        highest_tick = lowest_tick + self.width
        ticks_labels = ['{0:<10}'.format(i)
                 for i in range(lowest_tick, highest_tick, 10)]
        slice_start = (start % 10)
        slice_end = (start % 10) + self.width
        ticks = ''.join(ticks_labels)[slice_start:slice_end]
        return ticks

    def _make_blob_icons(self, start):
        icon = u'\u00A4'
        if start < 10:
            icons = u'.' + icon * (self.width - 1)
        else:
            icons = icon * self.width
        if start > 10:
            start = 10
        return [('blob', icons[:start]),
                ('blob_selected', icons[start]),
                ('blob', icons[start + 1:])]


