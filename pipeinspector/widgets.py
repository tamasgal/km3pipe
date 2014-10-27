# coding=utf-8
# Filename: widgets.py
"""
GUI elements for the pipeinspector.

"""
from __future__ import division, absolute_import, print_function

import urwid
import math


class BlobFrame(urwid.Frame):
    def __init__(self):
        items = []
        for i in range(100):
            items.append(ItemWidget(i, "Item {0}".format(i)))
        browser_header = urwid.AttrMap(urwid.Text('selected:'), 'head')
        browser_listbox = urwid.ListBox(urwid.SimpleListWalker(items))
        inner_frame = urwid.Frame(browser_listbox, header=browser_header)
        line_box = urwid.AttrMap(urwid.LineBox(inner_frame), 'body')
        urwid.Frame.__init__(self, line_box)
        self.overlay = None


class ItemWidget (urwid.WidgetWrap):

    def __init__ (self, id, description):
        self.id = id
        self.content = 'item %s: %s...' % (str(id), description[:25])
        self.item = [
            ('fixed', 15, urwid.Padding(urwid.AttrWrap(
                urwid.Text('item %s' % str(id)), 'body', 'focus'), left=2)),
            urwid.AttrWrap(urwid.Text('%s' % description), 'body', 'focus'),
        ]
        w = urwid.Columns(self.item)
        self.__super.__init__(w)

    def selectable (self):
        return True

    def keypress(self, size, key):
        return key


class BlobWidget(urwid.Pile):
    signals = ['blob_selected']
    def __init__(self):
        self.width = 50
        self.size = (0,)
        self.index = 0
        urwid.Pile.__init__(self, [urwid.Text('', wrap='clip'), 
                                   urwid.Text('', wrap='clip'), 
                                   urwid.Text('', wrap='clip')])

    def goto_blob(self, position):
        self.index = position
        self._emit_blob_selected()
        self.draw()

    def previous_blob(self, step=1):
        self.index -= step
        if self.index <= 0:
            self.index = 0
        self._emit_blob_selected()
        self.draw()

    def next_blob(self, step=1):
        self.index += step
        self._emit_blob_selected()
        self.draw()

    def draw(self):
        self.widget_list[0].set_text(self._make_blob_icons(self.index))
        self.widget_list[1].set_text(self._make_ruler(self.index))
        self.widget_list[2].set_text(self._make_scale_labels(self.index))

    def render(self, size, focus):
        self.size = size
        self.draw()
        return urwid.Pile.render(self, size, focus)

    def _emit_blob_selected(self):
        urwid.emit_signal(self, 'blob_selected', self.index)

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


