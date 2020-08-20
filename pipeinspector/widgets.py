# Filename: widgets.py
"""
GUI elements for the pipeinspector.

"""

import urwid
import pprint
import math
import sys

from pipeinspector.settings import UI

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"


class BlobBrowser(urwid.Frame):
    def __init__(self):
        self.items = []
        self.cursor_position = 0

        self.header = urwid.AttrMap(urwid.Text("Keys:"), "head")

        self.listbox = urwid.ListBox(urwid.SimpleListWalker(self.items))
        self.frame = urwid.Frame(self.listbox, header=self.header)
        line_box = urwid.AttrMap(urwid.LineBox(self.frame), "body")
        urwid.Frame.__init__(self, line_box)
        self.overlay = None
        self.popup = None

    def load(self, blob):
        del self.listbox.body[:]
        new_items = []
        for key in sorted(blob.keys()):
            item_widget = ItemWidget(key, blob[key])
            new_items.append(item_widget)
            urwid.connect_signal(item_widget, "key_selected", self.key_selected)
        self.listbox.body.extend(new_items)
        self.listbox.set_focus(self.cursor_position)

    def key_selected(self, data):
        def formatter(obj):
            if hasattr(obj, "__insp__"):
                return obj.__insp__()
            if hasattr(obj, "size"):
                output = ""
                for obj in data:
                    output += str(obj) + "\n"
                return output
            return pprint.pformat(obj)

        content = [urwid.Text(line) for line in formatter(data).split("\n")]
        self.popup = urwid.ListBox(content)
        popup_box = urwid.LineBox(self.popup)
        self.overlay = urwid.Overlay(
            popup_box,
            self.body,
            "center",
            ("relative", 80),
            "middle",
            ("relative", 80),
        )
        self.body = self.overlay

    def keypress(self, size, key):
        input = urwid.Frame.keypress(self, size, key)
        self.cursor_position = self.listbox.focus_position
        if self.overlay:
            if input in UI.keys["escape"]:
                self.body = self.overlay.bottom_w
                self.overlay = None
        else:
            return input


class ItemWidget(urwid.WidgetWrap):
    signals = ["key_selected"]

    def __init__(self, key, data):
        self.key = key
        self.data = data
        try:
            size_label = str(len(data)) + " items"
        except TypeError:
            size_label = str(sys.getsizeof(data)) + " bytes"

        type_label = str(type(data)).split("'")[1]

        self.item = [
            (
                "fixed",
                35,
                urwid.Padding(urwid.AttrWrap(urwid.Text(key), "body", "focus"), left=2),
            ),
            urwid.AttrWrap(urwid.Text(type_label), "body", "focus"),
            urwid.AttrWrap(urwid.Text(size_label), "body", "focus"),
        ]
        w = urwid.Columns(self.item)
        self.__super.__init__(w)

    def selectable(self):
        return True

    def keypress(self, size, key):
        if key == "x":
            urwid.emit_signal(self, "key_selected", self.data)
        return key


class BlobWidget(urwid.Pile):
    signals = ["blob_selected"]

    def __init__(self):
        self.width = 50
        self.size = (0,)
        self.index = 0
        urwid.Pile.__init__(
            self,
            [
                urwid.Text("", wrap="clip"),
                urwid.Text("", wrap="clip"),
                urwid.Text("", wrap="clip"),
            ],
        )

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
        urwid.emit_signal(self, "blob_selected", self.index)

    def _make_ruler(self, start):
        if start <= 10:
            start = 0
        else:
            start -= 10
        segment = "|    '    "
        repeat = int(math.ceil(self.width / len(segment)) + 1)
        ruler = segment * repeat
        slice_start = start % len(segment)
        slice_end = (start % len(segment)) + self.width
        return ruler[slice_start:slice_end]

    def _make_scale_labels(self, start):
        if start <= 10:
            start = 0
        else:
            start -= 10
        lowest_tick = int(math.floor(start / 10) * 10)
        highest_tick = lowest_tick + self.width
        ticks_labels = [
            "{0:<10}".format(i) for i in range(lowest_tick, highest_tick, 10)
        ]
        slice_start = start % 10
        slice_end = (start % 10) + self.width
        ticks = "".join(ticks_labels)[slice_start:slice_end]
        return ticks

    def _make_blob_icons(self, start):
        icon = "B"
        if start < 10:
            icons = "." + icon * (self.width - 1)
        else:
            icons = icon * self.width
        if start > 10:
            start = 10
        return [
            ("blob", icons[:start]),
            ("blob_selected", icons[start]),
            ("blob", icons[start + 1 :]),
        ]
