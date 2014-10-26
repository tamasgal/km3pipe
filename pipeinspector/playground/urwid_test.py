try:
    import urwid
except ImportError:
    print("Could not import the Python module 'urwid'.")
    raise SystemExit

import random


def handle_input(input):
    if input in UI.keys['escape']:
        raise urwid.ExitMainLoop()
    if input in UI.keys['left'] + UI.keys['right']:
        main_frame.set_focus('footer')
        next_blob()
    if input in UI.keys['down'] + UI.keys['up']:
        main_frame.set_focus('body')


class UI(object):
    """Represents the settings for the UI."""
    fg = 'light gray'
    bg = 'black'

    palette = [
        ('default', fg, bg),
        ('highlight', fg+',standout', bg),
        ('header', 'white', 'dark cyan'),
        ('footer', 'light gray', 'dark blue'),
        ('body','dark cyan', '', 'standout'),
        ('focus','dark red', '', 'standout'),
        ('head','light red', 'black'),
        ('blob', 'yellow', 'dark cyan'),
        ('blob_scale', 'dark cyan', 'black'),
        ] 

    keys = {
        'select': ('return','enter'),
        'inspect': ('x','X'),
        'escape': ('esc','q','Q'),
        'left': ('left','h'),
        'right': ('right','l'),
        'up': ('up','k'),
        'down': ('down','j'),
        'goto':('g','G'),
        'help':('?',),
        }


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


class BlobSelector(urwid.WidgetWrap):
    def __init__ (self, description):
        self.content = description
        self.item = [
            urwid.AttrWrap(urwid.Text('%s' % description), 'blob', 'focus'),
        ]
        w = urwid.Columns(self.item)
        self.__super.__init__(w)

    def selectable (self):
        return True

    def keypress(self, size, key):
        return key


def next_blob():
    pass

header = urwid.AttrWrap(urwid.Text("The header!", align='center'), 'header')
footer = urwid.AttrWrap(urwid.Text("The footer"), 'footer')


items = []
for i in range(100):
    items.append(ItemWidget(i, "Item {0}".format(i)))

browser_header = urwid.AttrMap(urwid.Text('selected:'), 'head')
browser_listbox = urwid.ListBox(urwid.SimpleListWalker(items))
browser_view = urwid.Frame(urwid.AttrWrap(browser_listbox, 'body'), header=browser_header)

blobs = []
for i in range(100):
    blobs.append((1, BlobSelector(u"\u00A4")))
blob_columns = urwid.AttrMap(urwid.Columns(blobs[:20]), 'footer')
blob_scale_labels = urwid.AttrMap(urwid.Text("1   5    10"), 'blob_scale')
blob_scale = urwid.AttrMap(urwid.Text("|    .    |"), 'blob_scale')
blobs = urwid.Pile([blob_scale_labels, blob_scale, blob_columns])

footer = urwid.Columns([urwid.Text('Info\nSecond kube'), blobs])

main_frame = urwid.AttrWrap(urwid.Frame(browser_view, focus_part='footer'), 'default')
main_frame.set_header(header)
main_frame.set_footer(footer)

loop = urwid.MainLoop(main_frame, UI.palette, unhandled_input=handle_input)
loop.run()
