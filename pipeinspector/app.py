import urwid

from pipeinspector.widgets import (BlobWidget, BlobSelector)


def handle_input(input):
    if input in UI.keys['escape']:
        raise urwid.ExitMainLoop()
    if input in UI.keys['left']:
        blobs.previous_blob()
    if input in UI.keys['right']:
        blobs.next_blob()
    if input in UI.keys['down'] + UI.keys['up']:
        main_frame.set_focus('body')


def filter_input(keys, raw):
    if len(keys) == 1:
        if keys[0] in UI.keys['up']:
            keys[0] = 'up'
        elif keys[0] in UI.keys['down']:
            keys[0] = 'down'
        elif len(keys[0]) == 4 and keys[0][0] == 'mouse press':
            if keys[0][1] == 4:
                keys[0] = 'up'
            elif keys[0][1] == 5:
                keys[0] = 'down'
    return keys




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




header = urwid.AttrWrap(urwid.Text("The header!", align='center'), 'header')
footer = urwid.AttrWrap(urwid.Text("The footer"), 'footer')


items = []
for i in range(100):
    items.append(ItemWidget(i, "Item {0}".format(i)))

browser_header = urwid.AttrMap(urwid.Text('selected:'), 'head')
browser_listbox = urwid.ListBox(urwid.SimpleListWalker(items))
browser_view = urwid.Frame(urwid.AttrWrap(browser_listbox, 'body'), header=browser_header)

blobs = BlobWidget()
footer = urwid.Columns([urwid.Text('Info\nSecond kube'), blobs])

main_frame = urwid.AttrWrap(urwid.Frame(browser_view, focus_part='body'), 'default')
main_frame.set_header(header)
main_frame.set_footer(footer)

loop = urwid.MainLoop(main_frame, UI.palette,
                      input_filter=filter_input,
                      unhandled_input=handle_input)
loop.run()
