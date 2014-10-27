import urwid

from pipeinspector.widgets import BlobBrowser, BlobWidget


def handle_input(input):
    """Handle any unhandled input."""
    if input in UI.keys['escape']:
        raise urwid.ExitMainLoop

def filter_input(keys, raw):
    """Adds fancy mouse wheel functionality and VI navigation to ListBox"""
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
        ('body', 'dark cyan', '', 'standout'),
        ('focus', 'dark red', '', 'standout'),
        ('head', 'light red', 'black'),
        ('blob', 'yellow', 'dark cyan'),
        ('blob_selected', 'dark cyan', 'yellow'),
        ('blob_scale', 'dark cyan', 'black'),
        ]

    keys = {
        'select': ('return', 'enter'),
        'inspect': ('x', 'X'),
        'escape': ('esc', 'q', 'Q'),
        'left': ('left', 'h'),
        'right': ('right', 'l'),
        'up': ('up', 'k'),
        'down': ('down', 'j'),
        'home': ('0', '^'),
        'end': ('$',),
        'goto': ('g', 'G'),
        'help': ('?',),
        }


class MainFrame(urwid.Frame):
    """
    Represents the main GUI

    """
    def __init__(self):
        self.header = urwid.AttrWrap(urwid.Text("The header!", align='center'),
                                     'header')

        self.blob_browser = BlobBrowser()

        self.info_area = urwid.Text('')
        self.blobs = BlobWidget()
        self.footer = urwid.Columns([self.info_area, self.blobs])

        self.frame = urwid.AttrWrap(urwid.Frame(self.blob_browser,
                                                header=self.header,
                                                footer=self.footer), 'default')
        urwid.Frame.__init__(self, self.frame)
        self.overlay = None

        urwid.connect_signal(self.blobs, 'blob_selected', self.blob_selected)
        self.blobs.goto_blob(0)

    def blob_selected(self, index):
        self.info_area.set_text("Blob: {0}".format(index))
        self.blob_browser.load_items()

    def keypress(self, size, key):
        input = urwid.Frame.keypress(self, size, key)
        if input is None:
            return
        if input in UI.keys['left']:
            self.blobs.previous_blob()
        elif input in UI.keys['right']:
            self.blobs.next_blob()
        elif input in [key.upper() for key in UI.keys['left']]:
            self.blobs.previous_blob(step=10)
        elif input in [key.upper() for key in UI.keys['right']]:
            self.blobs.next_blob(step=10)
        elif input in UI.keys['home']:
            self.blobs.goto_blob(0)
        else:
            return self.body.keypress(size, input)


def main():
    loop = urwid.MainLoop(MainFrame(), UI.palette,
                      input_filter=filter_input,
                      unhandled_input=handle_input)
    loop.run()


if __name__ == '__main__':
    main()