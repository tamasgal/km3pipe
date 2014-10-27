__author__ = 'tamasgal'


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
