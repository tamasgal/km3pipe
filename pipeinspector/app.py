# coding=utf-8
# Filename: app.py
"""
PipeInspector

"""
import urwid

from pipeinspector.gui import MainFrame
from pipeinspector.settings import UI

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





def main():
    loop = urwid.MainLoop(MainFrame(), UI.palette,
                      input_filter=filter_input,
                      unhandled_input=handle_input)
    loop.run()


if __name__ == '__main__':
    main()