# coding=utf-8
# Filename: widgets.py
"""
GUI elements for the pipeinspector.

"""
import urwid


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


class BlobWidget(urwid.Pile):
    def __init__(self):
        self.width = 20
        self.size = (0,)
        urwid.Pile.__init__(self, [urwid.Text('', wrap='clip'), 
                                   urwid.Text('', wrap='clip'), 
                                   urwid.Text('', wrap='clip')])
    
    def draw(self):
        self.widget_list[0].set_text(".OOOOOOOOOOOOOOOOOOOO")
        self.widget_list[1].set_text("|    '    |    '    |")
        self.widget_list[2].set_text("0         10        20")

    def render(self, size, focus):
        self.size = size
        self.draw()
        return urwid.Pile.render(self, size, focus)



