import curses


class PipeInspector(object):
    def __init__(self, stdscr):
        print("Initialising PipeInspector")
        self.stdscr = stdscr
        self.max_row = None
        self.max_col = None
        self.origin = (0, 0)

        self._update_dimensions()

        self._windows = []
        self._pads = []

        self._setup_curses()
        self._create_windows()
        # self._create_pads()
        self.refresh()

    def test(self):
        self.stdscr.addstr("Pretty text", curses.color_pair(1))

    def run(self):
        self.refresh()
        while True:
            try:
                c = self.stdscr.getkey()
                if c == "q":
                    raise SystemExit
            except curses.error:
                self.refresh()

    def refresh(self):
        # resize = curses.is_term_resized(self.max_row, self.max_col)
        # if resize:
        #    self._update_dimensions()
        #    self.stdscr.clear()
        #    curses.resizeterm(self.max_row, self.max_col)
        for pad in self._pads:
            pad.refresh(0, 0, 1, 1, self.max_row - 1, self.max_col - 1)
        for window in self._windows:
            window.box()
            window.refresh()
        self.stdscr.refresh()

    def _setup_curses(self):
        curses.noecho()
        curses.cbreak()
        self.stdscr.keypad(True)
        curses.init_pair(1, curses.COLOR_RED, curses.COLOR_WHITE)
        self.stdscr.border(0)
        self.stdscr.immedok(True)

    def _update_dimensions(self):
        self.max_row, self.max_col = self.stdscr.getmaxyx()
        self.max_row -= 1
        self.max_col -= 1

    def _create_windows(self):
        x = 20
        y = 7
        height = 5
        width = 40
        win = curses.newwin(height, width, y, x)
        win.box()
        win.immedok(True)
        self._windows.append(win)

    def _create_pads(self):
        pad = curses.newpad(100, 100)
        pad.box()
        pad.immedok(True)
        for y in range(0, 100):
            for x in range(0, 100):
                try:
                    pad.addch(y, x, ord("a") + (x * x + y * y) % 26)
                except curses.error:
                    pass
        self._pads.append(pad)


def main(stdscr):
    stdscr.clear()
    pipe_inspector = PipeInspector(stdscr)
    pipe_inspector.run()


if __name__ == "__main__":
    curses.wrapper(main)
