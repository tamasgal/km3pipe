#!/usr/bin/env python

import curses 

screen = curses.initscr()
screen.immedok(True)

try:
    screen.border(0)

    box1 = curses.newwin(20, 20, 5, 5)
    box1.immedok(True)

    box1.box()    
    box1.addstr("Hello World of Curses!")

    screen.getch()

finally:
    curses.endwin()
