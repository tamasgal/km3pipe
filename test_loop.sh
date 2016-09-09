#!/bin/bash
pip install -U pytest-watch
ptw --ext=.py,.pyx --beforerun "python setup.py build_ext --inplace"
