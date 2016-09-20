#!/bin/bash
pip install -U pytest-watch
ptw --ext=.py,.pyx,.so --beforerun "python setup.py build_ext --inplace"
