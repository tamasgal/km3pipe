#!/bin/bash
BUILD_CMD="python setup.py build_ext --inplace"
${BUILD_CMD}
pip install -U pytest-watch
ptw --ext=.py,.pyx,.so --beforerun "${BUILD_CMD}"
