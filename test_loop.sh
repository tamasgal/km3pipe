#!/bin/bash
BUILD_CMD="python setup.py build_ext --inplace"
${BUILD_CMD}
# pip install -U pytest-watch
py.test
ptw --ext=.py,.pyx --beforerun "${BUILD_CMD}"
